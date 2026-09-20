"""Learning what an action really needs, by failing to do it.

An action model written by hand is a guess about the world, and the world
answers. ``toggle`` opens a door -- until a door is locked, and then the
agent stands in front of it toggling and nothing happens. The information is
right there in the transition: the action was possible by the model, it
happened, and the state did not change the way the model said it would.

What is learned here is **preconditions**, not effects, and that is a
deliberate limit. A missing precondition shows up as a failure, which is
observable; a missing effect shows up as something true that nobody
predicted, which is also observable but much easier to get wrong, because
anything else in the world may have caused it. Effects are P2.8's problem,
where there is a developer to ask.

The method is the same one P2.3 uses for rules, narrowed to a shape where it
is cheap. A candidate precondition is a fluent that held in every attempt
that worked and failed to hold in at least one that did not, lifted so that
it talks about the action's parameters rather than about the particular
door. Candidates are ranked by how many failures they explain and how much
evidence stands behind them, and nothing is proposed from a single
observation -- an action seen to fail once has told you almost nothing.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.terms import Atom, Const, Term, Var
from .actions import Action, ActionLibrary, Literal
from .domain import State

__all__ = ["ActionLearner", "Attempt", "Proposal", "MIN_EVIDENCE"]

#: Successes needed before anything is proposed.
#:
#: One, not two, and the reason is :func:`_lifted`: a candidate has to be a
#: fact about the action's own arguments, so a single success and a single
#: failure already leave only a handful of differences rather than the whole
#: world. Zero successes is still nothing -- an action that has only ever
#: failed has told you it does not work, not why.
MIN_EVIDENCE = 1


@dataclass(frozen=True)
class Attempt:
    """One try at one action, and whether it did what the model promised."""

    action: Atom
    before: State
    after: State
    worked: bool

    @property
    def name(self) -> str:
        return self.action.predicate


@dataclass(frozen=True)
class Proposal:
    """A precondition the evidence says is missing, and what backs it."""

    action: str
    literal: Literal
    #: Failures this would have prevented.
    explains: int
    #: Successful attempts in which it held, i.e. that it does not break.
    supported_by: int
    #: Failures it does not account for.
    leaves: int

    @property
    def complete(self) -> bool:
        """True when this alone accounts for every failure seen."""
        return self.leaves == 0

    def describe(self) -> str:
        state = "explains every failure" if self.complete else f"leaves {self.leaves}"
        return (
            f"{self.action} also needs {self.literal} "
            f"({self.explains} failure(s) explained, {state}; "
            f"held in {self.supported_by} success(es))"
        )

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "needs": str(self.literal),
            "explains": self.explains,
            "supported_by": self.supported_by,
            "leaves": self.leaves,
            "complete": self.complete,
        }

    def __str__(self) -> str:
        return self.describe()


class ActionLearner:
    """Watches attempts and works out what the action model left out."""

    def __init__(self, library: ActionLibrary, min_evidence: int = MIN_EVIDENCE) -> None:
        self.library = library
        self.min_evidence = min_evidence
        self.attempts: list[Attempt] = []

    # -- watching -----------------------------------------------------------

    def record(
        self, action: Atom, before: State, after: State, worked: Optional[bool] = None
    ) -> Attempt:
        """Note one attempt. ``worked`` is judged from the state if not given."""
        if worked is None:
            worked = self._did_what_it_said(action, before, after)
        attempt = Attempt(action, before, after, worked)
        self.attempts.append(attempt)
        return attempt

    def _did_what_it_said(self, action: Atom, before: State, after: State) -> bool:
        """Whether the modelled effects actually appeared."""
        schema = self.library.get(action.predicate, len(action.args))
        if schema is None:
            return False
        binding = dict(zip((v.name for v in schema.params), action.args))
        for effect in schema.adds:
            if _ground(effect, binding) not in after.fluents:
                return False
        for effect in schema.cancels:
            if _ground(effect, binding) in after.fluents:
                return False
        return True

    # -- concluding ---------------------------------------------------------

    def propose(self, action: Optional[str] = None) -> list[Proposal]:
        """What the evidence says is missing, best first."""
        proposals: list[Proposal] = []
        by_action: dict[str, list[Attempt]] = defaultdict(list)
        for attempt in self.attempts:
            by_action[attempt.name].append(attempt)

        for name, attempts in by_action.items():
            if action is not None and name != action:
                continue
            proposals += self._for_action(name, attempts)
        proposals.sort(
            key=lambda p: (p.complete, p.explains, p.supported_by), reverse=True
        )
        return proposals

    def _for_action(self, name: str, attempts: Sequence[Attempt]) -> list[Proposal]:
        successes = [a for a in attempts if a.worked]
        failures = [a for a in attempts if not a.worked]
        if not failures or len(successes) < self.min_evidence:
            # One failure and no successes is a mystery, not a lesson. Saying
            # so is better than proposing whatever happened to differ.
            return []

        schema = next(
            (
                self.library.get(name, len(a.action.args))
                for a in attempts
                if self.library.get(name, len(a.action.args)) is not None
            ),
            None,
        )
        if schema is None:
            return []

        known = {str(literal) for literal in schema.needs}
        shared = _lifted(successes[0], schema)
        for success in successes[1:]:
            shared &= _lifted(success, schema)

        proposals = []
        for atom in sorted(shared, key=str):
            if str(atom) in known:
                continue
            missing = [
                f for f in failures if atom not in _lifted(f, schema, raw=True)
            ]
            if not missing:
                continue
            proposals.append(
                Proposal(
                    action=name,
                    literal=Literal(atom),
                    explains=len(missing),
                    supported_by=len(successes),
                    leaves=len(failures) - len(missing),
                )
            )

        # The other shape: something true whenever it failed and never when it
        # worked, which is a precondition the action needs *absent*.
        blocking = _lifted(failures[0], schema)
        for failure in failures[1:]:
            blocking &= _lifted(failure, schema)
        for atom in sorted(blocking, key=str):
            if any(atom in _lifted(s, schema) for s in successes):
                continue
            literal = Literal(atom, absent=True)
            if str(literal) in known:
                continue
            proposals.append(
                Proposal(
                    action=name,
                    literal=literal,
                    explains=len(failures),
                    supported_by=len(successes),
                    leaves=0,
                )
            )
        return proposals

    def best(self, action: Optional[str] = None) -> Optional[Proposal]:
        proposals = self.propose(action)
        return proposals[0] if proposals else None

    def apply(self, proposal: Proposal) -> Action:
        """Fold a proposal into the library, returning the revised action.

        Revising rather than replacing: the action keeps its name, its
        effects and its note, and gains one condition. Anything that held a
        reference to the old schema gets the new one by looking it up, which
        is the only way a plan made under the old model can be noticed as
        stale rather than silently re-justified.
        """
        old = self.library.get(proposal.action)
        if old is None:
            raise KeyError(f"no action named {proposal.action}")
        revised = Action(
            name=old.name,
            params=old.params,
            needs=old.needs + (proposal.literal,),
            causes=old.causes,
            cost=old.cost,
            note=old.note,
            authority=old.authority,
            source=f"{old.source} + learned",
            line=old.line,
        )
        self.library.replace(revised)
        return revised

    # -- reporting ----------------------------------------------------------

    def describe(self) -> str:
        worked = sum(1 for a in self.attempts if a.worked)
        lines = [
            f"{len(self.attempts)} attempt(s): {worked} did what the model said, "
            f"{len(self.attempts) - worked} did not."
        ]
        proposals = self.propose()
        if not proposals:
            lines.append("Nothing to conclude yet.")
        else:
            lines += [f"  {p.describe()}" for p in proposals[:5]]
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self.attempts)

    def __repr__(self) -> str:
        return f"ActionLearner({len(self.attempts)} attempt(s))"


def _lifted(attempt: Attempt, schema: Action, raw: bool = False) -> set:
    """The state before an attempt, said in terms of the action's parameters.

    ``pick_up(red_key_1, 3, 2)`` in a state containing ``door_locked(d1)``
    teaches nothing about ``d1``; the same attempt in a state containing
    ``door_locked(red_key_1)`` would be about the thing being acted on. Only
    facts whose arguments are all the action's own arguments survive, which
    keeps the candidate set small and the conclusions about the action rather
    than about the room.
    """
    reverse: dict = {}
    for param, value in zip(schema.params, attempt.action.args):
        reverse.setdefault(value, param)

    lifted = set()
    for atom in attempt.before.atoms:
        if not atom.args:
            lifted.add(atom)
            continue
        if not all(arg in reverse for arg in atom.args):
            continue
        lifted.add(Atom(atom.predicate, tuple(reverse[arg] for arg in atom.args)))
    return lifted


def _ground(atom: Atom, binding: dict) -> Atom:
    args = tuple(binding.get(a.name, a) if isinstance(a, Var) else a for a in atom.args)
    return Atom(atom.predicate, args)
