"""A plan, and why each step of it is there.

A list of actions is not an answer to "what should I do" -- it is an answer
you have to take on trust. What makes a plan inspectable is the *causal
link*: this step is here because it makes ``closed(v1)`` false, which the
next step needs, which is how the goal gets met. Links are computed from the
plan itself by simulating it forward, so they cannot disagree with it; the
same property the proof trees have over the rules.

A step nothing depends on is a step with no reason to exist, and
:meth:`Plan.check` reports it rather than letting it ride. That has caught
more encoding mistakes than any other check here: an action that seems
necessary and is not usually means a precondition was written too weakly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from ..core.terms import Atom
from ..inference.proof import ACTION, DERIVED, FACT, ProofNode
from .actions import Action, Literal
from .domain import Goal, State

__all__ = ["Plan", "Step", "Link", "GOAL"]

#: The consumer of a link that feeds the goal rather than a later step.
GOAL = -1


@dataclass(frozen=True)
class Link:
    """One step's effect, consumed by a later step or by the goal.

    ``fluent`` is what was established; ``producer`` made it true and
    ``consumer`` needs it. ``consumer == GOAL`` means it is part of the
    answer rather than scaffolding for it.
    """

    fluent: Atom
    producer: int
    consumer: int
    #: True when the producer made the fluent *stop* holding.
    removes: bool = False

    @property
    def feeds_goal(self) -> bool:
        return self.consumer == GOAL

    def describe(self, steps: Sequence["Step"]) -> str:
        if self.feeds_goal:
            if self.removes:
                return f"clears {self.fluent}, which the goal asks to be false"
            return f"gives {self.fluent}, which the goal asks for"
        consumer = steps[self.consumer].action
        if self.removes:
            return f"clears {self.fluent}, which {consumer} needs out of the way"
        return f"gives {self.fluent}, which {consumer} needs"


@dataclass(frozen=True)
class Step:
    """One action in a plan, ground, at a known position."""

    index: int
    action: Atom
    schema: Optional[Action] = None
    #: The ground fluents that had to hold for it.
    needs: tuple[Atom, ...] = ()
    #: The ground fluents that had to *not* hold for it. Kept separately
    #: because a step whose whole job is to clear one of these is a real
    #: step with a real reason, and a plan that only tracks positive
    #: preconditions calls it redundant.
    needs_absent: tuple[Atom, ...] = ()
    adds: tuple[Atom, ...] = ()
    cancels: tuple[Atom, ...] = ()

    @property
    def name(self) -> str:
        return self.action.predicate

    @property
    def cost(self) -> int:
        return self.schema.cost if self.schema is not None else 1

    @property
    def note(self) -> str:
        return self.schema.note if self.schema is not None else ""

    def __str__(self) -> str:
        return str(self.action)


@dataclass
class Plan:
    """An ordered list of steps, the links that justify them, and the state
    each one leaves behind."""

    steps: list[Step] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    #: The state after every step, so ``trace[i]`` is the world before step i.
    trace: list[State] = field(default_factory=list)
    goal: Goal = field(default_factory=Goal)
    #: Why planning stopped, when it stopped without a plan.
    reason: str = ""
    #: How far the horizon was pushed before this came back.
    horizon: int = 0
    seconds: float = 0.0
    #: True when the goal was already met and nothing needed doing.
    already: bool = False

    # -- the shape of it ----------------------------------------------------

    @property
    def found(self) -> bool:
        return self.already or bool(self.steps)

    @property
    def cost(self) -> int:
        return sum(step.cost for step in self.steps)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self) -> Iterator[Step]:
        return iter(self.steps)

    def __bool__(self) -> bool:
        return self.found

    # -- why a step is there ------------------------------------------------

    def reasons(self, index: int) -> list[Link]:
        """The links this step produces: everything it is there for."""
        return [link for link in self.links if link.producer == index]

    def support(self, index: int) -> list[Link]:
        """The links this step consumes: what had to happen first."""
        return [link for link in self.links if link.consumer == index]

    def why(self, index: int) -> str:
        """One line saying what this step is for."""
        produced = self.reasons(index)
        if not produced:
            return "nothing later depends on it"
        return "; ".join(link.describe(self.steps) for link in produced)

    def proof(self, index: int) -> ProofNode:
        """Why this step can happen: its preconditions, back to the start.

        Every child is either a fact that was true before the plan began or
        an earlier step that made it true, so the tree bottoms out in the
        observed world rather than in an assumption.
        """
        step = self.steps[index]
        node = ProofNode(
            conclusion=step.action,
            kind=ACTION,
            rule_instance=step.schema.describe() if step.schema else None,
            rule_source=step.schema.source if step.schema else None,
            rule_label=step.note or None,
        )
        for need in step.needs:
            producer = self._producer_of(need, before=index)
            if producer is None:
                node.children.append(
                    ProofNode(conclusion=need, kind=FACT, rule_label="observed")
                )
            else:
                child = self.proof(producer)
                node.children.append(
                    ProofNode(
                        conclusion=need,
                        kind=DERIVED,
                        rule_instance=f"{self.steps[producer].action} causes {need}",
                        children=[child],
                    )
                )
        for absent in step.needs_absent:
            remover = self._remover_of(absent, before=index)
            if remover is None:
                node.children.append(
                    ProofNode(
                        conclusion=absent, kind=FACT, negated=True,
                        rule_label="never held",
                    )
                )
            else:
                node.children.append(
                    ProofNode(
                        conclusion=absent,
                        kind=DERIVED,
                        negated=True,
                        rule_instance=f"{self.steps[remover].action} cancels {absent}",
                        children=[self.proof(remover)],
                    )
                )
        return node

    def proofs(self) -> list[ProofNode]:
        return [self.proof(i) for i in range(len(self.steps))]

    def _remover_of(self, fluent: Atom, before: int) -> Optional[int]:
        """The latest step before ``before`` that cleared ``fluent``."""
        for index in range(before - 1, -1, -1):
            if fluent in self.steps[index].cancels:
                return index
            if fluent in self.steps[index].adds:
                return None
        return None

    def _producer_of(self, fluent: Atom, before: int) -> Optional[int]:
        """The latest step at or before ``before`` that established ``fluent``."""
        for index in range(before - 1, -1, -1):
            if fluent in self.steps[index].adds:
                return index
            if fluent in self.steps[index].cancels:
                return None  # it was removed and must have come back another way
        return None

    # -- checking it --------------------------------------------------------

    def check(self) -> list[str]:
        """Complaints about the plan, as plain sentences. Empty is good."""
        problems = []
        for index, step in enumerate(self.steps):
            if not self.reasons(index):
                problems.append(
                    f"step {index + 1} ({step.action}) has no reason: nothing "
                    "later needs what it does, and the goal does not ask for it"
                )
        if self.trace and self.goal and not self.goal.satisfied_by(self.trace[-1]):
            missing = ", ".join(str(a) for a in self.goal.missing(self.trace[-1]))
            problems.append(f"the plan ends without {missing}")
        return problems

    # -- saying it ----------------------------------------------------------

    def describe(self) -> str:
        if self.already:
            return "nothing to do: it is already true"
        if not self.steps:
            return f"no plan: {self.reason or 'none found'}"
        lines = [
            f"{len(self.steps)} step(s), cost {self.cost}, "
            f"found at horizon {self.horizon} in {self.seconds * 1000:.0f}ms"
        ]
        for index, step in enumerate(self.steps):
            lines.append(f"  {index + 1}. {step.action}")
            lines.append(f"       {self.why(index)}")
        return "\n".join(lines)

    def brief(self) -> str:
        """One line a person can act on."""
        if self.already:
            return "Nothing to do — it is already true."
        if not self.steps:
            return f"No plan — {self.reason or 'none found'}."
        first = self.steps[0]
        rest = len(self.steps) - 1
        tail = f", then {rest} more step{'s' if rest != 1 else ''}" if rest else ""
        return f"{_sentence(first.action)}{tail} — to make {self.goal.describe()} true."

    def to_dict(self) -> dict:
        return {
            "found": self.found,
            "steps": [
                {
                    "index": index + 1,
                    "action": str(step.action),
                    "why": self.why(index),
                    "needs": [str(n) for n in step.needs],
                }
                for index, step in enumerate(self.steps)
            ],
            "cost": self.cost,
            "horizon": self.horizon,
            "seconds": round(self.seconds, 4),
            "goal": self.goal.describe(),
            "reason": self.reason,
            "brief": self.brief(),
        }

    def __str__(self) -> str:
        return self.describe()


def build_links(
    steps: Sequence[Step],
    goal: Goal,
    start: State,
    trace: Optional[Sequence[State]] = None,
) -> list[Link]:
    """Work out what each step is for, by following what it produces.

    A precondition is satisfied by the latest earlier step that established
    it, or by the starting world. Goal literals are satisfied by the latest
    step that established them, full stop -- a goal already true at the start
    needs no step and produces no link.

    ``trace`` lets a step take credit for something it did not literally add.
    A state constraint means a step can make ``in_front(key)`` true by moving,
    without ``in_front`` appearing anywhere in its effects, and a link builder
    that only reads effects calls that step purposeless. Watching the state
    instead of the effect list is the difference between explaining what the
    plan does and explaining what its actions were written to do.
    """
    links: list[Link] = []

    def became_true(fluent: Atom, before: int) -> Optional[int]:
        if trace is None:
            return None
        for index in range(min(before, len(trace) - 1) - 1, -1, -1):
            if fluent in trace[index + 1].fluents and fluent not in trace[index].fluents:
                return index
        return None

    def producer(fluent: Atom, before: int) -> Optional[int]:
        for index in range(before - 1, -1, -1):
            if fluent in steps[index].adds:
                return index
            if fluent in steps[index].cancels:
                return None
        return became_true(fluent, before)

    def remover(fluent: Atom, before: int) -> Optional[int]:
        for index in range(before - 1, -1, -1):
            if fluent in steps[index].cancels:
                return index
            if fluent in steps[index].adds:
                return None
        return None

    for index, step in enumerate(steps):
        for need in step.needs:
            source = producer(need, index)
            if source is not None:
                links.append(Link(need, source, index))
        for absent in step.needs_absent:
            # Only a link if something actually had to clear it. A fluent
            # that was never true needed no help being false.
            if not _was_ever_true(absent, steps, index, start):
                continue
            source = remover(absent, index)
            if source is not None:
                links.append(Link(absent, source, index, removes=True))

    horizon = len(steps)
    for wanted in goal.wanted:
        source = producer(wanted, horizon)
        if source is not None:
            links.append(Link(wanted, source, GOAL))
    for unwanted in goal.unwanted:
        source = remover(unwanted, horizon)
        if source is not None:
            links.append(Link(unwanted, source, GOAL, removes=True))

    return links


def _was_ever_true(fluent: Atom, steps: Sequence[Step], before: int, start: State) -> bool:
    if fluent in start.fluents:
        return True
    return any(fluent in steps[i].adds for i in range(before))


def _sentence(atom: Atom) -> str:
    """``open_valve(v1)`` -> ``Open valve v1``, without inventing grammar."""
    words = atom.predicate.replace("_", " ")
    args = " ".join(str(a) for a in atom.args)
    text = f"{words} {args}".strip()
    return text[:1].upper() + text[1:]
