"""Agency: deciding, not just answering.

Everything before this phase answered questions. This package is the part
that does something about them -- and the gap between the two is wider than
it looks, because an answer that is wrong is a wrong answer, while an action
that is wrong has already happened.

Four pieces:

:mod:`~neuralmind.agency.actions`
    What an action needs and what it changes, in a small block syntax.
    Fluents are inferred from the effects, so nobody declares what changes.
:mod:`~neuralmind.agency.domain`
    The world the actions act on: fixed facts, what holds now, what is wanted.
:mod:`~neuralmind.agency.planner`
    Incremental search with clingo, growing the horizon one step at a time,
    under a wall-clock budget, keeping "no plan of this length" apart from
    "ran out of time".
:mod:`~neuralmind.agency.execute`
    Carrying a plan out and noticing when a surprise has actually broken it.

:class:`Agency` wires them to the safety kernel, which is the part that
matters most here. A plan is a proposal, not a permission: every step goes
through the autonomy gate before it comes back as something to do, and the
level each step needs is **inferred from whether the library can undo it**
rather than declared. An action nothing can reverse is one a person should
agree to, and the mind can work that out by reading its own action library
instead of being told.

That inference only ever adds caution. It cannot raise a step's autonomy
above what the host granted, for the same reason stakes cannot (design rule
11): guessing wrong about reversibility should be annoying, never dangerous.

Usage::

    agency = Agency()
    agency.learn_actions('''
    action open_valve(V):
        needs  valve(V), closed(V)
        causes open(V), -closed(V)
    ''')
    agency.observe(["valve(v1)", "closed(v1)"])
    choice = agency.decide("open(v1)")
    choice.brief()        # "Open valve v1 — to make open(v1) true."
    choice.plan.proof(0)  # why that step can happen
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from ..core.terms import Atom
from ..kernel.autonomy import L0, L1, L2, L3, AutonomyGate, Decision
from .actions import (
    Action,
    ActionError,
    ActionLibrary,
    Literal,
    parse_actions,
)
from .domain import Domain, Goal, State, as_atoms
from .execute import Execution, Progress, Surprise
from .plan import GOAL, Link, Plan, Step
from .planner import (
    SAT,
    TIMEOUT,
    UNSAT,
    PlanningError,
    Planner,
    possible_actions,
    to_asp,
)

__all__ = [
    "Agency",
    "Choice",
    # actions
    "Action", "ActionLibrary", "ActionError", "Literal", "parse_actions",
    # world
    "Domain", "State", "Goal", "as_atoms",
    # planning
    "Planner", "PlanningError", "to_asp", "SAT", "UNSAT", "TIMEOUT",
    "Plan", "Step", "Link", "GOAL",
    # doing
    "Execution", "Progress", "Surprise",
    # levels, re-exported so a caller need not reach into the kernel
    "L0", "L1", "L2", "L3",
]

#: What a step needs before it may happen, when nothing says otherwise. A
#: reversible step is a suggestion the mind may act on; one with no way back
#: waits for a person.
REVERSIBLE_NEEDS = L1
IRREVERSIBLE_NEEDS = L2


@dataclass
class Choice:
    """A plan, plus what the mind is allowed to do about it."""

    plan: Plan
    decisions: list[Decision] = field(default_factory=list)
    goal: Goal = field(default_factory=Goal)

    @property
    def found(self) -> bool:
        return self.plan.found

    @property
    def allowed(self) -> list[Step]:
        """The leading run of steps the mind may carry out by itself.

        A run, not a set: a plan is ordered, so permission to do step 3
        without permission for step 2 is permission to do nothing. Stopping
        at the first blocked step is what makes this honest.
        """
        out: list[Step] = []
        for step, decision in zip(self.plan.steps, self.decisions):
            if not decision:
                break
            out.append(step)
        return out

    @property
    def blocked(self) -> Optional[tuple[Step, Decision]]:
        """The first step the mind may not take alone, if there is one."""
        for step, decision in zip(self.plan.steps, self.decisions):
            if not decision:
                return step, decision
        return None

    @property
    def needs_person(self) -> bool:
        return self.blocked is not None

    def brief(self) -> str:
        """One line: what to do, and who has to agree to it."""
        line = self.plan.brief()
        blocked = self.blocked
        if blocked is None:
            return line
        step, decision = blocked
        if not self.allowed:
            return f"{line} I need you to confirm {step.action} first."
        return f"{line} I can do the first {len(self.allowed)}; {step.action} needs you."

    def describe(self) -> str:
        lines = [self.plan.describe()]
        if self.decisions:
            lines.append("")
            lines.append("what I may do:")
            for step, decision in zip(self.plan.steps, self.decisions):
                mark = "ok" if decision else "needs a person"
                lines.append(f"  {str(step.action):28} {mark}  ({decision.reason})")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "plan": self.plan.to_dict(),
            "allowed": [str(s.action) for s in self.allowed],
            "needs_person": self.needs_person,
            "decisions": [d.to_dict() for d in self.decisions],
            "brief": self.brief(),
        }

    def __bool__(self) -> bool:
        return self.found

    def __str__(self) -> str:
        return self.describe()


class Agency:
    """Actions, a world, a planner and the gate that decides what may happen.

    The gate is not optional and not bypassable from here: :meth:`decide`
    always runs every step through it. A caller who wants the raw plan can
    read :attr:`Choice.plan`, which is honest -- the plan is public, the
    permission is separate, and nothing pretends a plan is an authorisation.
    """

    def __init__(
        self,
        library: Optional[ActionLibrary] = None,
        gate: Optional[AutonomyGate] = None,
        planner: Optional[Planner] = None,
        state: Optional[State] = None,
    ) -> None:
        self.library = library if library is not None else ActionLibrary()
        self.gate = gate if gate is not None else AutonomyGate()
        self.planner = planner if planner is not None else Planner()
        self.state = state if state is not None else State()
        self.rules: tuple = ()

    # -- what it knows how to do -------------------------------------------

    def learn_actions(self, source: Union[str, ActionLibrary]) -> "Agency":
        """Take on a set of action definitions."""
        library = parse_actions(source) if isinstance(source, str) else source
        for action in library:
            self.library.add(action)
        for predicate in library.fluents - self.library.fluents:
            self.library.declare_fluent(predicate)
        # A state read before the actions were known split its facts with the
        # wrong idea of what changes, so it is re-sorted rather than kept.
        if len(self.state):
            self.state = State.of(self.state.atoms, self.library)
        return self

    def learn_state_rules(self, source: str) -> "Agency":
        """Take on what follows from a state, as opposed to what changes it."""
        from .domain import parse_rules

        parsed = parse_rules(source)
        self.rules = self.rules + parsed
        for rule in parsed:
            if rule.head is not None:
                self.library.declare_fluent(rule.head.predicate)
        return self

    def observe(self, facts: Iterable[Union[str, Atom]]) -> "Agency":
        """Replace what is believed to hold right now."""
        self.state = State.of(facts, self.library)
        return self

    # -- deciding -----------------------------------------------------------

    def domain(self, goal: Union[str, Atom, Iterable, None] = None) -> Domain:
        return Domain(
            self.library,
            self.state,
            Goal.of(goal) if goal is not None else Goal(),
            self.rules,
        )

    def authority_for(self, action: Action) -> int:
        """What level this action needs: declared, or read off reversibility."""
        if action.authority is not None:
            return action.authority
        return (
            REVERSIBLE_NEEDS
            if self.library.reversible(action)
            else IRREVERSIBLE_NEEDS
        )

    def possible(self) -> list:
        """Ground actions whose preconditions hold in the current state."""
        from .planner import possible_actions

        return possible_actions(self.domain())

    def decide(
        self,
        goal: Union[str, Atom, Iterable[Union[str, Atom]]],
        confirmed: Iterable[str] = (),
    ) -> Choice:
        """Work out what to do, and how much of it may be done unsupervised.

        ``confirmed`` names actions a person has already agreed to, by name
        or by their ground form, so a second call after a confirmation gets
        further rather than asking again.
        """
        target = Goal.of(goal)
        plan = self.planner.plan(self.domain(), target)
        agreed = set(confirmed)
        decisions = [
            self.gate.check(
                str(step.action),
                needs=self.authority_for(step.schema) if step.schema else L2,
                confirmed=str(step.action) in agreed or step.name in agreed,
            )
            for step in plan.steps
        ]
        return Choice(plan=plan, decisions=decisions, goal=target)

    def execution(
        self, goal: Union[str, Atom, Iterable[Union[str, Atom]]]
    ) -> Execution:
        """An execution that replans when the world breaks the plan."""
        return Execution(self.domain(goal), planner=self.planner)

    # -- reporting ----------------------------------------------------------

    def self_report(self) -> str:
        irreversible = self.library.irreversible()
        parts = [
            f"{len(self.library)} action(s), "
            f"{len(self.library.fluents)} thing(s) that change."
        ]
        if irreversible:
            names = ", ".join(a.name for a in irreversible)
            parts.append(f"No way back from: {names}.")
        parts.append(self.gate.describe() + ".")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "library": self.library.summary(),
            "irreversible": [a.name for a in self.library.irreversible()],
            "state": self.state.describe(),
            "autonomy": self.gate.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"Agency({len(self.library)} action(s), {len(self.state)} fact(s), "
            f"{self.gate.level})"
        )
