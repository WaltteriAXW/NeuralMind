"""The world the actions act on: what is fixed, what holds now, what is wanted.

A :class:`Domain` is an action library plus a state. The split matters: the
library is knowledge ("a valve can be opened when it is closed") and the state
is observation ("v1 is closed right now"). The first is learned or written once
and survives; the second is replaced every time the mind looks again.

Nothing here names a domain in the "which industry is this" sense. The atoms
arrive from observations the same way every other fact does, and a valve is a
valve because something said ``valve(v1)``, not because a host declared a
sector.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from ..core.parser import parse_atom
from ..core.terms import Atom
from .actions import Action, ActionError, ActionLibrary

__all__ = ["Domain", "State", "Goal", "as_atoms", "parse_rules"]


def parse_rules(source: Union[str, Sequence]) -> tuple:
    """Read state constraints with the ordinary rule parser.

    Deliberately the same parser the knowledge base uses. A second rule
    syntax for planning would be a second thing to learn and a second place
    for the two halves of the system to disagree about what a rule means.
    """
    if not isinstance(source, str):
        return tuple(source)
    from ..core.parser import parse_program

    return tuple(parse_program(source, "state-rules", check=False).rules)


def as_atoms(items: Iterable[Union[str, Atom]]) -> tuple[Atom, ...]:
    """Accept strings or atoms; everything downstream gets atoms."""
    return tuple(parse_atom(i) if isinstance(i, str) else i for i in items)


@dataclass(frozen=True)
class State:
    """What holds at one moment: the fixed facts and the changing ones."""

    statics: frozenset[Atom] = frozenset()
    fluents: frozenset[Atom] = frozenset()

    @classmethod
    def of(
        cls,
        facts: Iterable[Union[str, Atom]],
        library: ActionLibrary,
    ) -> "State":
        """Split a flat set of observed facts into fixed and changing.

        The library decides which is which, so a host hands over what it sees
        and never has to sort it.
        """
        changing = library.fluents
        statics, fluents = set(), set()
        for atom in as_atoms(facts):
            if not atom.is_ground:
                raise ActionError(f"the state must be ground; got {atom}")
            (fluents if atom.positive.predicate in changing else statics).add(atom)
        return cls(frozenset(statics), frozenset(fluents))

    @property
    def atoms(self) -> frozenset[Atom]:
        return self.statics | self.fluents

    def holds(self, atom: Union[str, Atom]) -> bool:
        target = parse_atom(atom) if isinstance(atom, str) else atom
        return target in self.atoms

    def difference(self, other: "State") -> tuple[frozenset[Atom], frozenset[Atom]]:
        """What appeared and what went away, moving from ``self`` to ``other``.

        This is how an unexpected observation becomes a reason to replan, and
        how the builder will later learn an action's effects from watching.
        """
        return (other.fluents - self.fluents, self.fluents - other.fluents)

    def with_fluents(self, fluents: Iterable[Atom]) -> "State":
        return State(self.statics, frozenset(fluents))

    def describe(self) -> str:
        return (
            f"{len(self.statics)} fixed fact(s), {len(self.fluents)} that change"
        )

    def __len__(self) -> int:
        return len(self.atoms)


@dataclass(frozen=True)
class Goal:
    """What the mind is trying to bring about.

    A conjunction of ground literals. Strong-negated members mean "and this
    must *stop* holding", which a goal language without them cannot say --
    and "make sure the valve is not open" is a thing people ask for.
    """

    literals: tuple[Atom, ...] = ()

    @classmethod
    def of(cls, goals: Union[str, Atom, Iterable[Union[str, Atom]]]) -> "Goal":
        if isinstance(goals, (str, Atom)):
            goals = [goals]
        return cls(as_atoms(goals))

    @property
    def wanted(self) -> tuple[Atom, ...]:
        return tuple(a for a in self.literals if not a.is_negated)

    @property
    def unwanted(self) -> tuple[Atom, ...]:
        return tuple(a.positive for a in self.literals if a.is_negated)

    def satisfied_by(self, state: State) -> bool:
        return all(a in state.atoms for a in self.wanted) and not any(
            a in state.atoms for a in self.unwanted
        )

    def missing(self, state: State) -> tuple[Atom, ...]:
        """The parts not yet true -- what a partial plan still owes.

        Against the whole state, fixed facts included. A goal may perfectly
        well name something that never changes -- "the key that fits this box"
        is part of what winning means even though no action could alter it --
        and checking only the changing half reports such a goal as forever
        unmet while the plan sits there having achieved it.
        """
        return tuple(a for a in self.wanted if a not in state.atoms) + tuple(
            a for a in self.unwanted if a in state.atoms
        )

    def __len__(self) -> int:
        return len(self.literals)

    def __iter__(self):
        return iter(self.literals)

    def describe(self) -> str:
        return ", ".join(str(a) for a in self.literals) or "nothing"

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Domain:
    """Actions, the world they act on, and what is wanted of it."""

    library: ActionLibrary
    state: State
    goal: Goal = field(default_factory=Goal)
    #: Things that are true *because* other things are, rather than because
    #: an action made them so -- "I am in front of the key" follows from where
    #: I am and which way I face, and no action causes it directly. Ordinary
    #: rules, read by the ordinary parser; the planner re-derives them at
    #: every time point instead of carrying them forward, which is what makes
    #: them different from the fluents actions change.
    rules: tuple = ()

    @classmethod
    def build(
        cls,
        actions: Union[str, ActionLibrary],
        facts: Iterable[Union[str, Atom]] = (),
        goal: Union[str, Atom, Iterable[Union[str, Atom]], None] = None,
        rules: Union[str, Sequence, None] = None,
    ) -> "Domain":
        from .actions import parse_actions

        library = parse_actions(actions) if isinstance(actions, str) else actions
        parsed = parse_rules(rules) if rules else ()
        derived = {r.head.predicate for r in parsed if r.head is not None}
        for predicate in derived:
            library.declare_fluent(predicate)
        return cls(
            library=library,
            state=State.of(facts, library),
            goal=Goal.of(goal) if goal is not None else Goal(),
            rules=parsed,
        )

    @property
    def derived(self) -> set[str]:
        """Predicates the rules conclude, which inertia must not carry."""
        return {r.head.predicate for r in self.rules if r.head is not None}

    # -- the questions worth asking before planning -------------------------

    def unreachable(self) -> tuple[Atom, ...]:
        """Goal atoms no action could ever establish.

        Worth checking first, because the answer is instant and it is the
        difference between "no plan within the budget" and "no plan, ever".
        A planner that cannot tell those apart sends you looking for a longer
        horizon that does not exist.
        """
        derived = self.derived
        return tuple(
            atom
            for atom in self.goal.wanted
            if atom not in self.state.atoms
            and atom.positive.predicate not in derived
            and not self.library.establishers(atom)
        )

    def already_satisfied(self) -> bool:
        return self.goal.satisfied_by(self.state)

    def to_dict(self) -> dict:
        return {
            "actions": len(self.library),
            "state": self.state.describe(),
            "goal": self.goal.describe(),
        }

    def __repr__(self) -> str:
        return (
            f"Domain({len(self.library)} action(s), {len(self.state)} fact(s), "
            f"goal: {self.goal.describe()})"
        )
