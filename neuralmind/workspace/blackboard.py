"""The blackboard: one shared place where kinds of reasoning meet.

Phase One was a pipeline. Text went in, atoms came out, the engine ran, an
answer appeared. That shape cannot accommodate a second kind of reasoning: an
arithmetic solver and a graph search have nothing to say to each other through
a pipe, because neither is downstream of the other.

A blackboard is the classical answer, and it is the right one here for a
specific reason -- everything on it is a ground atom with a justification, so a
result posted by the arithmetic specialist is indistinguishable, to the logic
specialist, from a fact that was given. The two reason over each other's
conclusions without either knowing the other exists, and the proof tree spans
both because every entry brought its own proof.

Three things live here:

**Facts.** Ground atoms with provenance. Posting one that is already present
is a no-op, which is what makes a fixpoint over several specialists terminate.

**Goals.** What is being worked on. A specialist may post new goals -- proving
``load_ok(beam1)`` needs ``mass(beam1, M)`` first -- and the controller works
them in priority order.

**Open questions.** Things nothing could settle. These are not failures; they
are the growth loop's input (P2.3) and the honest content of an ``unknown``.

Change tracking is the part that makes it efficient. Every post is stamped with
a revision, so a specialist can ask what has appeared since it last ran instead
of re-deriving the world. That is the same idea as the semi-naive delta in the
forward chainer, one level up.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from ..core.terms import Atom
from ..inference.proof import ProofNode

__all__ = ["Workspace", "Entry", "Goal", "OpenQuestion"]


@dataclass(frozen=True)
class Entry:
    """One fact on the blackboard, and the proof that put it there."""

    atom: Atom
    proof: ProofNode
    #: Which specialist posted it, or ``"given"`` for input.
    source: str = "given"
    confidence: float = 1.0
    #: The blackboard revision at which this appeared.
    revision: int = 0

    def __str__(self) -> str:
        return f"{self.atom}  [{self.source}]"


@dataclass(frozen=True)
class Goal:
    """Something to establish, and how badly it is wanted.

    ``priority`` orders the agenda. A goal posted in service of another
    inherits a higher priority than the one that asked for it, so the
    controller finishes what it started before widening.
    """

    atom: Atom
    priority: float = 1.0
    #: The goal this one was raised for, if any.
    parent: Optional[Atom] = None

    def __str__(self) -> str:
        return str(self.atom)


@dataclass(frozen=True)
class OpenQuestion:
    """Something nothing on the blackboard could settle, and why.

    The ``reason`` matters more than the question. "No specialist accepted it"
    and "the arithmetic specialist ran out of budget" call for different
    responses, and a bare list of unanswered goals cannot tell them apart.
    """

    atom: Atom
    reason: str
    asked_by: str = ""

    def __str__(self) -> str:
        return f"{self.atom}: {self.reason}"


class Workspace:
    """Facts, goals and open questions, with a revision counter over them."""

    def __init__(self) -> None:
        self._entries: dict[Atom, Entry] = {}
        self._goals: list[Goal] = []
        self._questions: list[OpenQuestion] = []
        self.revision = 0

    # -- facts --------------------------------------------------------------

    def post(
        self,
        atom: Atom,
        proof: ProofNode,
        source: str = "given",
        confidence: float = 1.0,
    ) -> bool:
        """Add a fact. Returns False if it was already known.

        Re-posting is deliberately cheap and deliberately silent: several
        specialists deriving the same atom by different routes is normal, and
        the first proof is kept because it is the one that was reachable
        first, not because it is better.
        """
        if atom in self._entries:
            return False
        self.revision += 1
        self._entries[atom] = Entry(
            atom=atom,
            proof=proof,
            source=source,
            confidence=confidence,
            revision=self.revision,
        )
        return True

    def holds(self, atom: Atom) -> bool:
        return atom in self._entries

    def entry(self, atom: Atom) -> Optional[Entry]:
        return self._entries.get(atom)

    def proof(self, atom: Atom) -> Optional[ProofNode]:
        entry = self._entries.get(atom)
        return entry.proof if entry is not None else None

    @property
    def facts(self) -> list[Atom]:
        return list(self._entries)

    def since(self, revision: int) -> list[Entry]:
        """Everything posted after ``revision`` -- a specialist's delta."""
        return [e for e in self._entries.values() if e.revision > revision]

    def by_predicate(self, predicate: str, arity: Optional[int] = None) -> list[Atom]:
        return [
            atom
            for atom in self._entries
            if atom.predicate == predicate and (arity is None or atom.arity == arity)
        ]

    # -- goals --------------------------------------------------------------

    def want(self, atom: Atom, priority: float = 1.0, parent: Optional[Atom] = None) -> Goal:
        goal = Goal(atom=atom, priority=priority, parent=parent)
        self._goals.append(goal)
        return goal

    def take_goal(self) -> Optional[Goal]:
        """The most wanted outstanding goal, removed from the agenda."""
        pending = [g for g in self._goals if not self.holds(g.atom)]
        self._goals = pending
        if not pending:
            return None
        best = max(pending, key=lambda g: g.priority)
        self._goals.remove(best)
        return best

    @property
    def goals(self) -> list[Goal]:
        return list(self._goals)

    def clear_goals(self) -> None:
        """Drop the agenda, keeping the facts. One query's wants are its own."""
        self._goals.clear()

    # -- open questions -----------------------------------------------------

    def ask(self, atom: Atom, reason: str, asked_by: str = "") -> OpenQuestion:
        question = OpenQuestion(atom=atom, reason=reason, asked_by=asked_by)
        self._questions.append(question)
        return question

    @property
    def questions(self) -> list[OpenQuestion]:
        return list(self._questions)

    # -- views --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, atom: Atom) -> bool:
        return atom in self._entries

    def __iter__(self) -> Iterator[Atom]:
        return iter(sorted(self._entries, key=str))

    def summary(self) -> dict:
        by_source: dict[str, int] = {}
        for entry in self._entries.values():
            by_source[entry.source] = by_source.get(entry.source, 0) + 1
        return {
            "facts": len(self._entries),
            "by_source": dict(sorted(by_source.items())),
            "goals": len(self._goals),
            "open_questions": len(self._questions),
            "revision": self.revision,
        }

    def __repr__(self) -> str:
        return (
            f"Workspace(facts={len(self._entries)}, goals={len(self._goals)}, "
            f"questions={len(self._questions)})"
        )
