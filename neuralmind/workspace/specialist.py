"""What a specialist is, and the one rule all of them obey.

A specialist is a kind of reasoning: deduction over rules, arithmetic over
constraints, dimensional analysis over quantities, search over a graph. Each is
better at its own job than a general engine would be, and none of them is
downstream of the others -- which is why they meet on a blackboard rather than
in a pipeline.

The contract is two methods:

``accepts(goal) -> float``
    How well this specialist expects to do, from 0.0 (not mine) to 1.0. The
    controller uses it to order the agenda, so a specialist that overclaims
    wastes budget and one that underclaims never runs. Answering honestly is
    the whole of a specialist's cooperation.

``run(goal, workspace, budget) -> Result``
    Do the work and return findings. Each finding is an atom **with a proof**,
    and that is the rule that makes the architecture hold together: a result
    posted by the arithmetic solver is, to the logic engine, indistinguishable
    from a given fact -- and a proof tree drawn afterwards spans both, because
    each specialist brought its own node.

A specialist that cannot answer says so. Returning an empty result with a
reason is a first-class outcome, not a failure: "the arithmetic specialist ran
out of budget" and "no specialist accepted this" are different things and the
controller reports which.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from ..core.terms import Atom
from ..inference.proof import SPECIALIST, ProofNode

__all__ = ["Specialist", "Result", "Finding", "Budget", "specialist_proof"]


@dataclass(frozen=True)
class Finding:
    """One conclusion, and the proof that stands behind it."""

    atom: Atom
    proof: ProofNode
    confidence: float = 1.0


@dataclass
class Result:
    """What a specialist produced, and what it could not.

    ``reason`` is filled in when nothing was found. It is shown to the user
    through the brief line, so it is written for a person: "the arithmetic
    specialist is not installed" rather than an exception class name.
    """

    findings: list[Finding] = field(default_factory=list)
    #: Goals this specialist needs before it can make progress.
    subgoals: list[Atom] = field(default_factory=list)
    reason: str = ""
    #: Wall time actually spent, in milliseconds.
    elapsed_ms: float = 0.0
    #: True if the specialist stopped because the budget ran out.
    exhausted: bool = False

    def __bool__(self) -> bool:
        return bool(self.findings)

    @classmethod
    def nothing(cls, reason: str) -> "Result":
        return cls(reason=reason)


class Budget:
    """A wall-clock allowance, checked rather than enforced.

    Nothing here can interrupt a specialist mid-call; Python has no safe way to
    do that, and a thread-based timeout would leave a half-solved solver state
    behind. So a budget is cooperative: specialists check :meth:`exhausted`
    between units of work and return what they have. That is also what makes
    "anytime" answers possible -- stopping early yields a partial result rather
    than nothing.
    """

    __slots__ = ("limit_ms", "started")

    def __init__(self, limit_ms: float) -> None:
        self.limit_ms = float(limit_ms)
        self.started = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.started) * 1000.0

    @property
    def remaining_ms(self) -> float:
        return max(0.0, self.limit_ms - self.elapsed_ms)

    @property
    def exhausted(self) -> bool:
        return self.elapsed_ms >= self.limit_ms

    def child(self, share: float = 1.0) -> "Budget":
        """A sub-budget for one call, never larger than what is left."""
        return Budget(min(self.remaining_ms, self.limit_ms * share))

    def __repr__(self) -> str:
        return f"Budget({self.remaining_ms:.1f}ms of {self.limit_ms:.1f}ms left)"


@runtime_checkable
class Specialist(Protocol):
    """The protocol every specialist implements."""

    name: str

    def accepts(self, goal: Atom, workspace) -> float:
        """0.0 to 1.0: how well this specialist expects to do on ``goal``."""
        ...

    def run(self, goal: Atom, workspace, budget: Budget) -> Result:
        """Work on ``goal``, reading and posting through ``workspace``."""
        ...

    def prepare(self) -> None:
        """Do one-time setup now, outside anyone's budget. Optional.

        Importing z3 costs ~30ms and building pint's unit registry ~260ms --
        several times a realistic per-query budget, and paid exactly once. A
        query that happens to be first should not be charged for it, so the
        controller calls this before the clock starts. Implementations must be
        idempotent and must not fail when their backend is missing.
        """
        ...


def specialist_proof(
    atom: Atom,
    name: str,
    explanation: str,
    because: Optional[list[ProofNode]] = None,
    confidence: Optional[float] = None,
) -> ProofNode:
    """Build the proof node for a specialist's conclusion.

    ``explanation`` is the step in the specialist's own terms -- "3.2 kN <=
    5.0 kN", "shortest path a -> c -> e" -- and ``because`` are the blackboard
    entries it used, so the tree stays connected across specialists.
    """
    return ProofNode(
        conclusion=atom,
        kind=SPECIALIST,
        rule_instance=explanation,
        rule_source=f"specialist:{name}",
        rule_label=name,
        children=list(because or []),
        confidence=confidence,
    )
