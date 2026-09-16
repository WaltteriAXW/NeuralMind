"""Which specialist gets a goal, and in what order.

Rule-based, and deliberately so. The roadmap has this becoming a small
classifier over embeddings once there are logged routing decisions to train on;
until those exist, a learned router would be a guess wearing a confidence
score. The rule here is simply to ask each specialist what it thinks of the
goal and believe it, which works because ``accepts`` is cheap and because a
specialist that overclaims wastes budget it also has to live within.

Ordering matters more than choosing. Several specialists often have something
to say about one goal -- the units specialist knows what ``3200 n`` means, the
arithmetic specialist knows whether it is under the limit -- and the answer
comes from running them in a sensible order rather than picking a winner. So
this returns a ranked list, not a single choice, and the controller works down
it until something sticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..core.terms import Atom

__all__ = ["Router", "Candidate"]


@dataclass(frozen=True)
class Candidate:
    """One specialist's claim on a goal."""

    specialist: object
    score: float

    @property
    def name(self) -> str:
        return getattr(self.specialist, "name", type(self.specialist).__name__)


class Router:
    """Ranks specialists for a goal by what they say about it."""

    def __init__(self, specialists: Sequence[object], threshold: float = 0.05) -> None:
        self.specialists = list(specialists)
        #: Below this a specialist is not worth the call.
        self.threshold = threshold

    def rank(self, goal: Atom, workspace) -> list[Candidate]:
        scored = []
        for specialist in self.specialists:
            try:
                score = float(specialist.accepts(goal, workspace))
            except Exception:
                # A specialist that cannot even judge a goal is not one to run
                # on it, and it should not take the whole query down with it.
                continue
            if score >= self.threshold:
                scored.append(Candidate(specialist, score))
        # Ties broken by registration order, so routing is deterministic --
        # design rule 6, and the thing that makes a replay reproducible.
        scored.sort(key=lambda candidate: -candidate.score)
        return scored

    def names(self) -> list[str]:
        return [getattr(s, "name", type(s).__name__) for s in self.specialists]

    def __len__(self) -> int:
        return len(self.specialists)
