"""Questions with known answers, checked after every change.

A canary is the cheapest possible regression test and the only one available at
runtime: a question the mind could already answer, and the answer it gave. If a
newly learned rule changes one, the rule broke something, and it does not
matter how good it looked.

What makes this work is that canaries are *derived from what the mind already
does*, not written separately. :meth:`CanarySet.capture` asks a set of goals
and records what came back; from then on any change that alters one of those
answers is caught. Phase One's own test cases are the first canaries.

Two properties are worth being explicit about.

**A canary checks the answer, not the proof.** A rule that reaches the same
conclusion by a better route is not a regression. A rule that reaches a
different conclusion is, whatever its justification.

**Silence counts.** An answer that was ``unknown`` and is now ``yes`` is a
change, and often the most dangerous kind -- it is what an over-general induced
rule does. So canaries record the status, all three values of it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.parser import parse_atom
from ..core.terms import Atom
from .layers import DEFAULT_LAYERS, LayerStack

__all__ = ["Canary", "CanarySet", "CanaryFailure"]


@dataclass(frozen=True)
class Canary:
    """One question and the answer it is expected to keep giving."""

    goal: Atom
    status: str
    #: Free text: why this one is worth watching.
    note: str = ""

    def __str__(self) -> str:
        return f"{self.goal} = {self.status}" + (f"  ({self.note})" if self.note else "")


@dataclass(frozen=True)
class CanaryFailure:
    """A canary that stopped giving its answer."""

    canary: Canary
    got: str

    def describe(self) -> str:
        return f"{self.canary.goal}: expected {self.canary.status}, got {self.got}"

    def __str__(self) -> str:
        return self.describe()


class CanarySet:
    """A set of canaries, and the check that runs them all."""

    def __init__(self, canaries: Iterable[Canary] = ()) -> None:
        self._canaries: list[Canary] = list(canaries)

    def __len__(self) -> int:
        return len(self._canaries)

    def __iter__(self):
        return iter(self._canaries)

    def add(self, goal, status: str, note: str = "") -> Canary:
        canary = Canary(_as_atom(goal), status, note)
        self._canaries.append(canary)
        return canary

    @classmethod
    def capture(
        cls,
        stack: LayerStack,
        goals: Iterable,
        layers: Sequence[str] = DEFAULT_LAYERS,
        note: str = "",
    ) -> "CanarySet":
        """Record what the mind currently answers, and freeze it.

        Derived from behaviour rather than written by hand, which is what makes
        it practical to have a lot of them.
        """
        engine = stack.engine(layers)
        found = [
            Canary(_as_atom(goal), engine.ask(_as_atom(goal), explain_answer=False).status, note)
            for goal in goals
        ]
        return cls(found)

    def check(
        self, stack: LayerStack, layers: Sequence[str] = DEFAULT_LAYERS
    ) -> list[CanaryFailure]:
        """Every canary that no longer gives its answer. Empty means healthy."""
        if not self._canaries:
            return []
        try:
            engine = stack.engine(layers)
        except Exception as exc:
            # A theory that will not even compile fails every canary, which is
            # the correct verdict and more useful than an exception here.
            return [CanaryFailure(c, f"the theory did not compile: {exc}") for c in self._canaries]
        failures = []
        for canary in self._canaries:
            try:
                got = engine.ask(canary.goal, explain_answer=False).status
            except Exception as exc:
                got = f"error: {exc}"
            if got != canary.status:
                failures.append(CanaryFailure(canary, got))
        return failures

    def healthy(self, stack: LayerStack, layers: Sequence[str] = DEFAULT_LAYERS) -> bool:
        return not self.check(stack, layers)

    def to_dict(self) -> dict:
        return {
            "count": len(self._canaries),
            "canaries": [str(c) for c in self._canaries],
        }


def _as_atom(goal) -> Atom:
    return goal if isinstance(goal, Atom) else parse_atom(goal)
