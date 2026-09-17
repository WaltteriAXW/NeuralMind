"""What the mind does not know, collected from what it already tells you.

Every part of this system already reports its own failures. The engine's
``why_not()`` names the body literal a rule stalled on; perception reports the
sentences it refused; the consistency layer reports which constraint broke; a
three-valued answer says ``unknown`` and carries the diagnosis behind it. None
of that was built for learning, and all of it is exactly what a learner needs.

So a gap is not inferred. It is *collected* -- read off a diagnosis that was
produced anyway, which is why a gap always comes with the evidence for it and
never has to be justified after the fact.

Four kinds, because they call for different responses:

``missing_rule``
    A goal nothing derives, where rules exist that nearly did. Induction has
    something to work with: examples of the target are in reach.
``missing_fact``
    A rule stalled on a literal that is simply absent. Nobody needs to learn a
    rule; somebody needs to be asked.
``unreadable``
    Perception refused a sentence. The fix is a lexicon entry, not a rule.
``inconsistent``
    An integrity constraint broke, or both polarities of an atom were derived.
    Something already believed is wrong, which is a repair rather than a
    growth problem.

Ordering matters more than completeness. A mind with a hundred gaps and no
sense of which to close first will close the easy ones forever, so each gap
carries how often it has been hit: the gap standing between the user and an
answer they keep asking for is the one worth work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.terms import Atom

__all__ = [
    "Gap",
    "GapCollector",
    "MISSING_RULE",
    "MISSING_FACT",
    "UNREADABLE",
    "INCONSISTENT",
]

MISSING_RULE = "missing_rule"
MISSING_FACT = "missing_fact"
UNREADABLE = "unreadable"
INCONSISTENT = "inconsistent"


@dataclass
class Gap:
    """One thing the mind cannot do, and the evidence that it cannot."""

    kind: str
    #: What was being asked for. ``None`` for an unreadable sentence.
    goal: Optional[Atom] = None
    #: Free text: the sentence, the constraint, the literal.
    subject: str = ""
    #: What the diagnosis already established on the way.
    established: tuple[Atom, ...] = ()
    #: The specific literal that stopped it, when there was one.
    missing: Optional[Atom] = None
    #: How many times this gap has been hit.
    hits: int = 1
    #: Where it came from: ``ask``, ``perception``, ``consistency``.
    source: str = ""

    @property
    def key(self) -> str:
        """What makes two gaps the same gap."""
        return f"{self.kind}:{self.goal or self.subject}"

    def describe(self) -> str:
        times = f" (×{self.hits})" if self.hits > 1 else ""
        if self.kind == UNREADABLE:
            return f"could not read: {self.subject}{times}"
        if self.kind == INCONSISTENT:
            return f"inconsistent: {self.subject}{times}"
        if self.kind == MISSING_FACT and self.missing is not None:
            return f"nothing says whether {self.missing}{times}"
        return f"nothing derives {self.goal}{times}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "goal": str(self.goal) if self.goal else None,
            "subject": self.subject,
            "missing": str(self.missing) if self.missing else None,
            "established": [str(a) for a in self.established],
            "hits": self.hits,
            "source": self.source,
        }

    def __str__(self) -> str:
        return self.describe()


class GapCollector:
    """Gathers gaps from the diagnoses the system produces anyway."""

    def __init__(self) -> None:
        self._gaps: dict[str, Gap] = {}

    # -- collecting ---------------------------------------------------------

    def from_answer(self, answer) -> Optional[Gap]:
        """Read a gap off a three-valued answer that was not ``yes``.

        The distinction between a missing *rule* and a missing *fact* is the
        useful one, and the diagnosis already draws it: a rule that stalled on
        a literal wants that literal supplied; a goal no rule even matches
        wants a definition.
        """
        status = getattr(answer, "status", "yes" if answer.holds else "no")
        if status == "yes":
            return None
        diagnosis = getattr(answer, "diagnosis", None)
        goal = answer.query
        if diagnosis is not None and diagnosis.attempts:
            best = max(diagnosis.attempts, key=lambda a: a.progress)
            if best.missing is not None:
                return self._add(
                    Gap(
                        kind=MISSING_FACT,
                        goal=goal,
                        subject=str(best.missing),
                        established=tuple(best.established),
                        missing=best.missing,
                        source="ask",
                    )
                )
        return self._add(Gap(kind=MISSING_RULE, goal=goal, source="ask"))

    def from_perception(self, perception) -> list[Gap]:
        """Every sentence the reader refused."""
        found = []
        for sentence in getattr(perception, "unparsed", ()):
            found.append(
                self._add(
                    Gap(
                        kind=UNREADABLE,
                        subject=str(sentence).split("  (")[0],
                        source="perception",
                    )
                )
            )
        return found

    def from_model(self, model) -> list[Gap]:
        """Integrity violations and contradictions -- things already believed
        that cannot both be true."""
        found = []
        for violation in getattr(model, "violations", ()):
            found.append(
                self._add(
                    Gap(kind=INCONSISTENT, subject=violation.describe(), source="consistency")
                )
            )
        for atom in getattr(model, "contradictions", ()):
            found.append(
                self._add(
                    Gap(
                        kind=INCONSISTENT,
                        goal=atom,
                        subject=f"{atom} and {atom.complement()} are both derived",
                        source="consistency",
                    )
                )
            )
        return found

    def _add(self, gap: Gap) -> Gap:
        existing = self._gaps.get(gap.key)
        if existing is not None:
            existing.hits += 1
            if gap.established and not existing.established:
                existing.established = gap.established
            return existing
        self._gaps[gap.key] = gap
        return gap

    # -- reading ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._gaps)

    def __iter__(self):
        return iter(self.ranked())

    def ranked(self, kind: Optional[str] = None) -> list[Gap]:
        """Most-hit first: the gap between the user and an answer they keep
        asking for is the one worth work."""
        found = [g for g in self._gaps.values() if kind is None or g.kind == kind]
        return sorted(found, key=lambda g: (-g.hits, g.key))

    def clear(self, gap: Gap) -> None:
        self._gaps.pop(gap.key, None)

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for gap in self._gaps.values():
            counts[gap.kind] = counts.get(gap.kind, 0) + 1
        return {"gaps": len(self._gaps), "by_kind": dict(sorted(counts.items()))}
