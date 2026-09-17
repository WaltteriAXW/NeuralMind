"""What the mind knows about itself, as facts it can reason over.

"Self-awareness" here is functional and worth describing that way to anyone
who asks: a set of ground atoms about the mind's own state, in the same
vocabulary as everything else, queried with the same engine. That is what makes
it testable, and it is the whole of the claim.

    has_specialist(arithmetic).      pack_active(inventory, 7).
    mode(full).                      budget_ms(20).
    stakes(medium).                  autonomy(l1).
    competence(logic, deduction, 98). gap(refund_policy).
    calibration(retail, 80, 76).

Two of those earn their place by being uncomfortable.

**competence** is what the mind is *observed* to be good at, not what it hopes.
It is updated from outcomes, so a pack that keeps being wrong says so, and the
brief line for a question in that area can open "Unsure —" instead of sounding
as confident as one in an area the mind handles well.

**calibration** is the mind checking its own confidence against what happened.
Saying 80% and being right 76% of the time is close; saying 80% and being right
40% of the time means the confidence means nothing, and only keeping the score
finds that out.

Everything here answers with a proof, because it is all just atoms:
"Can you act here?" -> "Only suggest -- stakes are medium and the host granted
L1", where both halves are facts and the rule joining them is readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.terms import Atom, Const

__all__ = ["SelfModel", "Competence", "Calibration"]

#: Rules over the self-facts, so a question about the mind gets a derivation.
SELF_RULES = """
%@ the mind may act only at the lower of what it was granted and stakes allow
can_act :- autonomy(l3).
can_act :- autonomy(l2), confirmed_by_person.
can_suggest :- autonomy(l1).
can_suggest :- can_act.

%@ a specialist that is not installed is a question it cannot take
cannot_answer(Kind) :- needs_specialist(Kind, S), not has_specialist(S).

%@ competence below half is worth warning about rather than hiding
unsure(Area) :- competence(Area, _Task, Score), Score < 50.

%@ confidence is miscalibrated when it is far from what was observed
overconfident(Context) :-
    calibration(Context, Claimed, Observed), Claimed - Observed > 20.
"""


@dataclass(frozen=True)
class Competence:
    """How well the mind has actually done at something."""

    area: str
    task: str
    correct: int = 0
    total: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def percent(self) -> int:
        return int(round(self.accuracy * 100))

    def describe(self) -> str:
        if not self.total:
            return f"{self.area}/{self.task}: never tried"
        return f"{self.area}/{self.task}: {self.percent}% of {self.total}"


@dataclass(frozen=True)
class Calibration:
    """Whether confidence at some level matches what actually happened."""

    context: str
    #: The confidence bucket, in percent: 80 means "said about 80% sure".
    claimed: int
    right: int = 0
    total: int = 0

    @property
    def observed(self) -> int:
        return int(round(100 * self.right / self.total)) if self.total else 0

    @property
    def gap(self) -> int:
        return self.claimed - self.observed

    def describe(self) -> str:
        if not self.total:
            return f"{self.context} @{self.claimed}%: no outcomes yet"
        return (
            f"{self.context} @{self.claimed}%: right {self.observed}% "
            f"of {self.total}"
        )


class SelfModel:
    """The mind's facts about itself, and the questions they answer."""

    def __init__(self, name: str = "mind") -> None:
        self.name = name
        self._facts: list[Atom] = []
        self._competence: dict[tuple[str, str], Competence] = {}
        self._calibration: dict[tuple[str, int], Calibration] = {}

    # -- recording ----------------------------------------------------------

    def state(self, predicate: str, *arguments) -> "SelfModel":
        """Set a single-valued fact, replacing whatever it was."""
        self._facts = [a for a in self._facts if a.predicate != predicate]
        self._facts.append(Atom(predicate, tuple(_const(a) for a in arguments)))
        return self

    def add(self, predicate: str, *arguments) -> "SelfModel":
        """Add a fact that can hold many times over."""
        atom = Atom(predicate, tuple(_const(a) for a in arguments))
        if atom not in self._facts:
            self._facts.append(atom)
        return self

    def drop(self, predicate: str, *arguments) -> "SelfModel":
        atom = Atom(predicate, tuple(_const(a) for a in arguments))
        self._facts = [a for a in self._facts if a != atom]
        return self

    def observed(self, area: str, task: str, correct: bool) -> Competence:
        """Record how an attempt actually went. This is what competence is."""
        key = (area, task)
        current = self._competence.get(key, Competence(area, task))
        updated = Competence(
            area, task, current.correct + int(correct), current.total + 1
        )
        self._competence[key] = updated
        return updated

    def claimed(self, context: str, confidence: int, right: bool) -> Calibration:
        """Record a confidence and whether it turned out to be justified."""
        bucket = int(round(confidence / 10.0) * 10)
        key = (context, bucket)
        current = self._calibration.get(key, Calibration(context, bucket))
        updated = Calibration(
            context, bucket, current.right + int(right), current.total + 1
        )
        self._calibration[key] = updated
        return updated

    # -- syncing with the rest ----------------------------------------------

    def sync(self, kernel=None, reading=None, specialists=None) -> "SelfModel":
        """Read the mind's current state off the parts that hold it.

        Nothing here is estimated. Every fact is copied from something that
        already knows it, which is what makes the self-report safe to show.
        """
        if specialists is not None:
            for name, installed in specialists.items():
                (self.add if installed else self.drop)("has_specialist", name)
        if kernel is not None:
            self.state("mode", _symbol(str(kernel.watchdog.mode)))
            self.state("autonomy", f"l{kernel.autonomy.level}")
            self.state("stakes", kernel.autonomy.stakes.level)
            self.state("quarantined", len(kernel.quarantine))
        if reading is not None:
            self._facts = [a for a in self._facts if a.predicate != "pack_active"]
            for facet in reading.facets:
                self.add("pack_active", facet.name, facet.weight)
            self.state("context", _symbol(reading.domain))
        return self

    def gap(self, concept: str) -> "SelfModel":
        """Something seen but not reasoned about. The growth loop's input."""
        return self.add("gap", _symbol(concept))

    # -- asking it about itself ---------------------------------------------

    @property
    def facts(self) -> list[Atom]:
        found = list(self._facts)
        for competence in self._competence.values():
            found.append(
                Atom(
                    "competence",
                    (
                        _const(_symbol(competence.area)),
                        _const(_symbol(competence.task)),
                        _const(competence.percent),
                    ),
                )
            )
        for calibration in self._calibration.values():
            found.append(
                Atom(
                    "calibration",
                    (
                        _const(_symbol(calibration.context)),
                        _const(calibration.claimed),
                        _const(calibration.observed),
                    ),
                )
            )
        return found

    def to_asp(self) -> str:
        return "\n".join(f"{atom}." for atom in sorted(self.facts, key=str))

    def engine(self, extra_rules: str = ""):
        from ..inference.engine import ReasoningEngine

        return ReasoningEngine(SELF_RULES + "\n" + extra_rules + "\n" + self.to_asp())

    def ask(self, goal: str):
        """Answer a question about the mind itself, with a proof."""
        return self.engine().ask(goal)

    # -- reporting ----------------------------------------------------------

    def unsure_about(self) -> list[str]:
        """Areas the mind has been observed to do badly at."""
        return sorted(
            c.area for c in self._competence.values() if c.total and c.percent < 50
        )

    def miscalibrated(self, tolerance: int = 20) -> list[Calibration]:
        """Confidence levels that did not hold up."""
        return sorted(
            (c for c in self._calibration.values() if c.total and abs(c.gap) > tolerance),
            key=lambda c: -abs(c.gap),
        )

    def describe(self) -> str:
        """At most three sentences, every clause traceable to a fact."""
        parts = []
        context = _first(self._facts, "context")
        mode = _first(self._facts, "mode")
        autonomy = _first(self._facts, "autonomy")
        stakes = _first(self._facts, "stakes")
        if context:
            parts.append(f"Context {context}")
        if mode:
            parts.append(f"running {mode}")
        first = ", ".join(parts) + "." if parts else ""

        second = ""
        if autonomy and stakes:
            second = f"May {_autonomy_verb(autonomy)}; stakes {stakes}."

        third = ""
        unsure = self.unsure_about()
        missing = [
            str(a.args[0].value)
            for a in self._facts
            if a.predicate == "missing_specialist"
        ]
        if unsure:
            third = "Unsure about " + ", ".join(unsure) + "."
        elif missing:
            third = "Cannot: " + ", ".join(sorted(missing)) + "."
        return " ".join(p for p in (first, second, third) if p)

    def to_dict(self) -> dict:
        return {
            "facts": [str(a) for a in self.facts],
            "competence": [c.describe() for c in self._competence.values()],
            "calibration": [c.describe() for c in self._calibration.values()],
            "unsure_about": self.unsure_about(),
            "miscalibrated": [c.describe() for c in self.miscalibrated()],
        }

    def __len__(self) -> int:
        return len(self.facts)

    def __repr__(self) -> str:
        return f"SelfModel({self.name!r}, {len(self.facts)} fact(s))"


# -- helpers ---------------------------------------------------------------


def _const(value) -> Const:
    return value if isinstance(value, Const) else Const(value)


def _symbol(text: str) -> str:
    import re

    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return cleaned or "unknown"


def _first(facts: Sequence[Atom], predicate: str) -> str:
    for atom in facts:
        if atom.predicate == predicate and atom.args:
            return str(atom.args[0].value)
    return ""


_AUTONOMY_VERBS = {
    "l0": "observe only",
    "l1": "answer and suggest",
    "l2": "act once confirmed",
    "l3": "act alone",
}


def _autonomy_verb(level: str) -> str:
    return _AUTONOMY_VERBS.get(level, level)
