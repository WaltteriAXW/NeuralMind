"""Rules for what the host reports but the mind cannot derive.

A log says ``alarm: true`` without saying why. That is a relation the host
knows and the mind does not, which the roadmap answers with induction over
the logged examples. Here the induction is narrow on purpose: a flag against
a measured quantity, looking for a **threshold**.

Narrow, because the alternative is worse. A general rule learner over a log
this size finds something for every flag, and most of what it finds is
coincidence dressed as a law. A threshold is one number, it either has a
counterexample in the log or it does not, and anybody can check it against
the process they actually run.

What is learned is an **implication, not an equivalence**, and the difference
is the honest part. "Whenever pressure is above 3.0 bar the alarm is on, with
no counterexample in 240 records" is a finding. "The alarm is on exactly when
pressure is above 3.0 bar" is a stronger claim the log does not support --
the alarm can stay on after the pressure falls, and a learner that rounded
its way past those cases would have produced a rule that is wrong in exactly
the situation an alarm exists for. So coverage is reported alongside
soundness, and what the rule does not explain becomes a question.

A threshold that survives is also given a **name**. ``high_pressure`` is then
a concept the rest of the builder can use -- it is what lets the effect
learner finally say when ``reset_alarm`` works, having had no way to express
"pressure is low enough" before. A learned concept earning its place by
explaining a second thing is the strongest evidence a log can offer that it
is real.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .survey import FLAG, QUANTITY, Finding, Survey

__all__ = ["Threshold", "learn_thresholds", "SCALE", "invented_name"]

#: The engine's arithmetic is integer, so a decimal threshold is written
#: scaled, with the scale recorded in the rule's own comment. Engineering
#: packs do this to real quantities too; it is not a workaround, it is what
#: fixed-point means.
SCALE = 10

#: Below this many examples either side, a threshold is not evidence.
MIN_EITHER_SIDE = 3


@dataclass(frozen=True)
class Threshold:
    """A learned implication: this quantity above this value means that flag."""

    flag: str
    quantity: str
    unit: Optional[str]
    above: float
    #: Records where the quantity was above and the flag was on.
    supports: int
    #: Records where it was above and the flag was off. Must be zero.
    counterexamples: int
    #: Records where the flag was on at all.
    flag_true: int
    #: Records considered.
    records: int

    @property
    def sound(self) -> bool:
        return self.counterexamples == 0

    @property
    def coverage(self) -> float:
        """Share of the flag's true cases this explains."""
        return self.supports / self.flag_true if self.flag_true else 0.0

    @property
    def complete(self) -> bool:
        return self.sound and self.coverage >= 1.0

    @property
    def name(self) -> str:
        return invented_name(self.quantity)

    def describe(self) -> str:
        unit = f" {self.unit}" if self.unit else ""
        head = (
            f"whenever {self.quantity} is above {self.above:g}{unit}, "
            f"{self.flag} is on"
        )
        if self.complete:
            return f"{head} — and only then ({self.supports} record(s), no exceptions)"
        return (
            f"{head} ({self.supports} record(s), no counterexamples), but that "
            f"explains only {self.coverage:.0%} of the {self.flag_true} record(s) "
            f"where {self.flag} was on"
        )

    def to_asp(self) -> str:
        """The rule, scaled, with the scaling written down beside it."""
        scaled = int(round(self.above * SCALE))
        unit = f" {self.unit}" if self.unit else ""
        return (
            f"% {self.quantity} is held as {SCALE}ths of a{unit or ' unit'}, so "
            f"{scaled} is {self.above:g}{unit}.\n"
            f"{self.name} :- {self.quantity}(V), V > {scaled}.\n"
            f"{self.flag} :- {self.name}."
        )

    def to_dict(self) -> dict:
        return {
            "flag": self.flag,
            "quantity": self.quantity,
            "above": self.above,
            "unit": self.unit,
            "concept": self.name,
            "supports": self.supports,
            "counterexamples": self.counterexamples,
            "coverage": round(self.coverage, 4),
            "sound": self.sound,
            "complete": self.complete,
        }

    def __str__(self) -> str:
        return self.describe()


def invented_name(quantity: str) -> str:
    """A name for the concept a threshold picks out."""
    return f"high_{quantity}"


def learn_thresholds(
    log: Sequence[dict], reading: Survey, min_either_side: int = MIN_EITHER_SIDE
) -> list[Threshold]:
    """Look for a quantity that explains each flag. Best first."""
    flags = [f for f in reading.findings if f.kind == FLAG]
    quantities = [f for f in reading.findings if f.kind == QUANTITY and f.unit]
    found: list[Threshold] = []

    for flag in flags:
        for quantity in quantities:
            threshold = _best_threshold(log, flag, quantity, min_either_side)
            if threshold is not None:
                found.append(threshold)

    found.sort(key=lambda t: (t.complete, t.coverage, t.supports), reverse=True)
    return found


def _best_threshold(
    log: Sequence[dict], flag: Finding, quantity: Finding, min_either_side: int
) -> Optional[Threshold]:
    pairs = []
    for record in log:
        raw, on = record.get(quantity.name), record.get(flag.name)
        value = _number(raw)
        if value is None or not isinstance(on, bool):
            continue
        pairs.append((value, on))
    if len(pairs) < 2 * min_either_side:
        return None

    flag_true = sum(1 for _, on in pairs if on)
    if flag_true == 0 or flag_true == len(pairs):
        return None  # a flag that never moves is explained by anything

    values = sorted({value for value, _ in pairs})
    best: Optional[Threshold] = None
    for low, high in zip(values, values[1:]):
        cut = round((low + high) / 2, 4)
        above = [(v, on) for v, on in pairs if v > cut]
        if len(above) < min_either_side or len(pairs) - len(above) < min_either_side:
            continue
        counterexamples = sum(1 for _, on in above if not on)
        if counterexamples:
            continue
        candidate = Threshold(
            flag=flag.name,
            quantity=quantity.name,
            unit=quantity.unit,
            above=cut,
            supports=len(above),
            counterexamples=0,
            flag_true=flag_true,
            records=len(pairs),
        )
        if best is None or candidate.supports > best.supports:
            best = candidate
    return best


def _number(raw) -> Optional[float]:
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    head = raw.strip().split()
    try:
        return float(head[0])
    except (ValueError, IndexError):
        return None
