"""Degrade modes: how the mind gets smaller rather than wrong.

Three modes, each strictly less capable and strictly more trustworthy:

``full``
    Every layer but the sandbox, every specialist, learning on.
``core only``
    The shipped core and nothing else. Everything learned, tentative or
    session-specific is dropped from the view -- not deleted, just not read.
    This is the floor: whatever went wrong, the mind still answers what it
    shipped knowing.
``observe only``
    Answer nothing. Take input, log it, and say why.

A watchdog steps down on repeated errors, budget overruns or canary failures,
and steps back up when the thing that caused the step-down has passed. Stepping
back up requires *evidence*, not just time: the canaries have to pass at the
current mode first, which is the difference between recovering and hoping.

Every change carries its reason, and the reasons are what
:meth:`~neuralmind.kernel.Kernel.self_report` prints. A mind that quietly
degraded and kept answering would be worse than one that failed outright,
because nobody would know which answers came from which mind.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .layers import CORE, DEFAULT_LAYERS

__all__ = [
    "Mode",
    "FULL",
    "CORE_ONLY",
    "OBSERVE_ONLY",
    "MODES",
    "MODE_LAYERS",
    "Watchdog",
    "ModeChange",
]

FULL = "full"
CORE_ONLY = "core only"
OBSERVE_ONLY = "observe only"

#: Most capable first. Stepping down moves right.
MODES = (FULL, CORE_ONLY, OBSERVE_ONLY)

#: Which layers each mode reads.
MODE_LAYERS = {
    FULL: DEFAULT_LAYERS,
    CORE_ONLY: (CORE,),
    OBSERVE_ONLY: (),
}


@dataclass(frozen=True)
class ModeChange:
    """One step up or down, and what caused it."""

    frm: str
    to: str
    reason: str
    at: float = field(default_factory=time.time)

    @property
    def direction(self) -> str:
        return "down" if MODES.index(self.to) > MODES.index(self.frm) else "up"

    def describe(self) -> str:
        return f"stepped {self.direction} from {self.frm} to {self.to}: {self.reason}"

    def to_dict(self) -> dict:
        return {
            "from": self.frm,
            "to": self.to,
            "direction": self.direction,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return self.describe()


class Mode:
    """The current mode, and which layers it reads."""

    def __init__(self, name: str = FULL) -> None:
        if name not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}")
        self.name = name

    @property
    def layers(self) -> Sequence[str]:
        return MODE_LAYERS[self.name]

    @property
    def answers(self) -> bool:
        return self.name != OBSERVE_ONLY

    @property
    def learns(self) -> bool:
        return self.name == FULL

    def __str__(self) -> str:
        return self.name


class Watchdog:
    """Steps the mode down on trouble, and back up only on evidence.

    ``tolerance`` is how many failures of one kind it takes to step down. One
    is too jumpy -- a single budget overrun on an unusually hard question is
    normal -- and a large number means the mind spends a long time answering
    from a state that has already gone wrong.
    """

    def __init__(self, tolerance: int = 3) -> None:
        self.mode = Mode(FULL)
        self.tolerance = tolerance
        self.history: list[ModeChange] = []
        self._strikes: dict[str, int] = {}

    # -- trouble ------------------------------------------------------------

    def report(self, kind: str, detail: str = "") -> Optional[ModeChange]:
        """Note something going wrong. Steps down once it keeps happening."""
        self._strikes[kind] = self._strikes.get(kind, 0) + 1
        if self._strikes[kind] < self.tolerance:
            return None
        return self.step_down(
            f"{self._strikes[kind]} {kind} failure(s)"
            + (f": {detail}" if detail else "")
        )

    def report_canary_failures(self, failures) -> Optional[ModeChange]:
        """A canary failure is not a strike: it is a step down on its own.

        A budget overrun means the mind was slow. A canary failure means it is
        now answering a question differently than it used to, which is the one
        symptom that cannot be waited out.
        """
        if not failures:
            return None
        detail = "; ".join(str(f) for f in list(failures)[:2])
        return self.step_down(f"a canary stopped agreeing: {detail}")

    def healthy(self) -> None:
        """Note that something went right, clearing the strikes."""
        self._strikes.clear()

    # -- moving -------------------------------------------------------------

    def step_down(self, reason: str) -> Optional[ModeChange]:
        index = MODES.index(self.mode.name)
        if index >= len(MODES) - 1:
            return None
        change = ModeChange(self.mode.name, MODES[index + 1], reason)
        self.mode = Mode(change.to)
        self.history.append(change)
        self._strikes.clear()
        return change

    def peek_up(self) -> Optional[str]:
        """The mode a step up would reach, or None at the top."""
        index = MODES.index(self.mode.name)
        return None if index == 0 else MODES[index - 1]

    def step_up(self, evidence: str) -> Optional[ModeChange]:
        """Recover one step. The caller must have checked that it is safe.

        Requiring a reason is not paperwork: the only safe way back up is to
        have re-run the canaries at the current mode and seen them pass, and
        naming that evidence is what makes the audit trail show whether anyone
        actually did.
        """
        if not evidence:
            raise ValueError(
                "stepping up requires evidence -- run the canaries and say so"
            )
        index = MODES.index(self.mode.name)
        if index == 0:
            return None
        change = ModeChange(self.mode.name, MODES[index - 1], evidence)
        self.mode = Mode(change.to)
        self.history.append(change)
        self._strikes.clear()
        return change

    # -- reporting ----------------------------------------------------------

    @property
    def last_change(self) -> Optional[ModeChange]:
        return self.history[-1] if self.history else None

    def describe(self) -> str:
        if not self.history:
            return f"{self.mode} (never degraded)"
        return f"{self.mode} — {self.history[-1].describe()}"

    def to_dict(self) -> dict:
        return {
            "mode": self.mode.name,
            "answers": self.mode.answers,
            "learns": self.mode.learns,
            "strikes": dict(self._strikes),
            "changes": [c.to_dict() for c in self.history],
        }
