"""How much the mind may do, and the rule that it is never the one deciding.

A wrong hint in a game is harmless. A wrong transfer is not. The mind is never
told where it is, so it cannot know which case it is in -- which means the safe
default is caution, and the way out of caution is a grant from the host rather
than a conclusion of its own.

Four levels:

``L0`` observe
    Take input, answer nothing, act on nothing.
``L1`` answer and suggest
    Answer questions and propose actions. Nothing happens without a person.
``L2`` act after confirmation
    Carry out an action once it is confirmed.
``L3`` act alone
    Act within limits the host set, unsupervised.

Two inputs decide the level in force, and they are not symmetric:

* the **grant** comes from the host and is the ceiling;
* the **stakes** are inferred from what the mind is seeing, and can only lower
  it.

That asymmetry is the whole design (roadmap design rule 11: caution goes up on
a guess, autonomy goes up only on a grant). Inferring stakes is guesswork, and
guesswork that could *raise* autonomy is a way for a misread observation to
authorise a transfer. Guesswork that only lowers it is a way for a misread
observation to be annoying. Those are not comparable risks, so the code makes
one of them impossible rather than unlikely.

High stakes cap the level at L2 no matter what the host granted: money moving
and decisions about people keep a person in the loop, and a host that wants
otherwise has to lower its own stakes assessment, not ask this to ignore it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

__all__ = [
    "Level",
    "Stakes",
    "AutonomyGate",
    "Decision",
    "L0",
    "L1",
    "L2",
    "L3",
    "LOW",
    "MEDIUM",
    "HIGH",
    "HIGH_STAKES_CEILING",
]

L0, L1, L2, L3 = 0, 1, 2, 3

LEVEL_NAMES = {
    L0: "L0: observe",
    L1: "L1: answer and suggest",
    L2: "L2: act after confirmation",
    L3: "L3: act alone",
}

LOW, MEDIUM, HIGH = "low", "medium", "high"

STAKES_ORDER = (LOW, MEDIUM, HIGH)

#: The highest level high stakes will allow, whatever the host granted.
HIGH_STAKES_CEILING = L2

#: What each stakes level allows on its own.
STAKES_CEILING = {LOW: L3, MEDIUM: L3, HIGH: HIGH_STAKES_CEILING}


class Level(int):
    """An autonomy level that prints as its name."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return LEVEL_NAMES.get(int(self), f"L{int(self)}")


@dataclass(frozen=True)
class Stakes:
    """How much could go wrong here, and what made it look that way.

    ``reasons`` matters as much as ``level``. "money moves and personal data
    is present" is something a host can argue with; a bare "high" is not.
    """

    level: str = LOW
    reasons: tuple[str, ...] = ()

    def raised_to(self, level: str, reason: str) -> "Stakes":
        """Raise the stakes. There is deliberately no way to lower them.

        Anything that inferred a risk is allowed to act on it. Nothing that
        inferred an absence of risk is allowed to act on *that*, because the
        absence of evidence for money moving is not evidence that it is not.
        """
        if STAKES_ORDER.index(level) <= STAKES_ORDER.index(self.level):
            return Stakes(self.level, tuple(dict.fromkeys(self.reasons + (reason,))))
        return Stakes(level, tuple(dict.fromkeys(self.reasons + (reason,))))

    @property
    def ceiling(self) -> int:
        return STAKES_CEILING[self.level]

    def describe(self) -> str:
        if not self.reasons:
            return self.level
        return f"{self.level} ({'; '.join(self.reasons)})"


@dataclass(frozen=True)
class Decision:
    """The gate's verdict on one action, and why."""

    action: str
    allowed: bool
    #: The level actually in force: the lower of the grant and the stakes cap.
    level: int
    granted: int
    stakes: Stakes
    #: ``act`` | ``confirm`` | ``suggest`` | ``refuse``
    outcome: str
    reason: str = ""
    #: True when the action needs a person to confirm it before it happens.
    needs_confirmation: bool = False

    def __bool__(self) -> bool:
        return self.allowed

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "allowed": self.allowed,
            "outcome": self.outcome,
            "level": LEVEL_NAMES.get(self.level, f"L{self.level}"),
            "granted": LEVEL_NAMES.get(self.granted, f"L{self.granted}"),
            "stakes": self.stakes.describe(),
            "needs_confirmation": self.needs_confirmation,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return f"{self.action}: {self.outcome} — {self.reason}"


class AutonomyGate:
    """Every action the mind would take passes through here.

    The gate holds no policy of its own about *which* actions are risky; that
    is what stakes are for. It holds the arithmetic: the level in force is the
    minimum of what the host granted and what the stakes allow, and that
    minimum is reported with every decision so a host can see which of the two
    bound it.
    """

    def __init__(self, granted: int = L1, stakes: Optional[Stakes] = None) -> None:
        self.granted = int(granted)
        self.stakes = stakes or Stakes()
        #: Every decision, for the audit trail a regulated host needs.
        self.log: list[Decision] = []

    # -- the two inputs -----------------------------------------------------

    def grant(self, level: int, by: str = "host") -> "AutonomyGate":
        """Set the ceiling. Only a host does this."""
        if not L0 <= int(level) <= L3:
            raise ValueError(f"autonomy level must be L0..L3, not {level!r}")
        self.granted = int(level)
        return self

    def raise_stakes(self, level: str, reason: str) -> "AutonomyGate":
        """Note something that makes this situation riskier. Never the reverse."""
        if level not in STAKES_ORDER:
            raise ValueError(f"stakes must be one of {', '.join(STAKES_ORDER)}")
        self.stakes = self.stakes.raised_to(level, reason)
        return self

    # -- the verdict --------------------------------------------------------

    @property
    def level(self) -> int:
        """The level in force: the lower of the grant and the stakes ceiling."""
        return min(self.granted, self.stakes.ceiling)

    @property
    def bound_by(self) -> str:
        if self.stakes.ceiling < self.granted:
            return "stakes"
        if self.granted < self.stakes.ceiling:
            return "grant"
        return "both"

    def check(self, action: str, needs: int = L2, confirmed: bool = False) -> Decision:
        """Decide what may happen to ``action``.

        ``needs`` is the level the action requires: a read is L1, a change to
        the world is L2, an unsupervised change is L3.
        """
        level = self.level
        if level >= L3 and needs <= L3 and not confirmed and needs <= level:
            decision = self._allow(action, "act", "acting within the granted limits")
        elif needs <= level and needs <= L1:
            decision = self._allow(action, "act", "answering is within the granted level")
        elif needs <= level and confirmed:
            decision = self._allow(action, "act", "confirmed by a person")
        elif needs <= level:
            decision = Decision(
                action=action,
                allowed=False,
                level=level,
                granted=self.granted,
                stakes=self.stakes,
                outcome="confirm",
                needs_confirmation=True,
                reason=f"{LEVEL_NAMES[level]} — this needs a person to confirm it",
            )
        elif level >= L1:
            decision = Decision(
                action=action,
                allowed=False,
                level=level,
                granted=self.granted,
                stakes=self.stakes,
                outcome="suggest",
                reason=self._why_short(needs),
            )
        else:
            decision = Decision(
                action=action,
                allowed=False,
                level=level,
                granted=self.granted,
                stakes=self.stakes,
                outcome="refuse",
                reason=self._why_short(needs),
            )
        self.log.append(decision)
        return decision

    def _allow(self, action: str, outcome: str, reason: str) -> Decision:
        return Decision(
            action=action,
            allowed=True,
            level=self.level,
            granted=self.granted,
            stakes=self.stakes,
            outcome=outcome,
            reason=reason,
        )

    def _why_short(self, needs: int) -> str:
        """Say which of the two bound the level -- a host can only fix one."""
        if self.stakes.ceiling < self.granted:
            return (
                f"{LEVEL_NAMES[needs]} is needed, but the stakes are "
                f"{self.stakes.describe()}, which caps this at "
                f"{LEVEL_NAMES[self.stakes.ceiling]}"
            )
        return (
            f"{LEVEL_NAMES[needs]} is needed and the host granted "
            f"{LEVEL_NAMES[self.granted]}"
        )

    def describe(self) -> str:
        return (
            f"{LEVEL_NAMES[self.level]}; granted {LEVEL_NAMES[self.granted]}, "
            f"stakes {self.stakes.describe()}, bound by {self.bound_by}"
        )

    def to_dict(self) -> dict:
        return {
            "level": LEVEL_NAMES[self.level],
            "granted": LEVEL_NAMES[self.granted],
            "stakes": self.stakes.level,
            "stakes_reasons": list(self.stakes.reasons),
            "bound_by": self.bound_by,
            "decisions": len(self.log),
        }
