"""The safety kernel: exploring without breaking.

This lands before the growth loop for one reason the roadmap is explicit
about: anything that changes itself needs a guaranteed way back, and building
the changing part first would mean a period where there was none.

Eight pieces, each solving one way a mind that learns can hurt itself:

:mod:`~neuralmind.kernel.layers`
    Knowledge stacked core → confirmed → tenant → session → sandbox, with the
    core read-only at runtime and the sandbox excluded from queries by default.
:mod:`~neuralmind.kernel.snapshot`
    Every risky step is a transaction that commits or leaves no trace.
:mod:`~neuralmind.kernel.canaries`
    Questions with known answers, checked after every change.
:mod:`~neuralmind.kernel.firewall`
    Safety, stratification, consistency with the core, and a budgeted dry run
    before a rule may enter anything.
:mod:`~neuralmind.kernel.quarantine`
    What failed is disabled with the reason recorded, never quietly deleted.
:mod:`~neuralmind.kernel.modes`
    full → core only → observe only, with a watchdog that steps down on
    trouble and back up only on evidence.
:mod:`~neuralmind.kernel.autonomy`
    L0–L3. Caution rises on a guess; autonomy rises only on a grant.
:mod:`~neuralmind.kernel.privacy`
    Personal data stays in its layer. Rules induced from it wait for a person.

:class:`Kernel` wires them together so the safe path is the easy one --
:meth:`Kernel.learn` snapshots, firewalls, checks canaries, and rolls back or
quarantines, and there is no shorter way to add a rule that skips any of it.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from ..core.terms import Atom
from .autonomy import (
    HIGH,
    L0,
    L1,
    L2,
    L3,
    LOW,
    MEDIUM,
    AutonomyGate,
    Decision,
    Stakes,
)
from .canaries import Canary, CanaryFailure, CanarySet
from .firewall import ADMITTED, REJECTED, Firewall, Verdict
from .layers import (
    CONFIRMED,
    CORE,
    DEFAULT_LAYERS,
    LAYER_ORDER,
    SANDBOX,
    SESSION,
    TENANT,
    Layer,
    LayerStack,
    ReadOnlyLayer,
)
from .modes import (
    CORE_ONLY,
    FULL,
    MODE_LAYERS,
    MODES,
    OBSERVE_ONLY,
    Mode,
    ModeChange,
    Watchdog,
)
from .privacy import HeldRule, PrivacyPolicy, Retention, Tag
from .quarantine import Quarantine, Quarantined
from .snapshot import RolledBack, Snapshot, Transaction, transaction

__all__ = [
    "Kernel",
    "Outcome",
    # layers
    "LayerStack", "Layer", "ReadOnlyLayer",
    "CORE", "CONFIRMED", "TENANT", "SESSION", "SANDBOX",
    "LAYER_ORDER", "DEFAULT_LAYERS",
    # transactions
    "Snapshot", "Transaction", "transaction", "RolledBack",
    # canaries
    "CanarySet", "Canary", "CanaryFailure",
    # firewall
    "Firewall", "Verdict", "ADMITTED", "REJECTED",
    # quarantine
    "Quarantine", "Quarantined",
    # modes
    "Watchdog", "Mode", "ModeChange", "FULL", "CORE_ONLY", "OBSERVE_ONLY",
    # autonomy
    "AutonomyGate", "Stakes", "Decision",
    "L0", "L1", "L2", "L3", "LOW", "MEDIUM", "HIGH",
    # privacy
    "PrivacyPolicy", "Tag", "Retention", "HeldRule",
]


class Outcome:
    """What happened to one attempt to learn something."""

    __slots__ = ("accepted", "verdict", "failures", "reason", "layer")

    def __init__(
        self,
        accepted: bool,
        verdict: Optional[Verdict] = None,
        failures: Sequence[CanaryFailure] = (),
        reason: str = "",
        layer: str = "",
    ) -> None:
        self.accepted = accepted
        self.verdict = verdict
        self.failures = list(failures)
        self.reason = reason
        self.layer = layer

    def __bool__(self) -> bool:
        return self.accepted

    def describe(self) -> str:
        if self.accepted:
            return f"accepted into {self.layer}"
        return f"rejected: {self.reason}"

    def to_dict(self) -> dict:
        payload = {"accepted": self.accepted, "reason": self.reason}
        if self.layer:
            payload["layer"] = self.layer
        if self.verdict is not None:
            payload["firewall"] = self.verdict.to_dict()
        if self.failures:
            payload["canaries"] = [f.describe() for f in self.failures]
        return payload

    def __str__(self) -> str:
        return self.describe()


class Kernel:
    """The eight mechanisms wired so the safe path is the only short one.

    Nothing here stops a caller reaching past it -- Python does not work that
    way, and a kernel that pretended otherwise would be lying. What it does is
    make the checked route the convenient one, and make everything it rejects
    visible afterwards rather than silent.
    """

    def __init__(
        self,
        name: str = "mind",
        tenant: Optional[str] = None,
        granted: int = L1,
        tolerance: int = 3,
    ) -> None:
        self.layers = LayerStack(name, tenant=tenant)
        self.canaries = CanarySet()
        self.firewall = Firewall(self.layers)
        self.quarantine = Quarantine()
        self.watchdog = Watchdog(tolerance=tolerance)
        self.autonomy = AutonomyGate(granted=granted)
        self.privacy = PrivacyPolicy()
        self.history: list[Outcome] = []

    # -- setting up ---------------------------------------------------------

    def load_core(self, source: str, name: str = "core") -> "Kernel":
        self.layers.load_core(source, name)
        return self

    def load_core_builtin(self, name: str) -> "Kernel":
        self.layers.load_core_builtin(name)
        return self

    def watch(self, goals: Iterable, note: str = "") -> CanarySet:
        """Record what the mind currently answers, and defend it from now on."""
        captured = CanarySet.capture(self.layers, goals, self.mode_layers, note)
        for canary in captured:
            self.canaries.add(canary.goal, canary.status, canary.note)
        return self.canaries

    # -- the guarded path ---------------------------------------------------

    @property
    def mode_layers(self) -> Sequence[str]:
        return self.watchdog.mode.layers or (CORE,)

    def learn(
        self, source: str, layer: str = CONFIRMED, name: str = "learned"
    ) -> Outcome:
        """Add a rule, but only if it survives everything.

        Firewall first, because a rule that cannot compile should not get as
        far as a snapshot. Then a transaction, so a rule that compiles and
        still breaks something leaves no trace. Then quarantine, so whatever
        produced it can be found rather than just the rule.
        """
        if not self.watchdog.mode.learns:
            return self._record(
                Outcome(False, reason=f"not learning in {self.watchdog.mode} mode")
            )
        if self.quarantine.holds(source):
            return self._record(
                Outcome(False, reason=f"quarantined: {self.quarantine._items[source].reason}")
            )

        verdict = self.firewall.check(source, name)
        if not verdict:
            self.quarantine.add(
                source, "rule", verdict.reason, evidence=(f"firewall: {verdict.check}",)
            )
            return self._record(
                Outcome(False, verdict=verdict, reason=verdict.describe())
            )

        with transaction(self.layers, f"learning {name}") as change:
            self.layers.add_rules(source, layer, name)
            failures = self.canaries.check(self.layers, self.mode_layers)
            if failures:
                change.rollback("; ".join(f.describe() for f in failures))

        if change.rolled_back:
            self.quarantine.add(
                source, "rule", "broke a canary",
                evidence=tuple(f.describe() for f in failures),
            )
            # A canary failure the rollback fixed is a rule that was caught --
            # the system working, not the system degrading. Stepping down is
            # for a failure that is still there after the way back was taken,
            # because that is the one nobody knows the cause of.
            still_failing = self.canaries.check(self.layers, self.mode_layers)
            if still_failing:
                self.watchdog.report_canary_failures(still_failing)
            else:
                self.watchdog.report("rejected change", failures[0].describe())
            return self._record(
                Outcome(False, verdict=verdict, failures=failures,
                        reason=f"a canary stopped agreeing: {failures[0].describe()}")
            )

        self.watchdog.healthy()
        return self._record(Outcome(True, verdict=verdict, layer=layer))

    def observe(self, facts: Iterable, layer: str = SESSION) -> list:
        """Take facts in, routing anything personal into a confined layer."""
        return self.privacy.admit(facts, self.layers, layer)

    def ask(self, goal, **kwargs):
        """Answer at the current mode. Observe-only answers nothing."""
        if not self.watchdog.mode.answers:
            return None
        return self.layers.engine(self.mode_layers).ask(goal, **kwargs)

    def decide(self, action: str, needs: int = L2, confirmed: bool = False) -> Decision:
        """Put an action through the autonomy gate. Nothing skips this."""
        return self.autonomy.check(action, needs, confirmed)

    # -- recovering ---------------------------------------------------------

    def recover(self) -> Optional[ModeChange]:
        """Step back up if the canaries pass *at the mode being stepped into*.

        Evidence, not time: a mode that recovers because a while has gone by
        recovers into the same fault.

        The check is against the target mode, not the current one. A degraded
        mode answers differently on purpose -- core-only cannot see the facts a
        canary was captured with, so checking there would say "still broken"
        forever and the mind would never come back up. The question that
        matters is whether the fuller view is healthy again.
        """
        target = self.watchdog.peek_up()
        if target is None:
            return None
        failures = self.canaries.check(self.layers, MODE_LAYERS[target] or (CORE,))
        if failures:
            return None
        return self.watchdog.step_up(
            f"{len(self.canaries)} canary/canaries pass at {target}"
        )

    def _record(self, outcome: Outcome) -> Outcome:
        self.history.append(outcome)
        return outcome

    # -- reporting ----------------------------------------------------------

    def self_report(self) -> str:
        """Three sentences: where it stands, what it refused, what it may do."""
        lines = [self.watchdog.describe() + "."]
        refused = sum(1 for o in self.history if not o)
        held = len(self.privacy.held)
        parts = []
        if refused:
            parts.append(f"{refused} change(s) refused")
        if len(self.quarantine):
            parts.append(f"{len(self.quarantine)} quarantined")
        if held:
            parts.append(f"{held} rule(s) awaiting human review")
        lines.append(
            ("Kept out: " + ", ".join(parts) + ".") if parts else "Nothing refused yet."
        )
        lines.append(self.autonomy.describe() + ".")
        return " ".join(lines)

    def to_dict(self) -> dict:
        return {
            "layers": self.layers.summary(),
            "mode": self.watchdog.to_dict(),
            "autonomy": self.autonomy.to_dict(),
            "privacy": self.privacy.to_dict(),
            "firewall": self.firewall.to_dict(),
            "quarantine": self.quarantine.to_dict(),
            "canaries": len(self.canaries),
        }

    def __repr__(self) -> str:
        return (
            f"Kernel(mode={self.watchdog.mode}, "
            f"autonomy={self.autonomy.level}, {self.layers!r})"
        )
