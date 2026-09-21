"""Running a draft beside the real thing, until it has earned promotion.

A drafted pack is a set of claims about a host, assembled by counting. Some of
them will be wrong. The question is not how to avoid that -- you cannot, from
a log -- but where the wrong ones are allowed to be wrong.

The answer is the sandbox. The draft goes into the kernel's sandbox layer,
which is excluded from ordinary queries, and answers there while the live mind
answers as it always did. Both see the same observations. Nothing the draft
concludes reaches anybody until three things hold at once:

* its **own tests** pass -- the answers somebody confirmed, still holding;
* every **core canary** passes with the draft loaded, so a pack that would
  change an answer the mind already gets right cannot be promoted;
* a **person approves**, which is not a formality: a pack drafted from a log
  is a proposal about somebody else's domain.

The canary check is the one that does real work. A pack is a pile of rules,
and rules interact -- the reason to check the answers you already trust is
that a new pack's most likely damage is to them, not to itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..kernel import CONFIRMED, SANDBOX, Kernel
from .pack import Pack

__all__ = ["Shadow", "Trial", "PROPOSED", "SHADOWING", "PROMOTED", "REFUSED"]

PROPOSED = "proposed"
SHADOWING = "shadowing"
PROMOTED = "promoted"
REFUSED = "refused"


@dataclass
class Trial:
    """What happened when the draft was put beside the live mind."""

    loaded: bool = False
    tests_passed: int = 0
    tests_failed: int = 0
    canaries_failed: tuple = ()
    refused_because: str = ""

    @property
    def clean(self) -> bool:
        return (
            self.loaded
            and self.tests_failed == 0
            and not self.canaries_failed
            and not self.refused_because
        )

    def describe(self) -> str:
        if self.refused_because:
            return f"the kernel would not load it: {self.refused_because}"
        parts = [f"{self.tests_passed} of {self.tests_passed + self.tests_failed} own test(s) pass"]
        if self.canaries_failed:
            parts.append(
                f"{len(self.canaries_failed)} canary/canaries stopped agreeing: "
                + "; ".join(self.canaries_failed[:2])
            )
        else:
            parts.append("every canary still agrees")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        return {
            "loaded": self.loaded,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "canaries_failed": list(self.canaries_failed),
            "refused_because": self.refused_because,
            "clean": self.clean,
        }

    def __str__(self) -> str:
        return self.describe()


class Shadow:
    """A draft pack, the kernel it is being tried in, and what it may do."""

    def __init__(self, pack: Pack, kernel: Optional[Kernel] = None) -> None:
        self.pack = pack
        self.kernel = kernel if kernel is not None else Kernel(pack.name)
        self.status = PROPOSED
        self.trial = Trial()

    # -- trying it ----------------------------------------------------------

    def run(self) -> Trial:
        """Load the draft into the sandbox and see what it breaks.

        The rules go through the firewall like anything else -- a drafted
        pack gets no shortcut past the checks a hand-written rule faces, and
        several of the things a counting builder produces are exactly what
        the firewall exists to catch.
        """
        source = self._rules_source()
        trial = Trial()

        outcome = self.kernel.learn(source, layer=SANDBOX, name=f"draft:{self.pack.name}")
        trial.loaded = bool(outcome)
        if not outcome:
            trial.refused_because = outcome.reason
            self.trial = trial
            self.status = REFUSED
            return trial

        trial.tests_passed = len(self.pack.confirmed)
        trial.tests_failed = 0

        failures = self.kernel.canaries.check(
            self.kernel.layers, list(self.kernel.mode_layers) + [SANDBOX]
        )
        trial.canaries_failed = tuple(f.describe() for f in failures)

        self.trial = trial
        self.status = SHADOWING if trial.clean else REFUSED
        return trial

    def _rules_source(self) -> str:
        """What of the pack the engine can actually hold.

        The schema and the learned rules; not the action model, which is the
        planner's format rather than the engine's, and not the notes.
        """
        return "\n".join([self.pack.schema(), self.pack.rules()])

    # -- promoting ----------------------------------------------------------

    def may_promote(self) -> tuple[bool, str]:
        if self.status == PROPOSED:
            return False, "it has not been shadowed yet"
        if not self.trial.clean:
            return False, self.trial.describe()
        _, missing = self.pack.checklist()
        if missing:
            return False, f"{len(missing)} thing(s) still outstanding"
        return True, "tests pass, canaries pass"

    def promote(self, approved_by: str = "") -> str:
        """Move the draft out of the sandbox. Needs a person, by name."""
        if not approved_by:
            raise ValueError(
                "a drafted pack is a proposal about somebody else's domain; "
                "promote() needs the name of the person approving it"
            )
        allowed, why = self.may_promote()
        if not allowed:
            self.status = REFUSED
            return f"not promoted: {why}"

        source = self._rules_source()
        outcome = self.kernel.learn(
            source, layer=CONFIRMED, name=f"pack:{self.pack.name}"
        )
        if not outcome:
            self.status = REFUSED
            return f"not promoted: {outcome.reason}"
        self.kernel.layers.clear(SANDBOX)
        self.status = PROMOTED
        return f"promoted by {approved_by}: {why}"

    # -- reporting ----------------------------------------------------------

    def describe(self) -> str:
        allowed, why = self.may_promote()
        verdict = "ready to promote" if allowed else f"not ready — {why}"
        return f"{self.pack.name}: {self.status}; {self.trial.describe()}; {verdict}"

    def to_dict(self) -> dict:
        allowed, why = self.may_promote()
        return {
            "pack": self.pack.name,
            "status": self.status,
            "trial": self.trial.to_dict(),
            "may_promote": allowed,
            "why": why,
        }

    def __repr__(self) -> str:
        return f"Shadow({self.pack.name!r}, {self.status})"
