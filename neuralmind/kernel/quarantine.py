"""Disabled with the reason recorded, never silently deleted.

When a rule or a pack turns out to cause failures, the tempting thing is to
remove it. That is wrong twice over: whoever added it learns nothing, and the
same thing gets added again next week by whatever produced it the first time.

So a quarantined item stays. It is inert -- nothing reads it, nothing derives
from it -- and it carries what it did and when. A person can look at the list,
see that the induced rule about refunds broke two canaries on the fourth of
the month, and fix the cause rather than the symptom. Releasing is explicit and
records who did it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

__all__ = ["Quarantine", "Quarantined"]


@dataclass
class Quarantined:
    """One disabled item and the case against it."""

    item: str
    kind: str
    reason: str
    #: What it broke: canary failures, a firewall verdict, an exception.
    evidence: tuple[str, ...] = ()
    at: float = field(default_factory=time.time)
    released_by: str = ""
    #: How many times this same item has been quarantined.
    occurrences: int = 1

    @property
    def active(self) -> bool:
        return not self.released_by

    def describe(self) -> str:
        detail = f" ({'; '.join(self.evidence[:2])})" if self.evidence else ""
        repeat = f" [{self.occurrences}x]" if self.occurrences > 1 else ""
        state = "" if self.active else f" — released by {self.released_by}"
        return f"{self.kind} {self.item}: {self.reason}{detail}{repeat}{state}"

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "kind": self.kind,
            "reason": self.reason,
            "evidence": list(self.evidence),
            "occurrences": self.occurrences,
            "active": self.active,
            "released_by": self.released_by or None,
        }

    def __str__(self) -> str:
        return self.describe()


class Quarantine:
    """The register of what has been disabled and why."""

    def __init__(self) -> None:
        self._items: dict[str, Quarantined] = {}

    def add(
        self,
        item,
        kind: str = "rule",
        reason: str = "",
        evidence: Iterable[str] = (),
    ) -> Quarantined:
        """Disable something. Adding the same item again counts it.

        The count is the useful part: one bad rule is an accident, the same bad
        rule four times is whatever keeps producing it.
        """
        key = str(item)
        existing = self._items.get(key)
        if existing is not None:
            existing.occurrences += 1
            existing.reason = reason or existing.reason
            existing.evidence = tuple(dict.fromkeys(existing.evidence + tuple(evidence)))
            existing.released_by = ""
            existing.at = time.time()
            return existing
        entry = Quarantined(
            item=key, kind=kind, reason=reason, evidence=tuple(evidence)
        )
        self._items[key] = entry
        return entry

    def holds(self, item) -> bool:
        entry = self._items.get(str(item))
        return entry is not None and entry.active

    def entry(self, item) -> Optional[Quarantined]:
        """The record for an item, active or released, or None."""
        return self._items.get(str(item))

    def why(self, item) -> str:
        """Why something is quarantined, for a message to a person."""
        entry = self._items.get(str(item))
        return entry.reason if entry is not None else ""

    def release(self, item, by: str) -> Quarantined:
        """Let something back in. Requires naming who decided that."""
        if not by:
            raise ValueError("releasing from quarantine requires naming who did it")
        entry = self._items.get(str(item))
        if entry is None:
            raise KeyError(f"{item} is not quarantined")
        entry.released_by = by
        return entry

    @property
    def active(self) -> list[Quarantined]:
        return [entry for entry in self._items.values() if entry.active]

    @property
    def repeat_offenders(self) -> list[Quarantined]:
        """Items quarantined more than once -- a symptom with a cause behind it."""
        return sorted(
            (e for e in self._items.values() if e.occurrences > 1),
            key=lambda e: -e.occurrences,
        )

    def __len__(self) -> int:
        return len(self.active)

    def __iter__(self):
        return iter(self.active)

    def to_dict(self) -> dict:
        return {
            "active": len(self.active),
            "total": len(self._items),
            "items": [entry.to_dict() for entry in self._items.values()],
        }
