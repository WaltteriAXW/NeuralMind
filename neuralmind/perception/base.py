"""The perception layer's contract.

Perception is the only place a neural network appears in this system. Its job
is narrow: turn raw input -- text, pixels, numbers -- into ground atoms the
symbolic engine can reason over, each tagged with a confidence. Nothing here
draws a conclusion; conclusions come from the rules.

Every perceptor returns a :class:`Perception` rather than bare atoms, because
the confidences and the discarded candidates are what the consistency layer
and the error analysis need later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Protocol, runtime_checkable

from ..core.terms import Atom
from ..knowledge.base import FactRecord, KnowledgeBase

__all__ = ["Perception", "Perceptor", "PerceptionError"]


class PerceptionError(Exception):
    """Raw input could not be turned into symbols."""


@dataclass
class Perception:
    """What a perceptor extracted from one input."""

    facts: list[FactRecord] = field(default_factory=list)
    #: Rules extracted from the input, for inputs that state general rules.
    rules: list[Any] = field(default_factory=list)
    source: str = "unknown"
    #: Anything the caller may want for debugging: raw scores, spans, timings.
    diagnostics: dict = field(default_factory=dict)
    #: Input fragments the perceptor could not interpret.
    unparsed: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.facts)

    def __iter__(self):
        return iter(self.facts)

    @property
    def atoms(self) -> list[Atom]:
        return [record.atom for record in self.facts]

    @property
    def mean_confidence(self) -> float:
        if not self.facts:
            return 0.0
        return sum(record.confidence for record in self.facts) / len(self.facts)

    @property
    def min_confidence(self) -> float:
        return min((record.confidence for record in self.facts), default=0.0)

    def filter(self, minimum_confidence: float) -> "Perception":
        """A copy keeping only facts at or above a confidence threshold."""
        return Perception(
            facts=[r for r in self.facts if r.confidence >= minimum_confidence],
            rules=list(self.rules),
            source=self.source,
            diagnostics=dict(self.diagnostics),
            unparsed=list(self.unparsed),
        )

    def extend(self, other: "Perception") -> "Perception":
        """Merge another perception into this one, in place."""
        self.facts.extend(other.facts)
        self.rules.extend(other.rules)
        self.unparsed.extend(other.unparsed)
        for key, value in other.diagnostics.items():
            if key == "proper_names":
                merged = set(self.diagnostics.get(key, ())) | set(value)
                self.diagnostics[key] = sorted(merged)
            else:
                self.diagnostics.setdefault(key, value)
        return self

    def into(self, kb: KnowledgeBase) -> KnowledgeBase:
        """Write the perceived facts and rules into a knowledge base."""
        kb.add_facts(self.facts)
        for rule in self.rules:
            kb.rules.add(rule)
        return kb

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "facts": [record.to_dict() for record in self.facts],
            "rules": [str(rule) for rule in self.rules],
            "unparsed": list(self.unparsed),
            "mean_confidence": round(self.mean_confidence, 4),
            "diagnostics": self.diagnostics,
        }


@runtime_checkable
class Perceptor(Protocol):
    """Anything that turns raw input into symbols."""

    name: str

    def perceive(self, raw: Any) -> Perception:
        """Extract facts from one input."""
        ...
