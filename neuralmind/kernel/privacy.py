"""Personal data stays where it came from.

The rule (roadmap design rule 12) is short and the consequences are not:
**a fact about a specific person or company never becomes shared knowledge.**
Not through promotion, not through a snapshot, not through a rule induced from
it, and not through another tenant's answer.

Three mechanisms, because there are three ways it leaks.

**Tagging on arrival.** A fact is personal because of what it is, not because
of where it ended up. So it is classified as it comes in -- by predicates the
host declared, by patterns that are personal in any domain (an email address,
an IBAN, a phone number), and optionally by an entity detector. Classifying
later would mean a window in which it was not yet protected.

**Confinement.** A tagged fact lives in the tenant or session layer, and
:meth:`~neuralmind.kernel.layers.LayerStack.promote` refuses to carry it out.
That is enforced in the layer stack rather than here, because the check has to
be on the path the data actually takes.

**Holding generalisations for review.** A rule induced *from* personal data is
a different object from a fact about one person, and often a legitimate one --
"customers who bought X return it" is not about anybody. But it can also be a
re-identification ("the customer born on this date in this town"), and nothing
here can tell those apart. So an induced rule that touched personal data is
held: it goes to the sandbox with a review flag, and only a person clears it.

**Retention.** Personal data has a clock. Facts past their limit are deleted on
:meth:`sweep`, and the sweep is something the host calls -- this does not start
a thread behind anyone's back.

On completeness: pattern matching finds what it matches. This is a mechanism
for enforcing a policy, not a guarantee that every personal datum is caught,
and a host that knows its own schema should declare its fields rather than
hoping the patterns cover them.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from ..core.terms import Atom, Const
from ..knowledge.base import FactRecord
from .layers import SESSION, TENANT, LayerStack

__all__ = [
    "PrivacyPolicy",
    "Tag",
    "PERSONAL_PATTERNS",
    "Retention",
    "HeldRule",
]

#: Patterns that are personal wherever they appear. Deliberately few and
#: deliberately specific: a loose pattern that tags everything makes the
#: mechanism useless, because a host will turn it off.
PERSONAL_PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.I),
    "phone": re.compile(r"^\+?\d[\d\s\-()]{7,}$"),
    "iban": re.compile(r"^[A-Z]{2}\d{2}[A-Z0-9]{10,30}$"),
    "national_id": re.compile(r"^\d{6}[-+A]\d{3}[0-9A-Z]$"),  # Finnish henkilötunnus
    "card": re.compile(r"^\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}$"),
}


@dataclass(frozen=True)
class Tag:
    """Why one fact counts as personal."""

    atom: Atom
    kind: str
    #: What made the call: ``declared``, ``pattern:email``, ``detector``.
    by: str
    at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        return f"{self.atom} [{self.kind} by {self.by}]"


@dataclass(frozen=True)
class Retention:
    """How long one kind of personal data may be kept."""

    kind: str
    seconds: float

    def expired(self, tagged_at: float, now: Optional[float] = None) -> bool:
        return (now if now is not None else time.time()) - tagged_at >= self.seconds


@dataclass
class HeldRule:
    """A rule induced from personal data, waiting for a person to look at it."""

    rule: object
    reason: str
    evidence: tuple[Atom, ...] = ()
    cleared: bool = False

    def describe(self) -> str:
        sample = ", ".join(str(a) for a in self.evidence[:3])
        return f"{self.rule}  — {self.reason}" + (f" (from {sample})" if sample else "")


class PrivacyPolicy:
    """Classifies facts, confines them, and holds what was learned from them."""

    def __init__(
        self,
        declared: Optional[dict[str, str]] = None,
        detector: Optional[Callable[[Atom], Optional[str]]] = None,
        retention: Optional[Sequence[Retention]] = None,
    ) -> None:
        #: ``predicate -> kind``, declared by the host. The reliable route.
        self.declared = dict(declared or {})
        #: Optional entity typer (GLiNER in the roadmap). Never required.
        self.detector = detector
        self.retention = {r.kind: r for r in (retention or ())}
        self._tags: dict[Atom, Tag] = {}
        self._held: list[HeldRule] = []

    # -- declaring ----------------------------------------------------------

    def declare(self, predicate: str, kind: str = "personal") -> "PrivacyPolicy":
        """Tell the policy that a predicate carries personal data.

        A host that knows its own schema should use this. Patterns are a
        backstop for what it did not think to declare, not a replacement.
        """
        self.declared[predicate] = kind
        return self

    def keep_for(self, kind: str, seconds: float) -> "PrivacyPolicy":
        self.retention[kind] = Retention(kind, seconds)
        return self

    # -- classifying --------------------------------------------------------

    def classify(self, atom: Atom) -> Optional[str]:
        """What kind of personal data this is, or None."""
        declared = self.declared.get(atom.predicate)
        if declared is not None:
            return declared
        for argument in atom.args:
            if not isinstance(argument, Const):
                continue
            text = str(argument.value)
            for kind, pattern in PERSONAL_PATTERNS.items():
                if pattern.match(text):
                    return kind
        if self.detector is not None:
            try:
                return self.detector(atom)
            except Exception:  # a detector fault must not open the gate
                return "personal"
        return None

    def tag(self, atom: Atom) -> Optional[Tag]:
        """Classify and record. Returns the tag, or None if it is not personal."""
        existing = self._tags.get(atom)
        if existing is not None:
            return existing
        kind = self.declared.get(atom.predicate)
        by = "declared"
        if kind is None:
            kind = self.classify(atom)
            by = f"pattern:{kind}" if kind in PERSONAL_PATTERNS else "detector"
        if kind is None:
            return None
        tag = Tag(atom=atom, kind=kind, by=by)
        self._tags[atom] = tag
        return tag

    def is_personal(self, atom: Atom) -> bool:
        return atom in self._tags or self.classify(atom) is not None

    @property
    def tags(self) -> list[Tag]:
        return list(self._tags.values())

    # -- taking data in -----------------------------------------------------

    def admit(
        self,
        facts: Iterable,
        layers: LayerStack,
        layer: str = SESSION,
        **kwargs,
    ) -> list[FactRecord]:
        """Add facts, routing anything personal into a confined layer.

        The routing is not advisory. A personal fact goes to the session or
        tenant layer whatever the caller asked for, because the caller asking
        for the wrong layer is exactly the mistake this exists to stop.
        """
        admitted: list[FactRecord] = []
        for fact in facts:
            atom = fact.atom if isinstance(fact, FactRecord) else _as_atom(fact)
            tag = self.tag(atom)
            target = layer
            if tag is not None and layer not in (TENANT, SESSION):
                target = SESSION
            if isinstance(fact, FactRecord):
                admitted.append(
                    layers.add_fact(
                        fact.atom,
                        target,
                        confidence=fact.confidence,
                        provenance=fact.provenance,
                        evidence=fact.evidence,
                    )
                )
            else:
                admitted.append(layers.add_fact(atom, target, **kwargs))
        return admitted

    # -- what was learned from it -------------------------------------------

    def review(self, rule, evidence: Sequence[Atom]) -> Optional[HeldRule]:
        """Hold a rule if personal data went into it. Returns the hold, or None.

        A generalisation over many people is usually fine and sometimes a
        re-identification, and nothing here can tell them apart -- so the
        decision goes to a person rather than to a heuristic.
        """
        personal = tuple(atom for atom in evidence if self.is_personal(atom))
        if not personal:
            return None
        held = HeldRule(
            rule=rule,
            reason=f"induced from {len(personal)} personal fact(s); needs human review",
            evidence=personal,
        )
        self._held.append(held)
        return held

    @property
    def held(self) -> list[HeldRule]:
        """Rules waiting for review, most recent last."""
        return [h for h in self._held if not h.cleared]

    def clear(self, held: HeldRule, by: str) -> HeldRule:
        """A person signs off on a held rule."""
        if not by:
            raise ValueError("clearing a held rule requires naming who cleared it")
        held.cleared = True
        held.reason += f"; cleared by {by}"
        return held

    # -- retention ----------------------------------------------------------

    def sweep(self, layers: LayerStack, now: Optional[float] = None) -> list[Atom]:
        """Delete personal facts past their retention limit.

        Called by the host, not by a timer: a background thread deleting data
        is a thing that happens without anyone watching, and this is exactly
        the kind of operation that should not.
        """
        removed: list[Atom] = []
        for atom, tag in list(self._tags.items()):
            limit = self.retention.get(tag.kind)
            if limit is None or not limit.expired(tag.at, now):
                continue
            for layer_name in (SESSION, TENANT):
                if layers[layer_name].knowledge.remove_fact(atom):
                    removed.append(atom)
            self._tags.pop(atom, None)
        return removed

    def forget_session(self, layers: LayerStack) -> int:
        """End a session: drop its layer and every tag that lived there."""
        atoms = [record.atom for record in layers[SESSION].knowledge.facts]
        layers.clear(SESSION)
        for atom in atoms:
            self._tags.pop(atom, None)
        return len(atoms)

    def to_dict(self) -> dict:
        kinds: dict[str, int] = {}
        for tag in self._tags.values():
            kinds[tag.kind] = kinds.get(tag.kind, 0) + 1
        return {
            "declared": dict(sorted(self.declared.items())),
            "tagged": len(self._tags),
            "kinds": dict(sorted(kinds.items())),
            "held_for_review": len(self.held),
            "retention": {k: v.seconds for k, v in sorted(self.retention.items())},
        }


def _as_atom(fact) -> Atom:
    if isinstance(fact, Atom):
        return fact
    from ..core.parser import parse_atom

    return parse_atom(fact)
