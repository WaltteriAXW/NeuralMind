"""Working out where it is, and being able to say why.

The mind is never told. It sees observations, turns them into shape facts
(:mod:`~neuralmind.self.signature`), and runs the facet rules over them with
the same engine it uses for everything else. That last part is the design: a
context hypothesis is a *derivation*, so "retail, because records carry stock
levels and prices in EUR" is the actual reason rather than a story told after
a score came out.

Four things the roadmap asks for, and what each means here.

**Facets first, domains second.** A facet is a building block -- money,
inventory, conversation -- and shows up in many places. A domain is a
combination of them. Recognising facets works even somewhere new, which is
why it is the level the mind commits to and the domain is only a further
guess. A domain's confidence is the weakest facet it rests on, minus a step.

**Stakes before recognition.** Caution does not wait for a domain. Money plus
an action list is high stakes whether or not the mind has worked out it is in
a bank, and :meth:`ContextDiscovery.apply` feeds that to the autonomy gate
immediately. It can only ever *lower* autonomy -- section 1c, and the reason
this is safe to act on while still uncertain.

**Contexts are not exclusive.** A training simulator is part game and part
engineering tool. Facets are reported with weights and blended, never switched.

**Unknown is a real answer.** Below a floor of evidence the honest report is
that the context is unresolved, and the mind stays on what it can do anywhere.

**Drift.** If new observations stop fitting, the mind re-evaluates rather than
holding on to an old guess.

That has to be judged on a *window* of recent observations, not on everything
seen so far, and the reason is worth stating: the signature accumulates. Once
the mind has seen a shop it has seen a shop for ever, so the evidence for the
old host never goes away and a comparison against the running total can never
notice anything. A second reading over the last few observations can, and the
difference between the two is what drift *is*.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from ..core.terms import Atom, Const
from ..inference.engine import ReasoningEngine
from ..inference.proof import ProofNode, explain
from .signature import Signature, SignatureReader, signature_of

__all__ = ["ContextDiscovery", "Reading", "Facet", "PROFILES", "UNRESOLVED"]

PROFILES = Path(__file__).parent / "profiles" / "facets.lp"

#: What the mind reports when nothing is established well enough.
UNRESOLVED = "unresolved"

#: Below this weight a facet is noise rather than evidence.
FLOOR = 3

#: The facets the stakes rules read. Naming them keeps the reason given to
#: the autonomy gate short enough to repeat without becoming noise.
_STAKES_FACETS = frozenset({"money", "parties", "policy"})

#: Evidence where one instance stands for all of them: a facet resting on
#: "a field carries a unit" rests equally on every field that does.
GENERIC = frozenset(
    {"quantity_field", "currency_field", "date_valued", "offers_action",
     "numeric_field", "fluent", "enumerated_field", "identifier_field"}
)

#: Predicates that are how a conclusion was reached, not why it holds.
INTERNAL = frozenset(
    {"facet", "suggests", "beaten", "present", "weight", "smaller",
     "domain", "best_domain", "outranked", "high_stakes", "medium_stakes",
     "stakes"}
)


@dataclass(frozen=True)
class Facet:
    """One building block the mind believes it is dealing with."""

    name: str
    #: 0..10, from the strongest single piece of evidence.
    weight: int
    proof: Optional[ProofNode] = None

    @property
    def confidence(self) -> float:
        return round(self.weight / 10.0, 2)

    def because(self) -> list[str]:
        """The signature facts this rests on, in their own terms.

        Only the observations. The rule machinery that picked the strongest
        piece of evidence -- ``suggests``, ``beaten`` -- is how the conclusion
        was reached and not why it is true, and putting it in an explanation
        for a person is noise wearing the clothes of rigour.
        """
        if self.proof is None:
            return []
        found = []
        for node in self.proof.walk():
            if node.conclusion.predicate in INTERNAL or node.negated:
                continue
            if node.is_leaf and str(node.conclusion) not in found:
                found.append(str(node.conclusion))
        return found

    def describe(self) -> str:
        reasons = ", ".join(self.because())
        return f"{self.name} ({self.confidence:.2f})" + (f" — {reasons}" if reasons else "")

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Reading:
    """What the mind currently makes of where it is."""

    facets: list[Facet] = field(default_factory=list)
    domain: str = UNRESOLVED
    domain_weight: int = 0
    stakes: str = "low"
    observations: int = 0
    #: True when nothing reached the evidence floor.
    unresolved: bool = True
    signature: Optional[Signature] = None
    #: Field names that contributed to no facet. In a host nobody planned for
    #: this is most of the vocabulary, and it is what the module builder
    #: (P2.8) is handed: the mind recognises the facets it knows and says
    #: plainly which words it could not place.
    unexplained: list[str] = field(default_factory=list)

    @property
    def domain_confidence(self) -> float:
        return round(self.domain_weight / 10.0, 2)

    def facet(self, name: str) -> Optional[Facet]:
        return next((f for f in self.facets if f.name == name), None)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.facets]

    def describe(self) -> str:
        """One sentence a person can argue with."""
        if self.unresolved:
            return (
                f"Context unresolved after {self.observations} observation(s); "
                f"stakes {self.stakes}."
            )
        strongest = self.facets[0]
        others = ", ".join(f"{f.name} {f.confidence:.2f}" for f in self.facets[1:4])
        head = f"{self.domain} ({self.domain_confidence:.2f})" if self.domain != UNRESOLVED else "no domain matched"
        return (
            f"{head}; facets {strongest.name} {strongest.confidence:.2f}"
            + (f", {others}" if others else "")
            + f"; stakes {self.stakes}. Because: {', '.join(strongest.because())}."
        )

    @property
    def covered(self) -> float:
        """The share of the vocabulary the facets account for.

        Low coverage with a confident domain is the combination worth
        distrusting: it means a few recognised words carried a reading that
        most of the observation does not support.
        """
        seen = len(self.signature.vocabulary) if self.signature else 0
        if not seen:
            return 1.0
        return round((seen - len(self.unexplained)) / seen, 2)

    def why_stakes(self) -> str:
        """The facets behind the stakes, named. One clause, and it repeats."""
        drivers = [f.name for f in self.facets if f.name in _STAKES_FACETS]
        if not drivers:
            return "no facet raises the stakes"
        return "facets " + ", ".join(sorted(drivers))

    def to_dict(self) -> dict:
        return {
            "domain": self.domain,
            "domain_confidence": self.domain_confidence,
            "unresolved": self.unresolved,
            "stakes": self.stakes,
            "observations": self.observations,
            "unexplained": list(self.unexplained),
            "covered": self.covered,
            "facets": [
                {
                    "name": f.name,
                    "confidence": f.confidence,
                    "because": f.because(),
                }
                for f in self.facets
            ],
        }

    def __str__(self) -> str:
        return self.describe()


class ContextDiscovery:
    """Reads observations and works out what kind of place this is.

    Holds no domain of its own. Everything it knows about what a shop or a
    bank looks like lives in ``profiles/facets.lp``, which is a file a person
    can read and disagree with.
    """

    def __init__(
        self,
        profiles: Optional[str] = None,
        intents=None,
        floor: int = FLOOR,
        drift_after: int = 3,
    ) -> None:
        self.rules = profiles if profiles is not None else PROFILES.read_text(encoding="utf-8")
        self.floor = floor
        #: How many observations that do not fit before the reading is redone.
        self.drift_after = drift_after
        self._intents = _default_intents() if intents is None else intents
        self.reader = SignatureReader(intents=self._intents)
        self.reading = Reading()
        #: Every time the reading changed materially, with what it changed to.
        self.changes: list[tuple[int, str, str]] = []
        self._misfits = 0
        self._extra: list[Atom] = []
        #: The last few payloads, re-read on their own to see what is arriving
        #: *now* rather than what has ever arrived.
        self._window: deque = deque(maxlen=max(drift_after, 3))
        #: The facets believed at the last settled point. Not the cumulative
        #: reading: that absorbs the new host as it arrives, so comparing the
        #: window against it compares the new evidence with itself.
        self._established: set[str] = set()

    # -- taking observations ------------------------------------------------

    def observe(self, payload: Any) -> Reading:
        """Fold one observation in and re-read the context."""
        before = self.reading.domain
        signature = self.reader.observe(payload)
        self.reading = self._read(signature)

        self._window.append(payload)
        if len(self._window) == self._window.maxlen:
            recent = self._read(
                signature_of(list(self._window), intents=self._intents)
            )
            fresh = set(recent.names)
            if not self._established:
                self._established = fresh
            elif fresh and _mostly_gone(self._established, fresh):
                # Drift is what was settled *disappearing*, not what is new
                # arriving. Measuring disjointness instead misses the whole
                # transition: while the window straddles both hosts it shows
                # everything, which looks like agreement with both.
                self._misfits += 1
            else:
                self._misfits = max(0, self._misfits - 1)
                if not self._misfits and fresh:
                    self._established = fresh

        if self.reading.domain != before:
            self.changes.append((self.reading.observations, before, self.reading.domain))
        return self.reading

    def observe_all(self, payloads: Iterable[Any]) -> Reading:
        for payload in payloads:
            self.observe(payload)
        return self.reading

    def note(self, *atoms: Atom) -> "ContextDiscovery":
        """Add a signature fact from outside -- ``personal_data_present``, say.

        The privacy layer knows things the shape reader cannot: whether a field
        holds personal data is a policy question, not a syntactic one.
        """
        for atom in atoms:
            if atom not in self._extra:
                self._extra.append(atom)
        if self.reader.signature.observations:
            self.reading = self._read(self.reader.signature)
        return self

    # -- drift ---------------------------------------------------------------

    @property
    def drifting(self) -> bool:
        """True when recent observations keep not fitting what is believed.

        A host that switches under the mind shows up here, and the honest
        response is to stop acting rather than to keep applying the old
        reading -- which is what :meth:`apply` does with it.
        """
        return self._misfits >= self.drift_after

    def reset(self) -> "ContextDiscovery":
        """Forget the reading and start again. What a detected switch deserves."""
        self.reader = SignatureReader(intents=self._intents)
        self.reading = Reading()
        self._misfits = 0
        self._window.clear()
        self._established = set()
        return self

    # -- using it -----------------------------------------------------------

    def apply(self, gate) -> Reading:
        """Tell the autonomy gate what the stakes look like.

        One direction only. An inferred context can make the mind more careful
        and can never make it freer -- a misread observation that raised
        autonomy is how a guess authorises a transfer.
        """
        reading = self.reading
        if reading.stakes in ("medium", "high"):
            # Short and stable: the gate's reasons accumulate, and re-reading
            # the context every observation would otherwise fill them with a
            # slightly different copy of the same sentence each time.
            gate.raise_stakes(reading.stakes, reading.why_stakes())
        if self.drifting:
            gate.raise_stakes(
                "high", "the observations stopped fitting; the context is being re-read"
            )
        return reading

    # -- reading the rules ---------------------------------------------------

    def _read(self, signature: Signature) -> Reading:
        facts = "\n".join(f"{atom}." for atom in list(signature.atoms) + self._extra)
        try:
            engine = ReasoningEngine(self.rules + "\n" + facts)
            model = engine.solve()
        except Exception:
            return Reading(observations=signature.observations, signature=signature)

        facets = []
        for atom in model.by_predicate("facet", 2):
            name, weight = (_value(a) for a in atom.args)
            if int(weight) < self.floor:
                continue
            try:
                proof = explain(model, atom)
            except Exception:  # pragma: no cover - proof is best-effort
                proof = None
            facets.append(Facet(str(name), int(weight), proof))
        facets.sort(key=lambda f: (-f.weight, f.name))

        domain, domain_weight = UNRESOLVED, 0
        for atom in model.by_predicate("best_domain", 2):
            name, weight = (_value(a) for a in atom.args)
            if int(weight) > domain_weight:
                domain, domain_weight = str(name), int(weight)

        stakes = "low"
        for level in ("high", "medium", "low"):
            if model.holds(Atom("stakes", (Const(level),))):
                stakes = level
                break

        return Reading(
            facets=facets,
            domain=domain,
            domain_weight=domain_weight,
            stakes=stakes,
            observations=signature.observations,
            unresolved=not facets,
            signature=signature,
            unexplained=_unexplained(signature, facets),
        )


def _mostly_gone(established: set, fresh: set, share: float = 0.6) -> bool:
    """True when most of what was believed is no longer being observed."""
    lost = established - fresh
    return len(lost) >= max(1, int(len(established) * share))


def _unexplained(signature: Signature, facets: Sequence[Facet]) -> list[str]:
    """Field names nothing the facets rest on ever mentions.

    Computed from the *kinds* of evidence that fired, not from the one proof
    each facet happens to display. A facet backed by ``quantity_field`` is
    backed by every quantity field there is, and crediting only the one the
    derivation names would report a reading as covering far less than it does.
    """
    explained: set[str] = set()
    kinds: set[str] = set()
    for facet in facets:
        if facet.proof is None:
            continue
        for node in facet.proof.walk():
            predicate = node.conclusion.predicate
            if predicate in INTERNAL or node.negated:
                continue
            kinds.add(predicate)
            # The specific evidence used: "record_with(stock)" explains stock
            # and says nothing about supplier.
            for argument in node.conclusion.args:
                explained.add(str(getattr(argument, "value", argument)))
    # Generic evidence: a facet resting on "there is a field with a unit" rests
    # on all of them equally, so every such field is accounted for. GENERIC is
    # deliberately a short list -- "record_with" is emitted for literally every
    # field and would mark the whole vocabulary explained for free.
    for atom in signature.atoms:
        if atom.predicate in kinds and atom.predicate in GENERIC:
            for argument in atom.args:
                explained.add(str(getattr(argument, "value", argument)))
    return [name for name in signature.vocabulary if name not in explained]


def _value(term) -> Any:
    return term.value if isinstance(term, Const) else str(term)


def _default_intents():
    """Load the shipped intent classifier, or None if it is not usable."""
    from .intent import IntentClassifier, numpy_available

    if not numpy_available():
        return None
    weights = Path(__file__).parent / "weights" / "intents.npz"
    if not weights.exists():
        return None
    try:
        return IntentClassifier.load(weights)
    except Exception:
        return None
