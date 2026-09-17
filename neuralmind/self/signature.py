"""What an observation *looks like*, said without naming a domain.

The mind is never told where it is. That is a design rule, and it only works
if the first step is honest about it: turning a payload into facts that do not
already presuppose an answer.

So a signature says things like "records carry a field named ``stock``",
"values are numbers with a currency attached", "something changed between one
observation and the next". It never says "this is a shop". Whether those facts
add up to a shop is a question for rules, which can be argued with, rather
than for a classifier here, which cannot.

Six kinds of fact, from the roadmap:

**structure** -- keys, nesting, lists of similar records, identifiers.
**values** -- numbers with unit strings, currencies, dates, enumerations,
booleans that flip.
**time** -- how regularly observations arrive, what changes between them
(fluents) and what never does.
**interaction** -- does the host offer an action list, ask questions, send a
reward or a score?
**vocabulary** -- the field and value names themselves, normalised.
**conversation** -- is this free text from a person?

Everything is emitted as ground atoms, because the whole point is that the
next layer reasons over them with the same engine as everything else. A
signature fact is evidence, and evidence you can print.

On the limits: this reads shape, not meaning. ``amount`` and ``menge`` are
different symbols here and the facet rules have to say so. The roadmap has
embeddings closing that gap later; what is here does not pretend to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from ..core.terms import Atom, Const

__all__ = ["Signature", "signature_of", "CURRENCIES", "UNIT_PATTERN"]

#: Currency codes and symbols worth recognising by shape alone.
CURRENCIES = {
    "eur", "usd", "gbp", "sek", "nok", "dkk", "chf", "jpy", "pln", "czk",
    "€", "$", "£", "¥",
}

#: "12 mm", "3.2 kN", "1.29 EUR" -- a number with something attached.
UNIT_PATTERN = re.compile(
    r"^\s*(-?\d+(?:[.,]\d+)?)\s*([A-Za-z€$£¥/·^]{1,12})\s*$"
)

#: ISO-ish dates, the one value format that is unambiguous across domains.
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?$")

#: Fields a host uses to offer the mind something to do.
ACTION_KEYS = {"actions", "available_actions", "action_list", "moves", "commands"}
#: Fields carrying a score or reward.
REWARD_KEYS = {"reward", "score", "points", "return", "fitness"}
#: Fields naming a tick or step.
TICK_KEYS = {"tick", "step", "frame", "turn", "t", "timestamp", "time"}


@dataclass
class Signature:
    """The shape of what has been observed so far, as ground atoms."""

    atoms: list[Atom] = field(default_factory=list)
    #: How many observations went into it.
    observations: int = 0
    #: Field names seen, in order of first appearance.
    vocabulary: list[str] = field(default_factory=list)
    #: Fields whose value changed between observations.
    fluents: list[str] = field(default_factory=list)
    #: Fields present every time and never changing.
    constants: list[str] = field(default_factory=list)

    def add(self, atom: Atom) -> bool:
        if atom in self.atoms:
            return False
        self.atoms.append(atom)
        return True

    def has(self, predicate: str, *arguments: str) -> bool:
        wanted = Atom(predicate, tuple(Const(a) for a in arguments))
        if arguments:
            return wanted in self.atoms
        return any(a.predicate == predicate for a in self.atoms)

    def to_asp(self) -> str:
        return "\n".join(f"{atom}." for atom in sorted(self.atoms, key=str))

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.predicate] = counts.get(atom.predicate, 0) + 1
        return {
            "observations": self.observations,
            "facts": len(self.atoms),
            "by_kind": dict(sorted(counts.items())),
            "vocabulary": list(self.vocabulary),
            "fluents": list(self.fluents),
        }

    def __len__(self) -> int:
        return len(self.atoms)

    def __repr__(self) -> str:
        return f"Signature({len(self.atoms)} fact(s) from {self.observations} observation(s))"


class SignatureReader:
    """Accumulates shape facts across a stream of observations.

    Stateful on purpose: "what changes between observations" cannot be read
    off one of them, and it is among the most telling things there is -- a
    field that moves every tick is a fluent, and a host with fluents and an
    action list is something you act in rather than something you query.
    """

    def __init__(self, intents=None) -> None:
        #: Optional intent classifier. A sentence has almost no shape, so
        #: without one the reader sees "free text asking a question" and
        #: nothing about what the question is *for*.
        self.intents = intents
        self.signature = Signature()
        self._previous: dict[str, Any] = {}
        self._seen_values: dict[str, set] = {}
        self._changed: set[str] = set()
        self._always: set[str] = set()

    def observe(self, payload: Any) -> Signature:
        """Fold one observation in and return the signature so far."""
        self.signature.observations += 1
        kind = _shape_of(payload)
        self._emit("payload_kind", kind)

        if kind == "text":
            self._read_text(str(payload))
        elif kind in ("record", "records"):
            records = payload if isinstance(payload, list) else [payload]
            self._emit("record_count", str(len(records)))
            if len(records) > 1 and _similar(records):
                self._emit0("list_of_similar_records")
            for record in records:
                self._read_record(record)
        elif kind == "sequence":
            self._emit0("sequence_payload")
        self._track_change(payload)
        return self.signature

    # -- reading a record ---------------------------------------------------

    def _read_record(self, record: dict, prefix: str = "") -> None:
        for key, value in record.items():
            name = _normalise(f"{prefix}{key}")
            if name not in self.signature.vocabulary:
                self.signature.vocabulary.append(name)
            self._emit("record_with", name)
            if isinstance(value, dict):
                self._emit0("nested_records")
                self._read_record(value, prefix=f"{key}_")
                continue
            if isinstance(value, (list, tuple)):
                if name in ACTION_KEYS or _normalise(key) in ACTION_KEYS:
                    self._emit0("action_list_offered")
                    for action in value[:12]:
                        self._emit("offers_action", _normalise(str(action)))
                self._emit("list_valued", name)
                continue
            self._read_value(name, key, value)

    def _read_value(self, name: str, key: str, value: Any) -> None:
        plain = _normalise(str(key))
        if plain in REWARD_KEYS and isinstance(value, (int, float)):
            self._emit0("reward_offered")
        if plain in TICK_KEYS:
            self._emit0("tick_offered")
        if isinstance(value, bool):
            self._emit("boolean_field", name)
            return
        if isinstance(value, (int, float)):
            self._emit("numeric_field", name)
            if _looks_like_identifier(key):
                self._emit("identifier_field", name)
            return
        text = str(value)
        if DATE_PATTERN.match(text):
            self._emit("date_valued", name)
            return

        match = UNIT_PATTERN.match(text)
        if match:
            unit = _normalise(match.group(2))
            if unit in CURRENCIES:
                self._emit0("value_with_currency")
                # Named as well as flagged: an explanation that says "because
                # price carries a currency" is worth more than one that says
                # "because a currency appeared", and it lets the mind report
                # which fields its reading does *not* account for.
                self._emit("currency_field", name)
                self._emit("currency", unit)
            else:
                self._emit0("quantity_with_unit")
                self._emit("quantity_field", name)
                self._emit("unit", unit)
            return
        if _looks_like_identifier(key) or _looks_like_code(text):
            self._emit("identifier_field", name)
        self._emit("text_field", name)
        # Small closed vocabularies are enumerations; large open ones are not.
        seen = self._seen_values.setdefault(name, set())
        seen.add(text)
        if 1 < len(seen) <= 8 and self.signature.observations >= len(seen) * 2:
            self._emit("enumerated_field", name)

    # -- reading free text --------------------------------------------------

    def _read_text(self, text: str) -> None:
        self._emit0("free_text")
        stripped = text.strip()
        if stripped.endswith("?"):
            self._emit0("asks_questions")
        words = stripped.split()
        if len(words) <= 40:
            self._emit0("utterance_length_short")
        lowered = stripped.lower()
        if re.search(r"\b(i|my|me|we|our)\b", lowered):
            self._emit0("first_person")
        if re.search(r"\b(you|your|please|can you|could you)\b", lowered):
            self._emit0("addresses_someone")
        if self.intents is None:
            return
        try:
            prediction = self.intents.predict(stripped)
        except Exception:
            return
        if not prediction.in_scope:
            # A real answer, and one the roadmap asks for by name: a service
            # desk meets these all day and saying so beats forcing an intent.
            self._emit0("out_of_scope")
            return
        self._emit("intent", prediction.intent)
        if prediction.family:
            self._emit("intent_family", prediction.family)

    # -- what moves ---------------------------------------------------------

    def _track_change(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        flat = _flatten(payload)
        if self._previous:
            for key, value in flat.items():
                if key in self._previous and self._previous[key] != value:
                    self._changed.add(key)
            self._always = {
                key
                for key in flat
                if key in self._previous and key not in self._changed
            }
        self._previous = flat
        for key in sorted(self._changed):
            self._emit("fluent", _normalise(key))
        if self._changed:
            self._emit0("state_changes_between_observations")
        if self.signature.observations >= 3 and not self._changed:
            self._emit0("static_between_observations")
        self.signature.fluents = sorted(_normalise(k) for k in self._changed)
        self.signature.constants = sorted(_normalise(k) for k in self._always)

    # -- emitting -----------------------------------------------------------

    def _emit(self, predicate: str, argument: str) -> None:
        self.signature.add(Atom(predicate, (Const(argument),)))

    def _emit0(self, predicate: str) -> None:
        self.signature.add(Atom(predicate, ()))


def signature_of(payloads: Iterable[Any], intents=None) -> Signature:
    """The signature of a whole stream, in one call."""
    reader = SignatureReader(intents=intents)
    signature = reader.signature
    for payload in payloads:
        signature = reader.observe(payload)
    return signature


# -- helpers ---------------------------------------------------------------


def _shape_of(payload: Any) -> str:
    if isinstance(payload, str):
        return "text"
    if isinstance(payload, dict):
        return "record"
    if isinstance(payload, (list, tuple)):
        return "records" if payload and isinstance(payload[0], dict) else "sequence"
    return "value"


def _similar(records: Sequence) -> bool:
    """True if the records share most of their keys -- a table, not a bag."""
    if len(records) < 2 or not all(isinstance(r, dict) for r in records):
        return False
    first = set(records[0])
    if not first:
        return False
    return all(len(first & set(r)) >= len(first) * 0.6 for r in records[1:])


def _normalise(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return cleaned or "field"


def _looks_like_identifier(key: str) -> bool:
    lowered = _normalise(key)
    return (
        lowered.endswith("_id")
        or lowered in ("id", "sku", "isbn", "iban", "code", "ref", "reference", "uuid")
        or lowered.startswith("id_")
    )


def _looks_like_code(text: str) -> bool:
    """An opaque code: uppercase letters and digits, no spaces."""
    stripped = text.strip()
    if len(stripped) < 3 or len(stripped) > 32 or " " in stripped:
        return False
    return bool(re.match(r"^[A-Z0-9][A-Z0-9\-_]{2,}$", stripped))


def _flatten(payload: dict, prefix: str = "") -> dict:
    flat: dict[str, Any] = {}
    for key, value in payload.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{name}."))
        elif isinstance(value, (list, tuple)):
            flat[name] = tuple(str(v) for v in value)
        else:
            flat[name] = value
    return flat
