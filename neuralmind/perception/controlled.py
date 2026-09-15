"""Controlled English -> logic.

Open-domain natural language is exactly where this architecture is weakest, so
this parser does not pretend to handle it. It covers a controlled subset --
the register used by rule-reasoning benchmarks like ProofWriter and by written
policies and specifications -- and reports anything outside that subset as
unparsed rather than guessing.

What it covers::

    Bob is blue.                      attr(bob, blue)
    Bob is not blue.                  not_attr(bob, blue)
    Bob is a cat.                     isa(bob, cat)
    The cat likes the dog.            rel(cat, likes, dog)
    All cats are mammals.             isa(X, mammal) :- isa(X, cat).
    If someone is round and blue      attr(X, green) :- attr(X, round),
      then they are green.                             attr(X, blue).
    Is Bob green?                     query: attr(bob, green)

Every output is a ground atom or a rule, so what comes out is auditable
against what went in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.program import Rule
from ..core.terms import Atom, Const, Literal, Term, Var
from ..knowledge.base import FactRecord
from .base import Perception
from .lexicon import (
    AUXILIARIES,
    base_verb,
    COPULAS,
    DETERMINERS,
    NEGATIONS,
    PRONOUNS,
    QUANTIFIED,
    normalise_symbol,
    singularise,
    split_sentences,
    strip_determiner,
    tokenize,
)

__all__ = ["ControlledEnglishParser", "TripleSchema", "Clause"]

#: Predicate names of the default triple schema.
ATTR = "attr"
ISA = "isa"
REL = "rel"
ACT = "act"
NEG_PREFIX = "not_"

_VARIABLE_NAMES = ("X", "Y", "Z", "W")


@dataclass
class TripleSchema:
    """How clauses become atoms.

    ``"triple"`` keeps an open predicate vocabulary -- ``rel(cat, likes, dog)``
    -- so new verbs never require new predicate symbols, which is what makes
    generic rules over unseen vocabulary possible. ``"direct"`` emits
    domain-shaped predicates instead: ``likes(cat, dog)``, ``blue(bob)``.
    """

    style: str = "triple"

    def attribute(self, subject: Term, attribute: str, negated: bool = False) -> Atom:
        if self.style == "direct":
            name = (NEG_PREFIX if negated else "") + normalise_symbol(attribute)
            return Atom(name, (subject,))
        return Atom((NEG_PREFIX if negated else "") + ATTR, (subject, Const(normalise_symbol(attribute))))

    def membership(self, subject: Term, class_name: str, negated: bool = False) -> Atom:
        if self.style == "direct":
            name = (NEG_PREFIX if negated else "") + normalise_symbol(class_name)
            return Atom(name, (subject,))
        return Atom((NEG_PREFIX if negated else "") + ISA, (subject, Const(normalise_symbol(class_name))))

    def action(self, subject: Term, verb: str, negated: bool = False) -> Atom:
        """An intransitive action: "the dog barks"."""
        if self.style == "direct":
            return Atom((NEG_PREFIX if negated else "") + normalise_symbol(verb), (subject,))
        return Atom(
            (NEG_PREFIX if negated else "") + ACT, (subject, Const(normalise_symbol(verb)))
        )

    def relation(self, subject: Term, verb: str, obj: Term, negated: bool = False) -> Atom:
        if self.style == "direct":
            name = (NEG_PREFIX if negated else "") + normalise_symbol(verb)
            return Atom(name, (subject, obj))
        return Atom(
            (NEG_PREFIX if negated else "") + REL,
            (subject, Const(normalise_symbol(verb)), obj),
        )


@dataclass
class Clause:
    """One parsed proposition, before it becomes an atom."""

    kind: str  # attribute | membership | relation
    subject: str
    value: str
    obj: Optional[str] = None
    negated: bool = False
    text: str = ""

    def entities(self) -> list[str]:
        return [e for e in (self.subject, self.obj) if e]


class ControlledEnglishParser:
    """Parses controlled English into facts, rules and queries."""

    name = "controlled-english"

    def __init__(self, schema: Optional[TripleSchema] = None, confidence: float = 1.0) -> None:
        self.schema = schema or TripleSchema()
        #: Rule-based extraction is deterministic, so the default is certainty.
        self.confidence = confidence
        self._bare_mentions: set[str] = set()
        self._determined_mentions: set[str] = set()

    # -- public API ------------------------------------------------------

    def perceive(self, raw: str) -> Perception:
        """Parse a passage into facts and rules."""
        perception = Perception(source=f"text:{self.name}")
        self._bare_mentions = set()
        self._determined_mentions = set()
        for sentence in split_sentences(raw):
            if not sentence.strip():
                continue
            try:
                self._parse_sentence(sentence, perception)
            except _ParseFailure as failure:
                perception.unparsed.append(f"{sentence}  ({failure})")
        perception.diagnostics["sentences"] = len(split_sentences(raw))
        perception.diagnostics["unparsed"] = len(perception.unparsed)
        # A noun phrase that never takes a determiner is almost always a proper
        # name in English; the output layer uses this to capitalise correctly.
        perception.diagnostics["proper_names"] = sorted(
            self._bare_mentions - self._determined_mentions
        )
        return perception

    def parse_question(self, question: str) -> Atom:
        """Turn a yes/no question into the atom to query.

        ``"Is Bob green?"`` -> ``attr(bob, green)``;
        ``"Does the cat like the dog?"`` -> ``rel(cat, like, dog)``.
        """
        tokens = tokenize(question.rstrip("?"))
        if not tokens:
            raise ValueError("empty question")
        lowered = [t.lower() for t in tokens]
        if lowered[0] in COPULAS:
            # "Is Bob green" -> "Bob is green"
            reordered = tokens[1:2] + tokens[0:1] + tokens[2:]
        elif lowered[0] in AUXILIARIES:
            # "Does the cat like the dog" -> "the cat like the dog"
            reordered = tokens[1:]
        else:
            reordered = tokens
        clauses = self._parse_clause_group(" ".join(reordered))
        if not clauses:
            raise ValueError(f"could not interpret the question {question!r}")
        return self._to_atom(clauses[0], {})

    # -- sentence dispatch ------------------------------------------------

    def _parse_sentence(self, sentence: str, perception: Perception) -> None:
        text = sentence.strip().rstrip(".!")
        lowered = text.lower()
        if lowered.startswith("if ") and " then " in lowered:
            self._parse_conditional(text, perception)
            return
        if lowered.startswith(("all ", "every ", "each ")):
            self._parse_universal(text, perception)
            return
        if _is_bare_plural_subject(text):
            # "Cats are animals." states a universal just as "All cats ..." does.
            self._parse_universal("all " + text, perception)
            return
        clauses = self._parse_clause_group(text)
        if not clauses:
            raise _ParseFailure("no clause recognised")
        bindings: dict[str, Term] = {}
        for clause in clauses:
            atom = self._to_atom(clause, bindings)
            if not atom.is_ground:
                raise _ParseFailure(
                    "a standalone statement about 'someone' or 'something' is a rule, "
                    "not a fact -- phrase it as 'If ... then ...'"
                )
            perception.facts.append(
                FactRecord(
                    atom=atom,
                    confidence=self.confidence,
                    provenance=f"perception:{self.name}",
                    evidence=sentence.strip(),
                )
            )

    def _parse_conditional(self, text: str, perception: Perception) -> None:
        body_text, _, head_text = text[3:].partition(" then ")
        if not body_text.strip() or not head_text.strip():
            raise _ParseFailure("an 'if ... then ...' sentence needs both halves")
        body_clauses = self._parse_clause_group(body_text)
        head_clauses = self._parse_clause_group(head_text)
        if not body_clauses or not head_clauses:
            raise _ParseFailure("could not read the condition or the consequence")
        bindings: dict[str, Term] = {}
        body_atoms = [self._to_atom(c, bindings) for c in body_clauses]
        # The consequence reuses the condition's bindings, which is how "they"
        # and "it" refer back to the individual the condition introduced.
        head_atoms = [self._to_atom(c, bindings) for c in head_clauses]
        for head in head_atoms:
            perception.rules.append(
                Rule(
                    head=head,
                    body=tuple(Literal(atom) for atom in body_atoms),
                    source=f"perception:{self.name}",
                    label=text.strip(),
                )
            )

    def _parse_universal(self, text: str, perception: Perception) -> None:
        tokens = tokenize(text)
        quantifier = tokens[0].lower()
        rest = tokens[1:]
        if not rest:
            raise _ParseFailure(f"'{quantifier}' needs a noun")
        # "All cats are mammals" / "Every cat that is blue is round"
        noun = singularise(rest[0])
        remainder = rest[1:]
        condition_text = ""
        if remainder and remainder[0].lower() in ("that", "which", "who"):
            inner = remainder[1:]
            inner_verb = _find_main_copula(inner)
            if inner_verb is None:
                raise _ParseFailure("a 'that ...' clause needs its own verb")
            # The relative clause runs until the outer clause's own verb, so
            # "All dogs that are big are loud" splits at the second "are".
            outer_verb = _find_main_copula(inner[inner_verb + 1 :])
            if outer_verb is None:
                raise _ParseFailure("nothing is stated about the quantified noun")
            split_at = inner_verb + 1 + outer_verb
            condition_text = " ".join(inner[:split_at])
            remainder = inner[split_at:]
        if not remainder:
            raise _ParseFailure("nothing stated about the quantified noun")
        variable = Var(_VARIABLE_NAMES[0])
        body_atoms = [self.schema.membership(variable, noun)]
        if condition_text:
            for clause in self._parse_clause_group("it " + condition_text):
                body_atoms.append(self._to_atom(clause, {"it": variable}))
        head_clauses = self._parse_clause_group("it " + " ".join(remainder))
        if not head_clauses:
            raise _ParseFailure("could not read what is stated about the noun")
        for clause in head_clauses:
            head = self._to_atom(clause, {"it": variable})
            perception.rules.append(
                Rule(
                    head=head,
                    body=tuple(Literal(atom) for atom in body_atoms),
                    source=f"perception:{self.name}",
                    label=text.strip(),
                )
            )

    # -- clause parsing ---------------------------------------------------

    def _parse_clause_group(self, text: str) -> list[Clause]:
        """Split conjoined text into clauses and parse each one.

        ``"someone is round and blue"`` is one clause with two complements,
        while ``"someone is round and the dog is blue"`` is two clauses. The
        difference is whether a fragment carries its own verb.
        """
        fragments = _split_conjuncts(text)
        clauses: list[Clause] = []
        for fragment in fragments:
            tokens = tokenize(fragment)
            if not tokens:
                continue
            if _find_main_copula(tokens) is None and not _find_verb_position(tokens):
                # A bare complement continues the previous clause: in
                # "someone is small and green and a lion", the last two
                # fragments have no verb of their own.
                if not clauses:
                    raise _ParseFailure(f"fragment {fragment!r} has no verb")
                clauses.append(self._continuation(clauses[-1], tokens, fragment))
                continue
            clauses.append(self._parse_clause(tokens, fragment))
        return clauses

    def _parse_clause(self, tokens: list[str], text: str) -> Clause:
        copula_at = _find_main_copula(tokens)
        if copula_at is not None:
            subject = " ".join(tokens[:copula_at])
            rest = tokens[copula_at + 1 :]
            negated = bool(rest) and rest[0].lower() in NEGATIONS
            if negated:
                rest = rest[1:]
            if not subject:
                raise _ParseFailure(f"{text!r} has no subject")
            if not rest:
                raise _ParseFailure(f"{text!r} says nothing about the subject")
            if rest[0].lower() in DETERMINERS:
                return Clause("membership", subject, " ".join(rest[1:]), negated=negated, text=text)
            # "Bob is bigger than Alice" -> a relation, not an attribute.
            if len(rest) > 2 and rest[-2].lower() in ("than", "to", "of"):
                return Clause(
                    "relation",
                    subject,
                    " ".join(rest[:-2] + [rest[-2]]),
                    obj=rest[-1],
                    negated=negated,
                    text=text,
                )
            complement = " ".join(rest)
            # A bare plural complement names a class: "Cats are mammals".
            # Adjectives ending in -ous/-ss/-us/-is survive singularise()
            # unchanged, so they stay attributes.
            if len(rest) == 1 and singularise(rest[0]) != rest[0].lower():
                return Clause(
                    "membership", subject, singularise(rest[0]), negated=negated, text=text
                )
            return Clause("attribute", subject, complement, negated=negated, text=text)

        verb_at = _find_verb_position(tokens)
        if not verb_at:
            raise _ParseFailure(f"{text!r} has no verb")
        start, end, negated = verb_at
        subject = " ".join(tokens[:start])
        verb = " ".join(tokens[start:end])
        obj_tokens = strip_determiner(tokens[end:])
        if not subject:
            raise _ParseFailure(f"{text!r} has no subject")
        if not obj_tokens:
            # "the dog barks" -- an intransitive action, not an attribute.
            return Clause("action", subject, verb, negated=negated, text=text)
        return Clause(
            "relation", subject, verb, obj=" ".join(obj_tokens), negated=negated, text=text
        )

    def _continuation(self, previous: Clause, tokens: list[str], text: str) -> Clause:
        """Attach a verbless fragment to the clause before it.

        The fragment's own shape decides its kind, not the previous clause's:
        "a lion" is a class even when it follows "is small".
        """
        if previous.kind == "relation":
            return Clause(
                kind="relation",
                subject=previous.subject,
                value=previous.value,
                obj=" ".join(strip_determiner(tokens)),
                negated=previous.negated,
                text=text,
            )
        if tokens[0].lower() in DETERMINERS:
            return Clause(
                kind="membership",
                subject=previous.subject,
                value=singularise(" ".join(strip_determiner(tokens))),
                negated=previous.negated,
                text=text,
            )
        if len(tokens) == 1 and singularise(tokens[0]) != tokens[0].lower():
            return Clause(
                kind="membership",
                subject=previous.subject,
                value=singularise(tokens[0]),
                negated=previous.negated,
                text=text,
            )
        return Clause(
            kind=previous.kind,
            subject=previous.subject,
            value=self._complement_value(tokens, previous.kind),
            negated=previous.negated,
            text=text,
        )

    def _complement_value(self, tokens: list[str], kind: str) -> str:
        if kind == "membership":
            return " ".join(strip_determiner(tokens))
        return " ".join(tokens)

    # -- clause -> atom ----------------------------------------------------

    def _to_atom(self, clause: Clause, bindings: dict[str, Term]) -> Atom:
        subject = self._term(clause.subject, bindings)
        if clause.kind == "attribute":
            return self.schema.attribute(subject, clause.value, clause.negated)
        if clause.kind == "membership":
            return self.schema.membership(subject, singularise(clause.value), clause.negated)
        if clause.kind == "action":
            return self.schema.action(subject, base_verb(clause.value), clause.negated)
        obj = self._term(clause.obj or "", bindings)
        return self.schema.relation(subject, base_verb(clause.value), obj, clause.negated)

    def _term(self, phrase: str, bindings: dict[str, Term]) -> Term:
        """Map a noun phrase to a constant, or to a shared variable.

        Quantified phrases ('someone', 'something') introduce a fresh variable.
        Pronouns bind to the individual already introduced, which is what makes
        'If someone is red then they are round' a single-variable rule.
        """
        raw_tokens = tokenize(phrase)
        tokens = strip_determiner(raw_tokens)
        if not tokens:
            raise _ParseFailure("empty noun phrase")
        key = tokens[-1].lower()
        if key not in PRONOUNS and key not in QUANTIFIED:
            symbol = normalise_symbol(" ".join(tokens))
            if len(raw_tokens) != len(tokens) or singularise(tokens[-1]) != key:
                # Took a determiner, or is plural: a class, not an individual.
                self._determined_mentions.add(symbol)
            else:
                self._bare_mentions.add(symbol)
        if key in PRONOUNS:
            if bindings:
                # 'it'/'they' refer to the first individual introduced.
                return next(iter(bindings.values()))
            variable = Var(_VARIABLE_NAMES[0])
            bindings[key] = variable
            return variable
        if key in QUANTIFIED:
            existing = bindings.get(key)
            if existing is not None:
                return existing
            variable = Var(_VARIABLE_NAMES[min(len(bindings), len(_VARIABLE_NAMES) - 1)])
            bindings[key] = variable
            return variable
        return Const(normalise_symbol(" ".join(tokens)))


class _ParseFailure(Exception):
    """This sentence is outside the controlled subset."""


def _split_conjuncts(text: str) -> list[str]:
    """Split on ', and' / 'and' / ',' without breaking inside a noun phrase."""
    normalised = text.replace(", and ", " and ").replace(",and ", " and ")
    parts: list[str] = []
    for comma_part in normalised.split(","):
        for fragment in comma_part.split(" and "):
            stripped = fragment.strip()
            if stripped:
                parts.append(stripped)
    return parts


def _is_bare_plural_subject(text: str) -> bool:
    """True for 'Cats are animals.' -- a plural noun with no determiner."""
    tokens = tokenize(text)
    if len(tokens) < 3 or tokens[0].lower() in DETERMINERS:
        return False
    if tokens[1].lower() not in ("are", "were"):
        return False
    return singularise(tokens[0]) != tokens[0].lower()


def _find_main_copula(tokens: Sequence[str]) -> Optional[int]:
    """Index of the first copular verb, or None."""
    for index, token in enumerate(tokens):
        if token.lower() in COPULAS:
            return index
    return None


def _find_verb_position(tokens: Sequence[str]) -> Optional[tuple[int, int, bool]]:
    """Locate a non-copular verb: returns (start, end, negated).

    The subject is assumed to be the noun phrase before the verb. With no POS
    tagger available, the heuristic is that the verb follows the subject's noun
    phrase -- so the first token after a determiner-led phrase, or the second
    token otherwise.
    """
    if len(tokens) < 2:
        return None
    index = 1 if tokens[0].lower() not in DETERMINERS else 2
    if index >= len(tokens):
        return None
    negated = False
    if tokens[index].lower() in AUXILIARIES:
        index += 1
        if index < len(tokens) and tokens[index].lower() in NEGATIONS:
            negated = True
            index += 1
    if index >= len(tokens):
        return None
    start = index
    end = index + 1
    return (start, end, negated)
