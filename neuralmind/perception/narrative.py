"""Reading free-form English theory text with a dependency parse.

The controlled grammar in :mod:`neuralmind.perception.controlled` is exact on
the register it was written for and helpless outside it. Real prose says

    Charlie is green, but often kind, even when he is blue and cold.

where the controlled version would say "Charlie is green. Charlie is kind.
Charlie is blue. Charlie is cold." Four facts, one sentence, a pronoun standing
in for the subject, and an "even when" that looks like a conditional and is not.

This extractor works from the parse instead of from a grammar. It reads the
patterns that actually occur in crowdsourced theory text:

* **attributes strung together.** Each sentence is split into spans and every
  predicative adjective in a span is collected, so "green, red and blue" is
  three facts and not one. Walking the dependency edges instead loses half of
  them: spaCy hangs a coordinated complement off the root rather than off the
  first complement, and a clausal subject carries no marker edge at all;
* **pronouns** resolved to the sentence's own subject, which is what makes the
  second half of the sentence above recoverable at all;
* **adverbial clauses as co-assertions.** "as he is green", "even when he is
  blue" assert; only *if*/*when*-marked clauses in a generic sentence condition;
* **generic subjects** -- "Young people who are nice", "A kind person who is
  blue" -- read as universally quantified rules, with the modifiers and the
  relative clause as conditions;
* **hedging discarded.** "rather", "often", "seems to", "tends to", "is going
  to be" carry no logical content here, and keeping them would invent
  predicates nobody asked about.

It is a heuristic reader of one register, not a semantic parser, and it is
wrong often enough that the measured numbers are quoted in the README rather
than described. What it is not is a language model: every fact it produces is
traceable to a specific dependency edge.
"""

from __future__ import annotations

from typing import Optional

from ..core.program import Rule
from ..core.terms import Atom, Const, Literal, Term, Var
from ..knowledge.base import FactRecord
from .base import Perception, PerceptionError
from .controlled import TripleSchema
from .lexicon import PLACEHOLDER_NOUNS, normalise_symbol, predicate_name

__all__ = ["NarrativeExtractor"]

#: Markers that make a subordinate clause a condition rather than an assertion.
CONDITIONAL_MARKERS = frozenset({"if", "when", "whenever", "unless", "provided"})

#: Verbs that carry no content of their own: "tends to be", "seems to be".
HEDGE_VERBS = frozenset(
    {"tend", "seem", "appear", "go", "look", "become", "get", "turn", "remain",
     "stay", "keep", "be", "have", "say", "call", "name", "consider"}
)

#: Nouns standing in for "any individual".
GENERIC_NOUNS = PLACEHOLDER_NOUNS | {
    "someone", "somebody", "anyone", "anybody", "everyone", "everybody",
    "guy", "guys", "folk", "folks", "creature", "creatures",
}

_PRONOUNS = frozenset({"he", "she", "it", "they", "him", "her", "them", "his", "their"})

#: Nouns used only to hedge an adjective: "on the big side", "of a kind sort".
HEDGE_NOUNS = frozenset({"side", "way", "sort", "kind", "type", "note"})

#: Verbs that equate a description with a name: "is named Dave".
NAMING_VERBS = frozenset({"name", "call", "know"})

#: Structural reliability weight; see the note in perception/text.py.
CONFIDENCE = 0.7


class NarrativeExtractor:
    """Extracts facts and rules from free-form theory prose."""

    name = "narrative"

    def __init__(
        self,
        model: str = "en_core_web_sm",
        schema: Optional[TripleSchema] = None,
        confidence: float = CONFIDENCE,
        resolve_pronouns: bool = True,
    ) -> None:
        #: Whether a pronoun subject may take the previous sentence's subject.
        self.resolve_pronouns = resolve_pronouns
        self._antecedent = None
        self.model = model
        self.schema = schema or TripleSchema()
        self.confidence = confidence
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            from .text import _load_spacy

            try:
                self._nlp = _load_spacy(self.model)
            except Exception as exc:  # pragma: no cover - environment dependent
                raise PerceptionError(
                    f"could not load the spaCy model {self.model!r}: {exc}"
                ) from exc
        return self._nlp

    # -- entry point --------------------------------------------------------

    def perceive(self, raw: str) -> Perception:
        perception = Perception(source=f"text:{self.name}")
        doc = self.nlp(raw)
        names: set[str] = set()
        # The antecedent of a bare "he"/"they" is almost always the individual
        # the previous sentence was about. Carrying it forward is what makes
        # "Charlie is cold. He is also rough." two facts about Charlie rather
        # than one fact and one unreadable sentence.
        self._antecedent = None
        for sentence in doc.sents:
            try:
                produced = self._read_sentence(sentence, perception, names)
            except Exception:  # a parse this reader cannot use
                produced = False
            if not produced:
                perception.unparsed.append(sentence.text.strip())
        perception.diagnostics["model"] = self.model
        perception.diagnostics["proper_names"] = sorted(names)
        perception.diagnostics["unparsed"] = len(perception.unparsed)
        return perception

    # -- sentences ----------------------------------------------------------

    def _read_sentence(self, sentence, perception: Perception, names: set) -> bool:
        subject = _subject_of(sentence.root)
        marker = _conditional_marker(sentence)
        if subject is not None and subject.pos_ == "PROPN":
            # "Charlie is green ... even when he is blue" describes Charlie.
            # A marker here is concessive, not a condition: the sentence says
            # nothing about anyone else.
            marker = None
        if marker is not None:
            return self._emit_conditional(sentence, marker, perception)
        if _is_generic(subject):
            return self._emit_generic_rule(sentence, subject, perception)
        return self._emit_facts(sentence, sentence.root, subject, perception, names)

    # -- rules ---------------------------------------------------------------

    def _emit_conditional(self, sentence, marker, perception: Perception) -> bool:
        """"If/when <condition> then <conclusion>", split by position.

        Working from the dependency edges alone loses half of these. In "When
        someone is kind yet can be cold and blue, they will also be very big"
        spaCy hangs "is kind" off the root as a clausal subject and "can be
        cold" as a complement clause, with the marker attached to neither -- so
        a walk over the condition clause's children finds "kind" and stops. The
        span between the marker and the consequent contains all three.
        """
        condition, conclusion = _split_at_marker(sentence, marker)
        if not condition or not conclusion:
            return False
        return self._rule_from_spans(sentence, condition, conclusion, perception)

    def _emit_generic_rule(self, sentence, subject, perception: Perception) -> bool:
        """"Young people who are nice are green" -- the subject is the condition."""
        described = {token.i for token in subject.subtree}
        conclusion = [t for t in sentence if t.i not in described]
        condition = [t for t in sentence if t.i in described]
        if not condition or not conclusion:
            return False
        return self._rule_from_spans(sentence, condition, conclusion, perception)

    def _rule_from_spans(self, sentence, condition, conclusion, perception) -> bool:
        variable = Var("X")
        body = self._literals_in(condition, variable)
        heads = self._literals_in(conclusion, variable)
        if not body or not heads:
            return False
        unique_body = tuple(dict.fromkeys(body))
        emitted = False
        for head in heads:
            if head.negated:
                continue  # a negative conclusion is not a definite rule
            if head in unique_body:
                continue  # "if X is blue then X is blue" says nothing
            perception.rules.append(
                Rule(
                    head=head.atom,
                    body=unique_body,
                    source=f"perception:{self.name}",
                    label=sentence.text.strip(),
                )
            )
            emitted = True
        return emitted

    # -- facts ---------------------------------------------------------------

    def _emit_facts(self, sentence, root, subject, perception, names: set) -> bool:
        if subject is None:
            return False
        named = _naming_target(root)
        if named is not None:
            # "The young person who is always feeling cold is named Dave."
            anchor = Const(normalise_symbol(named.text))
            names.add(normalise_symbol(named.text))
            described = [t for t in subject.subtree]
        else:
            anchor = self._entity(subject, names)
            described = list(sentence)
        if anchor is None and self.resolve_pronouns and _is_pronoun(subject):
            anchor = self._antecedent
        if anchor is None:
            return False
        self._antecedent = anchor
        produced = False
        for literal in self._literals_in(described, anchor):
            atom = literal.atom if not literal.negated else _negate(literal.atom)
            perception.facts.append(
                FactRecord(
                    atom=atom,
                    confidence=self.confidence,
                    provenance=f"perception:{self.name}",
                    evidence=sentence.text.strip(),
                )
            )
            produced = True
        return produced

    # -- a span of tokens to literals ----------------------------------------

    def _literals_in(self, tokens, term: Term) -> list[Literal]:
        """Everything a stretch of text predicates of one individual.

        Every predicative adjective in the span becomes an attribute. That is
        blunt, and it is right for this register: the corpus expresses all of
        its content as "is <adjective>", however elaborately it phrases it.
        Nouns are read as classes only when they are a copula's complement, so
        "resembles the rainbow" contributes nothing.
        """
        literals: list[Literal] = []
        for token in tokens:
            modifier = token.pos_ == "ADJ" or (
                # spaCy tags some adjectives as nouns in this register --
                # "a kind person" comes back with kind/NOUN. The dependency
                # says modifier either way.
                token.dep_ == "amod" and token.pos_ in ("NOUN", "VERB")
            )
            if modifier and not _is_decorative(token):
                literals.append(
                    Literal(
                        self.schema.attribute(term, token.lemma_), _is_negated(token)
                    )
                )
            elif token.dep_ in ("attr", "oprd") and token.pos_ in ("NOUN", "PROPN"):
                if token.lemma_.lower() in GENERIC_NOUNS:
                    continue  # "is a red person" says red, caught as an ADJ
                literals.append(
                    Literal(
                        self.schema.membership(term, token.lemma_), _is_negated(token)
                    )
                )
        for token in tokens:
            if token.pos_ != "VERB" or token.lemma_.lower() in HEDGE_VERBS:
                continue
            for obj in token.children:
                if obj.dep_ in ("dobj", "obj") and obj.pos_ in ("NOUN", "PROPN"):
                    other = self._entity(obj, set())
                    if other is not None and obj.lemma_.lower() not in HEDGE_NOUNS:
                        literals.append(
                            Literal(
                                self.schema.relation(
                                    term, predicate_name(token.lemma_), other
                                ),
                                _is_negated(token),
                            )
                        )
        return list(dict.fromkeys(literals))

    def _entity(self, token, names: set) -> Optional[Term]:
        """A noun token as a logic constant, or None if it names no individual."""
        if token.pos_ == "PRON":
            return None
        compounds = [c.text for c in token.children if c.dep_ == "compound"]
        # "That guy Fred" -- the name is the apposition, not the placeholder.
        for child in token.children:
            if child.dep_ in ("appos",) and child.pos_ == "PROPN":
                names.add(normalise_symbol(child.text))
                return Const(normalise_symbol(child.text))
        if token.lemma_.lower() in GENERIC_NOUNS and not compounds:
            return None
        text = " ".join(compounds + [token.text])
        if token.pos_ == "PROPN":
            names.add(normalise_symbol(text))
        return Const(normalise_symbol(text))


# -- parse helpers ---------------------------------------------------------


def _is_pronoun(token) -> bool:
    return token is not None and (
        token.pos_ == "PRON" or token.text.lower() in _PRONOUNS
    )


def _naming_target(root):
    """The name in "... is named Dave", or None.

    The naming verb is usually the root itself ("The young person ... *is
    named* Dave"), but it can be subordinate ("... who is known as Dave"), so
    both are checked.
    """
    for verb in [root] + [c for c in root.children if c.pos_ == "VERB"]:
        if verb.lemma_.lower() not in NAMING_VERBS:
            continue
        for child in verb.children:
            if child.dep_ in ("oprd", "attr", "dobj") and child.pos_ == "PROPN":
                return child
    return None


def _conditional_marker(sentence):
    """The if/when that makes this sentence a rule, or None.

    Position decides. A conditional opens with its marker ("When someone is
    kind, they are big") or announces the consequent with "then". A marker
    late in a sentence with no "then" is modifying something -- "look green
    when ill" is a fact about looking green, not a rule about being ill.
    """
    first = next(iter(sentence), None)
    has_then = any(t.text.lower() == "then" for t in sentence)
    for token in sentence:
        if token.text.lower() not in CONDITIONAL_MARKERS:
            continue
        if token.pos_ not in ("SCONJ", "ADV"):
            continue
        if first is not None and token.i - first.i <= 1:
            return token
        if has_then and token.i < max(
            t.i for t in sentence if t.text.lower() == "then"
        ):
            return token
    return None


def _split_at_marker(sentence, marker) -> tuple[list, list]:
    """Condition and conclusion token spans, split around the marker.

    With an explicit "then" the split is exact. Without one, the condition is
    the marker's own clause and the conclusion is whatever lies outside it.
    """
    consequent = next(
        (t for t in sentence if t.i > marker.i and t.text.lower() == "then"), None
    )
    if consequent is not None:
        condition = [t for t in sentence if marker.i < t.i < consequent.i]
        conclusion = [t for t in sentence if t.i > consequent.i]
        return condition, conclusion
    # Without "then", the main clause's own subject marks the boundary:
    # "When someone is kind yet can be cold and blue, they will also be big."
    # The marker's clause subtree is not enough -- spaCy attaches the
    # coordinated "can be cold" to the root, outside it.
    main_subject = _subject_of(sentence.root)
    if main_subject is not None and main_subject.i > marker.i:
        condition = [t for t in sentence if marker.i < t.i < main_subject.i]
        conclusion = [t for t in sentence if t.i >= main_subject.i]
        if condition and conclusion:
            return condition, conclusion
    span = [t.i for t in marker.head.subtree]
    if not span:
        return [], []
    low, high = min(span), max(span)
    condition = [t for t in sentence if low <= t.i <= high and t.i != marker.i]
    conclusion = [t for t in sentence if not (low <= t.i <= high)]
    return condition, conclusion


def _is_decorative(token) -> bool:
    """True for an adjective that modifies a hedge rather than the subject."""
    head = token.head
    if token.dep_ == "amod" and head.pos_ in ("NOUN", "PROPN"):
        # "a rough side" hedges "rough"; "the red rainbow" does not predicate.
        return head.lemma_.lower() not in HEDGE_NOUNS and head.dep_ not in (
            "attr",
            "oprd",
            "nsubj",
            "nsubjpass",
            "dobj",
        )
    return False


def _is_negated(token) -> bool:
    """Whether the clause governing this token is negated."""
    current = token
    for _ in range(6):
        if any(child.dep_ == "neg" for child in current.children):
            return True
        if current.head is current:
            break
        current = current.head
    return False


def _subject_of(verb):
    """The clause's subject, preferring a name over a placeholder.

    "That guy Fred sure is nice" attaches both "guy" and "Fred" as subjects of
    the same verb. Taking the first gives the placeholder and loses the name.
    """
    subjects = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")]
    if not subjects:
        return None
    named = [c for c in subjects if c.pos_ == "PROPN"]
    return named[0] if named else subjects[0]


def _is_generic(subject) -> bool:
    """True if the subject denotes any individual rather than a named one."""
    if subject is None:
        return False
    if subject.pos_ == "PRON":
        # "someone"/"anyone" quantify; "he"/"she"/"they" point at an
        # individual named earlier, and reading them as generic turns a fact
        # about that individual into a rule about everyone.
        return subject.lemma_.lower() in GENERIC_NOUNS
    definite = any(
        c.dep_ == "det" and c.text.lower() in ("the", "this", "that")
        for c in subject.children
    )
    if subject.lemma_.lower() in GENERIC_NOUNS:
        # "The young person ..." names one individual, however generic the noun.
        return not definite
    if subject.tag_ in ("NNS", "NNPS"):  # a bare plural is a generic
        return not any(
            c.dep_ == "det" and c.text.lower() == "the" for c in subject.children
        )
    return any(
        c.dep_ == "det" and c.text.lower() in ("a", "an", "any", "every", "each")
        for c in subject.children
    )


def _negate(atom: Atom) -> Atom:
    return Atom(f"not_{atom.predicate}", atom.args)
