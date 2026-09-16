"""Text perception: English sentences in, ground atoms out.

Two extractors, used for different jobs rather than as fallbacks for each
other:

* :class:`~neuralmind.perception.controlled.ControlledEnglishParser` handles
  sentences that state *rules* ("If something is a mammal then it is warm
  blooded"). A dependency parse does not tell you that a sentence is an
  implication; the grammar does.
* :class:`SpacyTripleExtractor` handles sentences that state *facts*, using
  spaCy's dependency parse. It copes with the messier phrasing real text
  contains -- modifiers, prepositions, conjunctions, passives.

spaCy is optional. Without it, :class:`TextPerceptor` uses the controlled
parser for everything and says so in its diagnostics, so a missing model
degrades coverage rather than breaking the pipeline.

On confidences: the numbers attached to spaCy-derived facts are *structural
reliability weights*, not calibrated probabilities. A subject-verb-object
triple read straight off the parse is more trustworthy than one recovered
through a preposition, and the weights encode that ordering so the consistency
layer can act on it. They are not the output of a probability model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Iterable, Optional, Sequence

from ..core.terms import Atom, Const, Term
from ..knowledge.base import FactRecord
from .base import Perception, PerceptionError
from .controlled import ControlledEnglishParser, TripleSchema
from .lexicon import normalise_symbol, split_sentences

__all__ = ["TextPerceptor", "SpacyTripleExtractor", "spacy_available", "CONFIDENCE"]

#: Structural reliability weights by extraction pattern (see module docstring).
CONFIDENCE = {
    "copula_attribute": 0.95,
    "copula_class": 0.95,
    "subject_verb_object": 0.90,
    "modifier_attribute": 0.85,
    "prepositional_relation": 0.75,
    "intransitive_action": 0.80,
    "conjunct_penalty": 0.95,
}


def spacy_available(model: str = "en_core_web_sm") -> bool:
    """True if spaCy and the given model are both installed."""
    try:
        import spacy  # noqa: F401
    except ImportError:
        return False
    try:
        _load_spacy(model)
    except Exception:
        return False
    return True


@lru_cache(maxsize=4)
def _load_spacy(model: str):
    import spacy

    return spacy.load(model)


class SpacyTripleExtractor:
    """Reads facts off a spaCy dependency parse.

    Only the patterns below are extracted; anything else is reported as
    unparsed. That is deliberate -- an open-domain extractor that guesses would
    feed the reasoner facts nobody can audit.
    """

    name = "spacy-dependency"

    def __init__(
        self,
        model: str = "en_core_web_sm",
        schema: Optional[TripleSchema] = None,
        minimum_confidence: float = 0.0,
    ) -> None:
        self.model = model
        self.schema = schema or TripleSchema()
        self.minimum_confidence = minimum_confidence
        self._nlp = None

    @property
    def nlp(self):
        if self._nlp is None:
            try:
                self._nlp = _load_spacy(self.model)
            except Exception as exc:  # pragma: no cover - environment dependent
                raise PerceptionError(
                    f"could not load the spaCy model {self.model!r}: {exc}. "
                    f"Install it with `python -m spacy download {self.model}`."
                ) from exc
        return self._nlp

    def perceive(self, raw: str) -> Perception:
        perception = Perception(source=f"text:{self.name}")
        doc = self.nlp(raw)
        for sentence in doc.sents:
            found = self._extract_sentence(sentence, perception)
            if not found:
                perception.unparsed.append(sentence.text.strip())
        perception.diagnostics["model"] = self.model
        perception.diagnostics["proper_names"] = sorted(
            {normalise_symbol(token.text) for token in doc if _is_proper_name(token)}
        )
        perception.diagnostics["sentences"] = len(list(doc.sents))
        perception.diagnostics["unparsed"] = len(perception.unparsed)
        return perception

    # -- extraction ------------------------------------------------------

    def _extract_sentence(self, sentence, perception: Perception) -> int:
        before = len(perception.facts)
        for token in sentence:
            if token.pos_ in ("VERB", "AUX"):
                self._from_predicate(token, sentence, perception)
            elif token.dep_ == "amod" and token.head.pos_ in ("NOUN", "PROPN"):
                # "the red ball" asserts that the ball is red.
                self._emit(
                    perception,
                    self.schema.attribute(self._entity(token.head), token.text),
                    CONFIDENCE["modifier_attribute"],
                    sentence.text,
                )
        return len(perception.facts) - before

    def _from_predicate(self, verb, sentence, perception: Perception) -> None:
        subjects = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            return
        negated = any(child.dep_ == "neg" for child in verb.children)
        subject_terms = [self._entity(s) for s in _with_conjuncts(subjects[0])]

        if verb.lemma_ == "be":
            self._from_copula(verb, subject_terms, negated, sentence, perception)
            return

        objects = [c for c in verb.children if c.dep_ in ("dobj", "obj", "dative", "attr")]
        if objects:
            for subject in subject_terms:
                for obj in _with_conjuncts(objects[0]):
                    self._emit(
                        perception,
                        self.schema.relation(subject, verb.lemma_, self._entity(obj), negated),
                        CONFIDENCE["subject_verb_object"],
                        sentence.text,
                    )
        prepositions = [c for c in verb.children if c.dep_ == "prep"]
        for preposition in prepositions:
            for pobj in [c for c in preposition.children if c.dep_ == "pobj"]:
                for subject in subject_terms:
                    self._emit(
                        perception,
                        self.schema.relation(
                            subject,
                            f"{verb.lemma_}_{preposition.text.lower()}",
                            self._entity(pobj),
                            negated,
                        ),
                        CONFIDENCE["prepositional_relation"],
                        sentence.text,
                    )
        if not objects and not prepositions:
            for subject in subject_terms:
                self._emit(
                    perception,
                    self.schema.action(subject, verb.lemma_, negated),
                    CONFIDENCE["intransitive_action"],
                    sentence.text,
                )

    def _from_copula(self, verb, subject_terms, negated, sentence, perception: Perception) -> None:
        complements = [c for c in verb.children if c.dep_ in ("acomp", "attr", "oprd")]
        for complement in complements:
            for item in _with_conjuncts(complement):
                for subject in subject_terms:
                    if item.pos_ in ("NOUN", "PROPN"):
                        atom = self.schema.membership(subject, item.lemma_, negated)
                        weight = CONFIDENCE["copula_class"]
                    else:
                        atom = self.schema.attribute(subject, item.text, negated)
                        weight = CONFIDENCE["copula_attribute"]
                    self._emit(perception, atom, weight, sentence.text)
        for preposition in [c for c in verb.children if c.dep_ == "prep"]:
            for pobj in [c for c in preposition.children if c.dep_ == "pobj"]:
                for subject in subject_terms:
                    self._emit(
                        perception,
                        self.schema.relation(
                            subject, preposition.text.lower(), self._entity(pobj), negated
                        ),
                        CONFIDENCE["prepositional_relation"],
                        sentence.text,
                    )

    def _entity(self, token) -> Term:
        """A noun token plus its compound modifiers, as a logic constant."""
        compounds = [c.text for c in token.children if c.dep_ == "compound"]
        phrase = " ".join(compounds + [token.text])
        return Const(normalise_symbol(phrase))

    def _emit(self, perception: Perception, atom: Atom, confidence: float, evidence: str) -> None:
        if confidence < self.minimum_confidence:
            return
        if any(record.atom == atom for record in perception.facts):
            return
        perception.facts.append(
            FactRecord(
                atom=atom,
                confidence=confidence,
                provenance=f"perception:{self.name}",
                evidence=evidence.strip(),
            )
        )


def _is_proper_name(token) -> bool:
    """Proper name by POS tag, or by the way the noun phrase is built.

    The small English model tags some first names as common nouns, so the tag
    alone is not enough. A capitalised singular noun that takes no determiner
    is a proper name in practice, which catches the cases the tagger misses.
    """
    if token.pos_ == "PROPN":
        return True
    if token.pos_ != "NOUN" or not token.text[:1].isupper():
        return False
    if token.tag_ in ("NNS", "NNPS"):  # plurals are classes, not individuals
        return False
    return not any(child.dep_ == "det" for child in token.children)


def _with_conjuncts(token) -> list:
    """A token plus anything coordinated with it ('blue and round')."""
    items = [token]
    for child in token.children:
        if child.dep_ == "conj":
            items.extend(_with_conjuncts(child))
    return items


class TextPerceptor:
    """The text perception layer: routes each sentence to the right extractor.

    Sentences that state general rules always go to the controlled grammar, for
    the reason in the module docstring -- a dependency parse does not tell you
    a sentence is an implication.

    Fact sentences go to the grammar first and to spaCy only for what the
    grammar declines. That ordering is measured, not assumed. On the
    ProofWriter corpus's fact sentences the grammar is exact and spaCy is not:

        split               grammar   spaCy
        NatLang              100.0%   100.0%
        depth-5              100.0%    91.5%
        birds-electricity    100.0%    61.1%

    spaCy's errors are linguistically reasonable ones -- it reads "the bald
    eagle" as an eagle that is bald, giving ``bald(eagle)`` where the corpus
    means a single entity ``bald_eagle``. Preferring the deterministic reader
    and falling back only where it refuses keeps that from happening, without
    giving up the coverage spaCy provides outside the controlled register.

    Outside that register neither reader is enough, and the default
    ``prefer="auto"`` picks between the grammar and
    :class:`~neuralmind.perception.narrative.NarrativeExtractor` per input --
    see :meth:`_choose_reader`. The four strategies:

    ``auto``
        Choose per input. Falls back to ``grammar`` with no spaCy installed.
    ``grammar``
        The controlled grammar, with spaCy for the facts it declines.
    ``spacy``
        The same, with the two reversed.
    ``narrative``
        The free-text reader for everything. Requires spaCy.
    """

    name = "text"

    def __init__(
        self,
        schema: Optional[TripleSchema] = None,
        model: str = "en_core_web_sm",
        use_spacy: bool = True,
        prefer: str = "auto",
    ) -> None:
        if prefer not in ("auto", "grammar", "spacy", "narrative"):
            raise ValueError(
                "prefer must be 'auto', 'grammar', 'spacy' or 'narrative', "
                f"not {prefer!r}"
            )
        self.schema = schema or TripleSchema()
        self.controlled = ControlledEnglishParser(self.schema)
        self.use_spacy = use_spacy and spacy_available(model)
        self.prefer = prefer
        self.spacy = SpacyTripleExtractor(model, self.schema) if self.use_spacy else None
        self.narrative = None
        if prefer in ("narrative", "auto") and self.use_spacy:
            from .narrative import NarrativeExtractor

            self.narrative = NarrativeExtractor(model, self.schema)
        if prefer == "narrative":
            if not self.use_spacy:
                raise PerceptionError(
                    "prefer='narrative' needs spaCy: the free-text reader works "
                    "from a dependency parse. Install it with "
                    "`python -m spacy download en_core_web_sm`."
                )

    def perceive(self, raw: str) -> Perception:
        combined = Perception(source=f"text:{self.name}")
        combined.diagnostics["spacy"] = self.use_spacy
        combined.diagnostics["prefer"] = self.prefer
        mode = self.prefer
        if mode == "auto":
            mode = self._choose_reader(raw)
            combined.diagnostics["chose"] = mode
        if mode == "narrative" and self.narrative is not None:
            # Free-form prose: the grammar cannot split rules from facts here,
            # because whether a sentence states a rule is itself a question
            # about its structure. The narrative reader decides per sentence.
            combined.extend(self.narrative.perceive(raw))
            combined.diagnostics["unparsed"] = len(combined.unparsed)
            return combined
        rule_sentences: list[str] = []
        fact_sentences: list[str] = []
        for sentence in split_sentences(raw):
            (rule_sentences if _states_a_rule(sentence) else fact_sentences).append(sentence)

        if rule_sentences:
            combined.extend(self.controlled.perceive(" ".join(rule_sentences)))

        if fact_sentences:
            if self.prefer == "spacy" and self.spacy is not None:
                first, second = self.spacy, self.controlled
            else:
                first, second = self.controlled, self.spacy
            combined.extend(self._read_facts(fact_sentences, first, second))

        combined.diagnostics["rule_sentences"] = len(rule_sentences)
        combined.diagnostics["fact_sentences"] = len(fact_sentences)
        combined.diagnostics["unparsed"] = len(combined.unparsed)
        return combined

    def _choose_reader(self, raw: str) -> str:
        """Pick the exact reader where it applies, the approximate one elsewhere.

        The controlled grammar refuses what it cannot read, and that refusal is
        the signal. If it reads every sentence, it is exact and the narrative
        reader would only add noise; if it refuses any, the text is outside its
        register and the narrative reader does better on the whole passage.

        Measured on the ProofWriter corpus, the two readers are near-opposites
        -- neither is better everywhere -- which is why this is chosen per
        input rather than configured once.
        """
        if self.narrative is None:
            return "grammar"
        try:
            attempt = self.controlled.perceive(raw)
        except Exception:
            return "narrative"
        return "grammar" if not attempt.unparsed else "narrative"

    def _read_facts(self, sentences: Sequence[str], first, second) -> Perception:
        """Read each sentence with ``first``, handing the rest to ``second``.

        Sentence by sentence rather than in bulk, because the point is to know
        exactly which sentences the preferred extractor declined. The fallback
        still runs as one batch, so the expensive parser is loaded once.
        """
        collected = Perception(source=f"text:{self.name}")
        leftovers: list[str] = []
        for sentence in sentences:
            attempt = first.perceive(sentence)
            if attempt.facts and not attempt.unparsed:
                collected.extend(attempt)
            else:
                leftovers.append(sentence)
        if leftovers:
            if second is not None:
                collected.extend(second.perceive(" ".join(leftovers)))
            else:
                collected.unparsed.extend(leftovers)
        return collected

    def parse_question(self, question: str) -> Atom:
        """Turn a yes/no question into the atom to query."""
        return self.controlled.parse_question(question)


def _states_a_rule(sentence: str) -> bool:
    lowered = sentence.strip().lower()
    if lowered.startswith("if ") and " then " in lowered:
        return True
    if lowered.startswith(("all ", "every ", "each ", "anyone ", "anything ", "no ")):
        return True
    from .controlled import _is_bare_plural_subject

    return _is_bare_plural_subject(sentence)
