"""Small English lexicon and symbol normalisation shared by the text layers.

Nothing statistical here -- these are the closed-class words a controlled
English grammar needs, plus the rules that turn an English noun phrase into a
well-formed logic constant.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = [
    "DETERMINERS",
    "COPULAS",
    "NEGATIONS",
    "QUANTIFIED",
    "PRONOUNS",
    "AUXILIARIES",
    "normalise_symbol",
    "singularise",
    "base_verb",
    "predicate_name",
    "PLACEHOLDER_NOUNS",
    "PARTICLES",
    "strip_determiner",
    "is_variable_phrase",
    "split_sentences",
]

DETERMINERS = frozenset({"a", "an", "the", "this", "that", "these", "those", "its", "their"})
COPULAS = frozenset({"is", "are", "was", "were", "be", "been", "am"})
NEGATIONS = frozenset({"not", "n't", "never", "no"})
#: Words that introduce a universally quantified individual.
QUANTIFIED = frozenset(
    {"someone", "something", "anyone", "anything", "everyone", "everything", "somebody", "anybody", "everybody"}
)
PRONOUNS = frozenset({"they", "it", "he", "she", "them", "him", "her", "its", "their"})
#: Head nouns that carry no meaning of their own -- "All big things are young"
#: is a statement about big individuals, not about a class called "thing".
#: Deliberately short: "animals" is a real class, and treating it as a
#: placeholder would quietly discard a condition.
PLACEHOLDER_NOUNS = frozenset(
    {"thing", "things", "people", "person", "persons", "one", "ones",
     "individual", "individuals"}
)
AUXILIARIES = frozenset({"does", "do", "did", "can", "will", "would", "should", "must", "has", "have", "had"})

_IRREGULAR_PLURALS = {
    "people": "person",
    "children": "child",
    "men": "man",
    "women": "woman",
    "mice": "mouse",
    "geese": "goose",
    "feet": "foot",
    "teeth": "tooth",
    "wolves": "wolf",
    "leaves": "leaf",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def normalise_symbol(phrase: str) -> str:
    """Turn an English phrase into a logic constant.

    ``"the Red Ball"`` becomes ``red_ball``. Anything that cannot be expressed
    as a bare ASP identifier is returned in a form the caller should quote.
    """
    cleaned = phrase.strip().strip("'\"").lower()
    cleaned = cleaned.replace("'s", "").replace("’s", "")
    cleaned = re.sub(r"[^a-z0-9]+", "_", cleaned).strip("_")
    if not cleaned:
        return "unknown"
    if cleaned[0].isdigit():
        cleaned = f"n_{cleaned}"
    return cleaned


def strip_determiner(tokens: list[str]) -> list[str]:
    """Drop a leading determiner from a noun phrase."""
    if tokens and tokens[0].lower() in DETERMINERS:
        return tokens[1:]
    return tokens


def singularise(word: str) -> str:
    """Crude but predictable plural -> singular, for 'All cats are ...' rules."""
    lower = word.lower()
    if lower in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[lower]
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith(("sses", "shes", "ches", "xes", "zes")):
        return lower[:-2]
    if lower.endswith("ss") or lower.endswith("us") or lower.endswith("is"):
        return lower
    if lower.endswith("s") and len(lower) > 2:
        return lower[:-1]
    return lower


_IRREGULAR_VERBS = {
    "has": "have", "does": "do", "is": "be", "are": "be", "was": "be",
    "were": "be", "goes": "go", "says": "say",
}


def base_verb(word: str) -> str:
    """Undo third-person agreement: ``chases`` -> ``chase``.

    Both text extractors run verbs through this, so the grammar's ``purrs``
    and spaCy's lemmatised ``purr`` land on the same predicate. Without it the
    two paths silently produce facts that never unify.
    """
    lower = word.strip().lower()
    if lower in _IRREGULAR_VERBS:
        return _IRREGULAR_VERBS[lower]
    if lower.endswith("ies") and len(lower) > 4:
        return lower[:-3] + "y"
    if lower.endswith(("shes", "ches", "xes", "zes", "sses", "oes")):
        return lower[:-2]
    if lower.endswith("ss") or lower.endswith("us"):
        return lower
    if lower.endswith("s") and len(lower) > 2:
        return lower[:-1]
    return lower


#: Particles and prepositions that belong to the verb rather than the object:
#: "the current runs through the circuit" has the two-word verb "runs through".
PARTICLES = frozenset(
    {"through", "after", "into", "onto", "up", "over", "with", "at", "on", "off",
     "out", "in", "from", "to", "by", "about", "around", "across", "against",
     "for", "down", "away"}
)


def predicate_name(phrase: str) -> str:
    """Normalise a verb phrase into a predicate identifier.

    The first word goes to its base form and the whole phrase is then
    normalised: "runs through" becomes ``run_through``. Both the perception
    layer and the ProofWriter loader call this, so the two sides cannot drift
    apart on spelling -- a mismatch then means a genuinely different reading.
    """
    words = phrase.strip().split()
    if not words:
        return "unknown"
    words = [base_verb(words[0])] + words[1:]
    return normalise_symbol(" ".join(words))


def is_variable_phrase(phrase: str) -> bool:
    """True if this noun phrase denotes 'any individual' rather than a named one."""
    head = phrase.strip().lower().split()
    if not head:
        return False
    return head[-1] in QUANTIFIED or head[-1] in PRONOUNS


def split_sentences(text: str) -> list[str]:
    """Split a passage into sentences, keeping terminal punctuation."""
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text.strip()) if part.strip()]
    return parts


def tokenize(text: str) -> list[str]:
    """Word-level tokenisation that keeps contractions and drops punctuation."""
    text = text.replace("n't", " not")
    return [t for t in re.findall(r"[A-Za-z0-9_'’-]+", text) if t]
