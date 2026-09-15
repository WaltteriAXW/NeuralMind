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
