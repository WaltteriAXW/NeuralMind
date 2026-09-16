"""The real ProofWriter corpus (Allen Institute for AI).

:mod:`neuralmind.datasets.proofwriter` *generates* problems in the ProofWriter
style. This module loads the actual dataset, which is the more honest test:
the phrasing is not one this project's grammar was written against, and the
splits include a crowdsourced-language variant and two hand-built rulebases
that nothing here was tuned for.

The corpus ships a gold symbolic form beside every sentence, which is what
makes failure attribution possible on real data -- a wrong answer can be
blamed on the extraction or on the reasoning, with evidence either way.

Two details of the CWA splits matter here:

* **Theories are definite.** Negation appears in questions, not in the rules,
  so every theory is plain stratified Datalog.
* **Questions are statements, and some are negative.** "The mouse is not blue."
  is true exactly when ``attr(mouse, blue)`` is not derivable -- the
  closed-world reading this engine already implements.

Download with ``python scripts/download_proofwriter.py``. The corpus is not
redistributed here; it is AllenAI's, under their terms.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator, Optional, Sequence

from ..perception.lexicon import normalise_symbol, predicate_name
from .proofwriter import Problem, Question

__all__ = ["load", "available", "default_root", "SPLITS", "WORLDS", "parse_representation"]

#: The CWA splits this loader understands.
SPLITS = (
    "depth-0", "depth-1", "depth-2", "depth-3", "depth-5",
    "birds-electricity", "NatLang",
)

#: ``("cow" "needs" "dog" "+")`` -- subject, predicate, object, polarity.
#:
#: Polarity is one of three markers, and the difference matters:
#:   ``+``  the triple holds
#:   ``~``  negation as failure, used for CWA rule conditions ("and not white")
#:   ``-``  strong negation, used by the questions ("The mouse is not blue.")
#: Matching only ``+`` and ``-`` silently drops every ``~`` condition, which
#: quietly turns a guarded rule into an unguarded one.
_TRIPLE = re.compile(r'\("([^"]*)" "([^"]*)" "([^"]*)" "([+~-])"\)')

#: Words the corpus uses for a universally quantified individual.
_VARIABLES = frozenset({"someone", "something", "somebody", "anyone", "anything"})


def default_root() -> Path:
    override = os.environ.get("NEURALMIND_DATA")
    if override:
        return Path(override) / "proofwriter"
    return Path(__file__).resolve().parents[2] / "data" / "proofwriter"


#: The two readings the corpus ships, and the two this project implements.
WORLDS = ("cwa", "owa")


def split_path(split: str, root: Optional[Path] = None, world: str = "cwa") -> Path:
    prefix = "" if world == "cwa" else f"{world}-"
    return (Path(root) if root else default_root()) / f"{prefix}{split}-test.jsonl"


def available(
    split: str = "depth-2", root: Optional[Path] = None, world: str = "cwa"
) -> bool:
    return split_path(split, root, world).exists()


def parse_representation(representation: str, schema: str = "direct") -> tuple:
    """Turn one gold triple into this project's schema.

    Under the default ``direct`` schema::

        ("dog" "is" "blue" "+")    -> (("blue", "dog"), True)
        ("cow" "needs" "dog" "-")  -> (("need", "cow", "dog"), False)
        ("dog" "is" "white" "~")   -> (("white", "dog"), False)

    The verb goes through the same :func:`base_verb` the perception layer uses,
    so a mismatch here means the extraction really differed -- not that the two
    sides spell the predicate differently.
    """
    match = _TRIPLE.fullmatch(representation.strip())
    if match is None:
        raise ValueError(f"unrecognised representation: {representation!r}")
    return _triple_to_tuple(*match.groups(), schema=schema)


#: How gold atoms are shaped. ``direct`` gives ``blue(dog)`` and
#: ``need(cow, dog)``; ``triple`` gives ``attr(dog, blue)`` and
#: ``rel(cow, need, dog)``.
#:
#: ``direct`` is the default because these theories need it. Under the triple
#: schema every atom shares the predicate ``attr/2``, so a rule like
#: "if smart and not white then round" makes ``attr`` depend negatively on
#: itself and predicate-level stratification rejects the whole theory. With one
#: predicate per attribute the strata are exactly what the corpus intends.
SCHEMAS = ("direct", "triple")


def _attribute(text: str) -> str:
    """Normalise a gold attribute, dropping a leading article.

    The corpus writes class membership as an attribute with its determiner
    intact -- ``("Arthur" "is" "a bird" "+")``. Keeping it would give the
    predicate ``a_bird`` while the perception layer, which strips determiners,
    produces ``bird``; the two would never match.
    """
    words = text.strip().split()
    if len(words) > 1 and words[0].lower() in ("a", "an", "the"):
        words = words[1:]
    return normalise_symbol(" ".join(words))


def _term(text: str) -> str:
    """A constant, or ``?`` for the corpus's quantified individual."""
    lowered = text.strip().lower()
    if lowered in _VARIABLES:
        return "?"
    return normalise_symbol(text)


def _triple_to_tuple(
    subject: str, predicate: str, obj: str, polarity: str, schema: str = "direct"
) -> tuple:
    # Both "~" and "-" are read as "this does not hold"; under the closed-world
    # assumption this engine implements, they coincide.
    positive = polarity == "+"
    is_attribute = predicate.strip().lower() == "is"
    if schema == "triple":
        if is_attribute:
            atom = ("attr", _term(subject), _attribute(obj))
        else:
            atom = ("rel", _term(subject), predicate_name(predicate), _term(obj))
    else:
        if is_attribute:
            atom = (_attribute(obj), _term(subject))
        else:
            atom = (predicate_name(predicate), _term(subject), _term(obj))
    return atom, positive


def _mark(atom: tuple, positive: bool, world: str) -> tuple:
    """Apply the negative marker this world uses.

    Under CWA the engine reads ``not_p`` as a separate positive predicate that
    the perception layer also produces, so the two sides match. Under OWA the
    corpus means *strong* negation -- a claim that ``p`` is false, which
    interacts with ``p`` -- and that is written ``-p``.
    """
    if positive:
        return atom
    prefix = "not_" if world == "cwa" else "-"
    return (f"{prefix}{atom[0]}",) + atom[1:]


def _facts_of(record: dict, schema: str, world: str = "cwa") -> set[tuple]:
    facts: set[tuple] = set()
    for triple in record.get("triples", {}).values():
        atom, positive = parse_representation(triple["representation"], schema)
        facts.add(_mark(atom, positive, world))
    return facts


def _rules_of(record: dict, schema: str, world: str = "cwa") -> list[tuple]:
    rules: list[tuple] = []
    for rule in record.get("rules", {}).values():
        triples = _TRIPLE.findall(rule["representation"])
        if len(triples) < 2:
            continue  # a rule needs at least one condition and a conclusion
        parsed = [_triple_to_tuple(*parts, schema=schema) for parts in triples]
        body = [_mark(atom, positive, world) for atom, positive in parsed[:-1]]
        head_atom, head_positive = parsed[-1]
        rules.append((_mark(head_atom, head_positive, world), body))
    return rules


def _questions_of(record: dict, schema: str, world: str = "cwa") -> list[Question]:
    questions: list[Question] = []
    for item in record.get("questions", {}).values():
        try:
            goal, positive = parse_representation(item["representation"], schema)
        except ValueError:
            continue
        # OWA answers are True, False or the string "Unknown". bool("Unknown")
        # is True, so the string has to be checked before the cast -- reading
        # every unknown as a yes would score 46% of the split wrong and look
        # like a reasoning failure.
        raw = item["answer"]
        unknown = isinstance(raw, str) and raw.strip().lower() == "unknown"
        questions.append(
            Question(
                text=item["question"],
                goal=goal,
                answer=False if unknown else bool(raw),
                depth=int(item.get("QDep") or 0),
                negated=not positive,
                unknown=unknown,
            )
        )
    return questions


def iter_records(
    split: str, root: Optional[Path] = None, world: str = "cwa"
) -> Iterator[dict]:
    path = split_path(split, root, world)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Fetch the corpus with "
            "`python scripts/download_proofwriter.py`."
        )
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load(
    split: str = "depth-2",
    root: Optional[Path] = None,
    limit: Optional[int] = None,
    questions_per_problem: Optional[int] = None,
    schema: str = "direct",
    world: str = "cwa",
) -> list[Problem]:
    """Load a split as :class:`~neuralmind.datasets.proofwriter.Problem` objects.

    The result plugs straight into :func:`neuralmind.evaluation.evaluate`, so
    the real corpus and the generated one are scored by the same code.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; choose from {', '.join(SPLITS)}")
    if schema not in SCHEMAS:
        raise ValueError(f"unknown schema {schema!r}; choose from {', '.join(SCHEMAS)}")
    if world not in WORLDS:
        raise ValueError(f"unknown world {world!r}; choose from {', '.join(WORLDS)}")
    problems: list[Problem] = []
    for record in iter_records(split, root, world):
        questions = _questions_of(record, schema, world)
        if questions_per_problem is not None:
            questions = questions[:questions_per_problem]
        if not questions:
            continue
        problems.append(
            Problem(
                theory=record["theory"],
                questions=questions,
                gold_facts=_facts_of(record, schema, world),
                gold_rules=_rules_of(record, schema, world),
                seed=record.get("id"),
                world=world,
            )
        )
        if limit is not None and len(problems) >= limit:
            break
    return problems
