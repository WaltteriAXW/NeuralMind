"""The real ProofWriter corpus: loading it, and what it exposed.

Every test here is skipped when the corpus is absent, so the suite still runs
on a fresh clone. Fetch it with `python scripts/download_proofwriter.py`.
"""

import pytest

from neuralmind.datasets import proofwriter_corpus as corpus

pytestmark = pytest.mark.skipif(
    not corpus.available("depth-2"), reason="ProofWriter corpus not downloaded"
)


# -- the representation format --------------------------------------------


@pytest.mark.parametrize(
    "representation,expected,positive",
    [
        ('("dog" "is" "blue" "+")', ("blue", "dog"), True),
        ('("cow" "needs" "dog" "+")', ("need", "cow", "dog"), True),
        # "~" is negation-as-failure in a CWA rule condition. Reading only
        # "+" and "-" silently drops it, turning a guarded rule into an
        # unguarded one -- which is exactly what happened here first time.
        ('("dog" "is" "white" "~")', ("white", "dog"), False),
        ('("mouse" "is" "blue" "-")', ("blue", "mouse"), False),
        # The corpus keeps the article; the perception layer strips it.
        ('("Arthur" "is" "a bird" "+")', ("bird", "arthur"), True),
        # Two-word predicates are real in the electricity rulebase.
        ('("current" "runs through" "circuit" "+")',
         ("run_through", "current", "circuit"), True),
        ('("bald eagle" "sees" "rabbit" "+")',
         ("see", "bald_eagle", "rabbit"), True),
    ],
)
def test_representations_map_onto_the_project_schema(representation, expected, positive):
    atom, is_positive = corpus.parse_representation(representation)
    assert atom == expected
    assert is_positive is positive


def test_the_quantified_word_becomes_a_variable():
    atom, _ = corpus.parse_representation('("someone" "is" "kind" "+")')
    assert atom == ("kind", "?")


def test_the_triple_schema_is_still_available():
    atom, _ = corpus.parse_representation('("dog" "is" "blue" "+")', schema="triple")
    assert atom == ("attr", "dog", "blue")


def test_unknown_split_and_schema_are_refused():
    with pytest.raises(ValueError, match="unknown split"):
        corpus.load("depth-9")
    with pytest.raises(ValueError, match="unknown schema"):
        corpus.load("depth-2", schema="nonsense", limit=1)


# -- the loaded problems ---------------------------------------------------


@pytest.fixture(scope="module")
def problems():
    return corpus.load("depth-2", limit=25)


def test_problems_carry_a_theory_gold_form_and_questions(problems):
    problem = problems[0]
    assert problem.theory and problem.gold_facts and problem.gold_rules
    assert problem.questions
    assert all(isinstance(q.answer, bool) for q in problem.questions)


def test_negative_questions_are_marked(problems):
    negatives = [q for p in problems for q in p.questions if q.negated]
    assert negatives, "the corpus is full of 'X is not Y' questions"
    assert all("not" in q.text.lower() for q in negatives)


def test_gold_rules_keep_their_negated_conditions(problems):
    bodies = [b for p in problems for _, body in p.gold_rules for b in body]
    assert any(literal[0].startswith("not_") for literal in bodies), (
        "no negated condition survived the loader"
    )


# -- end to end ------------------------------------------------------------


@pytest.fixture(scope="module")
def perceptor():
    from neuralmind.perception.controlled import TripleSchema
    from neuralmind.perception.text import TextPerceptor

    return TextPerceptor(schema=TripleSchema("direct"))


@pytest.mark.parametrize("split", ["depth-0", "depth-2", "depth-5"])
def test_synthetic_language_splits_are_solved_exactly(split, perceptor):
    """The reasoning is exact, so these must be perfect or something regressed."""
    from neuralmind.evaluation import evaluate

    report = evaluate(corpus.load(split, limit=40), perceptor=perceptor)
    assert report.theory_mismatches == 0, "a theory was not extracted exactly"
    assert report.accuracy == 1.0, report.describe()


def test_the_hand_built_rulebases_are_solved(perceptor):
    from neuralmind.evaluation import evaluate

    report = evaluate(corpus.load("birds-electricity", limit=60), perceptor=perceptor)
    assert report.accuracy > 0.97, report.describe()


def test_no_failure_is_ever_the_engine(perceptor):
    """Perception may fail on real text; the reasoner may not."""
    from neuralmind.evaluation import ENGINE, evaluate

    for split in ("depth-3", "birds-electricity", "NatLang"):
        report = evaluate(corpus.load(split, limit=30), perceptor=perceptor)
        assert report.failure_breakdown().get(ENGINE, 0) == 0, split


def test_crowdsourced_language_is_much_harder(perceptor):
    """NatLang is the honest limit of a controlled grammar, and is recorded
    as such rather than quietly excluded."""
    from neuralmind.evaluation import evaluate

    report = evaluate(corpus.load("NatLang", limit=40), perceptor=perceptor)
    assert 0.3 < report.accuracy < 0.9, report.describe()
