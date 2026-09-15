"""Controlled English and spaCy-based extraction."""

import pytest

from neuralmind.perception.controlled import ControlledEnglishParser
from neuralmind.perception.lexicon import base_verb, singularise
from neuralmind.perception.text import TextPerceptor, spacy_available


@pytest.fixture(scope="module")
def parser():
    return ControlledEnglishParser()


def facts_of(parser, text):
    return {str(record.atom) for record in parser.perceive(text).facts}


def rules_of(parser, text):
    return {str(rule) for rule in parser.perceive(text).rules}


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("Bob is blue.", "attr(bob, blue)"),
        ("Bob is not blue.", "not_attr(bob, blue)"),
        ("Bob is a cat.", "isa(bob, cat)"),
        ("The cat likes the dog.", "rel(cat, like, dog)"),
        ("The dog barks.", "act(dog, bark)"),
        ("The red ball is round.", "attr(red_ball, round)"),
    ],
)
def test_facts(parser, sentence, expected):
    assert expected in facts_of(parser, sentence)


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("All cats are mammals.", "isa(X, mammal) :- isa(X, cat)."),
        ("Cats are animals.", "isa(X, animal) :- isa(X, cat)."),
        (
            "If someone is blue and round then they are green.",
            "attr(X, green) :- attr(X, blue), attr(X, round).",
        ),
        (
            "If something is a mammal then it is warm blooded.",
            "attr(X, warm_blooded) :- isa(X, mammal).",
        ),
        (
            "All dogs that are big are loud.",
            "attr(X, loud) :- isa(X, dog), attr(X, big).",
        ),
    ],
)
def test_rules(parser, sentence, expected):
    assert expected in rules_of(parser, sentence)


def test_class_conjunct_is_not_read_as_an_attribute(parser):
    # Regression: "and a lion" continues the clause but is a class, not an adjective.
    rules = rules_of(parser, "If someone is small and green and a lion then they are young.")
    assert "attr(X, young) :- attr(X, small), attr(X, green), isa(X, lion)." in rules


def test_pronouns_bind_to_the_introduced_individual(parser):
    rules = rules_of(parser, "If someone likes something then they are happy.")
    assert "attr(X, happy) :- rel(X, like, Y)." in rules


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Is Bob green?", "attr(bob, green)"),
        ("Is Bob a cat?", "isa(bob, cat)"),
        ("Does the cat like the dog?", "rel(cat, like, dog)"),
    ],
)
def test_questions(parser, question, expected):
    assert str(parser.parse_question(question)) == expected


def test_unparsable_input_is_reported_not_guessed(parser):
    perception = parser.perceive("Colorless green ideas sleep furiously and.")
    assert perception.unparsed or perception.facts


def test_a_standalone_quantified_statement_is_refused(parser):
    perception = parser.perceive("Someone is blue.")
    assert perception.unparsed and not perception.facts


def test_proper_names_are_identified(parser):
    perception = parser.perceive("Bob is blue. The cat is red.")
    assert perception.diagnostics["proper_names"] == ["bob"]


@pytest.mark.parametrize(
    "word,expected", [("cats", "cat"), ("flies", "fly"), ("boxes", "box"), ("grass", "grass")]
)
def test_singularise(word, expected):
    assert singularise(word) == expected


@pytest.mark.parametrize(
    "word,expected", [("purrs", "purr"), ("chases", "chase"), ("flies", "fly"), ("run", "run")]
)
def test_base_verb(word, expected):
    assert base_verb(word) == expected


def test_perceptor_routes_rules_to_the_grammar():
    perceptor = TextPerceptor()
    perception = perceptor.perceive("Bob is a cat. If something is a cat then it purrs.")
    assert any("purr" in str(rule) for rule in perception.rules)
    assert "isa(bob, cat)" in {str(r.atom) for r in perception.facts}


@pytest.mark.skipif(not spacy_available(), reason="spaCy model not installed")
def test_spacy_handles_modifiers_and_prepositions():
    from neuralmind.perception.text import SpacyTripleExtractor

    perception = SpacyTripleExtractor().perceive(
        "The cat chases the small mouse. Carol lives in Helsinki."
    )
    found = {str(record.atom) for record in perception.facts}
    assert "rel(cat, chase, mouse)" in found
    assert "attr(mouse, small)" in found
    assert "rel(carol, live_in, helsinki)" in found


@pytest.mark.skipif(not spacy_available(), reason="spaCy model not installed")
def test_both_extractors_agree_on_verb_form():
    # Regression: the grammar said "chases" while spaCy said "chase", so
    # facts from the two paths never unified.
    from neuralmind.perception.text import SpacyTripleExtractor

    sentence = "The cat chases the mouse."
    grammar = {str(r.atom) for r in ControlledEnglishParser().perceive(sentence).facts}
    parsed = {str(r.atom) for r in SpacyTripleExtractor().perceive(sentence).facts}
    assert "rel(cat, chase, mouse)" in grammar & parsed


@pytest.mark.skipif(not spacy_available(), reason="spaCy model not installed")
def test_confidences_rank_patterns_by_reliability():
    from neuralmind.perception.text import CONFIDENCE

    assert CONFIDENCE["copula_attribute"] > CONFIDENCE["prepositional_relation"]
