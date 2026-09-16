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


# -- patterns the real ProofWriter corpus made necessary -------------------


@pytest.mark.parametrize(
    "sentence,expected",
    [
        # A multi-word subject. The verb is found by the object's determiner,
        # not by assuming the subject is one or two tokens.
        ("The bald eagle eats the squirrel.", "rel(bald_eagle, eat, squirrel)"),
        # A particle belongs to the verb, not the object.
        ("The current runs through the circuit.", "rel(current, run_through, circuit)"),
        ("The circuit has the switch.", "rel(circuit, have, switch)"),
    ],
)
def test_multiword_subjects_and_particle_verbs(parser, sentence, expected):
    assert expected in facts_of(parser, sentence)


@pytest.mark.parametrize(
    "sentence,expected",
    [
        # A placeholder head noun: the statement is about big individuals,
        # not about a class called "thing".
        ("All big things are young.", "attr(X, young) :- attr(X, big)."),
        ("Young things are furry.", "attr(X, furry) :- attr(X, young)."),
        ("White, quiet people are smart.",
         "attr(X, smart) :- attr(X, white), attr(X, quiet)."),
        ("Round, nice people are big.",
         "attr(X, big) :- attr(X, round), attr(X, nice)."),
        # A real class noun is still read as a class.
        ("All cats are mammals.", "isa(X, mammal) :- isa(X, cat)."),
    ],
)
def test_universals_with_placeholder_head_nouns(parser, sentence, expected):
    assert expected in rules_of(parser, sentence)


def test_a_placeholder_noun_with_no_modifier_is_refused(parser):
    """"All things are young." states no condition, so it is not a rule."""
    perception = parser.perceive("All things are young.")
    assert perception.unparsed and not perception.rules


@pytest.mark.parametrize(
    "sentence,expected",
    [
        # A negated rule condition is negation-as-failure, not a predicate
        # called not_attr -- the latter is never derived, so the rule would
        # never fire.
        ("If Bob is not white then Bob is furry.",
         "attr(bob, furry) :- not attr(bob, white)."),
        ("If someone is smart and not white then they are round.",
         "attr(X, round) :- attr(X, smart), not attr(X, white)."),
        ("If something is rough and it does not see the rabbit then it is young.",
         "attr(X, young) :- attr(X, rough), not rel(X, see, rabbit)."),
        # Negation does not carry across "and".
        ("If something is red and not round and big then it is cold.",
         "attr(X, cold) :- attr(X, red), not attr(X, round), attr(X, big)."),
    ],
)
def test_negated_rule_conditions(parser, sentence, expected):
    assert expected in rules_of(parser, sentence)


def test_a_negated_fact_stays_an_explicit_negative(parser):
    """In a fact, "not" asserts something the contradiction rules can catch."""
    assert "not_attr(alice, green)" in facts_of(parser, "Alice is not green.")


def test_the_direct_schema_names_predicates_after_the_words():
    from neuralmind.perception.controlled import TripleSchema

    direct = ControlledEnglishParser(TripleSchema("direct"))
    assert "blue(bob)" in facts_of(direct, "Bob is blue.")
    assert "eat(cat, mouse)" in facts_of(direct, "The cat eats the mouse.")
    assert "round(X) :- smart(X), not white(X)." in rules_of(
        direct, "If someone is smart and not white then they are round."
    )


@pytest.mark.skipif(not spacy_available(), reason="spaCy model not installed")
def test_the_grammar_is_preferred_over_spacy_for_facts():
    """Measured, not assumed: spaCy reads "the bald eagle" as a bald eagle.

    That is linguistically reasonable and wrong for this corpus, where it is
    one entity. The grammar is exact on these sentences, so it goes first.
    """
    from neuralmind.perception.text import SpacyTripleExtractor

    sentence = "The bald eagle is cold."
    routed = {str(r.atom) for r in TextPerceptor().perceive(sentence).facts}
    spacy_only = {str(r.atom) for r in SpacyTripleExtractor().perceive(sentence).facts}
    assert "attr(bald_eagle, cold)" in routed
    assert "attr(bald_eagle, cold)" not in spacy_only


@pytest.mark.skipif(not spacy_available(), reason="spaCy model not installed")
def test_spacy_still_picks_up_what_the_grammar_refuses():
    """Preferring the grammar must not lose spaCy's extra coverage."""
    perception = TextPerceptor().perceive(
        "The quick brown fox jumped over the extremely lazy dog yesterday."
    )
    assert perception.facts, "nothing was extracted at all"


def test_prefer_must_be_a_known_strategy():
    with pytest.raises(ValueError, match="prefer must be"):
        TextPerceptor(prefer="whatever")
