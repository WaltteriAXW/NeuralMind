"""The free-text reader, and the routing that chooses between the two readers.

Every case here is a sentence that broke the extractor while it was being
written, kept so the fix cannot be undone quietly. The phrasing comes from the
ProofWriter NatLang split, which is crowdsourced -- nobody wrote these
sentences to be parsed.
"""

import pytest

from neuralmind.perception.controlled import ControlledEnglishParser, TripleSchema
from neuralmind.perception.text import TextPerceptor, spacy_available

pytestmark = pytest.mark.skipif(
    not spacy_available(), reason="spaCy model not installed"
)


@pytest.fixture(scope="module")
def reader():
    from neuralmind.perception.narrative import NarrativeExtractor

    return NarrativeExtractor()


def facts_of(reader, text):
    return {str(record.atom) for record in reader.perceive(text).facts}


def rules_of(reader, text):
    return {str(rule) for rule in reader.perceive(text).rules}


# -- facts -----------------------------------------------------------------


def test_a_sentence_with_four_predications_gives_four_facts(reader):
    """The sentence the reader exists for.

    One clause, a coordinated complement, a pronoun standing in for the named
    subject, and a subordinate clause -- and it asserts all four attributes.
    """
    assert facts_of(reader, "Charlie is green, but often kind, even when he is blue and cold.") == {
        "attr(charlie, green)",
        "attr(charlie, kind)",
        "attr(charlie, blue)",
        "attr(charlie, cold)",
    }


def test_a_late_marker_on_a_named_subject_is_not_a_condition(reader):
    """"when she is ill" describes Erin; it quantifies over nobody.

    Reading it as a rule would claim that anything ill looks green.
    """
    perception = reader.perceive("Erin looks green when she is ill.")
    assert not perception.rules
    assert {str(r.atom) for r in perception.facts} == {
        "attr(erin, green)",
        "attr(erin, ill)",
    }


def test_a_pronoun_subject_takes_the_previous_sentence_subject(reader):
    assert facts_of(reader, "Charlie is cold. He is also rough.") == {
        "attr(charlie, cold)",
        "attr(charlie, rough)",
    }


def test_pronoun_resolution_can_be_switched_off():
    from neuralmind.perception.narrative import NarrativeExtractor

    strict = NarrativeExtractor(resolve_pronouns=False)
    perception = strict.perceive("Charlie is cold. He is also rough.")
    assert "attr(charlie, rough)" not in {str(r.atom) for r in perception.facts}
    assert perception.unparsed == ["He is also rough."]


def test_a_description_that_ends_in_a_name_is_about_the_name(reader):
    """"The young person ... is named Dave" -- the facts are Dave's."""
    perception = reader.perceive(
        "The young person who is always feeling cold is named Dave."
    )
    assert {str(r.atom) for r in perception.facts} == {
        "attr(dave, young)",
        "attr(dave, cold)",
    }
    assert perception.diagnostics["proper_names"] == ["dave"]


def test_hedging_verbs_carry_no_content(reader):
    """"tends to be rather quiet" says quiet -- not tending, not rather."""
    assert facts_of(reader, "Gary tends to be rather quiet.") == {"attr(gary, quiet)"}


def test_a_hedging_noun_contributes_its_modifier_only(reader):
    """"on the big side" means big. There is no side."""
    assert facts_of(reader, "Anne is on the big side.") == {"attr(anne, big)"}


def test_negation_reaches_the_whole_clause(reader):
    assert "not_attr(harry, green)" in facts_of(reader, "Harry is not green.")


def test_a_content_verb_with_an_object_is_a_relation(reader):
    assert "rel(lion, chase, mouse)" in facts_of(reader, "The lion chases the mouse.")


# -- rules -----------------------------------------------------------------


def test_a_conditional_without_then_splits_at_the_main_subject(reader):
    """spaCy hangs "can be cold" off the root, outside the marker's clause.

    Walking the condition clause's own children finds "kind" and stops, which
    drops two of the three conditions and makes the rule far too general.
    """
    assert rules_of(
        reader, "When someone is kind yet can be cold and blue, they will also be very big."
    ) == {"attr(X, big) :- attr(X, kind), attr(X, cold), attr(X, blue)."}


def test_a_generic_subject_carries_its_modifiers_and_relative_clause(reader):
    """"look round" is a conjoined verb; its complement is a condition too."""
    assert rules_of(
        reader, "Young people who are nice and look round are also going to be green."
    ) == {"attr(X, green) :- attr(X, young), attr(X, nice), attr(X, round)."}


def test_an_explicit_then_splits_the_rule(reader):
    assert "attr(X, round) :- attr(X, blue)." in rules_of(
        reader, "If someone is blue then they are round."
    )


def test_a_definite_pronoun_does_not_quantify(reader):
    """"He is kind" is a fact about someone; "someone is kind" is a rule.

    Treating every pronoun as generic turned facts about named individuals
    into rules about everybody.
    """
    perception = reader.perceive("Bob is nice. He is kind.")
    assert not perception.rules


def test_a_definite_generic_noun_is_not_read_as_a_quantifier(reader):
    """"The young person" is one individual; "a young person" is anyone young.

    The definite reading names somebody the text has not identified yet, so
    the sentence is refused rather than turned into a rule about everyone --
    which is the answer the corpus wants, since the name arrives later ("...
    is named Dave").
    """
    definite = reader.perceive("The young person is nice.")
    indefinite = reader.perceive("A young person is nice.")
    assert not definite.rules and definite.unparsed
    assert indefinite.rules and not indefinite.facts


def test_a_conclusion_already_in_the_condition_is_not_emitted(reader):
    """"If X is blue then X is blue" is a rule that says nothing."""
    assert not rules_of(reader, "If someone is blue then they are blue.")


def test_facts_are_weighted_below_the_grammars(reader):
    """The reader guesses where the grammar refuses, and the weight says so."""
    from neuralmind.perception.narrative import CONFIDENCE as NARRATIVE
    from neuralmind.perception.text import CONFIDENCE as STRUCTURAL

    assert NARRATIVE < STRUCTURAL["copula_attribute"]
    assert all(record.confidence == NARRATIVE
               for record in reader.perceive("Bob is blue.").facts)


def test_an_unreadable_sentence_is_reported_not_guessed(reader):
    perception = reader.perceive("Meanwhile, elsewhere.")
    assert perception.unparsed and not perception.facts and not perception.rules


# -- routing ---------------------------------------------------------------


def test_auto_picks_the_grammar_for_controlled_text():
    """The grammar is exact on its register, so auto must not override it."""
    perceptor = TextPerceptor(prefer="auto")
    perception = perceptor.perceive("Bob is blue. If something is a cat then it purrs.")
    assert perception.diagnostics["chose"] == "grammar"
    assert "act(X, purr) :- isa(X, cat)." in {str(r) for r in perception.rules}


def test_auto_picks_the_narrative_reader_for_prose():
    """The grammar refusing any sentence is the signal that it is out of register."""
    perceptor = TextPerceptor(prefer="auto")
    perception = perceptor.perceive(
        "Charlie is green, but often kind, even when he is blue and cold."
    )
    assert perception.diagnostics["chose"] == "narrative"
    assert "attr(charlie, kind)" in {str(r.atom) for r in perception.facts}


def test_the_grammar_refuses_what_the_narrative_reader_reads():
    """The premise of the routing rule, asserted rather than assumed."""
    prose = "Charlie is green, but often kind, even when he is blue and cold."
    assert ControlledEnglishParser().perceive(prose).unparsed


def test_the_narrative_reader_honours_the_schema():
    from neuralmind.perception.narrative import NarrativeExtractor

    direct = NarrativeExtractor(schema=TripleSchema("direct"))
    assert "kind(charlie)" in facts_of(
        direct, "Charlie is green, but often kind, even when he is blue and cold."
    )


def test_narrative_mode_without_spacy_says_so():
    from neuralmind.perception.base import PerceptionError

    with pytest.raises(PerceptionError, match="spaCy"):
        TextPerceptor(use_spacy=False, prefer="narrative")
