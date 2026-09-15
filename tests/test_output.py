"""Rendering: proof trees, JSON, and template NLG."""

import json

import pytest

from neuralmind.core.parser import parse_atom
from neuralmind.inference.engine import ReasoningEngine
from neuralmind.output.nlg import Realiser, indefinite_article, third_person
from neuralmind.output.render import render_model, render_proof, render_violations
from neuralmind.output.serialize import model_document, to_json

PROGRAM = """
parent(alice, bob). parent(bob, carol).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
"""


@pytest.fixture
def engine():
    return ReasoningEngine(PROGRAM)


def test_proof_renders_as_a_tree(engine):
    text = render_proof(engine.ask("ancestor(alice, carol)").proof)
    assert text.splitlines()[0].startswith("ancestor(alice, carol)")
    assert "└──" in text and "[given]" in text


def test_render_proof_handles_a_missing_proof():
    assert "no proof" in render_proof(None)


def test_proof_serialises_to_nested_json(engine):
    document = json.loads(to_json(engine.ask("ancestor(alice, carol)").proof))
    assert document["conclusion"] == "ancestor(alice, carol)"
    assert document["because"][0]["kind"] == "fact"


def test_model_document_groups_by_predicate(engine):
    document = model_document(engine.model)
    assert set(document["atoms"]) == {"parent", "ancestor"}
    assert document["summary"]["consistent"] is True


def test_render_model_can_restrict_predicates(engine):
    assert "parent" not in render_model(engine.model, ["ancestor"])


def test_render_violations_when_there_are_none():
    assert render_violations([]) == "no violations"


@pytest.mark.parametrize("word,article", [("apple", "an"), ("cat", "a"), ("hour", "an"), ("user", "a")])
def test_indefinite_article(word, article):
    assert indefinite_article(word) == article


@pytest.mark.parametrize("verb,inflected", [("like", "likes"), ("watch", "watches"), ("fly", "flies")])
def test_agreement(verb, inflected):
    assert third_person(verb) == inflected


@pytest.mark.parametrize(
    "atom,expected",
    [
        ("attr(bob, blue)", "bob is blue"),
        ("isa(bob, cat)", "bob is a cat"),
        ("isa(bob, elephant)", "bob is an elephant"),
        ("rel(cat, chase, mouse)", "cat chases mouse"),
        ("act(dog, bark)", "dog barks"),
        ("not_attr(bob, blue)", "bob is not blue"),
        ("not_rel(cat, chase, mouse)", "cat does not chase mouse"),
    ],
)
def test_realising_atoms(atom, expected):
    assert Realiser().realise(parse_atom(atom)) == expected


def test_unregistered_predicates_degrade_readably():
    assert Realiser().realise(parse_atom("ancestor(alice, dave)")) == "alice is the ancestor of dave"


def test_proper_names_are_capitalised():
    realiser = Realiser().learn_names(["bob"])
    assert realiser.realise(parse_atom("attr(bob, blue)")) == "Bob is blue"


def test_proof_becomes_a_paragraph_in_premise_order():
    engine = ReasoningEngine(
        "isa(bob, cat). isa(X, mammal) :- isa(X, cat). attr(X, warm) :- isa(X, mammal)."
    )
    prose = Realiser().learn_names(["bob"]).realise_proof(engine.ask("attr(bob, warm)").proof)
    assert prose.index("Bob is a cat") < prose.index("Bob is warm")
    assert prose.startswith("Bob is a cat.") and "Therefore" in prose
    # Registering the name capitalises it everywhere, not just sentence-initially.
    assert "bob" not in prose


def test_a_false_answer_says_so_plainly():
    engine = ReasoningEngine("isa(bob, cat).")
    assert Realiser().realise_answer(engine.ask("isa(bob, dog)")).startswith("No")
