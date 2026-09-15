"""Parsing the ASP/Datalog subset."""

import pytest

from neuralmind.core.parser import ParseError, parse_atom, parse_program, parse_rule
from neuralmind.core.terms import Const


def test_facts_rules_and_constraints():
    program = parse_program(
        "p(a). q(X) :- p(X). :- q(a).", "t.lp"
    )
    assert len(program.rules) == 3
    assert program.rules[0].is_fact
    assert program.rules[1].head is not None and program.rules[1].body
    assert program.rules[2].is_constraint


def test_comments_and_directives():
    program = parse_program("% a comment\np(a).\n#show p/1.\n#const n=3.", "t.lp")
    assert program.shown == {("p", 1)}
    assert program.constants["n"] == Const(3)


def test_anonymous_variables_are_distinct():
    rule = parse_rule("q(X) :- p(X, _, _).")
    names = [str(a) for a in rule.body[0].atom.args]
    assert names[1] != names[2]
    # ...and stay valid clingo variables.
    assert all(name[0] == "_" and name[1].isupper() for name in names[1:])


def test_arithmetic_precedence():
    rule = parse_rule("p(X) :- q(A), q(B), X = A + B * 2.")
    assert str(rule).endswith("X = (A+(B*2)).")


def test_negation_and_comparison():
    rule = parse_rule("p(X) :- q(X), not r(X), X != a.")
    assert rule.negative_literals and rule.comparisons


def test_labels_attach_to_the_next_rule():
    program = parse_program("%@ No self ancestors\n:- ancestor(X, X).", "t.lp")
    assert program.rules[0].label == "No self ancestors"


def test_unsupported_constructs_are_kept_for_clingo():
    program = parse_program("1 { a(X) : b(X) } 1 :- c(X).\nc(1).", "t.lp")
    assert program.requires_asp == ("choice rules",)
    assert program.raw_asp and len(program.rules) == 1


@pytest.mark.parametrize("source", ["p(a)", "p(a) :- .", ":- .", "p(("])
def test_malformed_input_raises(source):
    with pytest.raises(ParseError):
        parse_program(source)


def test_parse_atom_rejects_trailing_input():
    assert parse_atom("parent(alice, bob)").arity == 2
    with pytest.raises(ParseError):
        parse_atom("parent(a) extra")


def test_strings_are_parsed_and_printed():
    atom = parse_atom('label("Hello, world")')
    assert atom.args[0] == Const("Hello, world", quoted=True)
    assert str(atom) == 'label("Hello, world")'
