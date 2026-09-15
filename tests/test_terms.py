"""The symbolic data model."""

import pytest

from neuralmind.core.terms import (
    Arith, Atom, Compare, Const, Literal, Var, evaluate, ground_term, is_ground, variables_in,
)


def test_atoms_print_as_asp():
    atom = Atom("parent", (Const("alice"), Var("X")))
    assert str(atom) == "parent(alice, X)"
    assert atom.signature == ("parent", 2)
    assert not atom.is_ground


def test_quoted_constants_round_trip():
    assert str(Const("Bob Smith", quoted=True)) == '"Bob Smith"'
    assert str(Const('say "hi"', quoted=True)) == '"say \\"hi\\""'


def test_grounding_evaluates_arithmetic():
    atom = Atom("sum", (Arith("+", Var("A"), Const(1)),))
    grounded = atom.ground({"A": Const(4)})
    assert grounded == Atom("sum", (Const(5),))


def test_grounding_leaves_unbound_expressions_alone():
    term = ground_term(Arith("+", Var("A"), Const(1)), {})
    assert isinstance(term, Arith) and not is_ground(term)


@pytest.mark.parametrize(
    "op,left,right,expected",
    [
        ("+", 7, 3, 10), ("-", 7, 3, 4), ("*", 7, 3, 21),
        ("/", 7, 3, 2), ("\\", 7, 3, 1), ("**", 2, 5, 32),
        ("/", -7, 3, -2),  # truncating, as in clingo
    ],
)
def test_arithmetic(op, left, right, expected):
    assert evaluate(Arith(op, Const(left), Const(right)), {}).value == expected


def test_division_by_zero_is_an_error():
    with pytest.raises(ValueError, match="division by zero"):
        evaluate(Arith("/", Const(1), Const(0)), {})


@pytest.mark.parametrize(
    "op,expected",
    [("=", False), ("!=", True), ("<", True), ("<=", True), (">", False), (">=", False)],
)
def test_comparisons(op, expected):
    assert Compare(op, Const(1), Const(2)).holds({}) is expected


def test_numbers_sort_before_symbols():
    assert Compare("<", Const(9), Const("a")).holds({}) is True


def test_variables_in_finds_every_occurrence():
    literal = Literal(Atom("p", (Var("X"), Arith("+", Var("Y"), Const(1)))))
    assert sorted(variables_in(literal)) == ["X", "Y"]


def test_as_tuple_gives_a_plain_view():
    assert Atom("parent", (Const("alice"), Const("bob"))).as_tuple() == ("parent", "alice", "bob")
