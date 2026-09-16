"""The optional clingo backend and the cross-check that uses it."""

import pytest

from neuralmind.core.parser import parse_program
from neuralmind.inference.clingo_backend import ClingoBackend, cross_check, clingo_available

pytestmark = pytest.mark.skipif(not clingo_available(), reason="clingo is not installed")

DATALOG = """
parent(alice, bob). parent(bob, carol).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
haschild(X) :- parent(X, _).
person(X) :- parent(X, _). person(Y) :- parent(_, Y).
childless(X) :- person(X), not haschild(X).
sum(A, B, S) :- num(A), num(B), S = A + B.
num(1). num(2).
"""


def test_engines_agree_on_a_datalog_program():
    assert cross_check(parse_program(DATALOG, "t.lp")).agree


def test_violations_correspond_to_unsat():
    program = parse_program(DATALOG + "\nparent(carol, alice).\n:- ancestor(X, X).", "t.lp")
    result = cross_check(program)
    assert result.agree
    assert result.python_violations > 0 and not result.clingo_satisfiable


def test_anonymous_variables_survive_the_round_trip():
    # Regression: a variable named for clingo must still be a variable there.
    source = parse_program("p(a, b). q(X) :- p(X, _).", "t.lp").to_asp()
    answer = ClingoBackend().solve(source).first()
    assert any(atom.predicate == "q" for atom in answer.atoms)


def test_choice_rules_run_through_the_backend():
    program = parse_program(
        "task(t1). task(t2). room(r1). room(r2).\n"
        "1 { assign(T, R) : room(R) } 1 :- task(T).\n"
        ":- assign(t1, r1).",
        "sched.lp",
    )
    result = ClingoBackend(models=0).solve(program)
    assert result.satisfiable and len(result.answer_sets) == 2
    for answer in result.answer_sets:
        assigned = answer.by_predicate("assign")
        assert len(assigned) == 2
        assert all(str(a) != "assign(t1, r1)" for a in assigned)


def test_unsatisfiable_program():
    assert not ClingoBackend().solve("p(a). :- p(a).").satisfiable


def test_cross_check_skips_asp_only_programs():
    program = parse_program("1 { a(X) : b(X) } 1 :- c(X).\nc(1).", "t.lp")
    assert cross_check(program).skipped
