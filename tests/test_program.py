"""Safety checking, stratification and body planning."""

import pytest

from neuralmind.core.parser import parse_program, parse_rule
from neuralmind.core.program import SafetyError, StratificationError


def test_unsafe_head_variable_is_rejected():
    with pytest.raises(SafetyError, match="head variable"):
        parse_rule("p(X, Y) :- q(X).").check_safety()


def test_unsafe_negated_variable_is_rejected():
    with pytest.raises(SafetyError, match="never bound"):
        parse_rule("p(X) :- q(X), not r(Y).").check_safety()


def test_unsafe_comparison_variable_is_rejected():
    with pytest.raises(SafetyError, match="never bound"):
        parse_rule("p(X) :- q(X), X < Y.").check_safety()


def test_equality_acts_as_an_assignment():
    steps = parse_rule("sum(A, B, S) :- d(A), d(B), S = A + B.").plan()
    assert [s.kind for s in steps] == ["match", "match", "assign"]


def test_plan_schedules_filters_as_soon_as_possible():
    steps = parse_rule("p(X, Y) :- q(X), X != a, r(Y).").plan()
    assert [s.kind for s in steps] == ["match", "filter", "match"]


def test_stratification_orders_negation():
    program = parse_program(
        "p(X) :- q(X). r(X) :- q(X), not p(X). q(1).", "t.lp"
    )
    strata = program.stratify()
    levels = program.stratum_of()
    assert levels[("r", 1)] > levels[("p", 1)]
    assert sum(len(s) for s in strata) == len(program.rules)


def test_recursion_through_negation_is_rejected():
    with pytest.raises(StratificationError, match="recursive"):
        parse_program("a(X) :- d(X), not b(X). b(X) :- d(X), not a(X). d(1).", "t.lp")


def test_positive_recursion_is_fine():
    program = parse_program(
        "e(1,2). t(X,Y) :- e(X,Y). t(X,Z) :- e(X,Y), t(Y,Z).", "t.lp"
    )
    assert len(program.stratify()) == 1


def test_merge_keeps_both_rule_sets():
    left = parse_program("p(a).", "l.lp")
    right = parse_program("q(b).", "r.lp")
    assert len(left.merge(right).rules) == 2


def test_to_asp_round_trips():
    source = "p(a).\nq(X) :- p(X), not r(X).\n:- q(b).\n#show q/1."
    program = parse_program(source, "t.lp")
    assert parse_program(program.to_asp(), "t2.lp").to_asp() == program.to_asp()
