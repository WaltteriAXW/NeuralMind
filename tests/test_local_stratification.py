"""Local stratification, and the bug that hid behind its absence.

Predicate-level stratification asks whether any predicate depends negatively on
itself. That is too coarse for real rule bases: ``p(a) :- not p(b).`` is
perfectly well defined, and 6-12% of the ProofWriter corpus looks like it.
"""

import pytest

from neuralmind.core.parser import parse_atom, parse_program
from neuralmind.core.program import GroundingLimit, StratificationError
from neuralmind.inference.clingo_backend import clingo_available, cross_check
from neuralmind.inference.engine import ReasoningEngine
from neuralmind.inference.forward import ForwardChainer

# "eat" depends negatively on "eat", but no ground atom depends on itself.
LOCALLY_STRATIFIED = """
cold(a). cold(b).
eat(eagle, squirrel) :- cold(X), not eat(X, eagle).
eat(eagle, rabbit) :- eat(eagle, squirrel).
"""

GENUINELY_RECURSIVE = """
d(1).
a(X) :- d(X), not b(X).
b(X) :- d(X), not a(X).
"""


def test_predicate_stratification_rejects_what_grounding_accepts():
    program = parse_program(LOCALLY_STRATIFIED, "t", check=False)
    with pytest.raises(StratificationError):
        program.stratify()
    strata = program.local_strata()
    assert sum(len(s) for s in strata) > 0


def test_the_engine_falls_back_to_local_stratification():
    chainer = ForwardChainer(parse_program(LOCALLY_STRATIFIED, "t", check=False))
    model = chainer.run()
    assert chainer.locally_stratified
    assert model.holds(parse_atom("eat(eagle, rabbit)"))


def test_check_accepts_a_locally_stratified_program():
    """check() must not reject what the engine can run."""
    parse_program(LOCALLY_STRATIFIED, "t").check()


def test_genuine_recursion_through_negation_is_still_rejected():
    program = parse_program(GENUINELY_RECURSIVE, "t", check=False)
    with pytest.raises(StratificationError):
        program.local_strata()
    with pytest.raises(StratificationError):
        program.check()


def test_grounding_is_bounded():
    source = "\n".join(f"c({i})." for i in range(40))
    source += "\np(A,B,C,D) :- c(A), c(B), c(C), c(D), not p(B,A,D,C)."
    program = parse_program(source, "t", check=False)
    with pytest.raises(GroundingLimit, match="past the"):
        program.ground(max_rules=1000)


def test_an_unstratifiable_and_huge_program_says_which_problem_it_has():
    source = "\n".join(f"c({i})." for i in range(40))
    source += "\np(A,B,C,D) :- c(A), c(B), c(C), c(D), not p(B,A,D,C)."
    program = parse_program(source, "t", check=False)
    with pytest.raises(StratificationError, match="too large to ground"):
        program.check()


def test_constant_universe_is_the_herbrand_universe():
    program = parse_program('p(a, 1). q(X) :- p(X, "s").', "t", check=False)
    assert {str(c) for c in program.constant_universe()} == {"a", "1", '"s"'}


@pytest.mark.skipif(not clingo_available(), reason="clingo is not installed")
def test_local_stratification_agrees_with_clingo():
    assert cross_check(parse_program(LOCALLY_STRATIFIED, "t", check=False)).agree


# -- the bug that only real data exposed ----------------------------------


def test_a_rule_with_no_positive_body_literal_fires():
    """Regression: such a rule was skipped forever.

    Its body has no predicate signatures, so the semi-naive delta guard --
    "skip unless something this rule reads has changed" -- never let it run.
    Every correctness test in the suite passed; only a real corpus with
    "chase(lion, tiger) :- not like(lion, tiger)." caught it.
    """
    engine = ReasoningEngine("q(b). p :- not q(a).")
    assert engine.ask("p").holds


def test_such_a_rule_still_respects_its_negation():
    engine = ReasoningEngine("q(a). p :- not q(a).")
    assert not engine.ask("p").holds


@pytest.mark.skipif(not clingo_available(), reason="clingo is not installed")
def test_negative_only_bodies_agree_with_clingo():
    for source in ("q(b). p :- not q(a).", "q(a). p :- not q(a).", "p :- not q."):
        assert cross_check(parse_program(source, "t")).agree
