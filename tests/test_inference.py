"""Forward chaining, proofs, and query answering."""

import pytest

from neuralmind.core.parser import parse_atom, parse_program
from neuralmind.inference.engine import ReasoningEngine
from neuralmind.inference.forward import ForwardChainer, ReasoningLimit, UnsupportedProgram, solve
from neuralmind.inference.proof import FACT, ProofError, explain

TRANSITIVE = """
parent(alice, bob). parent(bob, carol). parent(carol, dave).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).
"""


def test_transitive_closure():
    model = solve(parse_program(TRANSITIVE, "t.lp"))
    assert len(model.by_predicate("ancestor")) == 6
    assert model.holds(parse_atom("ancestor(alice, dave)"))


def test_closed_world_negation():
    model = solve(
        parse_program(
            "p(1). p(2). q(2). r(X) :- p(X), not q(X).", "t.lp"
        )
    )
    assert [str(a) for a in model.by_predicate("r")] == ["r(1)"]


def test_arithmetic_in_the_head():
    model = solve(parse_program("d(3). d(4). s(S) :- d(A), d(B), A < B, S = A + B.", "t.lp"))
    assert [str(a) for a in model.by_predicate("s")] == ["s(7)"]


def test_integrity_constraints_report_their_instances():
    model = solve(parse_program("p(1). p(2).\n%@ only one p\n:- p(X), p(Y), X != Y.", "t.lp"))
    assert not model.consistent
    assert "only one p" in model.violations[0].describe()


def test_proof_tree_shape():
    model = solve(parse_program(TRANSITIVE, "t.lp"))
    proof = explain(model, parse_atom("ancestor(alice, dave)"))
    assert proof.depth == 4
    assert {str(a) for a in proof.premises()} == {
        "parent(alice, bob)", "parent(bob, carol)", "parent(carol, dave)"
    }
    assert all(leaf.kind == FACT for leaf in proof.leaves())


def test_proofs_terminate_on_recursive_predicates():
    # Every ancestor atom must have a finite proof despite the recursion.
    model = solve(parse_program(TRANSITIVE, "t.lp"))
    for atom in model.by_predicate("ancestor"):
        assert explain(model, atom).size < 20


def test_explaining_a_false_atom_raises():
    model = solve(parse_program(TRANSITIVE, "t.lp"))
    with pytest.raises(ProofError, match="closed-world"):
        explain(model, parse_atom("ancestor(dave, alice)"))


def test_runaway_arithmetic_hits_a_limit():
    program = parse_program("p(0). p(Y) :- p(X), Y = X + 1.", "t.lp")
    with pytest.raises(ReasoningLimit, match="atom limit"):
        ForwardChainer(program, max_atoms=500).run()


def test_limits_can_be_soft():
    program = parse_program("p(0). p(Y) :- p(X), Y = X + 1.", "t.lp")
    model = ForwardChainer(program, max_atoms=200, strict=False).run()
    assert model.truncated and len(model) >= 200


def test_asp_only_programs_are_refused_with_a_pointer():
    program = parse_program("1 { a(X) : b(X) } 1 :- c(X).\nc(1).", "t.lp")
    with pytest.raises(UnsupportedProgram, match="clingo"):
        ForwardChainer(program)


def test_query_with_variables_returns_bindings():
    engine = ReasoningEngine(TRANSITIVE)
    answer = engine.ask("ancestor(alice, X)")
    assert answer.holds
    assert {str(a) for a in answer.atoms} == {
        "ancestor(alice, bob)", "ancestor(alice, carol)", "ancestor(alice, dave)"
    }
    assert len(answer.proofs) == 3


def test_failed_query_explains_where_the_rule_stalled():
    engine = ReasoningEngine(TRANSITIVE)
    answer = engine.ask("ancestor(dave, alice)")
    assert not answer.holds
    assert answer.diagnosis is not None and answer.diagnosis.has_candidates
    assert "parent(dave" in answer.diagnosis.describe()


def test_failed_query_with_no_matching_rule():
    engine = ReasoningEngine(TRANSITIVE)
    answer = engine.ask("unrelated(a, b)")
    assert not answer.holds and not answer.diagnosis.has_candidates


def test_adding_facts_invalidates_the_model():
    engine = ReasoningEngine(TRANSITIVE)
    assert not engine.ask("ancestor(dave, erin)").holds
    engine.add_facts(["parent(dave, erin)"])
    assert engine.ask("ancestor(alice, erin)").holds


def test_alternative_justifications_are_recorded():
    model = solve(parse_program("p(a). q(a). r(X) :- p(X). r(X) :- q(X).", "t.lp"))
    assert len(model.alternatives(parse_atom("r(a)"))) == 2
