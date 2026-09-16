"""Fuzzy semantics and the Type 5 consistency layer."""

import pytest

from neuralmind.consistency.fuzzy import FuzzySemantics, p_mean, p_mean_error
from neuralmind.consistency.layer import ConsistencyLayer
from neuralmind.core.parser import parse_atom
from neuralmind.knowledge.base import KnowledgeBase


def test_tnorms_behave_as_documented():
    product = FuzzySemantics(tnorm="product")
    lukasiewicz = FuzzySemantics(tnorm="lukasiewicz")
    godel = FuzzySemantics(tnorm="godel")
    assert product.conjoin([0.5, 0.5]) == pytest.approx(0.25)
    assert lukasiewicz.conjoin([0.5, 0.5]) == pytest.approx(0.0)
    assert godel.conjoin([0.5, 0.8]) == pytest.approx(0.5)


def test_implication_is_satisfied_when_the_body_is_false():
    semantics = FuzzySemantics()
    assert semantics.implies(0.0, 0.0) == pytest.approx(1.0)
    assert semantics.implies(1.0, 0.0) == pytest.approx(0.0)


def test_forall_punishes_a_single_violation_harder_than_a_mean():
    values = [1.0] * 9 + [0.0]
    assert p_mean_error(values) < sum(values) / len(values)


def test_exists_rewards_a_single_strong_case():
    assert p_mean([0.0] * 9 + [1.0]) > 0.0


def test_unknown_connectives_are_rejected():
    with pytest.raises(ValueError, match="unknown t-norm"):
        FuzzySemantics(tnorm="nope")


def _uncertain_kb():
    kb = KnowledgeBase("t")
    kb.add_rules("%@ no two colours\n:- attr(X, red), attr(X, blue).")
    return kb


def test_a_rule_resting_on_weak_facts_scores_low():
    kb = _uncertain_kb()
    kb.add_rules("attr(X, warm) :- attr(X, furry).")
    kb.add_fact("attr(bob, furry)", confidence=0.55)
    engine = kb.engine()
    report = ConsistencyLayer(kb).check(
        engine.model, {record.atom: record.confidence for record in kb.facts}
    )
    assert report.crisp_consistent  # nothing is broken crisply...
    assert not report.consistent  # ...but the support is too weak to trust
    assert report.weak_rules


def test_certain_facts_satisfy_the_same_rule_fully():
    kb = _uncertain_kb()
    kb.add_rules("attr(X, warm) :- attr(X, furry).")
    kb.add_fact("attr(bob, furry)", confidence=1.0)
    report = ConsistencyLayer(kb).check(
        kb.engine().model, {record.atom: record.confidence for record in kb.facts}
    )
    assert report.consistent and report.satisfaction == pytest.approx(1.0)


def test_hard_violations_are_reported():
    kb = _uncertain_kb()
    kb.add_facts(["attr(bob, red)", "attr(bob, blue)"])
    report = ConsistencyLayer(kb).check(
        kb.engine().model, {record.atom: 1.0 for record in kb.facts}
    )
    assert not report.crisp_consistent
    assert "no two colours" in report.describe()


class _Slot:
    """A minimal stand-in for a perceptor's SlotDistribution."""

    def __init__(self, name, probabilities):
        from neuralmind.core.terms import Atom, Const

        self._name = name
        self.probabilities = {Const(value): p for value, p in probabilities.items()}
        self._Atom, self._Const = Atom, Const

    def atom(self, value):
        return self._Atom("digit", (self._Const(self._name), value))

    def top(self, k=3):
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:k]

    @property
    def best(self):
        return max(self.probabilities.items(), key=lambda kv: kv[1])


def test_resolve_prefers_the_consistent_reading_over_the_likely_one():
    kb = KnowledgeBase("sum").load_builtin("mnist_sum")
    kb.add_fact("expected_sum(3)")
    left = _Slot("d0", {1: 0.9, 7: 0.08, 4: 0.02})
    right = _Slot("d1", {8: 0.6, 2: 0.3, 3: 0.1})
    best = ConsistencyLayer(kb).resolve([left, right], candidates_per_slot=3, top_k=1)
    assert best
    digits = [int(r.atom.args[1].value) for r in best[0].facts if r.atom.predicate == "digit"]
    assert digits == [1, 2]  # not [1, 8], which breaks the sum rule
    assert best[0].revised


def test_resolve_keeps_the_top_prediction_when_it_is_already_consistent():
    kb = KnowledgeBase("sum").load_builtin("mnist_sum")
    kb.add_fact("expected_sum(9)")
    left = _Slot("d0", {1: 0.9, 7: 0.1})
    right = _Slot("d1", {8: 0.8, 2: 0.2})
    best = ConsistencyLayer(kb).resolve([left, right], top_k=1)
    assert best and not best[0].revised


def test_resolve_returns_nothing_when_no_candidate_works():
    kb = KnowledgeBase("sum").load_builtin("mnist_sum")
    kb.add_fact("expected_sum(99)")
    left = _Slot("d0", {1: 0.9, 7: 0.1})
    right = _Slot("d1", {8: 0.8, 2: 0.2})
    assert ConsistencyLayer(kb).resolve([left, right]) == []


def test_resolve_needs_at_least_one_distribution():
    with pytest.raises(ValueError, match="at least one"):
        ConsistencyLayer(KnowledgeBase()).resolve([])


def test_confidence_propagates_through_a_derivation_chain():
    kb = KnowledgeBase("chain")
    kb.add_rules("b(X) :- a(X). c(X) :- b(X).")
    kb.add_fact("a(bob)", confidence=0.8)
    report = ConsistencyLayer(kb).check(
        kb.engine().model, {record.atom: record.confidence for record in kb.facts}
    )
    # Product t-norm over a single-literal body keeps the value, so the
    # conclusion is exactly as good as the observation it rests on.
    assert report.confidence(parse_atom("c(bob)")) == pytest.approx(0.8)
    assert parse_atom("c(bob)") in dict(report.uncertain_conclusions(threshold=0.9))


def test_two_independent_derivations_raise_confidence():
    kb = KnowledgeBase("both")
    kb.add_rules("c(X) :- a(X). c(X) :- b(X).")
    kb.add_fact("a(bob)", confidence=0.6)
    kb.add_fact("b(bob)", confidence=0.6)
    report = ConsistencyLayer(kb).check(
        kb.engine().model, {record.atom: record.confidence for record in kb.facts}
    )
    assert report.confidence(parse_atom("c(bob)")) == pytest.approx(0.84)  # 0.6 + 0.6 - 0.36


def test_a_long_chain_erodes_confidence():
    kb = KnowledgeBase("long")
    kb.add_rules("b(X) :- a(X), a2(X). c(X) :- b(X).")
    kb.add_fact("a(bob)", confidence=0.9)
    kb.add_fact("a2(bob)", confidence=0.9)
    report = ConsistencyLayer(kb).check(
        kb.engine().model, {record.atom: record.confidence for record in kb.facts}
    )
    assert report.confidence(parse_atom("c(bob)")) == pytest.approx(0.81)
