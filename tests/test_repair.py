"""Correcting a knowledge base by complaining about specific conclusions."""

import pytest

from neuralmind.core.parser import parse_atom
from neuralmind.induction.repair import ADD_RULE, REMOVE_RULE, SPECIALISE, Corrector
from neuralmind.knowledge.base import KnowledgeBase


@pytest.fixture
def birds():
    kb = KnowledgeBase("birds")
    kb.add_facts(["bird(tweety)", "bird(pingu)", "bird(eagle)", "penguin(pingu)"])
    kb.add_rules("flies(X) :- bird(X).")
    return kb


def test_an_overgeneral_rule_is_narrowed(birds):
    """The repair worth having: the rule was nearly right."""
    repairs = Corrector(birds).reject("flies(pingu)")
    best = repairs[0]
    assert best.kind == SPECIALISE
    assert str(best.add[0]) == "flies(X) :- bird(X), not penguin(X)."
    assert best.clean


def test_the_cost_of_each_alternative_is_measured(birds):
    """Dropping the rule fixes the complaint and breaks two other answers."""
    dropping = [r for r in Corrector(birds).reject("flies(pingu)") if r.kind == REMOVE_RULE]
    assert dropping
    broken = {str(a) for a in dropping[0].breaks}
    assert broken == {"flies(eagle)", "flies(tweety)"}
    assert not dropping[0].clean


def test_a_clean_repair_is_ranked_first(birds):
    repairs = Corrector(birds).reject("flies(pingu)")
    assert repairs[0].clean
    assert repairs == sorted(repairs, key=lambda r: r.score())


def test_applying_a_repair_changes_the_answers(birds):
    repair = Corrector(birds).reject("flies(pingu)")[0]
    repair.apply(birds)
    derived = {str(a) for a in birds.engine().model.by_predicate("flies")}
    assert derived == {"flies(tweety)", "flies(eagle)"}


def test_a_missing_conclusion_can_be_asserted_or_derived():
    kb = KnowledgeBase("family")
    kb.add_facts(
        ["parent(m, j)", "parent(j, a)", "parent(m, l)", "parent(l, o)"]
    )
    repairs = Corrector(kb).expect("grandparent(m, a)")
    kinds = {r.kind for r in repairs}
    assert "add-fact" in kinds, "asserting it outright is always an option"
    assert ADD_RULE in kinds, "and so is learning the rule"
    rule = [r for r in repairs if r.kind == ADD_RULE][0]
    assert str(rule.add[0]) == "grandparent(A, B) :- parent(A, C), parent(C, B)."


def test_a_learned_rule_reports_what_else_it_derives():
    kb = KnowledgeBase("family")
    kb.add_facts(["parent(m, j)", "parent(j, a)", "parent(m, l)", "parent(l, o)"])
    rule = [r for r in Corrector(kb).expect("grandparent(m, a)") if r.kind == ADD_RULE][0]
    # It is right, but the user should see that it concludes more than asked.
    assert "grandparent(m, o)" in {str(a) for a in rule.introduces}


def test_an_asserted_fact_is_offered_for_withdrawal():
    kb = KnowledgeBase("t")
    kb.add_facts(["p(a)"])
    repairs = Corrector(kb).reject("p(a)")
    assert repairs[0].kind == "retract-fact"
    assert repairs[0].clean


def test_nothing_is_proposed_when_there_is_nothing_to_fix(birds):
    assert Corrector(birds).reject("flies(nobody)") == []
    assert Corrector(birds).expect("flies(tweety)") == []


def test_the_knowledge_base_is_never_modified_by_diagnosis(birds):
    before = birds.to_asp()
    Corrector(birds).reject("flies(pingu)")
    assert birds.to_asp() == before


def test_a_policy_rule_is_corrected_by_one_counterexample():
    """The case this is for: a rule that is right until it is not."""
    kb = KnowledgeBase("policy")
    kb.add_facts(
        ["employee(dana)", "employee(tom)", "role(dana, analyst)",
         "role(tom, analyst)", "suspended(tom)"]
    )
    kb.add_rules("may_read(E) :- role(E, analyst).")
    best = Corrector(kb).reject("may_read(tom)")[0]
    assert str(best.add[0]) == "may_read(E) :- role(E, analyst), not suspended(E)."
    assert best.clean


def test_repairs_serialise(birds):
    document = Corrector(birds).reject("flies(pingu)")[0].to_dict()
    assert document["kind"] == SPECIALISE and document["clean"] is True


def test_diagnose_handles_both_complaints_together(birds):
    repairs = Corrector(birds).diagnose(
        expected=["flies(nobody)"], rejected=["flies(pingu)"]
    )
    assert any(r.kind == SPECIALISE for r in repairs)
    assert any(r.kind == "add-fact" for r in repairs)
