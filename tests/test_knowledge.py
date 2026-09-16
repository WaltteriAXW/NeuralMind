"""Knowledge bases, provenance and the RDF bridge."""

import pytest

from neuralmind.core.parser import parse_atom
from neuralmind.knowledge.base import KnowledgeBase, builtin_rulesets, rule_path


def test_bundled_rulesets_all_parse_and_check():
    for name in builtin_rulesets():
        kb = KnowledgeBase(name).load_builtin(name)
        kb.program()  # raises on unsafe or unstratified rules


def test_facts_keep_provenance_and_confidence():
    kb = KnowledgeBase()
    kb.add_fact("parent(alice, bob)", confidence=0.8, provenance="perception:text")
    record = kb.fact_record(parse_atom("parent(alice, bob)"))
    assert record.confidence == 0.8 and not record.certain
    assert kb.uncertain_facts() == [record]


def test_reasserting_keeps_the_higher_confidence():
    kb = KnowledgeBase()
    kb.add_fact("p(a)", confidence=0.6)
    kb.add_fact("p(a)", confidence=0.9)
    kb.add_fact("p(a)", confidence=0.7)
    assert kb.confidence_of(parse_atom("p(a)")) == 0.9


def test_retain_drops_low_confidence_facts():
    kb = KnowledgeBase()
    kb.add_fact("p(a)", confidence=0.4)
    kb.add_fact("p(b)", confidence=0.95)
    dropped = kb.retain(0.5)
    assert len(dropped) == 1 and len(kb) == 1


def test_non_ground_facts_are_rejected():
    with pytest.raises(ValueError, match="ground"):
        KnowledgeBase().add_fact("parent(X, bob)")


def test_rule_files_separate_facts_from_rules():
    kb = KnowledgeBase().add_rules("p(a). q(X) :- p(X).")
    assert len(kb) == 1 and len(kb.rules.derivation_rules) == 1


def test_missing_ruleset_names_the_alternatives():
    with pytest.raises(FileNotFoundError, match="available"):
        rule_path("nonexistent")


def test_rdf_round_trip():
    rdflib = pytest.importorskip("rdflib")
    from neuralmind.knowledge.rdf import from_graph, to_graph

    atoms = [parse_atom("parent(alice, bob)"), parse_atom("cat(felix)")]
    recovered = from_graph(to_graph(atoms))
    assert {str(a) for a in recovered} == {str(a) for a in atoms}


def test_rdf_reifies_n_ary_predicates():
    pytest.importorskip("rdflib")
    from neuralmind.knowledge.rdf import to_graph

    graph = to_graph([parse_atom("rel(cat, chase, mouse)")])
    assert len(graph) == 4  # one type triple plus one per argument
