"""The end-to-end pipeline and its policies."""

import json

import pytest

from neuralmind.knowledge.base import KnowledgeBase
from neuralmind.pipeline import NeuralMindPipeline, ViolationPolicy

PASSAGE = (
    "Bob is a cat. All cats are mammals. "
    "If something is a mammal then it is warm blooded."
)


def test_english_in_proof_out():
    result = NeuralMindPipeline().run(PASSAGE, question="Is Bob warm blooded?")
    assert result.holds
    assert result.proof.depth == 3
    assert "Bob is a cat" in result.explanation


def test_result_serialises_to_json():
    result = NeuralMindPipeline().run(PASSAGE, question="Is Bob warm blooded?")
    document = json.loads(result.to_json())
    assert document["answer"]["holds"] is True
    assert document["answer"]["proofs"][0]["because"]


def test_a_false_question_carries_a_diagnosis():
    result = NeuralMindPipeline().run(PASSAGE, question="Is Bob green?")
    assert not result.holds
    assert result.answer.diagnosis is not None


def test_questions_can_be_atoms_or_english():
    pipeline = NeuralMindPipeline()
    pipeline.run(PASSAGE)
    assert pipeline.ask("attr(bob, warm_blooded)").holds
    assert pipeline.ask("Is Bob warm blooded?").holds


class _StubPerceptor:
    """A perceptor with fixed output, so the test exercises the pipeline's
    confidence policy rather than whichever text extractor is installed."""

    name = "stub"

    def perceive(self, raw):
        from neuralmind.core.parser import parse_atom
        from neuralmind.knowledge.base import FactRecord
        from neuralmind.perception.base import Perception

        return Perception(
            source="stub",
            facts=[
                FactRecord(parse_atom("attr(bob, blue)"), 0.40, "stub"),
                FactRecord(parse_atom("attr(bob, round)"), 0.99, "stub"),
            ],
        )


def test_low_confidence_facts_are_dropped_and_reported():
    kb = KnowledgeBase("t")
    pipeline = NeuralMindPipeline(kb, perceptor=_StubPerceptor(), minimum_confidence=0.9)
    result = pipeline.run("anything")
    assert any("dropped" in note and "attr(bob, blue)" in note for note in result.notes)
    assert [str(record.atom) for record in kb.facts] == ["attr(bob, round)"]


def test_contradictions_are_flagged_not_hidden():
    kb = KnowledgeBase("t").load_builtin("triples")
    result = NeuralMindPipeline(kb).run("Bob is blue. Bob is not blue.")
    assert not result.consistent
    assert any("violation" in note for note in result.notes)


def test_reject_policy_refuses_to_answer():
    kb = KnowledgeBase("t").load_builtin("triples")
    pipeline = NeuralMindPipeline(kb, on_violation=ViolationPolicy.REJECT)
    result = pipeline.run("Bob is blue. Bob is not blue.", question="Is Bob blue?")
    assert result.rejected and result.answer is None


def test_from_rules_accepts_bundled_names_and_source():
    pipeline = NeuralMindPipeline.from_rules("q(X) :- p(X).")
    pipeline.knowledge.add_fact("p(a)")
    assert pipeline.ask("q(a)").holds


def test_pipeline_without_input_queries_what_is_known():
    kb = KnowledgeBase("t").load_builtin("family")
    kb.add_facts(["parent(a, b)", "parent(b, c)"])
    assert NeuralMindPipeline(kb).ask("ancestor(a, c)").holds
