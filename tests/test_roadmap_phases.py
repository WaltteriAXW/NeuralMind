"""The roadmap's milestones, asserted rather than claimed.

Each test is the "done-when" criterion from the blueprint's phase table, so a
regression that breaks a milestone fails here by name.
"""

import pytest

from neuralmind.knowledge.base import KnowledgeBase

try:
    from neuralmind.datasets.mnist import digit_pairs, load_mnist, mnist_available
    from neuralmind.perception.vision import DigitPerceptor, checkpoint_available

    VISION_READY = mnist_available() and checkpoint_available()
except ImportError:  # pragma: no cover - NumPy missing
    VISION_READY = False

vision = pytest.mark.skipif(not VISION_READY, reason="MNIST data or checkpoint missing")


@pytest.fixture(scope="module")
def perceptor():
    return DigitPerceptor()


@pytest.fixture(scope="module")
def pairs():
    _, test = load_mnist()
    return digit_pairs(test, 100, seed=1)


def test_phase_0_the_solver_is_callable_from_python():
    """Phase 0: call the solver from Python and get a result back."""
    from neuralmind.inference.engine import ReasoningEngine

    assert ReasoningEngine("p(a). q(X) :- p(X).").ask("q(a)").holds


@vision
def test_phase_1_mnist_pair_addition_above_90_percent(perceptor, pairs):
    """Phase 1: >90% end-to-end accuracy on digit-pair addition."""
    correct = 0
    for pair in pairs:
        kb = KnowledgeBase("mnist").load_builtin("mnist_sum")
        perceptor.perceive(pair.images, slots=["d0", "d1"]).into(kb)
        answer = kb.engine().ask("sum(S)")
        total = answer.atoms[0].args[0].value if answer.holds else None
        correct += total == pair.total
    assert correct / len(pairs) > 0.90, f"only {correct}/{len(pairs)}"


def test_phase_2_knowledge_base_answers_hand_written_cases(family_kb):
    """Phase 2: query the KB directly, with no neural component, and be right."""
    engine = family_kb.engine()
    cases = {
        "ancestor(maria, elias)": True,
        "grandmother(maria, aino)": True,
        "aunt(liisa, aino)": True,
        "older(maria, juho)": True,
        "sibling(juho, liisa)": True,
        "ancestor(elias, maria)": False,
        "uncle(juho, aino)": False,  # juho is aino's father, not her uncle
    }
    for query, expected in cases.items():
        assert engine.ask(query).holds is expected, query
    assert engine.consistent


def test_phase_3_perception_feeds_the_knowledge_base():
    """Phase 3: real English produces facts the Phase 2 engine consumes."""
    from neuralmind.datasets.proofwriter import generate
    from neuralmind.evaluation import evaluate

    report = evaluate(generate(20, seed=11))
    assert report.accuracy > 0.80, report.describe()
    assert report.theory_mismatches == 0


@vision
def test_phase_4_consistency_layer_repairs_a_misread_digit(perceptor):
    """Phase 4: hard rules catch what noisy perception got wrong."""
    from neuralmind.consistency.layer import ConsistencyLayer

    _, test = load_mnist()
    attempted = repaired = 0
    for pair in digit_pairs(test, 300, seed=2):
        distributions = perceptor.distributions(pair.images, slots=["d0", "d1"])
        predicted = [int(d.best[0].value) for d in distributions]
        truth = [pair.left_label, pair.right_label]
        if predicted == truth:
            continue
        attempted += 1
        kb = KnowledgeBase("mnist").load_builtin("mnist_sum")
        kb.add_fact(f"expected_sum({pair.total})")
        best = ConsistencyLayer(kb).resolve(distributions, candidates_per_slot=4, top_k=1)
        if not best:
            continue
        resolved = [
            int(r.atom.args[1].value) for r in best[0].facts if r.atom.predicate == "digit"
        ]
        repaired += resolved == truth
    assert attempted > 0, "the classifier made no mistakes to repair"
    assert repaired / attempted >= 0.5, f"repaired only {repaired}/{attempted}"


def test_phase_5_output_is_legible_as_json_and_as_prose(family_kb):
    """Phase 5: a non-technical reader can see *why*, not just *what*."""
    import json

    from neuralmind.output.nlg import Realiser
    from neuralmind.output.serialize import to_json

    engine = family_kb.engine()
    answer = engine.ask("aunt(liisa, aino)")
    document = json.loads(to_json(answer.proof))
    assert document["because"], "the proof must nest its reasons"
    prose = Realiser().learn_names(["liisa", "aino", "maria", "juho"]).realise_proof(answer.proof)
    assert "because" in prose and prose.endswith(".")


def test_phase_6_domain_pilot_decides_and_audits():
    """Phase 6: the pilot answers, explains, and audits its own configuration."""
    kb = KnowledgeBase("policy").load_builtin("access_policy")
    kb.add_facts(
        [
            "employee(dana)", "employee(erik)",
            "has_role(dana, analyst)", "has_role(erik, engineer)", "has_role(erik, auditor)",
            "grants(analyst, read, internal)",
            "resource(wiki)", "classification(wiki, internal)",
            "resource(ledger)", "classification(ledger, confidential)",
            "clearance(dana, internal)", "clearance(erik, restricted)",
            "conflicting_duties(engineer, auditor)",
            "request(q1, dana, read, wiki)", "request(q2, dana, read, ledger)",
        ]
    )
    engine = kb.engine()
    assert engine.ask("granted(q1)").holds
    assert engine.ask("denied(q2)").holds
    assert engine.ask("denial_reason(q2, insufficient_clearance)").holds
    # The proof is the access-review evidence.
    proof = engine.ask("granted(q1)").proof
    assert {str(a) for a in proof.premises()} >= {
        "request(q1, dana, read, wiki)", "has_role(dana, analyst)", "clearance(dana, internal)"
    }
    # And the configuration itself is audited, independently of any request.
    assert any("Separation of duties" in v.describe() for v in engine.violations)


def test_phase_7_failures_are_attributed_not_just_counted():
    """Phase 7: say what fraction of failures are perception vs reasoning."""
    from neuralmind.datasets.proofwriter import generate
    from neuralmind.evaluation import ENGINE, evaluate

    report = evaluate(generate(30, seed=21))
    assert report.total > 100
    # Every failure must be attributed, and none may be an engine bug.
    assert all(outcome.failure for outcome in report.failures)
    assert report.failure_breakdown().get(ENGINE, 0) == 0
    assert "perception" in report.describe() or "no failures" in report.describe()
