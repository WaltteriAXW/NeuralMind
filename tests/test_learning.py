"""Gradients through the logic: the semantic loss and weak supervision."""

import pytest

# The learning layer is NumPy-only by design, so skip the module rather than
# failing collection where NumPy is absent.
np = pytest.importorskip("numpy")

from neuralmind.knowledge.base import KnowledgeBase
from neuralmind.learning.semantic_loss import (
    NeuralPredicate, SemanticLoss, TableTooLarge, softmax_backward,
)


def _mnist_available() -> bool:
    from neuralmind.datasets.mnist import mnist_available

    return mnist_available()


@pytest.fixture(scope="module")
def sum_loss():
    kb = KnowledgeBase("sum").load_builtin("mnist_sum")
    slots = [
        NeuralPredicate.over_integers("digit", "d0", 10),
        NeuralPredicate.over_integers("digit", "d1", 10),
    ]
    return SemanticLoss(kb, slots)


def normalised(rng, shape):
    values = rng.random(shape) + 0.05
    return values / values.sum(axis=1, keepdims=True)


# -- the truth tensor ------------------------------------------------------


def test_truth_table_is_computed_by_the_symbolic_engine(sum_loss):
    table = sum_loss.truth_table("sum(7)")
    assert table.shape == (10, 10)
    # Exactly the pairs that add to 7, as the rule says.
    expected = np.array([[a + b == 7 for b in range(10)] for a in range(10)])
    assert np.array_equal(table, expected)


def test_truth_table_respects_integrity_constraints():
    kb = KnowledgeBase("t").add_rules(
        "total(S) :- d(a, X), d(b, Y), S = X + Y.\n"
        "%@ the two values must differ\n"
        ":- d(a, X), d(b, X)."
    )
    slots = [NeuralPredicate.over_integers("d", k, 5) for k in ("a", "b")]
    table = SemanticLoss(kb, slots).truth_table("total(4)")
    # (2, 2) sums to 4 but breaks the constraint, so it entails nothing.
    assert table[0, 4] and table[4, 0] and table[1, 3]
    assert not table[2, 2]


def test_truth_table_is_cached(sum_loss):
    before = sum_loss.engine_calls
    sum_loss.truth_table("sum(7)")
    sum_loss.truth_table("sum(7)")
    assert sum_loss.engine_calls == before  # already computed in an earlier test


def test_evidence_changes_what_is_entailed():
    kb = KnowledgeBase("t").add_rules("ok :- d(a, X), allowed(X).")
    slots = [NeuralPredicate.over_integers("d", "a", 4)]
    loss = SemanticLoss(kb, slots)
    assert not loss.truth_table("ok").any()
    assert loss.truth_table("ok", evidence=["allowed(2)"]).tolist() == [
        False, False, True, False
    ]


def test_an_oversized_table_is_refused_with_the_reason():
    slots = [NeuralPredicate.over_integers("d", k, 50) for k in "abcd"]
    with pytest.raises(TableTooLarge, match=r"\*\* slots"):
        SemanticLoss(KnowledgeBase(), slots, max_table=1000)


def test_at_least_one_slot_is_required():
    with pytest.raises(ValueError, match="at least one"):
        SemanticLoss(KnowledgeBase(), [])


# -- probability -----------------------------------------------------------


def test_probability_is_an_exact_weighted_model_count(sum_loss):
    uniform = np.full((2, 10), 0.1)
    # Eight of the hundred pairs sum to seven.
    assert sum_loss.probability(uniform, "sum(7)") == pytest.approx(0.08)


def test_probability_sums_to_one_over_every_possible_sum(sum_loss):
    rng = np.random.default_rng(0)
    distributions = normalised(rng, (2, 10))
    total = sum(sum_loss.probability(distributions, f"sum({s})") for s in range(19))
    assert total == pytest.approx(1.0)


def test_a_confident_correct_reading_has_high_probability(sum_loss):
    distributions = np.full((2, 10), 0.001)
    distributions[0, 3] = distributions[1, 4] = 0.991
    assert sum_loss.probability(distributions, "sum(7)") > 0.9


def test_mismatched_shapes_are_rejected(sum_loss):
    with pytest.raises(ValueError, match="expected probabilities shaped"):
        sum_loss.loss_and_gradient(np.full((3, 10), 0.1), "sum(7)")
    with pytest.raises(ValueError, match="values but the"):
        sum_loss.loss_and_gradient(np.full((2, 5), 0.2), "sum(7)")


# -- gradients -------------------------------------------------------------


def _numeric_gradient(loss, distributions, query, epsilon=1e-6):
    gradient = np.zeros_like(distributions)
    for row in range(distributions.shape[0]):
        for column in range(distributions.shape[1]):
            plus = distributions.copy()
            minus = distributions.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            gradient[row, column] = (
                loss.loss_and_gradient(plus, query)[0]
                - loss.loss_and_gradient(minus, query)[0]
            ) / (2 * epsilon)
    return gradient


def test_gradient_matches_finite_differences(sum_loss):
    rng = np.random.default_rng(1)
    distributions = normalised(rng, (2, 10))
    _, analytic = sum_loss.loss_and_gradient(distributions, "sum(7)")
    numeric = _numeric_gradient(sum_loss, distributions, "sum(7)")
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_gradient_matches_finite_differences_with_three_slots():
    kb = KnowledgeBase("t").add_rules(
        "total(S) :- d(a, X), d(b, Y), d(c, Z), S = X + Y + Z."
    )
    slots = [NeuralPredicate.over_integers("d", k, 4) for k in ("a", "b", "c")]
    loss = SemanticLoss(kb, slots)
    rng = np.random.default_rng(2)
    distributions = normalised(rng, (3, 4))
    _, analytic = loss.loss_and_gradient(distributions, "total(4)")
    numeric = _numeric_gradient(loss, distributions, "total(4)")
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_gradient_points_toward_the_correct_reading(sum_loss):
    distributions = np.full((2, 10), 0.1)
    _, gradient = sum_loss.loss_and_gradient(distributions, "sum(0)")
    # Only 0 + 0 gives zero, so mass should move to 0 in both slots and
    # nowhere else. The loss falls as p[0] rises, so its gradient is negative.
    assert gradient[0, 0] < 0 and gradient[1, 0] < 0
    assert np.all(gradient[:, 1:] == 0)


def test_an_unsatisfiable_query_gives_a_flat_gradient(sum_loss):
    distributions = np.full((2, 10), 0.1)
    loss, gradient = sum_loss.loss_and_gradient(distributions, "sum(99)")
    assert loss > 20 and np.all(gradient == 0)


def test_softmax_backward_matches_finite_differences():
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(4, 6))
    weights = rng.normal(size=(4, 6))

    def scalar(values):
        shifted = values - values.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return float((exponentiated / exponentiated.sum(axis=1, keepdims=True) * weights).sum())

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    analytic = softmax_backward(probabilities, weights)

    numeric = np.zeros_like(logits)
    epsilon = 1e-6
    for row in range(logits.shape[0]):
        for column in range(logits.shape[1]):
            plus, minus = logits.copy(), logits.copy()
            plus[row, column] += epsilon
            minus[row, column] -= epsilon
            numeric[row, column] = (scalar(plus) - scalar(minus)) / (2 * epsilon)
    assert np.allclose(analytic, numeric, atol=1e-6)


# -- most probable assignment ---------------------------------------------


def test_most_probable_assignment_obeys_the_rules(sum_loss):
    distributions = np.full((2, 10), 0.001)
    distributions[0, 1] = 0.9
    distributions[1, 8] = 0.8  # the likeliest reading sums to 9, not 3
    distributions[1, 2] = 0.19
    found = sum_loss.most_probable_assignment(distributions, "sum(3)")
    assert found is not None
    atoms, _ = found
    assert [str(a) for a in atoms] == ["digit(d0, 1)", "digit(d1, 2)"]


def test_most_probable_assignment_returns_none_when_nothing_fits(sum_loss):
    assert sum_loss.most_probable_assignment(np.full((2, 10), 0.1), "sum(99)") is None


# -- the trainer -----------------------------------------------------------


def test_training_reduces_the_loss(sum_loss):
    from neuralmind.learning.weak import WeaklySupervisedTrainer, WeakExample
    from neuralmind.perception.nn import ConvNet

    rng = np.random.default_rng(0)
    # Two easily separable "digits": a bright top half is 0, bottom half is 1.
    def image(value):
        canvas = np.zeros((28, 28), dtype=np.float32)
        canvas[:14] = 1.0 if value == 0 else 0.0
        canvas[14:] = 1.0 if value == 1 else 0.0
        return canvas + rng.normal(0, 0.02, (28, 28)).astype(np.float32)

    examples = []
    for _ in range(48):
        left, right = int(rng.integers(0, 2)), int(rng.integers(0, 2))
        examples.append(
            WeakExample(images=np.stack([image(left), image(right)]),
                        query=f"sum({left + right})")
        )

    trainer = WeaklySupervisedTrainer(ConvNet(seed=0), sum_loss, learning_rate=3e-3)
    first = trainer.train_step(examples[:16])
    for _ in range(12):
        trainer.train_step(examples[:16])
    last = trainer.train_step(examples[:16])
    assert last < first


def test_the_trainer_rejects_a_wrong_number_of_images(sum_loss):
    from neuralmind.learning.weak import WeaklySupervisedTrainer, WeakExample
    from neuralmind.perception.nn import ConvNet

    trainer = WeaklySupervisedTrainer(ConvNet(seed=0), sum_loss)
    batch = [WeakExample(images=np.zeros((3, 28, 28), dtype=np.float32), query="sum(4)")]
    with pytest.raises(ValueError, match="one per neural"):
        trainer.train_step(batch)


def test_report_records_that_no_slot_labels_were_used():
    from neuralmind.learning.weak import WeakTrainingReport

    assert WeakTrainingReport().to_dict()["slot_labels_used_in_training"] == 0


def test_slots_must_share_a_domain_size():
    slots = [
        NeuralPredicate.over_integers("d", "a", 4),
        NeuralPredicate.over_integers("d", "b", 7),
    ]
    with pytest.raises(ValueError, match="same number of values"):
        SemanticLoss(KnowledgeBase(), slots)


def test_unsatisfiable_queries_are_counted_not_silent():
    kb = KnowledgeBase("sum").load_builtin("mnist_sum")
    slots = [
        NeuralPredicate.over_integers("digit", "d0", 10),
        NeuralPredicate.over_integers("digit", "d1", 10),
    ]
    loss = SemanticLoss(kb, slots)
    assert loss.unsatisfiable_queries == 0
    loss.loss_and_gradient(np.full((2, 10), 0.1), "sum(99)")
    loss.loss_and_gradient(np.full((2, 10), 0.1), "sum(7)")
    assert loss.unsatisfiable_queries == 1


def test_an_empty_batch_is_rejected(sum_loss):
    from neuralmind.learning.weak import WeaklySupervisedTrainer
    from neuralmind.perception.nn import ConvNet

    with pytest.raises(ValueError, match="at least one example"):
        WeaklySupervisedTrainer(ConvNet(seed=0), sum_loss).train_step([])


@pytest.mark.skipif(not _mnist_available(), reason="MNIST data missing")
def test_the_committed_weakly_supervised_checkpoint_holds_its_claim():
    """Guard the README's headline number against drift.

    This checkpoint was trained without a single digit label. If it ever stops
    clearing the bar, the claim in the README is wrong and this fails first.
    """
    from pathlib import Path

    from neuralmind.datasets.mnist import load_mnist
    from neuralmind.perception.nn import ConvNet

    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "neuralmind/perception/weights/mnist_weak.npz"
    )
    if not checkpoint.exists():  # pragma: no cover - only if weights are stripped
        pytest.skip("weakly supervised checkpoint not present")
    _, test = load_mnist()
    accuracy = ConvNet.from_checkpoint(checkpoint).accuracy(test.images, test.labels)
    assert accuracy > 0.97, f"weakly supervised checkpoint fell to {accuracy:.4f}"
