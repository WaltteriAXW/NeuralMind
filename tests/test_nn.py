"""The NumPy network: shapes, gradients, training and persistence."""

import numpy as np
import pytest

from neuralmind.perception.nn import (
    Conv2D, ConvNet, Dense, Flatten, MaxPool2D, ReLU, _softmax, _softmax_cross_entropy,
)


def _numeric_gradient(function, tensor, epsilon=1e-4):
    """Central-difference gradient of a scalar function w.r.t. a tensor."""
    gradient = np.zeros_like(tensor)
    iterator = np.nditer(tensor, flags=["multi_index"])
    while not iterator.finished:
        index = iterator.multi_index
        original = tensor[index]
        tensor[index] = original + epsilon
        plus = function()
        tensor[index] = original - epsilon
        minus = function()
        tensor[index] = original
        gradient[index] = (plus - minus) / (2 * epsilon)
        iterator.iternext()
    return gradient


@pytest.mark.parametrize("layer_factory,shape", [
    (lambda: Conv2D(2, 3, 3, np.random.default_rng(1)), (2, 2, 6, 6)),
    (lambda: Dense(7, 4, np.random.default_rng(1)), (3, 7)),
    (lambda: MaxPool2D(2), (2, 2, 4, 4)),
    (lambda: ReLU(), (3, 5)),
    (lambda: Flatten(), (2, 3, 4, 4)),
])
def test_backward_matches_finite_differences(layer_factory, shape):
    rng = np.random.default_rng(0)
    layer = layer_factory()
    x = rng.normal(size=shape).astype(np.float64)
    # A random but fixed projection turns the layer output into a scalar.
    seed_output = layer.forward(x.copy(), training=True)
    weights = rng.normal(size=seed_output.shape)

    def loss():
        return float((layer.forward(x.copy()) * weights).sum())

    layer.forward(x.copy(), training=True)
    analytic = layer.backward(weights.copy())
    numeric = _numeric_gradient(loss, x)
    assert np.allclose(analytic, numeric, atol=1e-4), np.abs(analytic - numeric).max()


def test_conv_parameter_gradients_match_finite_differences():
    rng = np.random.default_rng(3)
    layer = Conv2D(1, 2, 3, rng)
    layer.weight = layer.weight.astype(np.float64)
    layer.bias = layer.bias.astype(np.float64)
    x = rng.normal(size=(2, 1, 5, 5))
    weights = rng.normal(size=layer.forward(x.copy()).shape)

    def loss():
        return float((layer.forward(x.copy()) * weights).sum())

    layer.forward(x.copy(), training=True)
    layer.backward(weights.copy())
    numeric = _numeric_gradient(loss, layer.weight)
    assert np.allclose(layer.grad_weight, numeric, atol=1e-4)


def test_maxpool_routes_each_gradient_to_one_input():
    layer = MaxPool2D(2)
    x = np.ones((1, 1, 2, 2))  # every element ties for the maximum
    layer.forward(x, training=True)
    grad = layer.backward(np.array([[[[1.0]]]]))
    assert grad.sum() == 1.0


def test_softmax_is_normalised_and_stable():
    probabilities = _softmax(np.array([[1000.0, 1000.0, 1000.0]]))
    assert np.allclose(probabilities, 1 / 3)


def test_cross_entropy_gradient_matches_finite_differences():
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(4, 3))
    labels = np.array([0, 1, 2, 1])
    _, analytic = _softmax_cross_entropy(logits, labels)
    numeric = _numeric_gradient(
        lambda: _softmax_cross_entropy(logits, labels)[0], logits
    )
    assert np.allclose(analytic, numeric, atol=1e-6)


def test_network_shapes_and_size():
    net = ConvNet()
    assert net.predict_proba(np.zeros((3, 28, 28))).shape == (3, 10)
    assert net.predict(np.zeros((28, 28))).shape == (1,)
    # Small by design: this is a perception model, not a foundation model.
    assert 1_000 < net.parameter_count < 50_000


def test_training_reduces_loss_on_a_separable_task():
    rng = np.random.default_rng(0)
    # Two classes: bright top half vs bright bottom half.
    images = np.zeros((64, 28, 28), dtype=np.float32)
    labels = np.zeros(64, dtype=np.int64)
    for index in range(64):
        labels[index] = index % 2
        if labels[index] == 0:
            images[index, :14, :] = 1.0
        else:
            images[index, 14:, :] = 1.0
    images += rng.normal(0, 0.05, images.shape).astype(np.float32)

    net = ConvNet(num_classes=2, seed=0)
    report = net.fit(images, labels, epochs=4, batch_size=16, verbose=False)
    assert report.losses[-1] < report.losses[0]
    assert net.accuracy(images, labels) > 0.9


def test_checkpoint_round_trip(tmp_path):
    net = ConvNet(seed=1)
    probabilities = net.predict_proba(np.zeros((2, 28, 28)))
    path = net.save(tmp_path / "net.npz")
    restored = ConvNet.from_checkpoint(path)
    assert np.allclose(restored.predict_proba(np.zeros((2, 28, 28))), probabilities)


def test_loading_a_mismatched_checkpoint_fails_loudly(tmp_path):
    path = ConvNet(num_classes=10).save(tmp_path / "net.npz")
    with pytest.raises(ValueError, match="architecture mismatch"):
        ConvNet(num_classes=3).load(path)
