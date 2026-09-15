"""A small convolutional network in NumPy.

This is the only place in the system where anything is *learned*. It exists to
make the point the blueprint insists on: "no LLM" does not mean "no neural
network". This is a ~13k-parameter classifier trained for one narrow job, with
no generative component anywhere -- exactly the kind of model that belongs in a
perception layer.

NumPy rather than PyTorch, for two reasons. The network is small enough that
the difference does not matter, and a pipeline whose only heavy dependency is
NumPy stays installable anywhere. Swapping in a torch model later means
implementing :meth:`predict_proba`; nothing above this layer needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional, Sequence

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "this layer needs NumPy. Install it with `pip install 'neuralmind[numeric]'`. "
        "The symbolic core and the text perception layer work without it."
    ) from exc

__all__ = ["ConvNet", "Conv2D", "MaxPool2D", "Dense", "ReLU", "Flatten", "Adam"]


class Layer:
    """Base class: forward pass, backward pass, and trainable parameters."""

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self) -> list[tuple[str, np.ndarray, np.ndarray]]:
        """``(name, value, gradient)`` triples for the optimiser."""
        return []


class Conv2D(Layer):
    """Valid-padding 2-D convolution, implemented with an im2col reshape."""

    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, rng=None) -> None:
        rng = rng or np.random.default_rng(0)
        fan_in = in_channels * kernel * kernel
        # He initialisation, appropriate for the ReLU that follows.
        self.weight = rng.normal(0.0, np.sqrt(2.0 / fan_in), (out_channels, fan_in)).astype(np.float32)
        self.bias = np.zeros(out_channels, dtype=np.float32)
        self.kernel = kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        n, c, h, w = x.shape
        k = self.kernel
        out_h, out_w = h - k + 1, w - k + 1
        cols = _im2col(x, k)  # (N, C*k*k, out_h*out_w)
        out = np.einsum("fp,npl->nfl", self.weight, cols) + self.bias[None, :, None]
        if training:
            self._cache = (x.shape, cols)
        return out.reshape(n, self.out_channels, out_h, out_w)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._cache is not None, "backward() called before forward(training=True)"
        shape, cols = self._cache
        n, c, h, w = shape
        grad_flat = grad.reshape(n, self.out_channels, -1)
        self.grad_weight = np.einsum("nfl,npl->fp", grad_flat, cols)
        self.grad_bias = grad_flat.sum(axis=(0, 2))
        grad_cols = np.einsum("fp,nfl->npl", self.weight, grad_flat)
        return _col2im(grad_cols, shape, self.kernel)

    def parameters(self):
        return [
            ("weight", self.weight, self.grad_weight),
            ("bias", self.bias, self.grad_bias),
        ]


class MaxPool2D(Layer):
    """Non-overlapping max pooling."""

    def __init__(self, size: int = 2) -> None:
        self.size = size
        self._cache: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        n, c, h, w = x.shape
        s = self.size
        h_trim, w_trim = h - h % s, w - w % s
        trimmed = x[:, :, :h_trim, :w_trim]
        blocks = trimmed.reshape(n, c, h_trim // s, s, w_trim // s, s)
        out = blocks.max(axis=(3, 5))
        if training:
            mask = blocks == out[:, :, :, None, :, None]
            # Ties would otherwise route the gradient to several inputs.
            mask = mask & (mask.cumsum(axis=3).cumsum(axis=5) == 1)
            self._cache = (x.shape, mask, h_trim, w_trim)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._cache is not None
        shape, mask, h_trim, w_trim = self._cache
        n, c, h, w = shape
        s = self.size
        expanded = mask * grad[:, :, :, None, :, None]
        out = np.zeros(shape, dtype=grad.dtype)
        out[:, :, :h_trim, :w_trim] = expanded.reshape(n, c, h_trim, w_trim)
        return out


class ReLU(Layer):
    def __init__(self) -> None:
        self._mask: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        mask = x > 0
        if training:
            self._mask = mask
        return x * mask

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._mask is not None
        return grad * self._mask


class Flatten(Layer):
    def __init__(self) -> None:
        self._shape: Optional[tuple] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._shape is not None
        return grad.reshape(self._shape)


class Dense(Layer):
    def __init__(self, in_features: int, out_features: int, rng=None) -> None:
        rng = rng or np.random.default_rng(0)
        scale = np.sqrt(2.0 / in_features)
        self.weight = rng.normal(0.0, scale, (in_features, out_features)).astype(np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.grad_weight = np.zeros_like(self.weight)
        self.grad_bias = np.zeros_like(self.bias)
        self._input: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        if training:
            self._input = x
        return x @ self.weight + self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        assert self._input is not None
        self.grad_weight = self._input.T @ grad
        self.grad_bias = grad.sum(axis=0)
        return grad @ self.weight.T

    def parameters(self):
        return [
            ("weight", self.weight, self.grad_weight),
            ("bias", self.bias, self.grad_bias),
        ]


class Adam:
    """Adam, with the usual defaults."""

    def __init__(self, learning_rate: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999) -> None:
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.step_count = 0
        self._moments: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def step(self, layers: Sequence[Layer]) -> None:
        self.step_count += 1
        bias1 = 1 - self.beta1**self.step_count
        bias2 = 1 - self.beta2**self.step_count
        for layer in layers:
            for _, value, gradient in layer.parameters():
                key = id(value)
                m, v = self._moments.setdefault(
                    key, (np.zeros_like(value), np.zeros_like(value))
                )
                m *= self.beta1
                m += (1 - self.beta1) * gradient
                v *= self.beta2
                v += (1 - self.beta2) * (gradient * gradient)
                value -= self.learning_rate * (m / bias1) / (np.sqrt(v / bias2) + self.epsilon)


@dataclass
class TrainingReport:
    """What happened during training, for the record."""

    epochs: int = 0
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    final_accuracy: float = 0.0
    parameters: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "final_accuracy": round(self.final_accuracy, 4),
            "loss_per_epoch": [round(x, 4) for x in self.losses],
            "accuracy_per_epoch": [round(x, 4) for x in self.accuracies],
            "parameters": self.parameters,
            "seconds": round(self.seconds, 1),
        }


class ConvNet:
    """A small CNN for 28x28 greyscale digits.

    The architecture is conv(1->16) - pool - conv(16->32) - pool - dense(10),
    about 13k parameters. Nothing exotic: the point is a task-specific model
    whose output is a probability vector the symbolic layer can act on.
    """

    def __init__(self, num_classes: int = 10, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self.num_classes = num_classes
        self.layers: list[Layer] = [
            Conv2D(1, 16, 3, rng),
            ReLU(),
            MaxPool2D(2),
            Conv2D(16, 32, 3, rng),
            ReLU(),
            MaxPool2D(2),
            Flatten(),
            Dense(32 * 5 * 5, num_classes, rng),
        ]

    # -- inference -------------------------------------------------------

    def forward(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x, training)
        return x

    def predict_proba(self, images: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Class probabilities for a batch of images, shaped ``(N, 28, 28)``."""
        x = _as_batch(images)
        outputs = [
            _softmax(self.forward(x[start : start + batch_size]))
            for start in range(0, len(x), batch_size)
        ]
        return np.concatenate(outputs, axis=0)

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(axis=1)

    def accuracy(self, images: np.ndarray, labels: np.ndarray) -> float:
        return float((self.predict(images) == np.asarray(labels)).mean())

    @property
    def parameter_count(self) -> int:
        return sum(value.size for layer in self.layers for _, value, _ in layer.parameters())

    # -- training --------------------------------------------------------

    def fit(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        epochs: int = 6,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        validation: Optional[tuple[np.ndarray, np.ndarray]] = None,
        seed: int = 0,
        verbose: bool = True,
    ) -> TrainingReport:
        """Train with Adam on cross-entropy loss."""
        import time

        x = _as_batch(images)
        y = np.asarray(labels).astype(np.int64)
        optimiser = Adam(learning_rate)
        rng = np.random.default_rng(seed)
        report = TrainingReport(parameters=self.parameter_count)
        started = time.time()

        for epoch in range(epochs):
            order = rng.permutation(len(x))
            total_loss = 0.0
            batches = 0
            for start in range(0, len(x), batch_size):
                index = order[start : start + batch_size]
                logits = self.forward(x[index], training=True)
                loss, grad = _softmax_cross_entropy(logits, y[index])
                for layer in reversed(self.layers):
                    grad = layer.backward(grad)
                optimiser.step(self.layers)
                total_loss += loss
                batches += 1
            mean_loss = total_loss / max(batches, 1)
            report.losses.append(mean_loss)
            if validation is not None:
                accuracy = self.accuracy(validation[0], validation[1])
                report.accuracies.append(accuracy)
                report.final_accuracy = accuracy
            if verbose:
                suffix = (
                    f", validation accuracy {report.accuracies[-1]:.4f}"
                    if report.accuracies
                    else ""
                )
                print(f"epoch {epoch + 1}/{epochs}: loss {mean_loss:.4f}{suffix}")
        report.epochs = epochs
        report.seconds = time.time() - started
        return report

    # -- persistence -----------------------------------------------------

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        for index, layer in enumerate(self.layers):
            for name, value, _ in layer.parameters():
                arrays[f"{index}.{name}"] = value
        np.savez_compressed(path, **arrays)
        return path

    def load(self, path: Path | str) -> "ConvNet":
        data = np.load(Path(path))
        for index, layer in enumerate(self.layers):
            for name, value, _ in layer.parameters():
                key = f"{index}.{name}"
                if key not in data:
                    raise ValueError(f"checkpoint is missing {key}; architecture mismatch")
                if data[key].shape != value.shape:
                    raise ValueError(
                        f"architecture mismatch at {key}: checkpoint holds "
                        f"{data[key].shape}, this network expects {value.shape}"
                    )
                value[...] = data[key]
        return self

    @classmethod
    def from_checkpoint(cls, path: Path | str, **kwargs) -> "ConvNet":
        return cls(**kwargs).load(path)


# -- functional helpers --------------------------------------------------


def _as_batch(images: np.ndarray) -> np.ndarray:
    x = np.asarray(images, dtype=np.float32)
    if x.ndim == 2:  # a single 28x28 image
        x = x[None, None, :, :]
    elif x.ndim == 3:  # (N, 28, 28)
        x = x[:, None, :, :]
    elif x.ndim != 4:
        raise ValueError(f"expected images shaped (N, H, W) or (N, C, H, W), got {x.shape}")
    return x


def _im2col(x: np.ndarray, kernel: int) -> np.ndarray:
    n, c, h, w = x.shape
    out_h, out_w = h - kernel + 1, w - kernel + 1
    cols = np.empty((n, c, kernel, kernel, out_h, out_w), dtype=x.dtype)
    for i in range(kernel):
        for j in range(kernel):
            cols[:, :, i, j] = x[:, :, i : i + out_h, j : j + out_w]
    return cols.reshape(n, c * kernel * kernel, out_h * out_w)


def _col2im(cols: np.ndarray, shape: tuple, kernel: int) -> np.ndarray:
    n, c, h, w = shape
    out_h, out_w = h - kernel + 1, w - kernel + 1
    reshaped = cols.reshape(n, c, kernel, kernel, out_h, out_w)
    out = np.zeros(shape, dtype=cols.dtype)
    for i in range(kernel):
        for j in range(kernel):
            out[:, :, i : i + out_h, j : j + out_w] += reshaped[:, :, i, j]
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum(axis=1, keepdims=True)


def _softmax_cross_entropy(logits: np.ndarray, labels: np.ndarray) -> tuple[float, np.ndarray]:
    probabilities = _softmax(logits)
    n = len(labels)
    loss = float(-np.log(probabilities[np.arange(n), labels] + 1e-12).mean())
    grad = probabilities.copy()
    grad[np.arange(n), labels] -= 1.0
    return loss, grad / n
