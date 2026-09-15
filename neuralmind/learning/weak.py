"""Weakly supervised training: learning perception from what the rules entail.

The network never sees a label for what it is looking at. It sees a *logical
consequence* of what it is looking at -- the sum of two digits, the outcome of
a rule, the answer to a query -- and the gradient of that consequence's
probability teaches it the rest.

This is the capability the reasoning-only pipeline does not have, and it is
what DeepProbLog and Scallop exist to provide. The mechanism here is exact
weighted model counting (:mod:`neuralmind.learning.semantic_loss`) rather than
circuit compilation, which is tractable precisely because the symbolic side of
the problem is small.

The neural side is unchanged: the same :class:`~neuralmind.perception.nn.ConvNet`,
the same Adam. Only the loss is different, and the loss is the logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np

from ..core.terms import Atom
from ..perception.nn import Adam, ConvNet, _as_batch, _softmax
from .semantic_loss import SemanticLoss, softmax_backward

__all__ = ["WeaklySupervisedTrainer", "WeakExample", "WeakTrainingReport"]


@dataclass
class WeakExample:
    """One training example: images for each slot, and what is known to follow.

    ``query`` is the logical statement known to be true of this example -- for
    digit addition, ``sum(11)``. There is deliberately no field for the true
    digits: the trainer could not use them if they were there.
    """

    images: np.ndarray  # (slots, H, W)
    query: Union[Atom, str, Sequence]
    evidence: tuple = ()


@dataclass
class WeakTrainingReport:
    """What happened, including the supervision that was deliberately withheld."""

    epochs: int = 0
    losses: list[float] = field(default_factory=list)
    #: Accuracy on the underlying task, measured but never trained on.
    slot_accuracies: list[float] = field(default_factory=list)
    query_accuracies: list[float] = field(default_factory=list)
    final_slot_accuracy: float = 0.0
    final_query_accuracy: float = 0.0
    seconds: float = 0.0
    labels_seen: int = 0  # stays zero; that is the point
    #: Examples whose label the rules cannot produce at all.
    unsatisfiable_examples: int = 0

    def to_dict(self) -> dict:
        return {
            "epochs": self.epochs,
            "loss_per_epoch": [round(x, 4) for x in self.losses],
            "slot_accuracy_per_epoch": [round(x, 4) for x in self.slot_accuracies],
            "query_accuracy_per_epoch": [round(x, 4) for x in self.query_accuracies],
            "final_slot_accuracy": round(self.final_slot_accuracy, 4),
            "final_query_accuracy": round(self.final_query_accuracy, 4),
            "slot_labels_used_in_training": self.labels_seen,
            "unsatisfiable_examples": self.unsatisfiable_examples,
            "seconds": round(self.seconds, 1),
        }


class WeaklySupervisedTrainer:
    """Trains a classifier through a logic program.

    Parameters
    ----------
    network:
        Anything with ``forward(x, training)``, ``layers``, and logits output.
        The bundled :class:`~neuralmind.perception.nn.ConvNet` qualifies.
    loss:
        A :class:`~neuralmind.learning.semantic_loss.SemanticLoss` whose slots
        line up with the rows of each example's ``images``.
    """

    def __init__(
        self,
        network: ConvNet,
        loss: SemanticLoss,
        learning_rate: float = 1e-3,
    ) -> None:
        self.network = network
        self.loss = loss
        self.optimiser = Adam(learning_rate)
        self.slots = len(loss.slots)

    # -- one step ---------------------------------------------------------

    def train_step(self, batch: Sequence[WeakExample]) -> float:
        """One gradient step over a batch. Returns the mean loss."""
        if not batch:
            raise ValueError("train_step needs at least one example")
        stacked, shape = self._stack(batch)
        logits = self.network.forward(stacked, training=True)
        probabilities = _softmax(logits)
        per_slot = probabilities.reshape(self.slots, len(batch), -1)

        total = 0.0
        gradient = np.zeros_like(per_slot)
        for index, example in enumerate(batch):
            value, slot_gradient = self.loss.loss_and_gradient(
                per_slot[:, index, :], example.query, example.evidence
            )
            total += value
            gradient[:, index, :] = slot_gradient
        gradient /= max(len(batch), 1)

        # Push the gradient on probabilities back through the softmax, then
        # through the network. Both slots share one network, so one backward
        # pass covers the whole batch.
        logit_gradient = softmax_backward(probabilities, gradient.reshape(probabilities.shape))
        grad = logit_gradient
        for layer in reversed(self.network.layers):
            grad = layer.backward(grad)
        self.optimiser.step(self.network.layers)
        return total / max(len(batch), 1)

    def _stack(self, batch: Sequence[WeakExample]) -> tuple[np.ndarray, tuple]:
        """Lay out a batch as (slot0 images ..., slot1 images ...)."""
        images = np.stack([example.images for example in batch])  # (B, slots, H, W)
        if images.shape[1] != self.slots:
            raise ValueError(
                f"each example needs {self.slots} images, one per neural "
                f"predicate; got {images.shape[1]}"
            )
        reordered = np.transpose(images, (1, 0, 2, 3))  # (slots, B, H, W)
        flat = reordered.reshape(-1, *images.shape[2:])
        return _as_batch(flat), images.shape

    # -- the loop ---------------------------------------------------------

    def fit(
        self,
        examples: Sequence[WeakExample],
        epochs: int = 10,
        batch_size: int = 32,
        seed: int = 0,
        validation: Optional[Callable[[], tuple[float, float]]] = None,
        verbose: bool = True,
    ) -> WeakTrainingReport:
        """Train on examples labelled only with their logical consequence.

        ``validation`` is an optional callable returning
        ``(slot_accuracy, query_accuracy)``. Slot accuracy is measured against
        labels the trainer never touches -- it is a diagnostic, not a signal.
        """
        import time

        rng = np.random.default_rng(seed)
        report = WeakTrainingReport()
        started = time.time()

        for epoch in range(epochs):
            order = rng.permutation(len(examples))
            total = 0.0
            steps = 0
            for start in range(0, len(examples), batch_size):
                batch = [examples[i] for i in order[start : start + batch_size]]
                total += self.train_step(batch)
                steps += 1
            mean_loss = total / max(steps, 1)
            report.losses.append(mean_loss)
            line = f"epoch {epoch + 1}/{epochs}: loss {mean_loss:.4f}"
            if validation is not None:
                slot_accuracy, query_accuracy = validation()
                report.slot_accuracies.append(slot_accuracy)
                report.query_accuracies.append(query_accuracy)
                report.final_slot_accuracy = slot_accuracy
                report.final_query_accuracy = query_accuracy
                line += (
                    f", digit accuracy {slot_accuracy:.4f}"
                    f", query accuracy {query_accuracy:.4f}"
                )
            if verbose:
                print(line, flush=True)
        report.epochs = epochs
        report.seconds = time.time() - started
        report.unsatisfiable_examples = self.loss.unsatisfiable_queries
        if report.unsatisfiable_examples and verbose:
            print(
                f"warning: {report.unsatisfiable_examples} example(s) had a label "
                "the rules cannot produce, and contributed nothing to training",
                flush=True,
            )
        return report

    # -- inference --------------------------------------------------------

    def predict_slots(self, images: np.ndarray) -> np.ndarray:
        """Per-slot predictions for one example's images."""
        probabilities = self.network.predict_proba(images)
        return probabilities.argmax(axis=1)
