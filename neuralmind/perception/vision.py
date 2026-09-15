"""Vision perception: images in, ground atoms out.

The digit perceptor wraps the small CNN from :mod:`neuralmind.perception.nn`
and reports what it sees as ``digit(Slot, Value)`` facts, each carrying the
softmax probability as its confidence. Unlike the structural weights the text
layer attaches, these *are* probabilities from a model, which is what makes
them usable by the Type 5 consistency layer.

The perceptor keeps the full distribution, not just the argmax. That matters:
when the rules say the top prediction is impossible, the second-best candidate
is where the correct answer usually is.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from ..core.terms import Atom, Const
from ..knowledge.base import FactRecord
from .base import Perception, PerceptionError

__all__ = ["DigitPerceptor", "DEFAULT_CHECKPOINT", "checkpoint_available"]

DEFAULT_CHECKPOINT = Path(__file__).parent / "weights" / "mnist_cnn.npz"


def checkpoint_available(path: Optional[Path] = None) -> bool:
    return (path or DEFAULT_CHECKPOINT).exists()


class DigitPerceptor:
    """Classifies 28x28 digit images into ``digit(Slot, Value)`` facts.

    Parameters
    ----------
    checkpoint:
        Trained weights. Defaults to the bundled MNIST checkpoint; pass
        ``None`` with ``network=`` to supply your own model.
    network:
        Anything with ``predict_proba(images) -> (N, 10)``. A torch model with
        that method drops straight in.
    predicate:
        Name of the emitted predicate.
    """

    name = "mnist-cnn"

    def __init__(
        self,
        checkpoint: Optional[Path] = DEFAULT_CHECKPOINT,
        network=None,
        predicate: str = "digit",
    ) -> None:
        self.predicate = predicate
        if network is not None:
            self.network = network
        else:
            from .nn import ConvNet

            path = Path(checkpoint) if checkpoint else DEFAULT_CHECKPOINT
            if not path.exists():
                raise PerceptionError(
                    f"no trained digit model at {path}. Train one with "
                    "`python scripts/train_mnist.py`, or pass network=."
                )
            self.network = ConvNet.from_checkpoint(path)

    # -- perception ------------------------------------------------------

    def perceive(self, raw, slots: Optional[Sequence[str]] = None) -> Perception:
        """Classify one image or a batch, emitting one fact per image.

        ``slots`` names the images; the default is ``d0``, ``d1``, ...
        """
        images = np.asarray(raw, dtype=np.float32)
        if images.ndim == 2:
            images = images[None, :, :]
        probabilities = self.network.predict_proba(images)
        names = list(slots) if slots is not None else [f"d{i}" for i in range(len(images))]
        if len(names) != len(images):
            raise ValueError(f"got {len(names)} slot names for {len(images)} images")

        perception = Perception(source=f"vision:{self.name}")
        distributions: dict[str, dict[int, float]] = {}
        for slot, row in zip(names, probabilities):
            predicted = int(row.argmax())
            perception.facts.append(
                FactRecord(
                    atom=Atom(self.predicate, (Const(slot), Const(predicted))),
                    confidence=float(row[predicted]),
                    provenance=f"perception:{self.name}",
                    evidence=f"image {slot}",
                )
            )
            distributions[slot] = {value: float(p) for value, p in enumerate(row)}
        perception.diagnostics["distributions"] = distributions
        perception.diagnostics["slots"] = names
        return perception

    def distributions(self, raw, slots: Optional[Sequence[str]] = None) -> list["SlotDistribution"]:
        """Per-image distributions, for the consistency layer to search over."""
        perception = self.perceive(raw, slots)
        raw_distributions = perception.diagnostics["distributions"]
        return [
            SlotDistribution(
                predicate=self.predicate,
                key=(Const(slot),),
                probabilities={
                    Const(value): probability
                    for value, probability in sorted(
                        raw_distributions[slot].items(), key=lambda kv: -kv[1]
                    )
                },
            )
            for slot in perception.diagnostics["slots"]
        ]


@dataclass
class SlotDistribution:
    """A distribution over the possible values of one perceived slot.

    ``digit(d0, ?)`` with a probability for each candidate value. This is the
    interface between a neural classifier and the consistency layer: the layer
    searches over these candidates for the most probable assignment that the
    rules actually allow.
    """

    predicate: str
    key: tuple
    probabilities: dict

    def atom(self, value) -> Atom:
        return Atom(self.predicate, self.key + (value,))

    def top(self, k: int = 3) -> list[tuple]:
        ranked = sorted(self.probabilities.items(), key=lambda kv: -kv[1])
        return ranked[:k]

    @property
    def best(self):
        return max(self.probabilities.items(), key=lambda kv: kv[1])

    @property
    def entropy(self) -> float:
        values = np.array(list(self.probabilities.values()), dtype=np.float64)
        values = values[values > 0]
        return float(-(values * np.log2(values)).sum())

    def __str__(self) -> str:
        best_value, best_probability = self.best
        return f"{self.predicate}{self.key} = {best_value} (p={best_probability:.3f})"
