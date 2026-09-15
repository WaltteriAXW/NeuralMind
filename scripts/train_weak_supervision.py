#!/usr/bin/env python3
"""Learn to read digits from sums alone.

The classifier is trained on pairs of MNIST images labelled only with their
sum. It is never shown a digit label -- not once, not for validation, not for
early stopping. The only training signal is the gradient of P(sum = s) through
the rule ``sum(S) :- digit(d0, A), digit(d1, B), S = A + B.``

Digit accuracy is measured afterwards purely as a diagnostic, to answer the
question the experiment is asking: can the logic supply the supervision the
labels did not?

    python scripts/train_weak_supervision.py --pairs 15000 --epochs 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralmind.datasets.mnist import digit_pairs, load_mnist
from neuralmind.knowledge.base import KnowledgeBase
from neuralmind.learning.semantic_loss import NeuralPredicate, SemanticLoss
from neuralmind.learning.weak import WeaklySupervisedTrainer, WeakExample
from neuralmind.perception.nn import ConvNet

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1] / "neuralmind/perception/weights/mnist_weak.npz"
)


def build_examples(pairs) -> list[WeakExample]:
    """Pairs labelled with the sum and nothing else."""
    return [
        WeakExample(images=pair.images, query=f"sum({pair.total})") for pair in pairs
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=15000)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    train, test = load_mnist()
    examples = build_examples(digit_pairs(train, args.pairs, seed=args.seed))
    test_pairs = digit_pairs(test, 1000, seed=99)
    test_examples = build_examples(test_pairs)

    kb = KnowledgeBase("sum").load_builtin("mnist_sum")
    slots = [
        NeuralPredicate.over_integers("digit", "d0", 10),
        NeuralPredicate.over_integers("digit", "d1", 10),
    ]
    loss = SemanticLoss(kb, slots)
    network = ConvNet(seed=args.seed)
    trainer = WeaklySupervisedTrainer(network, loss, learning_rate=args.learning_rate)

    # Held-out digit labels, used only to report progress.
    held_out_images = test.images[:2000]
    held_out_labels = test.labels[:2000]

    def validate() -> tuple[float, float]:
        digit_accuracy = network.accuracy(held_out_images, held_out_labels)
        stacked = np.stack([e.images for e in test_examples])
        left = network.predict(stacked[:, 0])
        right = network.predict(stacked[:, 1])
        totals = np.array([p.total for p in test_pairs])
        return digit_accuracy, float(((left + right) == totals).mean())

    print(
        f"training {network.parameter_count} parameters on {len(examples)} pairs\n"
        f"supervision: the sum only -- zero digit labels\n"
    )
    report = trainer.fit(
        examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        validation=validate,
    )

    network.save(args.out)
    metadata = {
        "experiment": "MNIST addition with sum-only supervision",
        "rule": "sum(S) :- digit(d0, A), digit(d1, B), S = A + B.",
        "parameters": network.parameter_count,
        "training_pairs": args.pairs,
        "digit_labels_used": 0,
        **report.to_dict(),
        "final_digit_accuracy_full_test_set": round(
            network.accuracy(test.images, test.labels), 4
        ),
        "symbolic_engine_calls_total": loss.engine_calls,
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("\n" + json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
