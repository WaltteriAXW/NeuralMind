#!/usr/bin/env python3
"""Train the digit perceptor used by the Phase 1 demo.

The checkpoint this writes is committed to the repository so the demo and the
tests run without a training step. Re-run it to reproduce the weights:

    python scripts/train_mnist.py --train-size 30000 --epochs 10

Weights trained here are ours outright -- as the blueprint's licensing section
notes, a dataset's terms govern redistributing the dataset, not the weights
someone trains on it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralmind.datasets.mnist import load_mnist
from neuralmind.perception.nn import ConvNet

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "neuralmind/perception/weights/mnist_cnn.npz"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=30000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    train, test = load_mnist()
    subset = train.subset(args.train_size, seed=args.seed)
    net = ConvNet(seed=args.seed)
    print(f"training {net.parameter_count} parameters on {len(subset)} images")
    report = net.fit(
        subset.images,
        subset.labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation=(test.images, test.labels),
        seed=args.seed,
    )
    net.save(args.out)

    metadata = {
        "architecture": "conv(1->16,3) relu pool conv(16->32,3) relu pool dense(800->10)",
        "parameters": net.parameter_count,
        "train_size": args.train_size,
        "dataset": "MNIST (train split)",
        **report.to_dict(),
        "test_accuracy": round(net.accuracy(test.images, test.labels), 4),
    }
    args.out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    print(f"saved {args.out} ({args.out.stat().st_size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
