"""MNIST loading, with no dependency beyond NumPy.

The files are the original IDX archives. Point ``NEURALMIND_DATA`` at a
directory containing them, or pass ``root=``; :func:`load_mnist` raises a
message naming the four files and where to get them if they are missing,
rather than silently downloading anything.
"""

from __future__ import annotations

import gzip
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "load_mnist",
    "MnistSplit",
    "DigitPair",
    "digit_pairs",
    "mnist_available",
    "default_root",
    "MNIST_URLS",
]

MNIST_URLS = {
    "train-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


@dataclass
class MnistSplit:
    """Images scaled to [0, 1] plus integer labels."""

    images: np.ndarray
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def subset(self, count: int, seed: int = 0) -> "MnistSplit":
        rng = np.random.default_rng(seed)
        index = rng.permutation(len(self.labels))[:count]
        return MnistSplit(self.images[index], self.labels[index])


def default_root() -> Path:
    """Where MNIST is expected to live."""
    override = os.environ.get("NEURALMIND_DATA")
    if override:
        return Path(override) / "mnist"
    return Path(__file__).resolve().parents[2] / "data" / "mnist"


def mnist_available(root: Optional[Path] = None) -> bool:
    directory = Path(root) if root else default_root()
    return all((directory / name).exists() for name in MNIST_URLS)


def load_mnist(root: Optional[Path] = None) -> tuple[MnistSplit, MnistSplit]:
    """Return ``(train, test)``."""
    directory = Path(root) if root else default_root()
    missing = [name for name in MNIST_URLS if not (directory / name).exists()]
    if missing:
        listing = "\n  ".join(f"{name}  <- {MNIST_URLS[name]}" for name in missing)
        raise FileNotFoundError(
            f"MNIST files missing from {directory}:\n  {listing}\n"
            "Download them there, or set NEURALMIND_DATA to a directory that has them."
        )
    train = MnistSplit(
        _read_images(directory / "train-images-idx3-ubyte.gz"),
        _read_labels(directory / "train-labels-idx1-ubyte.gz"),
    )
    test = MnistSplit(
        _read_images(directory / "t10k-images-idx3-ubyte.gz"),
        _read_labels(directory / "t10k-labels-idx1-ubyte.gz"),
    )
    return train, test


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count, rows, columns = struct.unpack(">IIII", handle.read(16))
        if magic != 2051:
            raise ValueError(f"{path} is not an IDX image file (magic {magic})")
        buffer = handle.read(count * rows * columns)
    array = np.frombuffer(buffer, dtype=np.uint8).reshape(count, rows, columns)
    return (array.astype(np.float32) / 255.0)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        magic, count = struct.unpack(">II", handle.read(8))
        if magic != 2049:
            raise ValueError(f"{path} is not an IDX label file (magic {magic})")
        buffer = handle.read(count)
    return np.frombuffer(buffer, dtype=np.uint8).astype(np.int64)


@dataclass
class DigitPair:
    """Two digit images and the sum the pipeline should work out."""

    left: np.ndarray
    right: np.ndarray
    left_label: int
    right_label: int

    @property
    def total(self) -> int:
        return int(self.left_label) + int(self.right_label)

    @property
    def images(self) -> np.ndarray:
        return np.stack([self.left, self.right])


def digit_pairs(split: MnistSplit, count: int = 100, seed: int = 0) -> list[DigitPair]:
    """Sample image pairs for the digit-addition task (blueprint Phase 1)."""
    rng = np.random.default_rng(seed)
    left_index = rng.integers(0, len(split), size=count)
    right_index = rng.integers(0, len(split), size=count)
    return [
        DigitPair(
            left=split.images[i],
            right=split.images[j],
            left_label=int(split.labels[i]),
            right_label=int(split.labels[j]),
        )
        for i, j in zip(left_index, right_index)
    ]
