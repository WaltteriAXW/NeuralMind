#!/usr/bin/env python3
"""Fetch MNIST into ``data/mnist`` so the Phase 1 and Phase 4 demos can run.

The dataset is not committed (12 MB), but the trained weights are, so this is
only needed to re-run the demos and milestone tests over real images -- not to
use the perceptor.

    python scripts/download_mnist.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralmind.datasets.mnist import MNIST_URLS, default_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=None, help="target directory")
    args = parser.parse_args()

    destination = args.dest or default_root()
    destination.mkdir(parents=True, exist_ok=True)

    for name, url in MNIST_URLS.items():
        target = destination / name
        if target.exists():
            print(f"have {name}")
            continue
        print(f"fetching {name} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            print(f"failed: {exc}")
            print(f"  download it by hand from {url} into {destination}")
            return 1
        print(f"{target.stat().st_size / 1024:.0f} KB")

    print(f"\nMNIST is ready in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
