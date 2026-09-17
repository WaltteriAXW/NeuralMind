#!/usr/bin/env python3
"""Fetch the two intent datasets the context layer is trained and tested on.

Neither is redistributed here; both come from their own repositories.

* **CLINC150** -- 150 intents across 10 domains plus a deliberate
  out-of-scope set (Larson et al. 2019, CC BY 3.0).
* **BANKING77** -- 77 fine-grained banking intents (PolyAI, CC BY 4.0).

    python scripts/download_contexts.py
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "contexts"

SOURCES = {
    "clinc150.json": (
        "https://raw.githubusercontent.com/clinc/oos-eval/master/data/data_full.json",
        "CLINC150, CC BY 3.0",
    ),
    "banking77-test.csv": (
        "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
        "master/banking_data/test.csv",
        "BANKING77, CC BY 4.0",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEST)
    args = parser.parse_args()
    args.dest.mkdir(parents=True, exist_ok=True)

    for name, (url, licence) in SOURCES.items():
        target = args.dest / name
        if target.exists():
            print(f"  {name:22} already present")
            continue
        print(f"  {name:22} fetching ({licence})")
        try:
            urllib.request.urlretrieve(url, target)
        except Exception as exc:
            print(f"    failed: {exc}", file=sys.stderr)
            print(f"    fetch it by hand from {url}", file=sys.stderr)
            return 1
        print(f"    {target.stat().st_size / 1e6:.1f} MB")
    print(f"\nready in {args.dest}")
    print("next: python scripts/train_intents.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
