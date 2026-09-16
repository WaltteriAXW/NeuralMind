#!/usr/bin/env python3
"""Fetch the ProofWriter corpus test splits into ``data/proofwriter``.

The corpus is the Allen Institute for AI's, distributed from their public
bucket; it is not redistributed in this repository. Only the CWA test splits
are kept, which is what the evaluation uses -- the full archive is 214 MB and
most of it is training data this project has no use for.

    python scripts/download_proofwriter.py

Reference: Tafjord, Dalvi & Clark, "ProofWriter: Generating Implications,
Proofs, and Abductive Statements over Natural Language" (2020), an updated
release of the RuleTaker datasets.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralmind.datasets.proofwriter_corpus import SPLITS, default_root

URL = (
    "https://aristo-data-public.s3.amazonaws.com/proofwriter/"
    "proofwriter-dataset-V2020.12.3.zip"
)
ARCHIVE_ROOT = "proofwriter-dataset-V2020.12.3/CWA"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--keep-archive", type=Path, default=None)
    args = parser.parse_args()

    destination = args.dest or default_root()
    destination.mkdir(parents=True, exist_ok=True)

    if all((destination / f"{s}-test.jsonl").exists() for s in SPLITS):
        print(f"already present in {destination}")
        return 0

    print(f"downloading {URL}\n  (214 MB; only the CWA test splits are kept)")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
        archive = Path(args.keep_archive or handle.name)
    try:
        urllib.request.urlretrieve(URL, archive)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print(f"  fetch it by hand from {URL}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(archive) as bundle:
        for split in SPLITS:
            member = f"{ARCHIVE_ROOT}/{split}/meta-test.jsonl"
            try:
                data = bundle.read(member)
            except KeyError:
                print(f"  missing from archive: {split}", file=sys.stderr)
                continue
            target = destination / f"{split}-test.jsonl"
            target.write_bytes(data)
            print(f"  {split:20s} {len(data) / 1e6:7.2f} MB")

    if args.keep_archive is None:
        archive.unlink(missing_ok=True)
    print(f"\nProofWriter test splits are ready in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
