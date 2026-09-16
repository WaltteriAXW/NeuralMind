#!/usr/bin/env python3
"""Fetch the ProofWriter corpus test splits into ``data/proofwriter``.

The corpus is the Allen Institute for AI's, distributed from their public
bucket; it is not redistributed in this repository. Only the test splits are
kept -- the full archive is 214 MB and most of it is training data this project
has no use for.

Both readings are fetched. The CWA splits answer True or False; the OWA splits
add Unknown, which is the whole point of them: about 46% of their questions are
ones the theory does not settle either way, and a system that cannot say so
scores them all wrong.

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

from neuralmind.datasets.proofwriter_corpus import SPLITS, WORLDS, default_root

URL = (
    "https://aristo-data-public.s3.amazonaws.com/proofwriter/"
    "proofwriter-dataset-V2020.12.3.zip"
)
ARCHIVE_ROOT = "proofwriter-dataset-V2020.12.3"


def _filename(world: str, split: str) -> str:
    return f"{'' if world == 'cwa' else world + '-'}{split}-test.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=None)
    parser.add_argument("--keep-archive", type=Path, default=None)
    args = parser.parse_args()

    destination = args.dest or default_root()
    destination.mkdir(parents=True, exist_ok=True)

    wanted = [
        (world, split, destination / _filename(world, split))
        for world in WORLDS
        for split in SPLITS
    ]
    if all(path.exists() for _, _, path in wanted):
        print(f"already present in {destination}")
        return 0

    print(f"downloading {URL}\n  (214 MB; only the CWA and OWA test splits are kept)")
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as handle:
        archive = Path(args.keep_archive or handle.name)
    try:
        urllib.request.urlretrieve(URL, archive)
    except Exception as exc:
        print(f"download failed: {exc}", file=sys.stderr)
        print(f"  fetch it by hand from {URL}", file=sys.stderr)
        return 1

    with zipfile.ZipFile(archive) as bundle:
        for world, split, target in wanted:
            member = f"{ARCHIVE_ROOT}/{world.upper()}/{split}/meta-test.jsonl"
            try:
                data = bundle.read(member)
            except KeyError:
                print(f"  missing from archive: {world}/{split}", file=sys.stderr)
                continue
            target.write_bytes(data)
            print(f"  {world}/{split:20s} {len(data) / 1e6:7.2f} MB")

    if args.keep_archive is None:
        archive.unlink(missing_ok=True)
    print(f"\nProofWriter test splits are ready in {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
