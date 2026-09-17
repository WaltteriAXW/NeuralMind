#!/usr/bin/env python3
"""Train the intent classifier the context layer uses to read an utterance.

A single sentence carries almost no structure, so shape alone sees "free text,
a question, first person" and stops. This is what lets the mind tell a banking
chat from a general assistant -- and, more importantly, tell both from an
utterance that belongs to neither.

Two real datasets, both fetched from their own repositories:

* **BANKING77** (PolyAI, CC BY 4.0) -- 77 fine-grained banking intents.
* **CLINC150** (CC BY 3.0) -- 150 intents across 10 domains, *plus* a
  deliberate out-of-scope set. That last part is the reason it is here: a
  service assistant meets inputs belonging to no intent it knows all day, and
  "not something I handle" is the answer that matters.

    python scripts/download_contexts.py     # fetch the data
    python scripts/train_intents.py         # train and save

The out-of-scope threshold is *calibrated*, not chosen: picked on held-out
data to keep a target share of genuinely in-scope utterances. Both halves
matter -- a threshold tuned only on in-scope data rejects nothing, one tuned
only on out-of-scope data rejects everything.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralmind.self.intent import IntentClassifier, numpy_available

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "contexts"
WEIGHTS = ROOT / "neuralmind" / "self" / "weights" / "intents.npz"


def load() -> tuple[list, dict, dict]:
    clinc = json.loads((DATA / "clinc150.json").read_text(encoding="utf-8"))
    with (DATA / "banking77-test.csv").open(encoding="utf-8") as handle:
        banking = list(csv.DictReader(handle))
    random.Random(7).shuffle(banking)
    split = int(len(banking) * 0.75)

    families = {intent: "assistant" for _text, intent in clinc["train"]}
    families.update({row["category"]: "banking" for row in banking})

    train = [(text, intent) for text, intent in clinc["train"]]
    train += [(row["text"], row["category"]) for row in banking[:split]]
    held_out = [(text, intent) for text, intent in clinc["test"][:800]]
    held_out += [(row["text"], row["category"]) for row in banking[split:]]
    return train, {"held_out": held_out, "clinc": clinc}, families


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recall", type=float, default=0.90,
                        help="share of in-scope utterances to keep (default: 0.90)")
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--out", type=Path, default=WEIGHTS)
    args = parser.parse_args()

    if not numpy_available():
        print("this needs NumPy: pip install neuralmind[numeric]", file=sys.stderr)
        return 1
    if not (DATA / "clinc150.json").exists():
        print("data missing: run python scripts/download_contexts.py", file=sys.stderr)
        return 1

    train, sets, families = load()
    print(f"training on {len(train):,} utterances")
    model = IntentClassifier().fit(train, families, min_df=args.min_df)
    clinc = sets["clinc"]
    model.calibrate(
        [text for text, _ in clinc["val"][:500]],
        [text for text, _ in clinc["oos_val"][:500]],
        target_recall=args.recall,
    )
    print(f"  {model.summary()}")

    held_out = sets["held_out"]
    family_right = sum(
        1 for text, intent in held_out
        if model.predict(text).family == families[intent]
    )
    intent_right = sum(
        1 for text, intent in held_out if model.predict(text).intent == intent
    )
    out_of_scope = [text for text, _ in clinc["oos_test"]]
    rejected = sum(1 for text in out_of_scope if not model.predict(text).in_scope)

    print(f"  held-out family   {family_right / len(held_out):6.1%}  ({len(held_out):,})")
    print(f"  held-out intent   {intent_right / len(held_out):6.1%}")
    print(f"  out-of-scope kept out {rejected / len(out_of_scope):6.1%}  ({len(out_of_scope):,})")

    model.save(args.out)
    size = args.out.stat().st_size + args.out.with_suffix(".json").stat().st_size
    print(f"\nsaved to {args.out.relative_to(ROOT)} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
