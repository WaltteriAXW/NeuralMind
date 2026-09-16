#!/usr/bin/env python3
"""Timing for the forward chainer on the cases that stress it.

Transitive closure over a chain is the classic worst case for naive
evaluation: every round re-derives every pair it already had, so the work
grows with the cube of the chain length while the answer grows with the square.
Semi-naive evaluation removes the extra factor, which shows up here as a flat
"scanned per atom" column -- the work tracks the size of the answer and
nothing else.

    python benchmarks/bench_inference.py
    python benchmarks/bench_inference.py --case chain-800 --repeats 3
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuralmind.core.parser import parse_program
from neuralmind.inference.forward import ForwardChainer

CHAIN = """
path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).
"""

CYCLE = """
reach(X, Y) :- link(X, Y).
reach(X, Z) :- link(X, Y), reach(Y, Z).
"""

JOIN = """
uncle(X, Z) :- sibling(X, Y), parent(Y, Z).
cousin(A, B) :- parent(P, A), parent(Q, B), sibling(P, Q).
"""


def chain(size: int) -> str:
    edges = "\n".join(f"edge(n{i}, n{i + 1})." for i in range(size))
    return edges + CHAIN


def cycle(size: int) -> str:
    links = "\n".join(f"link(n{i}, n{(i + 1) % size})." for i in range(size))
    return links + CYCLE


def wide_join(size: int) -> str:
    facts = []
    for i in range(size):
        facts.append(f"parent(p{i}, c{i}).")
        facts.append(f"sibling(p{i}, p{(i + 1) % size}).")
    return "\n".join(facts) + JOIN


CASES = {
    "chain-40": lambda: chain(40),
    "chain-80": lambda: chain(80),
    "chain-120": lambda: chain(120),
    "chain-200": lambda: chain(200),
    "chain-400": lambda: chain(400),
    "cycle-30": lambda: cycle(30),
    "cycle-60": lambda: cycle(60),
    "join-200": lambda: wide_join(200),
    "join-800": lambda: wide_join(800),
}


def run(name: str, source: str, repeats: int = 1) -> dict:
    program = parse_program(source, name)
    best = None
    model = None
    scanned = _ScanCounter()
    for attempt in range(repeats):
        with scanned.measuring(active=attempt == 0):
            started = time.perf_counter()
            model = ForwardChainer(program, max_justifications=1).run()
            elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    assert model is not None
    return {
        "case": name,
        "seconds": best,
        "atoms": len(model),
        "iterations": model.iterations,
        "scanned": scanned.total,
        "per_atom": scanned.total / max(len(model), 1),
    }


class _ScanCounter:
    """Counts atoms the matcher looks at, by wrapping the index lookup."""

    def __init__(self) -> None:
        self.total = 0

    def measuring(self, active: bool = True):
        from contextlib import contextmanager

        from neuralmind.inference.model import AtomIndex

        @contextmanager
        def scope():
            if not active:
                yield
                return
            original = AtomIndex.candidates
            self.total = 0

            def counting(index_self, pattern):
                found = original(index_self, pattern)
                self.total += len(found)
                return found

            AtomIndex.candidates = counting
            try:
                yield
            finally:
                AtomIndex.candidates = original

        return scope()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    wanted = args.case or list(CASES)
    header = (
        f"{'case':<12} {'seconds':>9} {'atoms':>8} {'rounds':>7} "
        f"{'scanned':>10} {'per atom':>9}"
    )
    print(header)
    print("-" * len(header))
    results = []
    for name in wanted:
        result = run(name, CASES[name](), args.repeats)
        results.append(result)
        print(
            f"{result['case']:<12} {result['seconds']:>9.3f} "
            f"{result['atoms']:>8} {result['iterations']:>7} "
            f"{result['scanned']:>10} {result['per_atom']:>9.1f}"
        )
    total = sum(r["seconds"] for r in results)
    print("-" * len(header))
    print(f"{'total':<12} {total:>9.3f}")
    print(
        "\n'per atom' is the number of atoms examined for each atom derived. "
        "Semi-naive\nevaluation keeps it flat as the problem grows; naive "
        "iteration does not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
