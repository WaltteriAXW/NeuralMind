#!/usr/bin/env python3
"""P2.2's first bar: 10,000 random, malformed and contradictory inputs.

The claim being tested is narrow and absolute: **the core is never modified
and no canary ever changes its answer**, whatever is thrown at the kernel.

Eight kinds of input, in roughly equal measure: junk bytes, unbalanced syntax,
unsafe rules, recursive negation, contradictions with the core, over-general
rules that would change a canary, unbounded recursion, and harmless-but-novel
rules. The last of those is why the run takes an hour -- they are accepted, so
the theory grows to ~1,600 rules and every later check runs against all of
them. That is the fuzz being adversarial about volume as well as content.

"The core is read-only" is checked as what it is: the shipped text, compared
byte for byte after every input. Checking it by asking questions instead would
fail for reasons that have nothing to do with safety, which is how the first
version of this harness tested the wrong thing.

    python benchmarks/fuzz_kernel.py

Exits non-zero on any breach, and names the input that caused it.
"""
import random, string, sys, time
from neuralmind.kernel import CONFIRMED, CORE, Kernel

random.seed(7)

CORE_RULES = """
mammal(X) :- cat(X).
mammal(X) :- dog(X).
warm_blooded(X) :- mammal(X).
:- cat(X), dog(X).
#open flies/1.
"""

def junk() -> str:
    kind = random.randrange(8)
    word = lambda: "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 6)))
    var = lambda: random.choice("XYZW")
    if kind == 0:   # random bytes
        return "".join(random.choices(string.printable, k=random.randint(1, 40)))
    if kind == 1:   # unbalanced syntax
        return f"{word()}({var()} :- {word()}({var()})."
    if kind == 2:   # unsafe rule
        return f"{word()}({var()}, {var()}) :- {word()}({var()})."
    if kind == 3:   # recursive negation
        a, b = word(), word()
        return f"{a}(X) :- cat(X), not {b}(X). {b}(X) :- cat(X), not {a}(X)."
    if kind == 4:   # contradicts the core
        return random.choice([
            "dog(X) :- cat(X).",
            "-mammal(X) :- cat(X).",
            "-warm_blooded(X) :- mammal(X).",
        ])
    if kind == 5:   # over-general: changes a canary
        return f"{random.choice(['flies','mammal','warm_blooded'])}(X) :- {word()}(X)."
    if kind == 6:   # runaway recursion
        return "big(X) :- cat(X). big(f(X)) :- big(X)."
    return f"{word()}({var()}) :- cat({var()})."   # harmless

def main() -> int:
    kernel = Kernel("fuzz")
    kernel.load_core(CORE_RULES)
    kernel.layers.add_fact("cat(bob)", CONFIRMED)
    kernel.layers.add_fact("dog(rex)", CONFIRMED)
    kernel.watch([
        "mammal(bob)", "warm_blooded(bob)", "mammal(rex)", "flies(bob)",
        "cat(bob)", "dog(bob)", "warm_blooded(rex)", "fish(bob)",
    ])
    baseline = [(str(c.goal), c.status) for c in kernel.canaries]
    # "The core is read-only at runtime" is a claim about the core itself, not
    # about answers drawn from it plus other layers. So it is checked as what
    # it is: the shipped text, unchanged.
    core_text = kernel.layers.core.to_asp()

    accepted = rejected = 0
    breaches = []
    started = time.perf_counter()
    for index in range(10_000):
        outcome = kernel.learn(junk(), CONFIRMED, f"fuzz{index}")
        accepted += bool(outcome)
        rejected += not outcome
        failures = kernel.canaries.check(kernel.layers, kernel.mode_layers)
        if failures:
            breaches.append((index, [f.describe() for f in failures]))
            break
        if kernel.layers.core.to_asp() != core_text:
            breaches.append((index, ["the core layer was modified"]))
            break
        if (index + 1) % 500 == 0:
            print(
                f"  {index + 1:6}/10000  accepted={accepted:5}  "
                f"rules={len(kernel.layers['confirmed'].knowledge.rules.derivation_rules):5}  "
                f"{time.perf_counter() - started:6.0f}s",
                flush=True,
            )
    elapsed = time.perf_counter() - started

    print(f"inputs      : 10,000 in {elapsed:.1f}s")
    print(f"accepted    : {accepted}")
    print(f"rejected    : {rejected}")
    print(f"mode        : {kernel.watchdog.mode}")
    print(f"quarantined : {len(kernel.quarantine)}")
    print(f"canaries    : {'ALL PASS' if not breaches else 'BREACH'}")
    for index, detail in breaches:
        print(f"  breach at input {index}: {detail}")
    after = [(str(c.goal), c.status) for c in kernel.canaries]
    print(f"baseline unchanged: {baseline == after}")
    by_check = kernel.firewall.to_dict()["by_check"]
    print(f"firewall rejections by check: {by_check}")
    return 1 if breaches else 0

if __name__ == "__main__":
    raise SystemExit(main())
