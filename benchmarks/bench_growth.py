#!/usr/bin/env python3
"""How many questions it takes to learn a relation, and how many it saves.

P2.3's first two bars, run end to end: every family relation learned by asking,
and active selection measured against random selection on the same tasks.

    python benchmarks/bench_growth.py

Random hits the question cap on every target, so its total is a floor on what
it costs rather than the cost. That works against the comparison, which is the
right direction for it to work in.
"""
from neuralmind import KnowledgeBase, ReasoningEngine
from neuralmind.growth.loop import GrowthLoop

FACTS = """
parent(maria, juho). parent(maria, liisa). parent(matti, juho). parent(matti, liisa).
parent(juho, aino).  parent(juho, eero).  parent(liisa, sanna).
parent(aino, taavi). parent(sanna, venla).
male(matti). male(juho). male(eero). male(taavi).
female(maria). female(liisa). female(aino). female(sanna). female(venla).
"""
SIBLING = "sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y."
TARGETS = {
    "grandparent/2": ("", "grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."),
    "sibling/2": ("", SIBLING),
    "aunt/2": (SIBLING, "aunt(X, Y) :- parent(P, Y), sibling(X, P), female(X)."),
    "ancestor/2": (
        "",
        "ancestor(X, Y) :- parent(X, Y). ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).",
    ),
}

totals = {"active": 0, "random": 0}
print(f"{'target':14} {'strategy':9} {'asked':>6} {'settled':>8} {'correct':>8}  definition")
for target, (support, truth) in TARGETS.items():
    base = FACTS + support
    oracle = ReasoningEngine(base + truth).model
    name = target.split("/")[0]
    gold = {a for a in oracle.atoms if a.predicate == name}
    seed = sorted(gold, key=str)[:1]
    for strategy in ("active", "random"):
        kb = KnowledgeBase().add_rules(base)
        session = GrowthLoop(kb, max_questions=80).learn(
            target, lambda a: oracle.holds(a), positive=seed, strategy=strategy
        )
        totals[strategy] += session.questions
        got = set()
        if session.rules:
            learned = ReasoningEngine(
                base + "\n".join(str(r) for r in session.rules)
            ).model
            got = {a for a in learned.atoms if a.predicate == name}
        rules = " | ".join(str(r) for r in session.rules)
        print(
            f"{target:14} {strategy:9} {session.questions:6} "
            f"{str(session.settled):>8} {str(got == gold):>8}  {rules[:70]}"
        )
ratio = totals["active"] / totals["random"] if totals["random"] else 0
print(f"\ntotal: active {totals['active']}, random {totals['random']}  "
      f"-> active needs {ratio:.0%} of random's questions")
