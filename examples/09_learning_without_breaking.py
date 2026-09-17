#!/usr/bin/env python3
"""Trying things, and always getting back.

A mind that learns will be wrong sometimes. It will induce a rule from two
examples, accept a pack that contradicts something it knew, read a sentence
badly. None of that is avoidable. What is avoidable is letting any of it reach
what the mind came with.

The safety kernel is eight mechanisms around that one idea. This shows four of
them doing their job on rules that are each wrong in a different way.
"""

from neuralmind.kernel import CONFIRMED, HIGH, L1, L2, L3, SESSION, Kernel

kernel = Kernel("workshop", tenant="acme", granted=L3)

# The core is shipped knowledge. It is read-only for the rest of this run.
kernel.load_core("""
mammal(X) :- cat(X).
mammal(X) :- dog(X).
warm_blooded(X) :- mammal(X).
:- cat(X), dog(X).
#open flies/1.
""")
kernel.layers.add_fact("cat(bob)", CONFIRMED)

# Canaries are captured from what the mind already answers, not written by
# hand. Anything that changes one of these answers is caught from now on.
kernel.watch(["mammal(bob)", "warm_blooded(bob)", "flies(bob)", "dog(bob)"])
print("watching:", ", ".join(str(c) for c in kernel.canaries), "\n")

# -- four ways for a rule to be wrong --------------------------------------

candidates = [
    ("a reasonable rule",   "furry(X) :- cat(X)."),
    ("unsafe",              "blocked(E, R) :- suspended(E)."),
    ("contradicts the core", "dog(X) :- cat(X)."),
    ("over-general",        "flies(X) :- cat(X)."),
    ("another good one",    "purrs(X) :- cat(X)."),
]
for label, rule in candidates:
    outcome = kernel.learn(rule)
    mark = "accepted" if outcome else "refused "
    print(f"  {mark}  {label:20} {rule}")
    if not outcome:
        print(f"            └─ {outcome.reason}")

print(f"\nquarantined, not deleted:")
for entry in kernel.quarantine:
    print(f"  {entry}")

print(f"\nstill answering: mammal(bob) = {kernel.ask('mammal(bob)').status}")
print(f"canaries healthy: {kernel.canaries.healthy(kernel.layers)}")

# -- personal data stays where it came from --------------------------------

print("\n" + "-" * 70)
kernel.privacy.declare("salary", "pay")
kernel.observe(["salary(bob, 50000)", 'email(bob, "bob@example.io")', "cat(felix)"])
for record in kernel.layers.facts(("core", "confirmed", "tenant", "session")):
    layer = kernel.layers.layer_of(record.atom)
    tagged = " [personal]" if kernel.privacy.is_personal(record.atom) else ""
    print(f"  {str(record.atom):34} in {layer}{tagged}")

from neuralmind import parse_atom
from neuralmind.kernel import ReadOnlyLayer

try:
    kernel.layers.promote(parse_atom("salary(bob, 50000)"), SESSION, CONFIRMED)
except ReadOnlyLayer as refusal:
    print(f"\n  promotion refused: {refusal}")

# -- autonomy: caution rises on a guess, freedom only on a grant -----------

print("\n" + "-" * 70)
print(f"  {kernel.autonomy.describe()}")
print(f"  {kernel.decide('transfer(500)', needs=L2)}")

kernel.autonomy.raise_stakes(HIGH, "money moves and personal data is present")
print(f"\n  {kernel.autonomy.describe()}")
print(f"  {kernel.decide('transfer(500)', needs=L2)}")
print(f"  {kernel.decide('transfer(500)', needs=L2, confirmed=True)}")

print("\n" + "-" * 70)
print(kernel.self_report())
