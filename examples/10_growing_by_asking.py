#!/usr/bin/env python3
"""A mind that notices what it does not know, and asks.

Phase One could learn a rule when you handed it examples and named the target.
This is the part that notices the gap on its own, works out which question
would settle it, and then *does not use the answer* until someone confirms it.

The teacher here is an oracle built from the true definition, which is what
turns "how many questions did that take?" into a number rather than an
impression. In a session the teacher is you.
"""

from neuralmind import ReasoningEngine, parse_atom
from neuralmind.growth import Grower
from neuralmind.kernel import Kernel

FAMILY = """
parent(maria, juho). parent(maria, liisa). parent(matti, juho). parent(matti, liisa).
parent(juho, aino).  parent(juho, eero).  parent(liisa, sanna).
parent(aino, taavi). parent(sanna, venla).
male(matti). male(juho). male(eero). male(taavi).
female(maria). female(liisa). female(aino). female(sanna). female(venla).
"""
TRUTH = "ancestor(X, Y) :- parent(X, Y). ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)."
oracle = ReasoningEngine(FAMILY + TRUTH).model

kernel = Kernel("family")
kernel.load_core(FAMILY)
# Canaries go on what we are *not* trying to learn. A canary on the target
# would refuse the lesson, since learning is meant to change that answer.
kernel.watch(["parent(maria, juho)", "male(matti)", "female(venla)"])
grower = Grower(kernel)

# -- notice ----------------------------------------------------------------

answer = grower.ask("ancestor(maria, taavi)")
print(f"ancestor(maria, taavi)? {answer.status}")
print("gaps noticed:", grower.questions(), "\n")

# -- ask -------------------------------------------------------------------

def teacher(atom):
    verdict = oracle.holds(atom)
    print(f"    Q: is {atom} true?   A: {'yes' if verdict else 'no'}")
    return verdict

print("working it out by asking:")
session = grower.grow(
    "ancestor/2", teacher, positive=[parse_atom("ancestor(maria, juho)")]
)
print(f"\nsettled after {session.questions} question(s):")
for rule in session.rules:
    print(f"    {rule}")

# -- but not believed ------------------------------------------------------

print("\n" + "-" * 70)
print("The rule is a *proposal*. Nothing has changed yet:")
print(f"    ancestor(maria, taavi)? {kernel.ask('ancestor(maria, taavi)').status}")
print(f"    memory: {grower.memory.summary()['by_state']}")

print("\nConfirming puts it through the safety kernel:")
for rule in session.rules:
    record, outcome = grower.confirm(rule, by="a reviewer")
    print(f"    {outcome}  {record.body}")

print(f"\n    ancestor(maria, taavi)? {kernel.ask('ancestor(maria, taavi)').status}")
proof = kernel.ask("ancestor(maria, taavi)").proof
if proof is not None:
    print()
    print(proof)

print("\n" + "-" * 70)
print(kernel.self_report())
