#!/usr/bin/env python3
"""Start here: reasoning with no neural component at all.

The blueprint's advice is to get the logic right before adding perception,
because debugging a wrong rule is much easier without a noisy input layer.
"""

from neuralmind import KnowledgeBase

kb = KnowledgeBase("family").load_builtin("family")
kb.add_facts(
    [
        "parent(maria, juho)", "parent(maria, liisa)", "parent(juho, aino)",
        "female(maria)", "male(juho)", "female(liisa)", "female(aino)",
        "born(maria, 1948)", "born(juho, 1972)", "born(aino, 1999)",
    ]
)

engine = kb.engine()

print("Who are Maria's descendants?")
print(engine.ask("ancestor(maria, X)"))

print("\nWhy is Liisa Aino's aunt?")
print(engine.ask("aunt(liisa, aino)").proof)

print("\nWhy is Juho *not* Aino's uncle?")
print(engine.ask("uncle(juho, aino)").diagnosis)

print("\nIs the knowledge base self-consistent?")
print("violations:", engine.violations or "none")

print("\nDoes an independent solver agree?")
print(engine.cross_check().report())
