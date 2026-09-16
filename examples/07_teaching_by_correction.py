#!/usr/bin/env python3
"""Building a domain from nothing, by stating facts and correcting mistakes.

Writing rules is the hard part of this architecture -- the source report calls
it the knowledge acquisition bottleneck and names it as the main cost. This is
the other way round: state what is true, let the system generalise, and correct
it when it overreaches. Each accepted rule becomes vocabulary for the next one,
so a domain is built up rather than written out.

Nothing here is specific to family relations. The learner takes its search
space from whatever the knowledge base already mentions, so the same session
works for an access policy, a chemical compatibility table, or a maintenance
schedule.
"""

from neuralmind.induction import Examples, LanguageBias, RuleLearner
from neuralmind.induction.repair import Corrector
from neuralmind.knowledge.base import KnowledgeBase
from neuralmind.output.render import render_proof

kb = KnowledgeBase("domain")

# --- 1. what is true -------------------------------------------------------

FACTS = [
    "parent(maria, juho)", "parent(maria, liisa)",
    "parent(juho, aino)", "parent(liisa, onni)",
    "female(maria)", "female(liisa)", "female(aino)",
    "male(juho)", "male(onni)",
]
kb.add_facts(FACTS)
people = ["maria", "juho", "liisa", "aino", "onni"]


def teach(target: str, examples: list[str], **options) -> None:
    """Generalise from examples, using whatever vocabulary already exists."""
    bias = LanguageBias.from_knowledge(
        kb, target, allow_recursion=False, allow_comparison=True, **options
    )
    hypothesis = RuleLearner(kb, bias, Examples.closed_world(examples, people)).learn()
    for rule in hypothesis.rules:
        kb.rules.add(rule)
    verdict = "learned" if hypothesis.correct else "INCOMPLETE"
    print(f"  {verdict:10s} {'; '.join(str(r) for r in hypothesis.rules) or '(nothing)'}")
    if hypothesis.underdetermined:
        print("             ! the examples do not single out one rule")


print("teaching, each rule built on the last:")
teach("sibling/2", ["sibling(juho, liisa)", "sibling(liisa, juho)"],
      max_variables=4, max_body=4)
teach("aunt/2", ["aunt(liisa, aino)"], max_variables=4, max_body=3)

# --- 2. where it overreached ----------------------------------------------

print("\nthe aunt rule came from a single example, and it shows: it picked up")
print("male(C), because in that one case the child's parent happened to be a")
print("man. Nothing in the data said otherwise -- which is why the learner")
print("flagged it as underdetermined rather than presenting it as settled.\n")

# Sanna is Liisa's sister, and Liisa -- a woman -- is Onni's parent. So Sanna
# is Onni's aunt, and the over-specific rule cannot see it.
kb.add_facts(["parent(maria, sanna)", "female(sanna)"])

model = kb.engine().model
print("  it says:", ", ".join(sorted(str(a) for a in model.by_predicate("aunt"))))
print("  but Sanna is Onni's aunt too, through Liisa.\n")
print("  > :expect aunt(sanna, onni)")
for repair in Corrector(kb).expect("aunt(sanna, onni)")[:2]:
    print("    " + repair.describe().replace("\n", "\n    "))

# --- 3. and where it lets too much through --------------------------------

print()
print("  Look at the second proposal: it drops male(C) and fixes the complaint,")
print("  and it reports that it would also derive aunt(juho, onni) -- which is")
print("  wrong, because Juho is an uncle. The rule needs female(A), and no")
print("  example given so far distinguishes that. The system is not hiding the")
print("  problem; it is showing exactly what the change would cost, which is")
print("  the only thing that makes accepting one safe.")

print("\nthe same machinery in the other direction. A deliberately loose rule:")
kb2 = KnowledgeBase("access")
kb2.add_facts(["staff(ana)", "staff(ben)", "contractor(ben)"])
kb2.add_rules("may_enter(P) :- staff(P).")
print("  may_enter(P) :- staff(P).")
print("  lets in:", ", ".join(sorted(str(a) for a in kb2.engine().model.by_predicate("may_enter"))))
print("\n  > :wrong may_enter(ben)")
best = Corrector(kb2).reject("may_enter(ben)")[0]
print("    " + best.describe().replace("\n", "\n    "))
best.apply(kb2)
print("\n  after accepting:",
      ", ".join(sorted(str(a) for a in kb2.engine().model.by_predicate("may_enter"))))

print("\n" + "=" * 68)
print("Every rule here was learned or repaired, not written, and each one")
print("still proves its conclusions like any other:\n")
print(render_proof(kb.engine().ask("aunt(liisa, aino)").proof))
