#!/usr/bin/env python3
"""Four kinds of reasoning, one proof.

A real question rarely needs only deduction. "Is this frame signed off?" needs
a rule, an inequality, two unit conversions and a path through an assembly
graph -- and the answer is only trustworthy if you can see all of it at once.

The specialists meet on a blackboard rather than in a pipeline, because none of
them is downstream of the others. Everything posted is a ground atom with a
justification, so a result from the constraint solver is indistinguishable, to
the logic engine, from a fact that was given -- and the proof tree spans all
four because each brought its own node.

Needs the optional specialists::

    pip install neuralmind[specialists]
"""

from neuralmind import Mind
from neuralmind.workspace.scenarios import WORKSHOP_FACTS, WORKSHOP_RULES
from neuralmind.workspace.specialists import installed

print("specialists:", ", ".join(
    f"{name} {'yes' if ready else 'NOT INSTALLED'}" for name, ready in installed().items()
))
print()

mind = Mind(budget_ms=500)
mind.add_rules(WORKSHOP_RULES)
mind.add_rules(WORKSHOP_FACTS)

# -- one question, four specialists ---------------------------------------

conclusion = mind.solve("signed_off")
print(f"signed_off? {conclusion.status}\n")
print(conclusion)
print(f"\n  consulted {', '.join(dict.fromkeys(conclusion.consulted))} "
      f"in {conclusion.rounds} rounds, {conclusion.elapsed_ms:.0f}ms\n")

# -- what each specialist adds over plain Datalog --------------------------

print("-" * 72)
print("A relation run backwards. Datalog needs one rule per direction;")
print("the solver needs the relation stated once.\n")
print(mind.solve("value(labour_cost, V)"))

print("\n" + "-" * 72)
print("Units are what make two numbers comparable at all.\n")
print(mind.solve("in_unit(beam_a_load, kn, V)"))

print("\n" + "-" * 72)
print("The *shortest* path -- a minimum over derivations, which a least")
print("model cannot express.\n")
print(mind.solve("distance(cut, ship, N)"))

# -- and what it says when it cannot answer --------------------------------

print("\n" + "-" * 72)
print("An honest 'no': beam B is over its rating, and the reason names")
print("the literal that failed rather than shrugging.\n")
beam_b = mind.solve("safe(beam_b)")
print(f"  safe(beam_b)? {beam_b.status} — {beam_b.reason}")

print("\n" + "-" * 72)
print(mind.self_report())
