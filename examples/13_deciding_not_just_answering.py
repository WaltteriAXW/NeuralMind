#!/usr/bin/env python3
"""A mind that decides, not just answers.

Everything before this could tell you whether something was true. This works
out what to do about it — and the difference that matters is not the plan, it
is that every step of the plan says why it is there, and that a plan is a
proposal rather than a permission.

Four things worth watching, in order: a step that exists only to clear the
way still has a reason; the proof for a step bottoms out in what was actually
observed; an action nobody can undo waits for a person, and the mind works
out which those are by reading its own action library; and when the world
breaks the plan, the plan is rebuilt rather than abandoned.
"""

from neuralmind.agency import Agency, Domain, L3, Planner
from neuralmind.agency.execute import Execution

WORKSHOP = """
% A valve can be opened when it is closed and not locked.
action unlock(V):
    needs  valve(V), locked(V)
    causes -locked(V)
    note   the key is on the panel

action lock(V):
    needs  valve(V), closed(V), not locked(V)
    causes locked(V)

action open_valve(V):
    needs  valve(V), closed(V), not locked(V)
    causes open(V), -closed(V)

action close_valve(V):
    needs  valve(V), open(V)
    causes closed(V), -open(V)

% Venting is the one that cannot be taken back.
action vent(V):
    needs  valve(V), open(V)
    causes vented(V)
"""

print("what it knows how to do")
print("-" * 74)
agency = Agency().learn_actions(WORKSHOP)
agency.observe(["valve(v1)", "closed(v1)", "locked(v1)"])
print(" ", agency.self_report())
print("  possible right now:", ", ".join(str(a) for a in agency.possible()))

print()
print("a plan, and what each step is for")
print("-" * 74)
choice = agency.decide("vented(v1)")
print(choice.plan.describe())

print()
print("why the second step can happen at all")
print("-" * 74)
print(choice.plan.proof(1))

print()
print("a plan is not a permission")
print("-" * 74)
print(choice.describe())
print()
print("  brief:", choice.brief())
print()
print("  nothing in the library can undo vent, and nobody said so —")
print("  it was worked out by looking for an action that puts back what")
print("  vent changes, and finding none.")

agency.gate.grant(L3)
print()
print("  after the host grants L3:", agency.decide("vented(v1)").brief())

print()
print("when the world breaks the plan")
print("-" * 74)
DOORS = """
action unlock(D):
    needs  door(D), locked(D), carrying(key)
    causes -locked(D)

action open_door(D):
    needs  door(D), shut(D), not locked(D)
    causes open(D), -shut(D)

action walk_through(D):
    needs  door(D), open(D)
    causes through(D)
"""
run = Execution(
    Domain.build(
        DOORS,
        facts=["door(d1)", "shut(d1)", "carrying(key)"],
        goal=["through(d1)"],
    )
)
print("  planned:", " then ".join(str(s.action) for s in run.plan))
run.step()
progress = run.step(["door(d1)", "shut(d1)", "locked(d1)", "carrying(key)"])
print("  then the door locked itself:")
print("   ", progress.surprise.describe())
print("  replanned:", " then ".join(str(s.action) for s in run.plan))
run.run()
print("  finished:", run.finished)

print()
print("  an irrelevant change would not have cost the plan — a surprise")
print("  matters only when it breaks a link the plan was relying on.")
