#!/usr/bin/env python3
"""One command that reproduces every number this project claims.

The measurements used to live in five places — a fuzz harness, a growth
benchmark, two `eval` invocations, an intent training script. Figures you have
to reassemble by hand quietly stop being true, so they are one ordered
curriculum now.

What this shows, in order: the stages and their prerequisites; a short run of
three of them; and the failure attribution, where seven categories in
upstream-first order mean every failure is counted once, under the cause that
is furthest upstream.

For the whole thing, use the command line — `neuralmind school`, or
`neuralmind school --full` for every question and 10,000 hostile inputs.
"""

from neuralmind.school import CATEGORIES, REMEDY, Evidence, School, attribute

school = School()

print("the curriculum")
print("-" * 78)
for stage in school.curriculum:
    gate = f"after {', '.join(stage.after)}" if stage.after else "no prerequisites"
    why_not = stage.why_not()
    runs = f"unavailable: {why_not}" if why_not else "runs"
    print(f"  {stage.name:26} {gate:34} {runs}")

print()
print("running three of them")
print("-" * 78)
report = School().run(quick=True, only=["proofwriter-cwa", "proofwriter-owa", "where-am-i"])
print(report.describe())

print()
print("every failure has exactly one attribution, most upstream first")
print("-" * 78)
for i, category in enumerate(CATEGORIES, 1):
    print(f"  {i}. {category:20} {REMEDY[category]}")

print()
print("so a question that ran out of time is a budget failure, and only that:")
timed_out = Evidence(out_of_budget=True, rules_stalled=True, missing_fact="parent(x, y)")
found = attribute(timed_out)
print(f"  {found.category}  —  {found.detail}")
print("  (it also stalled and also lacked a fact; counting all three is how a")
print("   report ends up telling you to write rules when a specialist is slow)")
