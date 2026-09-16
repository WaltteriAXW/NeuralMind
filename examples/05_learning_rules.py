#!/usr/bin/env python3
"""Learning the rules themselves, instead of writing them.

The knowledge acquisition bottleneck has two halves. Example 04 removed one:
the perception layer can learn from what the rules entail, so it needs no
labels. This removes the other: given examples of a relation, the learner
searches for the rule that defines it.

The search is generate-test-constrain. Candidate clauses are enumerated
shortest-first inside a declared bias, the real inference engine decides which
examples each one covers, and any body that covers nothing prunes every
extension of itself -- which is what makes the search finish.
"""

from neuralmind import Examples, KnowledgeBase, LanguageBias, RuleLearner
from neuralmind.output.render import render_proof

EDGES = [
    ("maria", "juho"), ("maria", "liisa"), ("juho", "aino"),
    ("liisa", "onni"), ("aino", "elias"),
]
PEOPLE = ["maria", "juho", "liisa", "aino", "onni", "elias"]

family = KnowledgeBase("family")
family.add_facts([f"parent({a}, {b})" for a, b in EDGES])

# --- a non-recursive relation ---------------------------------------------

grandparents = [
    f"grandparent({a}, {c})"
    for a, b in EDGES
    for c in [d for e, d in EDGES if e == b]
]
examples = Examples.closed_world(grandparents, PEOPLE)
bias = LanguageBias.for_target(
    "grandparent/2", ["parent/2"], allow_recursion=False, max_variables=3, max_body=2
)
print(bias.describe())
print(f"examples: {examples.summary()}\n")

hypothesis = RuleLearner(family, bias, examples).learn()
print(hypothesis)

# --- a recursive one ------------------------------------------------------

closure = set(EDGES)
changed = True
while changed:
    changed = False
    for a, b in list(closure):
        for c, d in EDGES:
            if b == c and (a, d) not in closure:
                closure.add((a, d))
                changed = True

print("\n" + "-" * 60)
examples = Examples.closed_world(
    [f"ancestor({a}, {b})" for a, b in sorted(closure)], PEOPLE
)
bias = LanguageBias.for_target(
    "ancestor/2", ["parent/2"], max_variables=3, max_body=2, max_clauses=3
)
learner = RuleLearner(family, bias, examples)
hypothesis = learner.learn(verbose=True)
print(hypothesis)

print("\nand the learned rules prove their own conclusions:")
print(render_proof(learner.explain(hypothesis, "ancestor(maria, elias)")))

# --- exceptions, which need negation --------------------------------------

print("\n" + "-" * 60)
birds = KnowledgeBase("birds")
names = ["tweety", "robin", "pingu", "skipper", "eagle"]
birds.add_facts([f"bird({b})" for b in names] + ["penguin(pingu)", "penguin(skipper)"])

examples = Examples.closed_world(["flies(tweety)", "flies(robin)", "flies(eagle)"], names)
bias = LanguageBias.for_target(
    "flies/1", ["bird/1", "penguin/1"],
    max_variables=1, max_body=2, allow_negation=True, allow_recursion=False,
)
print(RuleLearner(birds, bias, examples).learn())
