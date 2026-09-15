#!/usr/bin/env python3
"""A CNN reads two digits; the logic computes and checks the sum.

This is the classic neurosymbolic experiment, and the point is the division of
labour: the sum is computed, not learned, so it is exactly right whenever the
perception is.
"""

from neuralmind import KnowledgeBase
from neuralmind.consistency.layer import ConsistencyLayer
from neuralmind.datasets.mnist import digit_pairs, load_mnist
from neuralmind.perception.vision import DigitPerceptor

_, test = load_mnist()
perceptor = DigitPerceptor()
pairs = digit_pairs(test, 200, seed=1)

correct = 0
first_mistake = None
for pair in pairs:
    kb = KnowledgeBase("mnist").load_builtin("mnist_sum")
    perceptor.perceive(pair.images, slots=["d0", "d1"]).into(kb)
    answer = kb.engine().ask("sum(S)")
    total = answer.atoms[0].args[0].value if answer.holds else None
    correct += total == pair.total
    if total != pair.total and first_mistake is None:
        first_mistake = pair

print(f"end-to-end accuracy: {correct}/{len(pairs)}")

if first_mistake is not None:
    print("\nthe first pair it got wrong, handed to the Type 5 layer:")
    distributions = perceptor.distributions(first_mistake.images, slots=["d0", "d1"])
    print(f"  truth: {first_mistake.left_label} + {first_mistake.right_label}"
          f" = {first_mistake.total}")
    print(f"  the CNN's reading: {[str(d) for d in distributions]}")

    kb = KnowledgeBase("mnist").load_builtin("mnist_sum")
    kb.add_fact(f"expected_sum({first_mistake.total})")
    best = ConsistencyLayer(kb).resolve(distributions, candidates_per_slot=4, top_k=1)
    print(f"  most probable reading the rules allow: {best[0] if best else 'none'}")
