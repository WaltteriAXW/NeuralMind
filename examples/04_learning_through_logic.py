#!/usr/bin/env python3
"""Learning from what the rules entail, rather than from labels.

Everywhere else in this project the logic runs forward: perception commits to
symbols and the engine draws conclusions. Here it runs backward. The network is
shown pairs of digit images labelled only with their *sum*, and the gradient of
P(sum = s) through the addition rule teaches it to read digits.

No digit label is used at any point in training.
"""

import numpy as np

from neuralmind import KnowledgeBase
from neuralmind.datasets.mnist import digit_pairs, load_mnist
from neuralmind.learning import NeuralPredicate, SemanticLoss, WeaklySupervisedTrainer, WeakExample
from neuralmind.perception.nn import ConvNet

kb = KnowledgeBase("sum").load_builtin("mnist_sum")
slots = [
    NeuralPredicate.over_integers("digit", "d0", 10),
    NeuralPredicate.over_integers("digit", "d1", 10),
]
loss = SemanticLoss(kb, slots)

# --- what the logic contributes, before any training -----------------------
#
# The rules decide which pairs of digits could produce a given sum. That is a
# boolean table with one axis per image, computed by the symbolic engine.

table = loss.truth_table("sum(4)")
print("pairs the rule allows for sum(4):")
print("  " + ", ".join(f"({a},{b})" for a, b in zip(*np.nonzero(table))))

uniform = np.full((2, 10), 0.1)
print(f"\nP(sum=4) if the classifier knows nothing: {loss.probability(uniform, 'sum(4)'):.3f}")

confident = np.full((2, 10), 0.001)
confident[0, 1] = confident[1, 3] = 0.991
print(f"P(sum=4) if it reads 1 and 3 confidently: {loss.probability(confident, 'sum(4)'):.3f}")

_, gradient = loss.loss_and_gradient(uniform, "sum(4)")
print("\nthe gradient tells the network where to put its mass:")
print(f"  slot 0, value 1: {gradient[0, 1]:+.3f}   (a pair that works -> push up)")
print(f"  slot 0, value 9: {gradient[0, 9]:+.3f}   (no pair works -> leave alone)")

# --- using it as the only training signal ----------------------------------

train, test = load_mnist()
pairs = digit_pairs(train, 3000, seed=5)
examples = [WeakExample(images=p.images, query=f"sum({p.total})") for p in pairs]

network = ConvNet(seed=0)
trainer = WeaklySupervisedTrainer(network, loss, learning_rate=2e-3)

held_images, held_labels = test.images[:1000], test.labels[:1000]
print(f"\ndigit accuracy before training: {network.accuracy(held_images, held_labels):.3f}")

report = trainer.fit(
    examples,
    epochs=4,
    batch_size=32,
    validation=lambda: (network.accuracy(held_images, held_labels), 0.0),
    verbose=False,
)
for epoch, (value, accuracy) in enumerate(zip(report.losses, report.slot_accuracies), 1):
    print(f"  epoch {epoch}: loss {value:.4f}, digit accuracy {accuracy:.3f}")

print(f"\ndigit labels used: {report.labels_seen}")
print(f"symbolic engine calls: {loss.engine_calls} (one per distinct assignment, then cached)")
