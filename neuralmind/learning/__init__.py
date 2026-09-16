"""Learning through the logic.

The rest of the project runs rules forward, from facts to conclusions. This
package runs them backward: given a conclusion known to be true, it computes
how a network's probabilities should move to make that conclusion more likely.

That turns a logical consequence into a training signal, which is what lets a
digit classifier learn from sums it is given without ever seeing a digit label.
Needs NumPy.
"""

from .semantic_loss import NeuralPredicate, SemanticLoss, TableTooLarge, softmax_backward
from .weak import WeakExample, WeaklySupervisedTrainer, WeakTrainingReport

__all__ = [
    "SemanticLoss",
    "NeuralPredicate",
    "TableTooLarge",
    "softmax_backward",
    "WeaklySupervisedTrainer",
    "WeakExample",
    "WeakTrainingReport",
]
