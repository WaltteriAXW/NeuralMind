"""Learning rules from examples.

The knowledge acquisition bottleneck has two halves. The perception layer can
learn from what the rules entail (:mod:`neuralmind.learning`); this package
learns the rules themselves, by generate-test-constrain search over a bounded
hypothesis space, with the inference engine judging every candidate.
"""

from .bias import LanguageBias, Signature
from .enumerate import candidate_bodies, candidate_literals
from .learn import Examples, Hypothesis, RuleLearner, learn_rules

__all__ = [
    "LanguageBias",
    "Signature",
    "Examples",
    "Hypothesis",
    "RuleLearner",
    "learn_rules",
    "candidate_literals",
    "candidate_bodies",
]
