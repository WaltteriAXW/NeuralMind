"""Fuzzy-logic semantics for the Type 5 layer.

Real Logic, in the sense of Logic Tensor Networks (Badreddine, d'Avila Garcez,
Serafini & Spranger, *Artificial Intelligence* 303:103649, 2022): the same
first-order formulas, evaluated over truth values in [0, 1] instead of {0, 1}.

Why this matters here. The crisp engine reasons over what perception
*asserted*; if the classifier was 51% sure and wrong, the crisp engine cannot
tell. Evaluating the same rules fuzzily over the confidences gives a
satisfaction degree per rule, so a rule that is technically satisfied by a
barely-believed fact scores low and can be flagged.

The connectives are configurable because the choice has real consequences. The
product t-norm matches a probabilistic reading and is the default; Łukasiewicz
is better behaved when many weak facts combine; Gödel (min) is the most
pessimistic and useful for hard safety checks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

__all__ = [
    "TNORMS",
    "TCONORMS",
    "IMPLICATIONS",
    "AGGREGATORS",
    "FuzzySemantics",
    "p_mean_error",
    "p_mean",
]

#: Conjunction.
TNORMS: dict[str, Callable[[float, float], float]] = {
    "product": lambda a, b: a * b,
    "lukasiewicz": lambda a, b: max(0.0, a + b - 1.0),
    "godel": lambda a, b: min(a, b),
}

#: Disjunction.
TCONORMS: dict[str, Callable[[float, float], float]] = {
    "product": lambda a, b: a + b - a * b,
    "lukasiewicz": lambda a, b: min(1.0, a + b),
    "godel": lambda a, b: max(a, b),
}

#: Implication ``a -> b``.
IMPLICATIONS: dict[str, Callable[[float, float], float]] = {
    # Reichenbach: the probabilistic reading, 1 - a + a*b.
    "reichenbach": lambda a, b: 1.0 - a + a * b,
    # Łukasiewicz residuum: 1 when b >= a, degrading linearly otherwise.
    "lukasiewicz": lambda a, b: min(1.0, 1.0 - a + b),
    # Gödel residuum: fully satisfied or exactly as true as the consequent.
    "godel": lambda a, b: 1.0 if a <= b else b,
    # Kleene-Dienes: the material implication read fuzzily.
    "kleene_dienes": lambda a, b: max(1.0 - a, b),
}


def p_mean(values: Sequence[float], p: float = 2.0) -> float:
    """Generalised mean -- the existential aggregator.

    Higher ``p`` approaches a maximum, so a single strongly satisfied
    instance can carry the formula.
    """
    values = [float(v) for v in values]
    if not values:
        return 0.0
    return (sum(v**p for v in values) / len(values)) ** (1.0 / p)


def p_mean_error(values: Sequence[float], p: float = 2.0) -> float:
    """The universal aggregator: a p-mean over *dissatisfaction*.

    This is the aggregator LTN uses for ``forall``. Unlike a plain mean it
    does not let a thousand satisfied groundings drown out one badly violated
    one -- and for a consistency check, the violation is the whole point.
    """
    values = [float(v) for v in values]
    if not values:
        return 1.0  # vacuously true
    error = (sum((1.0 - v) ** p for v in values) / len(values)) ** (1.0 / p)
    return 1.0 - error


AGGREGATORS: dict[str, Callable[[Sequence[float], float], float]] = {
    "forall": p_mean_error,
    "exists": p_mean,
}


@dataclass
class FuzzySemantics:
    """A choice of connectives, applied consistently across the layer."""

    tnorm: str = "product"
    implication: str = "reichenbach"
    #: Exponent for the universal aggregator; higher punishes violations harder.
    p: float = 2.0

    def __post_init__(self) -> None:
        if self.tnorm not in TNORMS:
            raise ValueError(f"unknown t-norm {self.tnorm!r}; choose from {sorted(TNORMS)}")
        if self.implication not in IMPLICATIONS:
            raise ValueError(
                f"unknown implication {self.implication!r}; choose from {sorted(IMPLICATIONS)}"
            )
        if self.p < 1:
            raise ValueError("p must be >= 1")

    def conjoin(self, values: Iterable[float]) -> float:
        """Fuzzy conjunction of a body."""
        operator = TNORMS[self.tnorm]
        result = 1.0
        for value in values:
            result = operator(result, float(value))
        return result

    def disjoin(self, values: Iterable[float]) -> float:
        operator = TCONORMS[self.tnorm]
        result = 0.0
        for value in values:
            result = operator(result, float(value))
        return result

    def negate(self, value: float) -> float:
        """Standard (strong) negation."""
        return 1.0 - float(value)

    def implies(self, antecedent: float, consequent: float) -> float:
        return IMPLICATIONS[self.implication](float(antecedent), float(consequent))

    def forall(self, values: Sequence[float]) -> float:
        return p_mean_error(values, self.p)

    def exists(self, values: Sequence[float]) -> float:
        return p_mean(values, self.p)
