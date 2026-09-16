"""Gradients through the logic: learning from what the rules entail.

Everything else in this project runs the logic *forward* -- perception commits
to symbols, the engine draws conclusions. This module runs it backward. Given a
neural network that is unsure what it is looking at, and a logical statement
that is known to be true, it computes how the network's probabilities should
move so that the statement becomes more likely.

That is what makes weak supervision possible. Train a digit classifier on pairs
of images labelled only with their *sum* -- never with a digit -- and the
gradient through the sum rule is enough to teach it the digits. No digit label
is ever seen. `scripts/train_weak_supervision.py` does exactly that.

## How it works

For a program with neural predicates over a finite domain, the probability that
a query follows is a weighted model count::

    P(query) = sum over assignments that entail the query of
               product over slots of P(slot = its value)

The assignments that entail the query do not depend on the probabilities at
all, only on the rules. So they are computed once, by the symbolic engine, and
stored as a boolean **truth tensor** with one axis per neural predicate. After
that the whole thing is multilinear: the probability is a tensor contraction,
and the gradient with respect to any slot is the same contraction with that
slot left out.

This is exact weighted model counting, which is what DeepProbLog and Scallop
compile to arithmetic circuits to compute. Here the circuit is a dense tensor,
which is the right representation when the domain is small (two digit slots is
a 10x10 table) and the wrong one when it is large -- the cost is
``|domain| ** slots``, and :class:`SemanticLoss` refuses to build a table
bigger than ``max_table`` rather than quietly exhausting memory.

The symbolic engine stays the source of truth throughout: every entry of the
tensor is the answer the engine gave for that assignment.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

try:
    import numpy as np
except ImportError as exc:  # pragma: no cover - environment dependent
    raise ImportError(
        "the learning layer needs NumPy. Install it with "
        "`pip install 'neuralmind[numeric]'`."
    ) from exc

from ..core.parser import parse_atom
from ..core.program import Program, Rule
from ..core.terms import Atom, Const
from ..inference.forward import ForwardChainer
from ..knowledge.base import KnowledgeBase

__all__ = ["NeuralPredicate", "SemanticLoss", "TableTooLarge", "softmax_backward"]

#: Probabilities are clamped away from zero before taking a logarithm.
EPSILON = 1e-12


class TableTooLarge(ValueError):
    """The truth tensor would be larger than the configured limit."""


@dataclass(frozen=True)
class NeuralPredicate:
    """A slot whose value a network predicts, e.g. ``digit(d0, ?)``.

    ``key`` holds the arguments that are known (``d0``); the last argument is
    the one the network is guessing, ranging over ``domain``.
    """

    predicate: str
    key: tuple
    domain: tuple

    @classmethod
    def over_integers(cls, predicate: str, key, size: int) -> "NeuralPredicate":
        """Convenience for the common case of a 0..n-1 integer domain."""
        keys = key if isinstance(key, tuple) else (key,)
        keys = tuple(k if isinstance(k, Const) else Const(k) for k in keys)
        return cls(predicate, keys, tuple(Const(value) for value in range(size)))

    def atom(self, value) -> Atom:
        return Atom(self.predicate, self.key + (value,))

    @property
    def size(self) -> int:
        return len(self.domain)

    def __str__(self) -> str:
        key = ", ".join(str(part) for part in self.key)
        return f"{self.predicate}({key}, ?) over {self.size} values"


@dataclass
class SemanticLoss:
    """Differentiates a query's probability with respect to neural outputs.

    Parameters
    ----------
    knowledge:
        The rules. Anything a :class:`~neuralmind.knowledge.base.KnowledgeBase`
        or :class:`~neuralmind.core.program.Program` can express, including
        integrity constraints -- an assignment that violates one counts as not
        entailing the query, so hard rules shape the gradient too.
    slots:
        One :class:`NeuralPredicate` per uncertain input, in the same order as
        the rows of the probability matrix passed to the methods below. All
        slots must share a domain size, because those probabilities arrive as
        one rectangular array from one network's output layer.
    max_table:
        Refuse to build a truth tensor with more than this many entries.
    """

    knowledge: Union[KnowledgeBase, Program]
    slots: Sequence[NeuralPredicate]
    max_table: int = 1_000_000
    _cache: dict = field(default_factory=dict, repr=False)
    _engine_calls: int = field(default=0, repr=False)
    _unsatisfiable: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        if not self.slots:
            raise ValueError("SemanticLoss needs at least one neural predicate")
        domains = {slot.size for slot in self.slots}
        if len(domains) > 1:
            # Probabilities arrive as one rectangular array, one row per slot,
            # because they come from one network's output layer.
            raise ValueError(
                "every slot must range over the same number of values, since "
                "probabilities are passed as a (slots, domain) array; got sizes "
                f"{[slot.size for slot in self.slots]}. Pad the domains, or use "
                "one SemanticLoss per group of same-sized slots."
            )
        size = 1
        for slot in self.slots:
            size *= slot.size
        if size > self.max_table:
            raise TableTooLarge(
                f"a truth tensor over {len(self.slots)} slots with domains of "
                f"{[s.size for s in self.slots]} needs {size} entries, over the "
                f"{self.max_table} limit. Exact weighted model counting is "
                "|domain| ** slots; past this point use a sampling or "
                "circuit-compilation approach instead."
            )
        self._shape = tuple(slot.size for slot in self.slots)

    # -- the truth tensor -------------------------------------------------

    def truth_table(
        self,
        query: Union[Atom, str, Sequence],
        evidence: Iterable[Union[Atom, str]] = (),
    ) -> np.ndarray:
        """Which assignments entail the query, as a boolean tensor.

        One axis per slot. Computed by running the symbolic engine on every
        assignment, and cached -- in weakly supervised training the same query
        recurs constantly (there are only 19 possible sums of two digits), so
        the engine runs a few hundred times in total rather than once per
        example.
        """
        goals = _as_atoms(query)
        given = tuple(sorted((_as_atom(a) for a in evidence), key=str))
        cache_key = (tuple(str(g) for g in goals), tuple(str(a) for a in given))
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        program = self._program()
        table = np.zeros(self._shape, dtype=bool)
        for index in itertools.product(*(range(slot.size) for slot in self.slots)):
            assignment = [
                slot.atom(slot.domain[position])
                for slot, position in zip(self.slots, index)
            ]
            table[index] = self._entails(program, assignment, given, goals)
        self._cache[cache_key] = table
        return table

    def _entails(
        self,
        program: Program,
        assignment: Sequence[Atom],
        evidence: Sequence[Atom],
        goals: Sequence[Atom],
    ) -> bool:
        """Run the engine on one assignment and report whether the goals hold."""
        self._engine_calls += 1
        facts = [
            Rule(atom, (), source="semantic-loss", label="given")
            for atom in (*assignment, *evidence)
        ]
        combined = Program(
            rules=facts + list(program.rules),
            shown=set(program.shown),
            constants=dict(program.constants),
        )
        model = ForwardChainer(combined).run()
        if not model.consistent:
            # A hard rule was broken, so this assignment supports nothing.
            return False
        return all(model.holds(goal) for goal in goals)

    def _program(self) -> Program:
        if isinstance(self.knowledge, Program):
            return self.knowledge
        return self.knowledge.program(check=False)

    # -- probability and gradient ------------------------------------------

    def probability(
        self,
        distributions: np.ndarray,
        query: Union[Atom, str, Sequence],
        evidence: Iterable[Union[Atom, str]] = (),
    ) -> float:
        """P(query) under the given per-slot probabilities.

        ``distributions`` has one row per slot, each row a probability
        distribution over that slot's domain.
        """
        table = self.truth_table(query, evidence)
        return float(_contract(table, self._check(distributions)))

    def loss_and_gradient(
        self,
        distributions: np.ndarray,
        query: Union[Atom, str, Sequence],
        evidence: Iterable[Union[Atom, str]] = (),
    ) -> tuple[float, np.ndarray]:
        """Negative log-likelihood of the query, and its gradient.

        The gradient has the same shape as ``distributions``: entry ``[i, v]``
        is how the loss changes as slot ``i`` puts more mass on value ``v``.
        """
        probabilities = self._check(distributions)
        table = self.truth_table(query, evidence)
        total = float(_contract(table, probabilities))
        loss = -float(np.log(max(total, EPSILON)))

        gradient = np.zeros_like(probabilities)
        if total <= EPSILON:
            # No assignment entails the query -- the rules and the label
            # disagree. There is no direction that helps, so the gradient is
            # flat, which would otherwise be an invisible no-op in a training
            # loop. Count it so a run can report how often it happened.
            self._unsatisfiable += 1
            return loss, gradient
        for index in range(len(self.slots)):
            # d P / d p_i = the same contraction with slot i left out.
            gradient[index] = -_contract_except(table, probabilities, index) / total
        return loss, gradient

    def most_probable_assignment(
        self,
        distributions: np.ndarray,
        query: Union[Atom, str, Sequence],
        evidence: Iterable[Union[Atom, str]] = (),
    ) -> Optional[tuple[list[Atom], float]]:
        """The likeliest assignment that entails the query, or None.

        The inference-time counterpart of the training-time gradient, and the
        same job :meth:`~neuralmind.consistency.layer.ConsistencyLayer.resolve`
        does -- computed here from the cached tensor, so it is a single argmax.
        """
        probabilities = self._check(distributions)
        table = self.truth_table(query, evidence)
        joint = _outer(probabilities) * table
        if not joint.any():
            return None
        flat = int(joint.argmax())
        index = np.unravel_index(flat, joint.shape)
        atoms = [
            slot.atom(slot.domain[position]) for slot, position in zip(self.slots, index)
        ]
        return atoms, float(joint[index])

    # -- helpers -----------------------------------------------------------

    def _check(self, distributions: np.ndarray) -> np.ndarray:
        array = np.asarray(distributions, dtype=np.float64)
        if array.ndim != 2 or array.shape[0] != len(self.slots):
            raise ValueError(
                f"expected probabilities shaped ({len(self.slots)}, domain), "
                f"got {array.shape}"
            )
        for index, slot in enumerate(self.slots):
            if array.shape[1] != slot.size:
                raise ValueError(
                    f"slot {index} ({slot}) has {slot.size} values but the "
                    f"probabilities give {array.shape[1]}"
                )
        return array

    @property
    def engine_calls(self) -> int:
        """How many times the symbolic engine has run."""
        return self._engine_calls

    @property
    def unsatisfiable_queries(self) -> int:
        """How many gradient requests had no satisfying assignment at all.

        Anything above zero means some example's label cannot be produced by
        the rules -- a mislabelled example, a domain that is too small, or a
        constraint that is too strong. Such an example contributes nothing to
        training, so this number should be watched rather than ignored.
        """
        return self._unsatisfiable


def _outer(probabilities: np.ndarray) -> np.ndarray:
    """Joint probability of every assignment, as a tensor."""
    joint = probabilities[0]
    for row in probabilities[1:]:
        joint = np.multiply.outer(joint, row)
    return joint


def _contract(table: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    """Weighted model count: contract the truth tensor with every slot."""
    result = table.astype(np.float64)
    # Highest axis first, so the indices of the axes still to go do not shift.
    for index in reversed(range(len(probabilities))):
        result = np.tensordot(result, probabilities[index], axes=([index], [0]))
    return result


def _contract_except(
    table: np.ndarray, probabilities: np.ndarray, skip: int
) -> np.ndarray:
    """Contract every slot except ``skip``, leaving a vector over its domain.

    This is exactly ``dP/dp_skip``: the count is multilinear in each slot, so
    the derivative with respect to one slot is the count with that slot's
    probabilities removed.
    """
    result = table.astype(np.float64)
    # Contract the other axes from the highest index down, so the positions of
    # the axes still to be contracted do not shift under us.
    for index in reversed(range(len(probabilities))):
        if index == skip:
            continue
        result = np.tensordot(result, probabilities[index], axes=([index], [0]))
    return result


def softmax_backward(probabilities: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    """Push a gradient on softmax outputs back to the logits.

    For the softmax Jacobian this reduces to
    ``p * (g - sum(g * p))`` per row, with no matrix ever formed.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    gradient = np.asarray(gradient, dtype=np.float64)
    weighted = (gradient * probabilities).sum(axis=-1, keepdims=True)
    return probabilities * (gradient - weighted)


def _as_atom(value: Union[Atom, str]) -> Atom:
    return parse_atom(value) if isinstance(value, str) else value


def _as_atoms(value: Union[Atom, str, Sequence]) -> tuple[Atom, ...]:
    if isinstance(value, (Atom, str)):
        return (_as_atom(value),)
    return tuple(_as_atom(item) for item in value)
