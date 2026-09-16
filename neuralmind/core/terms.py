"""Symbolic data model: the vocabulary every layer of the pipeline speaks.

The perception layer emits :class:`Atom` instances, the knowledge base stores
them, the inference engine derives new ones, and the output layer renders them.
Everything here is immutable and hashable so atoms can live in sets and act as
dictionary keys during fixpoint computation.

The syntax printed by ``str()`` is a subset of ASP/Datalog that ``clingo``
accepts verbatim, which is what lets the optional clingo backend re-check any
program the pure-Python engine solves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence, Union

__all__ = [
    "Term",
    "Var",
    "Const",
    "Arith",
    "Atom",
    "Compare",
    "Literal",
    "Substitution",
    "evaluate",
    "ground_term",
    "is_ground",
    "variables_in",
    "COMPARISON_OPS",
]

#: Comparison operators usable in a rule body.
COMPARISON_OPS = ("!=", "<=", ">=", "=", "<", ">")

#: A binding of variables to ground terms.
Substitution = Mapping[str, "Const"]


class Term:
    """Base class for anything that can appear as an argument of an atom."""

    __slots__ = ()


@dataclass(frozen=True)
class Var(Term):
    """A logic variable. Written with a leading capital letter, as in ASP."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Term):
    """A constant: a symbol (``alice``), a quoted string, or an integer."""

    value: Union[str, int]
    quoted: bool = False

    def __str__(self) -> str:
        if self.quoted:
            escaped = str(self.value).replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        return str(self.value)

    @property
    def is_number(self) -> bool:
        return isinstance(self.value, int)


@dataclass(frozen=True)
class Arith(Term):
    """An arithmetic expression, evaluated once its variables are bound."""

    op: str
    left: Term
    right: Term

    def __str__(self) -> str:
        return f"({self.left}{self.op}{self.right})"


@dataclass(frozen=True)
class Atom:
    """A predicate applied to terms, e.g. ``parent(alice, bob)``."""

    predicate: str
    args: tuple[Term, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(str(a) for a in self.args)})"

    @property
    def arity(self) -> int:
        return len(self.args)

    @property
    def signature(self) -> tuple[str, int]:
        """``(predicate, arity)`` -- the key used by every predicate index."""
        return (self.predicate, len(self.args))

    @property
    def is_ground(self) -> bool:
        return all(is_ground(a) for a in self.args)

    def ground(self, subst: Substitution) -> "Atom":
        """Return a copy with every bound variable replaced by its value."""
        if not self.args:
            return self
        return Atom(self.predicate, tuple(ground_term(a, subst) for a in self.args))

    def as_tuple(self) -> tuple:
        """Plain-Python view of a ground atom: ``('parent', 'alice', 'bob')``."""
        return (self.predicate,) + tuple(
            a.value if isinstance(a, Const) else str(a) for a in self.args
        )


@dataclass(frozen=True)
class Compare:
    """A built-in comparison body literal, e.g. ``S = A + B`` or ``X != Y``."""

    op: str
    left: Term
    right: Term

    def __str__(self) -> str:
        return f"{self.left} {self.op} {self.right}"

    def holds(self, subst: Substitution) -> bool:
        """Evaluate the comparison under ``subst``.

        ``=`` doubles as assignment in ASP, but the engine only ever calls this
        once both sides are fully bound, so here it is a plain equality test.
        """
        left = evaluate(self.left, subst)
        right = evaluate(self.right, subst)
        if self.op == "=":
            return left == right
        if self.op == "!=":
            return left != right
        # Ordering comparisons: numbers compare numerically, symbols
        # lexicographically, and numbers sort before symbols (as in clingo).
        lkey = _order_key(left)
        rkey = _order_key(right)
        if self.op == "<":
            return lkey < rkey
        if self.op == "<=":
            return lkey <= rkey
        if self.op == ">":
            return lkey > rkey
        if self.op == ">=":
            return lkey >= rkey
        raise ValueError(f"unknown comparison operator {self.op!r}")


@dataclass(frozen=True)
class Literal:
    """A positive or default-negated atom in a rule body."""

    atom: Atom
    negated: bool = False

    def __str__(self) -> str:
        return f"not {self.atom}" if self.negated else str(self.atom)

    @property
    def signature(self) -> tuple[str, int]:
        return self.atom.signature


def _order_key(const: Const) -> tuple[int, Union[int, str]]:
    return (0, const.value) if isinstance(const.value, int) else (1, str(const.value))


def is_ground(term: Term) -> bool:
    """True if ``term`` contains no variables."""
    if isinstance(term, Var):
        return False
    if isinstance(term, Arith):
        return is_ground(term.left) and is_ground(term.right)
    return True


def variables_in(obj: Union[Term, Atom, Compare, Literal]) -> Iterator[str]:
    """Yield the names of every variable occurring in ``obj``."""
    if isinstance(obj, Var):
        yield obj.name
    elif isinstance(obj, Arith):
        yield from variables_in(obj.left)
        yield from variables_in(obj.right)
    elif isinstance(obj, Atom):
        for arg in obj.args:
            yield from variables_in(arg)
    elif isinstance(obj, Compare):
        yield from variables_in(obj.left)
        yield from variables_in(obj.right)
    elif isinstance(obj, Literal):
        yield from variables_in(obj.atom)


def ground_term(term: Term, subst: Substitution) -> Term:
    """Substitute bound variables in ``term``; arithmetic stays unevaluated."""
    if isinstance(term, Var):
        return subst.get(term.name, term)
    if isinstance(term, Arith):
        left = ground_term(term.left, subst)
        right = ground_term(term.right, subst)
        expr = Arith(term.op, left, right)
        return evaluate(expr, {}) if is_ground(expr) else expr
    return term


def evaluate(term: Term, subst: Substitution) -> Const:
    """Fully evaluate ``term`` under ``subst`` down to a :class:`Const`."""
    if isinstance(term, Const):
        return term
    if isinstance(term, Var):
        try:
            return subst[term.name]
        except KeyError:
            raise ValueError(f"unbound variable {term.name} in arithmetic") from None
    if isinstance(term, Arith):
        left = evaluate(term.left, subst)
        right = evaluate(term.right, subst)
        if not (left.is_number and right.is_number):
            raise ValueError(f"arithmetic on non-numeric terms: {left} {term.op} {right}")
        a, b = int(left.value), int(right.value)
        if term.op == "+":
            return Const(a + b)
        if term.op == "-":
            return Const(a - b)
        if term.op == "*":
            return Const(a * b)
        if term.op == "/":
            if b == 0:
                raise ValueError("division by zero")
            # Truncating division, matching clingo's integer division.
            return Const(int(a / b))
        if term.op in ("\\", "%"):
            if b == 0:
                raise ValueError("modulo by zero")
            return Const(a - b * int(a / b))
        if term.op == "**":
            return Const(a**b)
        raise ValueError(f"unknown arithmetic operator {term.op!r}")
    raise TypeError(f"cannot evaluate term of type {type(term).__name__}")


def atom(predicate: str, *args: Union[Term, str, int]) -> Atom:
    """Convenience constructor: ``atom('parent', 'alice', 'bob')``.

    Strings starting with a capital letter become variables, everything else
    becomes a constant -- the same convention the parser uses.
    """
    return Atom(predicate, tuple(term(a) for a in args))


def term(value: Union[Term, str, int]) -> Term:
    """Coerce a Python value into a :class:`Term`."""
    if isinstance(value, Term):
        return value
    if isinstance(value, bool):
        raise TypeError("booleans are not terms; use a 0/1 constant or a predicate")
    if isinstance(value, int):
        return Const(value)
    if isinstance(value, str):
        if value and (value[0].isupper() or value[0] == "_"):
            return Var(value)
        if value and value[0].islower() and value.replace("_", "").isalnum():
            return Const(value)
        return Const(value, quoted=True)
    raise TypeError(f"cannot turn {value!r} into a term")


def format_body(body: Sequence[Union[Literal, Compare]]) -> str:
    """Render a rule body as comma-separated ASP source."""
    return ", ".join(str(part) for part in body)
