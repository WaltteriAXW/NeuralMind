"""Generating candidate clauses inside a language bias.

Three filters do most of the work, and between them they remove the large
majority of syntactically possible clauses before any of them is tested:

* **safety** -- every head variable must be bound by the body, the same check
  the inference engine applies. An unsafe clause has no finite grounding.
* **connectivity** -- a body literal that shares no variable with the rest of
  the clause constrains nothing, it just multiplies the search space.
* **variable renaming** -- ``p(A,B) :- q(A,C), r(C,B)`` and
  ``p(A,B) :- q(A,D), r(D,B)`` are the same clause. Canonical renaming
  collapses them to one.
"""

from __future__ import annotations

import itertools
from typing import Iterator, Optional, Sequence

from ..core.program import Rule, SafetyError
from ..core.terms import Atom, Literal, Var
from .bias import LanguageBias, Signature

__all__ = [
    "variable_pool",
    "head_atom",
    "candidate_literals",
    "candidate_bodies",
    "canonical_key",
    "is_connected",
    "build_clause",
]

#: Variable names used for the pool, in order.
_NAMES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def variable_pool(count: int) -> tuple[Var, ...]:
    """``A``, ``B``, ``C``, ... up to ``count`` variables."""
    if count > len(_NAMES):
        raise ValueError(f"at most {len(_NAMES)} variables are supported")
    return tuple(Var(_NAMES[index]) for index in range(count))


def head_atom(bias: LanguageBias) -> Atom:
    """The head shared by every candidate: ``target(A, B, ...)``."""
    return Atom(bias.target.name, variable_pool(bias.target.arity))


def candidate_literals(bias: LanguageBias) -> list[Literal]:
    """Every body literal the bias admits, with the head literal excluded.

    A body literal identical to the head would make the clause a tautology, so
    it is dropped here rather than filtered later.
    """
    pool = variable_pool(bias.max_variables)
    head = head_atom(bias)
    literals: list[Literal] = []
    for signature in bias.usable_predicates:
        for arguments in itertools.product(pool, repeat=signature.arity):
            atom = Atom(signature.name, arguments)
            if atom == head:
                continue  # p(A,B) :- p(A,B) is vacuous
            literals.append(Literal(atom))
            if bias.allow_negation and signature != bias.target:
                # The target is never negated, which keeps every hypothesis
                # stratified and keeps coverage monotone in the body length.
                literals.append(Literal(atom, negated=True))
    return literals


def is_connected(head: Atom, body: Sequence[Literal]) -> bool:
    """True if every body literal is linked to the head through shared variables.

    Starting from the head's variables, repeatedly absorb any literal sharing a
    variable with what is already reached. A literal never reached is talking
    about something unrelated.
    """
    if not body:
        return True
    reached = {arg.name for arg in head.args if isinstance(arg, Var)}
    remaining = list(body)
    progress = True
    while progress and remaining:
        progress = False
        for literal in list(remaining):
            variables = {a.name for a in literal.atom.args if isinstance(a, Var)}
            if variables & reached or not variables:
                reached |= variables
                remaining.remove(literal)
                progress = True
    return not remaining


def canonical_key(head: Atom, body: Sequence[Literal]) -> tuple:
    """A key equal for clauses that differ only by variable renaming.

    Variables are renumbered in order of first appearance, scanning the head
    and then the body. The body is compared as a set, since a conjunction does
    not care about order.
    """
    mapping: dict[str, str] = {}

    def rename(atom: Atom) -> str:
        parts = []
        for argument in atom.args:
            if isinstance(argument, Var):
                if argument.name not in mapping:
                    mapping[argument.name] = f"V{len(mapping)}"
                parts.append(mapping[argument.name])
            else:
                parts.append(str(argument))
        return f"{atom.predicate}({','.join(parts)})"

    head_key = rename(head)
    body_keys = []
    for literal in body:
        text = rename(literal.atom)
        body_keys.append(f"not {text}" if literal.negated else text)
    return (head_key, frozenset(body_keys))


def build_clause(head: Atom, body: Sequence[Literal]) -> Optional[Rule]:
    """Make a :class:`Rule` if the clause is safe, else None."""
    rule = Rule(head=head, body=tuple(body), source="induction")
    try:
        rule.plan()  # runs the engine's own safety check
    except SafetyError:
        return None
    return rule


def candidate_bodies(
    bias: LanguageBias,
    size: int,
    literals: Optional[Sequence[Literal]] = None,
) -> Iterator[tuple[Rule, tuple[Literal, ...], tuple]]:
    """Yield every admissible clause of exactly ``size`` body literals.

    Each item is ``(rule, body, canonical key)``. Duplicates under variable
    renaming are emitted once.
    """
    pool = list(literals) if literals is not None else candidate_literals(bias)
    head = head_atom(bias)
    seen: set[tuple] = set()
    for body in itertools.combinations(pool, size):
        if bias.require_connected and not is_connected(head, body):
            continue
        key = canonical_key(head, body)
        if key in seen:
            continue
        seen.add(key)
        rule = build_clause(head, body)
        if rule is None:
            continue
        yield rule, body, key
