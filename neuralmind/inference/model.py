"""The result of reasoning: derived atoms plus *why* each one holds.

A :class:`Model` is a set of ground atoms, but the part that matters for this
project is :attr:`Model.justifications` -- for every derived atom, the rule
instances that produced it. That record is what the output layer turns into a
proof tree, so the explanation is a by-product of inference rather than
something reconstructed afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional

from ..core.program import Rule
from ..core.terms import Atom, Const, Substitution, Var

__all__ = ["Justification", "Violation", "Model", "AtomIndex", "match_atom"]


@dataclass(frozen=True)
class Justification:
    """One rule instance that derives an atom."""

    rule: Rule
    bindings: tuple[tuple[str, Const], ...]
    support: tuple[Atom, ...] = ()
    negative_support: tuple[Atom, ...] = ()
    depth: int = 0

    @property
    def substitution(self) -> dict[str, Const]:
        return dict(self.bindings)

    def instance(self) -> str:
        """The rule with its variables filled in -- readable in a proof tree."""
        subst = self.substitution
        head = self.rule.head.ground(subst) if self.rule.head is not None else None
        body = ", ".join(str(a) for a in self.support)
        body_neg = ", ".join(f"not {a}" for a in self.negative_support)
        parts = ", ".join(p for p in (body, body_neg) if p)
        if head is None:
            return f":- {parts}."
        return f"{head}." if not parts else f"{head} :- {parts}."


@dataclass(frozen=True)
class Violation:
    """An integrity constraint whose body was satisfied -- a broken hard rule."""

    rule: Rule
    bindings: tuple[tuple[str, Const], ...]
    support: tuple[Atom, ...] = ()
    negative_support: tuple[Atom, ...] = ()

    @property
    def substitution(self) -> dict[str, Const]:
        return dict(self.bindings)

    def describe(self) -> str:
        label = self.rule.label or self.rule.origin()
        support = ", ".join(str(a) for a in self.support)
        return f"{label} violated by: {support}" if support else f"{label} violated"

    def __str__(self) -> str:
        return self.describe()


class AtomIndex:
    """Ground atoms indexed by predicate and by each constant argument.

    Lookup picks the most selective bound argument, so matching
    ``parent(alice, X)`` scans only the atoms whose first argument is
    ``alice`` rather than every ``parent`` fact. The forward chainer keeps one
    of these for the whole model and a second for each round's newly derived
    atoms, which is what makes semi-naive evaluation cheap.
    """

    __slots__ = ("by_signature", "by_argument", "_count")

    def __init__(self) -> None:
        self.by_signature: dict[tuple[str, int], list[Atom]] = {}
        self.by_argument: dict[tuple, list[Atom]] = {}
        self._count = 0

    def add(self, atom: Atom) -> None:
        self.by_signature.setdefault(atom.signature, []).append(atom)
        for position, argument in enumerate(atom.args):
            if isinstance(argument, Const):
                key = (atom.signature, position, argument)
                self.by_argument.setdefault(key, []).append(atom)
        self._count += 1

    def candidates(self, pattern: Atom) -> list[Atom]:
        """Atoms that could match ``pattern``, via the most selective index."""
        best: Optional[list[Atom]] = None
        for position, argument in enumerate(pattern.args):
            if isinstance(argument, Const):
                bucket = self.by_argument.get((pattern.signature, position, argument))
                if bucket is None:
                    return []  # nothing has that constant in that position
                if best is None or len(bucket) < len(best):
                    best = bucket
        if best is None:
            return self.by_signature.get(pattern.signature, [])
        return best

    def signatures(self) -> set[tuple[str, int]]:
        return set(self.by_signature)

    def has(self, signature: tuple[str, int]) -> bool:
        return signature in self.by_signature

    def __len__(self) -> int:
        return self._count

    def clear(self) -> None:
        self.by_signature.clear()
        self.by_argument.clear()
        self._count = 0


@dataclass
class Model:
    """A set of ground atoms with provenance and an argument index."""

    atoms: set[Atom] = field(default_factory=set)
    justifications: dict[Atom, list[Justification]] = field(default_factory=dict)
    depth: dict[Atom, int] = field(default_factory=dict)
    violations: list[Violation] = field(default_factory=list)
    iterations: int = 0
    #: Set when reasoning stopped early because a limit was hit.
    truncated: Optional[str] = None
    #: The program this model came from, when the solver recorded it. Needed
    #: to read an *absence* correctly: whether an underivable atom is false or
    #: merely unknown is a property of the program, not of the model.
    program: Optional[object] = None

    index: AtomIndex = field(default_factory=AtomIndex, repr=False)

    # -- population ------------------------------------------------------

    def add(self, atom: Atom, justification: Optional[Justification] = None, depth: int = 0) -> bool:
        """Add a ground atom. Returns True if it was not already present."""
        is_new = atom not in self.atoms
        if is_new:
            self.atoms.add(atom)
            self.depth[atom] = depth
            self._index(atom)
        if justification is not None:
            self.justifications.setdefault(atom, []).append(justification)
        return is_new

    def _index(self, atom: Atom) -> None:
        self.index.add(atom)

    # -- lookup ----------------------------------------------------------

    def __contains__(self, atom: Atom) -> bool:
        return atom in self.atoms

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self) -> Iterator[Atom]:
        return iter(sorted(self.atoms, key=str))

    def by_predicate(self, predicate: str, arity: Optional[int] = None) -> list[Atom]:
        """All atoms for a predicate, optionally restricted to one arity."""
        if arity is not None:
            return list(self.index.by_signature.get((predicate, arity), ()))
        found: list[Atom] = []
        for (name, _), atoms in self.index.by_signature.items():
            if name == predicate:
                found.extend(atoms)
        return found

    def candidates(self, pattern: Atom) -> list[Atom]:
        """Atoms that could match ``pattern``, using the most selective index."""
        return self.index.candidates(pattern)

    def query(self, pattern: Atom) -> list[tuple[Atom, dict[str, Const]]]:
        """Every atom matching ``pattern``, with the variable bindings used."""
        results = []
        for candidate in self.candidates(pattern):
            bindings = match_atom(pattern, candidate, {})
            if bindings is not None:
                results.append((candidate, bindings))
        results.sort(key=lambda pair: str(pair[0]))
        return results

    def holds(self, pattern: Atom) -> bool:
        """True if any atom matches ``pattern`` (closed-world otherwise)."""
        if pattern.is_ground:
            return pattern in self.atoms
        return bool(self.query(pattern))

    # -- provenance ------------------------------------------------------

    def is_input(self, atom: Atom) -> bool:
        """True if the atom was given rather than derived."""
        return any(not j.support and j.rule.is_fact for j in self.justifications.get(atom, ()))

    def best_justification(self, atom: Atom) -> Optional[Justification]:
        """The shallowest justification whose supports are all strictly shallower.

        Restricting to strictly-decreasing depth is what keeps proof trees
        finite when a predicate is recursive: ``ancestor`` can support
        ``ancestor``, but never the instance that produced it.
        """
        own_depth = self.depth.get(atom, 0)
        usable = [
            j
            for j in self.justifications.get(atom, ())
            if all(self.depth.get(s, 0) < own_depth for s in j.support)
        ]
        if not usable:
            return None
        return min(usable, key=lambda j: (len(j.support), j.depth))

    def alternatives(self, atom: Atom) -> list[Justification]:
        """All recorded justifications for an atom, shallowest first."""
        return sorted(self.justifications.get(atom, ()), key=lambda j: (j.depth, len(j.support)))

    @property
    def contradictions(self) -> list[Atom]:
        """Atoms derived alongside their own strong negation.

        ``p(x)`` and ``-p(x)`` together say the world both is and is not a
        certain way. clingo rejects such a program outright; the Python engine
        reports it, because knowing *which* pair clashed is the useful part.
        The positive atom of each pair is returned, once.
        """
        found = [
            atom
            for atom in self.atoms
            if not atom.is_negated and atom.complement() in self.atoms
        ]
        return sorted(found, key=str)

    @property
    def coherent(self) -> bool:
        """True if nothing was derived together with its own strong negation."""
        return not self.contradictions

    @property
    def consistent(self) -> bool:
        """True if no integrity constraint was violated and nothing contradicts."""
        return not self.violations and self.coherent

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for atom in self.atoms:
            counts[atom.predicate] = counts.get(atom.predicate, 0) + 1
        return {
            "atoms": len(self.atoms),
            "predicates": dict(sorted(counts.items())),
            "iterations": self.iterations,
            "violations": len(self.violations),
            "contradictions": [str(a) for a in self.contradictions],
            "consistent": self.consistent,
            "truncated": self.truncated,
        }


def match_atom(
    pattern: Atom, fact: Atom, bindings: Substitution
) -> Optional[dict[str, Const]]:
    """One-way match of a (partially ground) pattern against a ground fact.

    Returns the extended bindings, or ``None`` if the two cannot match. This is
    matching rather than full unification: ``fact`` is always ground, which is
    all forward chaining ever needs.
    """
    if pattern.predicate != fact.predicate or len(pattern.args) != len(fact.args):
        return None
    result = dict(bindings)
    for pattern_arg, fact_arg in zip(pattern.args, fact.args):
        if isinstance(pattern_arg, Var):
            existing = result.get(pattern_arg.name)
            if existing is None:
                result[pattern_arg.name] = fact_arg
            elif existing != fact_arg:
                return None
        elif isinstance(pattern_arg, Const):
            if pattern_arg != fact_arg:
                return None
        else:
            raise ValueError(
                f"cannot match unevaluated expression '{pattern_arg}' in {pattern}; "
                "its variables must be bound by an earlier body literal"
            )
    return result
