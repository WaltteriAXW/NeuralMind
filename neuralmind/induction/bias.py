"""The language bias: which rules the learner is allowed to consider.

Inductive logic programming without a bias is a search over an infinite space.
The bias is what makes it finite -- which predicates may appear in a body, how
many variables are available, how long a clause may get. Every ILP system has
one; being explicit about it is the difference between a search that terminates
and one that does not.

Choosing it is a real modelling decision, not a tuning knob. A bias that is too
tight cannot express the target rule and the learner will return nothing; one
that is too loose makes the search enormous. :meth:`LanguageBias.space_size`
reports how big the space is before you wait on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from typing import Iterable, Optional, Sequence, Union

from ..core.terms import Atom

__all__ = ["Signature", "LanguageBias"]


@dataclass(frozen=True, order=True)
class Signature:
    """A predicate name with its arity, e.g. ``parent/2``."""

    name: str
    arity: int

    @classmethod
    def parse(cls, text: Union[str, "Signature", tuple]) -> "Signature":
        """Accept ``"parent/2"``, ``("parent", 2)`` or a Signature."""
        if isinstance(text, Signature):
            return text
        if isinstance(text, tuple):
            return cls(str(text[0]), int(text[1]))
        name, _, arity = str(text).partition("/")
        if not arity.isdigit():
            raise ValueError(f"expected 'name/arity', got {text!r}")
        return cls(name.strip(), int(arity))

    def __str__(self) -> str:
        return f"{self.name}/{self.arity}"


@dataclass
class LanguageBias:
    """Bounds on the clauses the learner may build.

    Parameters
    ----------
    target:
        The predicate being learned. It is the head of every candidate clause.
    body_predicates:
        What may appear in a body. The target itself is added automatically
        when ``allow_recursion`` is set.
    max_variables:
        Size of the variable pool. The head consumes ``target.arity`` of them,
        so the rest are available as intermediates -- ``grandparent`` needs one,
        so a pool of three suffices for it.
    max_body:
        Longest body to consider.
    max_clauses:
        How many clauses the hypothesis may contain. Learned one at a time by
        sequential covering, so this is a ceiling rather than a target.
    allow_recursion:
        Whether the target may appear in a body. Needed for ``ancestor``;
        turn it off when the target is known to be non-recursive, which shrinks
        the space considerably.
    allow_negation:
        Whether body literals may be negated. The target is never negated even
        when this is on, which keeps every hypothesis stratified.
    require_connected:
        Drop clauses whose body has a literal sharing no variable with the rest.
        Such clauses are almost never what is wanted and there are a great many
        of them, so this is on by default.
    """

    target: Signature
    body_predicates: tuple[Signature, ...] = ()
    max_variables: int = 4
    max_body: int = 3
    max_clauses: int = 4
    allow_recursion: bool = True
    allow_negation: bool = False
    #: Whether ``X != Y`` may appear in a body. Most relational definitions
    #: need it -- a sibling is a child of the same parent who is *not the same
    #: person* -- and without it they cannot be expressed at all.
    allow_comparison: bool = False
    require_connected: bool = True
    #: Predicates dropped by :meth:`from_knowledge` to keep the space finite.
    omitted_predicates: tuple = ()

    def __post_init__(self) -> None:
        self.target = Signature.parse(self.target)
        self.body_predicates = tuple(
            Signature.parse(p) for p in self.body_predicates
        )
        if self.max_variables < self.target.arity:
            raise ValueError(
                f"max_variables ({self.max_variables}) must be at least the "
                f"target's arity ({self.target.arity}); the head needs one "
                "variable per argument"
            )
        if self.max_body < 1:
            raise ValueError("max_body must be at least 1")
        if self.max_clauses < 1:
            raise ValueError("max_clauses must be at least 1")

    @classmethod
    def for_target(
        cls,
        target: Union[str, Signature],
        background: Iterable[Union[str, Signature]] = (),
        **kwargs,
    ) -> "LanguageBias":
        """Build a bias from ``'grandparent/2'``-style strings."""
        return cls(
            target=Signature.parse(target),
            body_predicates=tuple(Signature.parse(p) for p in background),
            **kwargs,
        )

    @classmethod
    def from_knowledge(
        cls,
        knowledge,
        target: Union[str, Signature],
        exclude: Iterable[Union[str, Signature]] = (),
        max_predicates: int = 12,
        **kwargs,
    ) -> "LanguageBias":
        """Derive a bias from whatever a knowledge base already contains.

        Everything the knowledge base mentions, apart from the target itself,
        becomes usable in a body. This is what lets a caller say "learn
        ``uncle/2``" without first working out which predicates are relevant --
        the search space is bounded by the domain rather than by the user.

        Wider is not free: the space grows as ``|literals| ** max_body``, so a
        knowledge base with many predicates is capped at ``max_predicates``
        (most frequent first) and the rest reported by
        :attr:`omitted_predicates`. A caller who knows better should still say
        so; this is a sensible default, not a substitute for knowing the domain.
        """
        signature = Signature.parse(target)
        skip = {signature} | {Signature.parse(e) for e in exclude}
        counts: dict[Signature, int] = {}
        for record in getattr(knowledge, "facts", []):
            found = Signature(record.atom.predicate, record.atom.arity)
            counts[found] = counts.get(found, 0) + 1
        rules = getattr(knowledge, "rules", None)
        if rules is not None:
            for rule in rules.rules:
                if rule.head is not None:
                    found = Signature(rule.head.predicate, rule.head.arity)
                    counts.setdefault(found, 0)
        ranked = sorted(counts, key=lambda sig: (-counts[sig], sig))
        usable = [sig for sig in ranked if sig not in skip]
        bias = cls(
            target=signature,
            body_predicates=tuple(usable[:max_predicates]),
            **kwargs,
        )
        bias.omitted_predicates = tuple(usable[max_predicates:])
        return bias

    @property
    def usable_predicates(self) -> tuple[Signature, ...]:
        """Everything that may appear in a body, recursion included."""
        predicates = list(self.body_predicates)
        if self.allow_recursion and self.target not in predicates:
            predicates.append(self.target)
        return tuple(predicates)

    def literal_count(self) -> int:
        """How many distinct body literals the bias admits."""
        total = 0
        for signature in self.usable_predicates:
            total += self.max_variables**signature.arity
        total *= 2 if self.allow_negation else 1
        if self.allow_comparison:
            # One "!=" per unordered pair of variables.
            total += self.max_variables * (self.max_variables - 1) // 2
        return total

    def space_size(self) -> int:
        """Upper bound on the number of clause bodies to consider.

        Before connectivity and safety filtering, and before the search prunes
        anything -- so it overstates the real work, usually by a lot. Use it to
        tell "this will finish" from "this will not".
        """
        literals = self.literal_count()
        return sum(comb(literals, size) for size in range(1, self.max_body + 1))

    def describe(self) -> str:
        predicates = ", ".join(str(p) for p in self.usable_predicates) or "(none)"
        omitted = (
            f"\n  omitted to bound the search: "
            + ", ".join(str(p) for p in self.omitted_predicates)
            if self.omitted_predicates
            else ""
        )
        return (
            f"learning {self.target} from {predicates}\n"
            f"  up to {self.max_body} body literal(s), {self.max_clauses} clause(s), "
            f"{self.max_variables} variables\n"
            f"  recursion {'on' if self.allow_recursion else 'off'}, "
            f"negation {'on' if self.allow_negation else 'off'}, "
            f"comparison {'on' if self.allow_comparison else 'off'}\n"
            f"  {self.literal_count()} candidate literals, "
            f"up to {self.space_size():,} bodies before pruning" + omitted
        )

    def __str__(self) -> str:
        return self.describe()
