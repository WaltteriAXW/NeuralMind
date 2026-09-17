"""Predicate invention: naming the conjunction that keeps coming back.

A knowledge base that grows by induction grows sideways. Each learned rule is
reasonable on its own and the set of them repeats itself, because the learner
has no way to say "this combination is a thing" -- it can only spell the
combination out again. Four rules in the access policy each begin

    request(Q, E, Action, Res), classification(Res, Class)

and nothing in the language notices.

Consolidation notices. It finds conjunctions that recur across rules, proposes
a predicate for each, and rewrites the rules to use it. Three things make that
safe rather than merely tidy:

**It is a rewriting, not a claim.** The invented predicate is defined as
exactly the conjunction it replaces, so the model cannot change.
:meth:`Consolidator.verify` checks that rather than trusting it, because "this
should be equivalent" is how equivalence-preserving transformations stop
preserving equivalence.

**Variables that escape are exported.** A variable used outside the conjunction
becomes an argument of the new predicate; one used only inside is existentially
quantified away. Getting that backwards is the way to silently change what a
rule means, so the split is computed rather than guessed.

**The name is the user's.** A machine-generated name (``concept_3``) is a
placeholder and is labelled as one. What the conjunction *means* is the part a
person knows and this does not, and pretending otherwise produces a knowledge
base nobody can read.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.program import Program, Rule
from ..core.terms import Atom, Compare, Literal, Term, Var, variables_in

__all__ = ["Consolidator", "Invention"]


@dataclass
class Invention:
    """A conjunction worth naming, and the rewrite that uses it."""

    name: str
    #: ``name(Exported...) :- conjunction.``
    definition: Rule
    #: ``(original rule, rewritten rule)`` for every rule that used it.
    rewrites: list[tuple[Rule, Rule]] = field(default_factory=list)
    #: How many rules contained the conjunction.
    occurrences: int = 0
    #: Literals saved across the program by the rewrite.
    saved: int = 0
    #: True while the name is a placeholder nobody has approved.
    provisional: bool = True

    def rename(self, name: str) -> "Invention":
        """Give the concept a real name. The point of a person being here."""
        mapping = {self.name: name}
        self.definition = _rename_head(self.definition, name)
        self.rewrites = [
            (old, _rename_body_predicate(new, self.name, name))
            for old, new in self.rewrites
        ]
        self.name = name
        self.provisional = False
        return self

    def describe(self) -> str:
        marker = " (placeholder name)" if self.provisional else ""
        return (
            f"{self.definition}{marker}\n"
            f"  used by {self.occurrences} rule(s), saving {self.saved} literal(s)"
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "definition": str(self.definition),
            "occurrences": self.occurrences,
            "saved": self.saved,
            "provisional": self.provisional,
            "rewrites": [[str(old), str(new)] for old, new in self.rewrites],
        }

    def __str__(self) -> str:
        return self.describe()


class Consolidator:
    """Finds repeated conjunctions and proposes predicates for them."""

    def __init__(
        self,
        program,
        min_occurrences: int = 2,
        min_size: int = 2,
        max_size: int = 3,
        prefix: str = "concept",
    ) -> None:
        self.program = (
            program.program(check=False) if hasattr(program, "program") else program
        )
        self.min_occurrences = min_occurrences
        self.min_size = min_size
        self.max_size = max_size
        self.prefix = prefix

    # -- finding ------------------------------------------------------------

    def propose(self, limit: int = 5) -> list[Invention]:
        """Conjunctions worth naming, most saved first."""
        groups: dict[tuple, list[tuple[Rule, tuple[Literal, ...]]]] = {}
        for rule in self.program.derivation_rules:
            # Negated literals count. "not cleared(E, Res)" is as much a part
            # of a recurring condition as a positive one, and a conjunction
            # that only ever appears alongside the same negation is exactly
            # the kind of thing worth a name.
            parts = [part for part in rule.body if isinstance(part, Literal)]
            for size in range(self.min_size, min(self.max_size, len(parts)) + 1):
                for combination in itertools.combinations(parts, size):
                    key = _shape(combination)
                    if key is None:
                        continue
                    groups.setdefault(key, []).append((rule, combination))

        found: list[Invention] = []
        used_names: set[str] = set()
        for key, members in groups.items():
            rules = {id(rule): rule for rule, _ in members}
            if len(rules) < self.min_occurrences:
                continue
            invention = self._build(key, members, used_names)
            if invention is not None:
                found.append(invention)
                used_names.add(invention.name)
        found.sort(key=lambda i: (-i.saved, -i.occurrences, i.name))
        return found[:limit]

    def _build(self, key, members, used_names: set[str]) -> Optional[Invention]:
        # One occurrence per rule: naming the same conjunction twice inside one
        # rule is a different transformation and not this one.
        seen: dict[int, tuple] = {}
        for rule, combination in members:
            seen.setdefault(id(rule), (rule, combination))
        occurrences = list(seen.values())
        if len(occurrences) < self.min_occurrences:
            return None

        name = self._name(used_names)
        first_rule, first_body = occurrences[0]
        exported = _exported(first_rule, first_body)
        if not exported:
            # A conjunction sharing nothing with the rest of its rule is a
            # standalone condition; naming it is legal but never a saving.
            return None

        definition = Rule(
            head=Atom(name, tuple(exported)),
            body=tuple(first_body),
            source="consolidation",
            label="invented by consolidation",
        )
        if not _safe(definition):
            return None

        rewrites: list[tuple[Rule, Rule]] = []
        saved = 0
        for rule, combination in occurrences:
            substitution = _align(first_body, combination)
            if substitution is None:
                continue
            arguments = tuple(substitution.get(_key(v), v) for v in exported)
            remaining = [part for part in rule.body if part not in combination]
            rewritten = Rule(
                head=rule.head,
                body=tuple(remaining) + (Literal(Atom(name, arguments)),),
                source=rule.source,
                line=rule.line,
                label=rule.label,
            )
            if not _safe(rewritten):
                continue
            rewrites.append((rule, rewritten))
            saved += len(combination) - 1
        if len(rewrites) < self.min_occurrences:
            return None
        return Invention(
            name=name,
            definition=definition,
            rewrites=rewrites,
            occurrences=len(rewrites),
            saved=saved,
        )

    def _name(self, used: set[str]) -> str:
        index = 1
        existing = {
            rule.head.predicate for rule in self.program.rules if rule.head is not None
        } | used
        while f"{self.prefix}_{index}" in existing:
            index += 1
        return f"{self.prefix}_{index}"

    # -- applying -----------------------------------------------------------

    def apply(self, invention: Invention) -> Program:
        """A new program with the conjunction replaced. The original is untouched."""
        replaced = {id(old): new for old, new in invention.rewrites}
        rules = [replaced.get(id(rule), rule) for rule in self.program.rules]
        rules.append(invention.definition)
        return Program(
            rules=rules,
            shown=set(self.program.shown),
            constants=dict(self.program.constants),
            raw_asp=list(self.program.raw_asp),
            requires_asp=self.program.requires_asp,
            open_world=self.program.open_world,
            open_predicates=set(self.program.open_predicates),
            closed_predicates=set(self.program.closed_predicates),
        )

    def verify(self, rewritten: Program) -> tuple[bool, str]:
        """Check that the rewrite changed nothing anyone can observe.

        Every predicate that existed before must derive exactly what it did.
        The invented predicate is new and is allowed to be there -- that is the
        one difference a correct consolidation makes.
        """
        from ..inference.forward import ForwardChainer

        try:
            before = ForwardChainer(self.program).run()
            after = ForwardChainer(rewritten).run()
        except Exception as exc:
            return False, f"the rewritten program did not run: {exc}"
        known = {atom.predicate for atom in before.atoms}
        shrunk = {a for a in after.atoms if a.predicate in known}
        if shrunk != before.atoms:
            lost = sorted(str(a) for a in before.atoms - shrunk)[:3]
            gained = sorted(str(a) for a in shrunk - before.atoms)[:3]
            return False, f"answers changed; lost {lost}, gained {gained}"
        return True, ""


# -- shapes and variables --------------------------------------------------


def _shape(combination: Sequence[Literal]) -> Optional[tuple]:
    """A conjunction's form up to variable renaming, or None if it is ground.

    Two conjunctions are the same shape when one becomes the other by renaming
    variables consistently. A fully ground conjunction has no shape worth
    sharing -- it names one situation, not a pattern.
    """
    mapping: dict[str, str] = {}
    shape = []
    for literal in sorted(combination, key=str):
        arguments = [literal.negated]
        for argument in literal.atom.args:
            if isinstance(argument, Var):
                mapping.setdefault(argument.name, f"V{len(mapping)}")
                arguments.append(mapping[argument.name])
            else:
                arguments.append(str(argument))
        shape.append((literal.atom.predicate, tuple(arguments)))
    if not mapping:
        return None
    return tuple(shape)


def _align(reference: Sequence[Literal], other: Sequence[Literal]) -> Optional[dict]:
    """Map the reference conjunction's variables onto another occurrence's."""
    left = sorted(reference, key=str)
    right = sorted(other, key=str)
    if len(left) != len(right):
        return None
    substitution: dict[str, Term] = {}
    for a, b in zip(left, right):
        if a.atom.predicate != b.atom.predicate or a.atom.arity != b.atom.arity:
            return None
        for x, y in zip(a.atom.args, b.atom.args):
            if isinstance(x, Var):
                if substitution.setdefault(x.name, y) != y:
                    return None
            elif x != y:
                return None
    return substitution


def _exported(rule: Rule, combination: Sequence[Literal]) -> list[Var]:
    """Variables the conjunction shares with the rest of its rule.

    These have to be arguments of the invented predicate. A variable used only
    inside the conjunction is existentially quantified and is projected away --
    getting that the wrong way round silently changes what the rule means.
    """
    inside: list[str] = []
    for literal in combination:
        for name in variables_in(literal.atom):
            if name not in inside:
                inside.append(name)
    outside: set[str] = set()
    if rule.head is not None:
        outside |= set(variables_in(rule.head))
    for part in rule.body:
        if part in combination:
            continue
        outside |= set(variables_in(part.atom if isinstance(part, Literal) else part))
    return [Var(name) for name in inside if name in outside]


def _safe(rule: Rule) -> bool:
    try:
        rule.check_safety()
    except Exception:
        return False
    return True


def _key(term: Term) -> str:
    return term.name if isinstance(term, Var) else str(term)


def _rename_head(rule: Rule, name: str) -> Rule:
    assert rule.head is not None
    return Rule(
        head=Atom(name, rule.head.args),
        body=rule.body,
        source=rule.source,
        line=rule.line,
        label=rule.label,
    )


def _rename_body_predicate(rule: Rule, old: str, new: str) -> Rule:
    body = tuple(
        Literal(Atom(new, part.atom.args), part.negated)
        if isinstance(part, Literal) and part.atom.predicate == old
        else part
        for part in rule.body
    )
    return Rule(rule.head, body, rule.source, rule.line, rule.label)
