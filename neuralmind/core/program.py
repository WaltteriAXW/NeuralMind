"""Rules and programs: the knowledge side of the pipeline.

A :class:`Program` is a set of rules plus the metadata the inference engine
needs to run them safely -- variable safety and a negation stratification.
Both checks happen once, up front, so a malformed knowledge base fails with a
pointed error instead of quietly computing the wrong fixpoint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence, Union

from .terms import Atom, Compare, Const, Literal, Var, format_body, ground_term, variables_in

__all__ = [
    "Rule",
    "Step",
    "Program",
    "SafetyError",
    "StratificationError",
    "GroundingLimit",
    "ProgramError",
]

BodyPart = Union[Literal, Compare]


class ProgramError(Exception):
    """Base class for problems with a logic program."""


class SafetyError(ProgramError):
    """A rule uses a variable that no positive body literal can bind."""


class StratificationError(ProgramError):
    """Negation is recursive, so no unique least model exists."""


class GroundingLimit(ProgramError):
    """Grounding the program would produce more rules than allowed."""


@dataclass(frozen=True)
class Rule:
    """``head :- body.`` -- with ``head=None`` meaning an integrity constraint.

    A rule with an empty body is a fact. A rule with no head is a constraint:
    any model in which the body holds is rejected, and the engine reports the
    offending instantiation instead of silently dropping the model.
    """

    head: Optional[Atom]
    body: tuple[BodyPart, ...] = ()
    source: Optional[str] = None
    line: Optional[int] = None
    label: Optional[str] = None

    def __str__(self) -> str:
        if not self.body:
            return f"{self.head}."
        if self.head is None:
            return f":- {format_body(self.body)}."
        return f"{self.head} :- {format_body(self.body)}."

    @property
    def is_fact(self) -> bool:
        return self.head is not None and not self.body

    @property
    def is_constraint(self) -> bool:
        return self.head is None

    @property
    def positive_literals(self) -> tuple[Literal, ...]:
        return tuple(p for p in self.body if isinstance(p, Literal) and not p.negated)

    @property
    def negative_literals(self) -> tuple[Literal, ...]:
        return tuple(p for p in self.body if isinstance(p, Literal) and p.negated)

    @property
    def comparisons(self) -> tuple[Compare, ...]:
        return tuple(p for p in self.body if isinstance(p, Compare))

    def origin(self) -> str:
        """Human-readable provenance, used in proof trees and error messages."""
        if self.label:
            return self.label
        if self.source and self.line:
            return f"{self.source}:{self.line}"
        return str(self)

    def plan(self) -> list["Step"]:
        """Order the body into executable steps, resolving ``=`` assignments.

        Positive literals are enumerated in source order. Comparisons and
        negative literals are scheduled as soon as their variables are bound,
        and an equality with exactly one unbound side acts as an assignment --
        which is how ``S = A + B`` binds ``S`` in ASP.

        Raises :class:`SafetyError` if any part can never become evaluable.
        """
        bound: set[str] = set()
        steps: list[Step] = []
        pending_cmp = list(self.comparisons)
        pending_neg = list(self.negative_literals)

        def flush() -> None:
            progress = True
            while progress:
                progress = False
                for cmp_ in list(pending_cmp):
                    needed = set(variables_in(cmp_))
                    if needed <= bound:
                        steps.append(Step("filter", compare=cmp_))
                        pending_cmp.remove(cmp_)
                        progress = True
                        continue
                    if cmp_.op != "=":
                        continue
                    for side, other in ((cmp_.left, cmp_.right), (cmp_.right, cmp_.left)):
                        if (
                            isinstance(side, Var)
                            and side.name not in bound
                            and set(variables_in(other)) <= bound
                        ):
                            steps.append(Step("assign", variable=side.name, expression=other))
                            bound.add(side.name)
                            pending_cmp.remove(cmp_)
                            progress = True
                            break
                for lit in list(pending_neg):
                    if set(variables_in(lit)) <= bound:
                        steps.append(Step("absent", literal=lit))
                        pending_neg.remove(lit)
                        progress = True

        flush()
        # Join order. Taking body literals in source order can force a
        # cartesian product: with "parent(P,A), parent(Q,B), sibling(P,Q)"
        # the first two literals share nothing, so every pair of parents is
        # enumerated before the sibling check rejects almost all of them.
        # Picking a literal that shares a variable with what is already bound
        # keeps each step indexed instead. Conjunction is commutative, so this
        # changes only the cost.
        remaining = list(enumerate(self.positive_literals))
        while remaining:
            choice = _best_join(remaining, bound)
            origin, lit = remaining.pop(choice)
            steps.append(Step("match", literal=lit, origin=origin))
            bound.update(variables_in(lit))
            flush()

        if pending_cmp or pending_neg:
            stuck = [str(x) for x in (*pending_cmp, *pending_neg)]
            unbound = sorted(
                {v for x in (*pending_cmp, *pending_neg) for v in variables_in(x)} - bound
            )
            raise SafetyError(
                f"unsafe rule at {self.origin()}: {', '.join(stuck)} can never be "
                f"evaluated because variable(s) {', '.join(unbound)} are never bound "
                f"by a positive body literal or an assignment"
            )
        if self.head is not None:
            missing = set(variables_in(self.head)) - bound
            if missing:
                raise SafetyError(
                    f"unsafe rule at {self.origin()}: head variable(s) "
                    f"{', '.join(sorted(missing))} are not bound by the body"
                )
        return steps

    def check_safety(self) -> None:
        """Raise :class:`SafetyError` unless every variable can be bound."""
        self.plan()


@dataclass(frozen=True)
class Step:
    """One executable step of a rule body, produced by :meth:`Rule.plan`."""

    kind: str  # match | absent | filter | assign
    literal: Optional[Literal] = None
    compare: Optional[Compare] = None
    variable: Optional[str] = None
    expression: Optional[object] = None
    #: Position of this literal in the rule as written. Execution may reorder
    #: the body for speed; proofs are rendered back in source order so what a
    #: reader sees still matches the rule they wrote.
    origin: Optional[int] = None

    def __str__(self) -> str:
        if self.kind == "assign":
            return f"assign {self.variable} := {self.expression}"
        return f"{self.kind} {self.literal or self.compare}"


@dataclass
class Program:
    """A logic program: rules, constraints, and declared output predicates."""

    rules: list[Rule] = field(default_factory=list)
    shown: set[tuple[str, int]] = field(default_factory=set)
    #: ``#const`` declarations, kept so they survive a round-trip to clingo.
    constants: dict = field(default_factory=dict)
    #: Statements kept verbatim because only clingo can evaluate them.
    raw_asp: list[str] = field(default_factory=list)
    #: Names of the features that put statements in :attr:`raw_asp`.
    requires_asp: tuple[str, ...] = ()

    def __len__(self) -> int:
        return len(self.rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.rules)

    def __str__(self) -> str:
        return "\n".join(str(r) for r in self.rules)

    # -- construction ----------------------------------------------------

    def add(self, rule: Rule) -> "Program":
        self.rules.append(rule)
        return self

    def extend(self, rules: Iterable[Rule]) -> "Program":
        self.rules.extend(rules)
        return self

    def merge(self, other: "Program") -> "Program":
        """Return a new program containing the rules of both, in order."""
        merged = Program(
            rules=list(self.rules) + list(other.rules),
            shown=set(self.shown) | set(other.shown),
            constants={**self.constants, **other.constants},
            raw_asp=list(self.raw_asp) + list(other.raw_asp),
            requires_asp=tuple(dict.fromkeys(self.requires_asp + other.requires_asp)),
        )
        return merged

    # -- views -----------------------------------------------------------

    @property
    def facts(self) -> list[Rule]:
        return [r for r in self.rules if r.is_fact]

    @property
    def constraints(self) -> list[Rule]:
        return [r for r in self.rules if r.is_constraint]

    @property
    def derivation_rules(self) -> list[Rule]:
        return [r for r in self.rules if r.head is not None and r.body]

    def predicates(self) -> set[tuple[str, int]]:
        sigs: set[tuple[str, int]] = set()
        for rule in self.rules:
            if rule.head is not None:
                sigs.add(rule.head.signature)
            for part in rule.body:
                if isinstance(part, Literal):
                    sigs.add(part.signature)
        return sigs

    def intensional_predicates(self) -> set[tuple[str, int]]:
        """Predicates that some rule derives (as opposed to plain input facts)."""
        return {r.head.signature for r in self.derivation_rules if r.head is not None}

    # -- static checks ---------------------------------------------------

    def check(self) -> "Program":
        """Run every static check. Returns ``self`` so it can be chained.

        A program that predicate-level stratification rejects may still be
        *locally* stratified, which is equally well defined and is what the
        engine falls back to. Checking only the coarse condition here would
        reject programs the engine can run perfectly well.
        """
        for rule in self.rules:
            rule.check_safety()
        try:
            self.stratify()
        except StratificationError:
            try:
                self.local_strata()
            except GroundingLimit as exc:
                raise StratificationError(
                    "negation is recursive at the predicate level, and the program "
                    f"is too large to ground and check atom by atom ({exc})"
                ) from exc
        return self

    def stratify(self) -> list[list[Rule]]:
        """Split the rules into strata that respect negation.

        Every rule lands in the stratum of its head predicate, and a rule that
        negates ``q`` is guaranteed to run only after ``q`` has reached its
        fixpoint. Recursion through negation has no such ordering and raises
        :class:`StratificationError`.
        """
        level: dict[tuple[str, int], int] = defaultdict(int)
        for sig in self.predicates():
            level[sig] = 0

        limit = len(level) + 1
        for _ in range(limit):
            changed = False
            for rule in self.rules:
                if rule.head is None:
                    continue
                head_sig = rule.head.signature
                for part in rule.body:
                    if not isinstance(part, Literal):
                        continue
                    want = level[part.signature] + (1 if part.negated else 0)
                    if want > level[head_sig]:
                        level[head_sig] = want
                        changed = True
            if not changed:
                break
        else:
            cycle = _negative_cycle(self.rules)
            raise StratificationError(
                "negation is recursive, so this program has no unique least model"
                + (f" (cycle through: {', '.join(cycle)})" if cycle else "")
                + ". Use the clingo backend for answer-set semantics instead."
            )

        max_level = max(level.values(), default=0)
        strata: list[list[Rule]] = [[] for _ in range(max_level + 1)]
        for rule in self.rules:
            if rule.head is None:
                continue  # constraints are checked after the whole fixpoint
            strata[level[rule.head.signature]].append(rule)
        return strata

    # -- local stratification ---------------------------------------------

    def constant_universe(self) -> list:
        """Every constant appearing in the program, in a stable order.

        This is the Herbrand universe used for grounding. Distinct from
        :attr:`constants`, which holds ``#const`` declarations.
        """
        seen: dict = {}
        for rule in self.rules:
            atoms = [rule.head] if rule.head is not None else []
            atoms.extend(part.atom for part in rule.body if isinstance(part, Literal))
            for atom in atoms:
                for argument in atom.args:
                    if isinstance(argument, Const):
                        seen.setdefault((argument.quoted, str(argument.value)), argument)
        return [seen[key] for key in sorted(seen)]

    def ground(self, max_rules: int = 200_000) -> "Program":
        """Instantiate every rule over the program's constants.

        Grounding is exponential in the number of variables per rule, so it is
        guarded by ``max_rules`` and raises :class:`GroundingLimit` rather than
        exhausting memory. This is a fallback path, not the normal one.
        """
        import itertools

        universe = self.constant_universe()
        grounded: list[Rule] = []
        for rule in self.rules:
            names = sorted(_rule_variables(rule))
            if not names:
                grounded.append(rule)
                continue
            if not universe:
                continue  # no constants to instantiate with
            count = len(universe) ** len(names)
            if len(grounded) + count > max_rules:
                raise GroundingLimit(
                    f"grounding {rule.origin()} over {len(universe)} constants and "
                    f"{len(names)} variables needs {count} instances, past the "
                    f"{max_rules} limit"
                )
            for assignment in itertools.product(universe, repeat=len(names)):
                subst = dict(zip(names, assignment))
                head = rule.head.ground(subst) if rule.head is not None else None
                body = tuple(_ground_part(part, subst) for part in rule.body)
                grounded.append(
                    Rule(head, body, source=rule.source, line=rule.line, label=rule.label)
                )
        return Program(
            rules=grounded,
            shown=set(self.shown),
            constants=dict(self.constants),
        )

    def local_strata(self, max_rules: int = 200_000) -> list[list[Rule]]:
        """Stratify the *ground* program, atom by atom.

        A program can be recursive through negation at the predicate level and
        still be perfectly well behaved at the ground level. Real rule bases
        are full of these::

            eat(bald_eagle, squirrel) :- cold(X), not eat(X, bald_eagle).

        ``eat`` depends negatively on ``eat``, so predicate-level
        stratification rejects the theory -- yet no ground atom depends on
        itself, so the program is *locally stratified* and has exactly the same
        kind of unique perfect model. 6-12% of the ProofWriter corpus is like
        this.

        Returns strata of ground rules. Raises :class:`StratificationError` if
        the ground program really does have a negative cycle.
        """
        ground = self.ground(max_rules=max_rules)
        level: dict = defaultdict(int)
        edges: list[tuple] = []
        for rule in ground.rules:
            if rule.head is None:
                continue
            level[rule.head]
            for part in rule.body:
                if isinstance(part, Literal):
                    level[part.atom]
                    edges.append((rule.head, part.atom, part.negated))

        for _ in range(len(level) + 1):
            changed = False
            for head, body_atom, negated in edges:
                want = level[body_atom] + (1 if negated else 0)
                if want > level[head]:
                    level[head] = want
                    changed = True
            if not changed:
                break
        else:
            raise StratificationError(
                "negation is recursive even after grounding, so this program has "
                "no unique model. Use the clingo backend for answer-set semantics."
            )

        strata: list[list[Rule]] = [[] for _ in range(max(level.values(), default=0) + 1)]
        for rule in ground.rules:
            if rule.head is not None:
                strata[level[rule.head]].append(rule)
        return strata

    def stratum_of(self) -> dict[tuple[str, int], int]:
        """Map each predicate to its stratum index."""
        mapping: dict[tuple[str, int], int] = {}
        for index, rules in enumerate(self.stratify()):
            for rule in rules:
                if rule.head is not None:
                    mapping[rule.head.signature] = index
        return mapping

    # -- serialisation ---------------------------------------------------

    def to_asp(self, include_shown: bool = True) -> str:
        """Render the program as clingo-compatible ASP source."""
        lines = [f"#const {name}={value}." for name, value in sorted(self.constants.items())]
        lines.extend(str(rule) for rule in self.rules)
        lines.extend(self.raw_asp)
        if include_shown and self.shown:
            lines.extend(f"#show {name}/{arity}." for name, arity in sorted(self.shown))
        return "\n".join(lines)


def _rule_variables(rule: Rule) -> set:
    """Every variable name in a rule's head and body."""
    names: set = set()
    if rule.head is not None:
        names.update(variables_in(rule.head))
    for part in rule.body:
        names.update(variables_in(part))
    return names


def _ground_part(part, subst: dict):
    """Substitute into one body part, keeping its kind."""
    if isinstance(part, Literal):
        return Literal(part.atom.ground(subst), negated=part.negated)
    return Compare(part.op, ground_term(part.left, subst), ground_term(part.right, subst))


def _best_join(remaining: Sequence[tuple], bound: set) -> int:
    """Index into ``remaining`` of the literal to match next.

    Prefers a literal already sharing a variable with something bound, since
    that one can be looked up by index rather than scanned. Ties, and the very
    first literal, go to whichever has the most constants, then to source
    order -- so a rule with no join structure behaves exactly as written.
    """
    best_index = 0
    best_score: Optional[tuple] = None
    for position, (origin, literal) in enumerate(remaining):
        variables = set(variables_in(literal))
        shares = bool(variables & bound)
        constants = sum(
            1 for argument in literal.atom.args if not isinstance(argument, Var)
        )
        score = (shares, constants, -origin)
        if best_score is None or score > best_score:
            best_score = score
            best_index = position
    return best_index


def _negative_cycle(rules: Sequence[Rule]) -> list[str]:
    """Best-effort: name the predicates involved in a recursive negation."""
    edges: dict[tuple[str, int], set[tuple[tuple[str, int], bool]]] = defaultdict(set)
    for rule in rules:
        if rule.head is None:
            continue
        for part in rule.body:
            if isinstance(part, Literal):
                edges[rule.head.signature].add((part.signature, part.negated))

    seen: set[tuple[str, int]] = set()
    path: list[tuple[str, int]] = []

    def visit(node: tuple[str, int], used_negation: bool) -> Optional[list[str]]:
        if node in path:
            if used_negation:
                start = path.index(node)
                return [f"{n}/{a}" for n, a in path[start:]]
            return None
        if node in seen and not used_negation:
            return None
        seen.add(node)
        path.append(node)
        for target, negated in edges.get(node, ()):
            found = visit(target, used_negation or negated)
            if found:
                return found
        path.pop()
        return None

    for sig in list(edges):
        found = visit(sig, False)
        if found:
            return found
        path.clear()
    return []
