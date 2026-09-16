"""Learning rules from examples.

This is the other half of the knowledge acquisition bottleneck. The perception
layer can now learn from what the rules entail
(:mod:`neuralmind.learning`); this module learns the rules themselves, from
examples of the relation they are supposed to define.

The method is generate-test-constrain, the loop at the heart of modern ILP
(Cropper & Morel, "Learning programs by learning from failures", *Machine
Learning* 110(4), 2021):

**generate**
    Enumerate clauses inside the language bias, shortest first, so the simplest
    rule that works is the one found.

**test**
    Run the actual inference engine on background knowledge plus the candidate
    and see which examples come out. The same engine that will run the learned
    rules judges them, so a rule that passes here behaves identically in use.

**constrain**
    A body that covers no positive example cannot start covering one when more
    literals are added -- a conjunction only ever narrows. So every superset of
    it is pruned unseen, which is what makes the search finish.

Multi-clause definitions are built by sequential covering: learn a clause, set
aside the examples it explains, repeat. Already-learned clauses stay in the
program while later ones are tested, which is what lets a recursive definition
bootstrap -- ``ancestor``'s base case is found first, and the recursive clause
is then able to fire on top of it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from ..core.parser import parse_atom
from ..core.program import Program, Rule
from ..core.terms import Atom, Const
from ..inference.forward import ForwardChainer
from ..inference.model import Model
from ..inference.proof import ProofNode, explain
from ..knowledge.base import KnowledgeBase
from .bias import LanguageBias, Signature
from .enumerate import (
    candidate_bodies, candidate_comparisons, candidate_literals, head_atom,
)

__all__ = ["Examples", "Hypothesis", "RuleLearner"]


@dataclass
class Examples:
    """What the target relation should and should not contain."""

    positive: list[Atom] = field(default_factory=list)
    negative: list[Atom] = field(default_factory=list)

    @classmethod
    def from_strings(
        cls, positive: Iterable[str], negative: Iterable[str] = ()
    ) -> "Examples":
        return cls(
            positive=[parse_atom(text) for text in positive],
            negative=[parse_atom(text) for text in negative],
        )

    @classmethod
    def closed_world(
        cls,
        positive: Iterable[Union[Atom, str]],
        constants: Iterable[Union[Const, str]],
        target: Optional[Signature] = None,
    ) -> "Examples":
        """Positives given; everything else over ``constants`` is negative.

        The closed-world assumption the rest of the system already makes,
        applied to example generation. Convenient, and honest only when the
        positives really are complete -- a missing positive becomes a negative
        the learner will work hard to exclude.
        """
        import itertools

        positives = [parse_atom(p) if isinstance(p, str) else p for p in positive]
        if not positives and target is None:
            raise ValueError("closed_world() needs either examples or a target")
        signature = target or Signature(positives[0].predicate, positives[0].arity)
        pool = [Const(c) if not isinstance(c, Const) else c for c in constants]
        known = {p for p in positives}
        negatives = [
            atom
            for arguments in itertools.product(pool, repeat=signature.arity)
            if (atom := Atom(signature.name, arguments)) not in known
        ]
        return cls(positive=positives, negative=negatives)

    def __len__(self) -> int:
        return len(self.positive) + len(self.negative)

    def summary(self) -> str:
        return f"{len(self.positive)} positive, {len(self.negative)} negative"


@dataclass
class Hypothesis:
    """A learned definition, with the evidence for how well it does."""

    rules: list[Rule] = field(default_factory=list)
    covered_positive: frozenset = frozenset()
    covered_negative: frozenset = frozenset()
    total_positive: int = 0
    total_negative: int = 0
    candidates_evaluated: int = 0
    seconds: float = 0.0
    #: Set when the search stopped without explaining every positive.
    incomplete_reason: Optional[str] = None
    #: Positives the background alone already entailed, before any learning.
    baseline_positive: int = 0
    #: Rules that fit the examples exactly as well as the ones chosen, keyed by
    #: the position of the clause they compete with.
    ties: dict = field(default_factory=dict)
    #: Rules removed from the background because they defined the target.
    stripped_from_background: tuple = ()

    @property
    def complete(self) -> bool:
        """True if every positive example follows from the hypothesis."""
        return len(self.covered_positive) == self.total_positive

    @property
    def consistent(self) -> bool:
        """True if no negative example follows from it."""
        return not self.covered_negative

    @property
    def correct(self) -> bool:
        return self.complete and self.consistent

    @property
    def underdetermined(self) -> bool:
        """True if the examples do not single out one rule.

        When several clauses cover exactly the same examples, picking one is an
        arbitrary choice, not a conclusion. Saying so is the difference between
        learning from the data and reading a preference into it.
        """
        return any(self.ties.values())

    @property
    def leaked(self) -> bool:
        """True if the background alone already entailed some positive example.

        Then the hypothesis is credited with coverage it did not earn, and the
        result means nothing. Almost always this is the target's own definition
        left in the background by accident.
        """
        return self.baseline_positive > 0

    @property
    def precision(self) -> float:
        covered = len(self.covered_positive) + len(self.covered_negative)
        return len(self.covered_positive) / covered if covered else 0.0

    @property
    def recall(self) -> float:
        return len(self.covered_positive) / self.total_positive if self.total_positive else 0.0

    def to_asp(self) -> str:
        return "\n".join(str(rule) for rule in self.rules)

    def describe(self) -> str:
        if not self.rules:
            lines = ["no rule found"]
        else:
            lines = ["learned:"]
            lines.extend(f"  {rule}" for rule in self.rules)
        lines.append(
            f"covers {len(self.covered_positive)}/{self.total_positive} positive "
            f"and {len(self.covered_negative)}/{self.total_negative} negative examples"
        )
        if self.leaked:
            lines.append(
                f"WARNING: the background alone already entailed "
                f"{self.baseline_positive}/{self.total_positive} positive example(s), "
                "so this coverage is not evidence the learned rules work. Remove the "
                "target's definition from the background (the learner does this "
                "automatically with strip_target=True)."
            )
        if self.stripped_from_background:
            lines.append(
                f"removed {len(self.stripped_from_background)} background rule(s) "
                f"defining the target before learning"
            )
        verdict = (
            "complete and consistent"
            if self.correct
            else ", ".join(
                filter(
                    None,
                    [
                        None if self.complete else "incomplete",
                        None if self.consistent else "inconsistent",
                    ],
                )
            )
        )
        lines.append(f"verdict: {verdict}")
        if self.underdetermined:
            lines.append(
                "the examples do not single out one rule -- these fit exactly as well:"
            )
            for index, alternatives in sorted(self.ties.items()):
                for rule in alternatives:
                    lines.append(f"  (instead of clause {index + 1})  {rule}")
            lines.append(
                "  more examples would decide between them; choosing one here "
                "would be a guess, not a conclusion"
            )
        if self.incomplete_reason:
            lines.append(f"stopped because: {self.incomplete_reason}")
        lines.append(
            f"{self.candidates_evaluated} candidate clause(s) tested in {self.seconds:.2f}s"
        )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "rules": [str(rule) for rule in self.rules],
            "covered_positive": len(self.covered_positive),
            "total_positive": self.total_positive,
            "covered_negative": len(self.covered_negative),
            "total_negative": self.total_negative,
            "complete": self.complete,
            "consistent": self.consistent,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "candidates_evaluated": self.candidates_evaluated,
            "seconds": round(self.seconds, 2),
            "incomplete_reason": self.incomplete_reason,
            "baseline_positive": self.baseline_positive,
            "leaked": self.leaked,
            "underdetermined": self.underdetermined,
            "ties": {
                str(index): [str(rule) for rule in rules]
                for index, rules in sorted(self.ties.items())
            },
            "stripped_from_background": [str(r) for r in self.stripped_from_background],
        }

    def __str__(self) -> str:
        return self.describe()


class RuleLearner:
    """Learns a definition of one predicate from positive and negative examples.

    Parameters
    ----------
    background:
        Facts and any rules already known. The learned clauses are added to
        this when candidates are tested, so learned rules can build on existing
        ones.
    bias:
        The hypothesis space. See :class:`~neuralmind.induction.bias.LanguageBias`.
    examples:
        What the target relation contains and does not.
    max_atoms:
        A candidate that derives more atoms than this is discarded as
        pathological rather than allowed to hang the search.
    budget:
        Stop after this many candidate evaluations and report what was found.
    strip_target:
        Remove anything in the background that already defines the target --
        rules and facts alike. On by default, because a definition left in the
        background entails the examples on its own and the learner would be
        credited with coverage it did not earn. Turn it off only when the
        target genuinely has a partial definition that the new clauses are
        meant to extend.
    """

    def __init__(
        self,
        background: Union[KnowledgeBase, Program],
        bias: LanguageBias,
        examples: Examples,
        max_atoms: int = 50_000,
        budget: int = 200_000,
        strip_target: bool = True,
    ) -> None:
        self.bias = bias
        self.examples = examples
        self.max_atoms = max_atoms
        self.budget = budget
        source = (
            background.program(check=False)
            if isinstance(background, KnowledgeBase)
            else background
        )
        self.stripped: tuple = ()
        if strip_target:
            signature = (bias.target.name, bias.target.arity)
            removed = [
                rule
                for rule in source.rules
                if rule.head is not None and rule.head.signature == signature
            ]
            if removed:
                source = Program(
                    rules=[rule for rule in source.rules if rule not in removed],
                    shown=set(source.shown),
                    constants=dict(source.constants),
                    raw_asp=list(source.raw_asp),
                )
                self.stripped = tuple(removed)
        self.background = source
        self._cache: dict[tuple, Optional[tuple]] = {}
        self._evaluated = 0
        self._literals = candidate_literals(bias) + candidate_comparisons(bias)

    # -- the search --------------------------------------------------------

    def learn(self, verbose: bool = False) -> Hypothesis:
        """Search for a definition, one clause at a time."""
        started = time.time()
        hypothesis_ties: dict[int, list[Rule]] = {}
        baseline = self._evaluate([])
        baseline_positive = len(baseline[0]) if baseline else 0
        if baseline_positive and verbose:
            print(
                f"  warning: the background alone already entails "
                f"{baseline_positive} positive example(s)"
            )
        learned: list[Rule] = []
        remaining = set(self.examples.positive)
        reason: Optional[str] = None

        for _ in range(self.bias.max_clauses):
            if not remaining:
                break
            found = self._find_clause(learned, remaining, verbose=verbose)
            if found is None:
                reason = (
                    "no further clause covers an unexplained example without "
                    "also covering a negative one"
                    if self._evaluated < self.budget
                    else f"evaluation budget of {self.budget} candidates was spent"
                )
                break
            clause, newly_covered, tied = found
            if tied:
                hypothesis_ties[len(learned)] = tied
            learned.append(clause)
            remaining -= newly_covered
            if verbose:
                print(f"  + {clause}   (+{len(newly_covered)} examples)")
        else:
            if remaining:
                reason = f"reached the {self.bias.max_clauses}-clause limit"

        covered = self._evaluate(learned)
        positive, negative = covered if covered else (frozenset(), frozenset())
        return Hypothesis(
            rules=learned,
            covered_positive=positive,
            covered_negative=negative,
            total_positive=len(self.examples.positive),
            total_negative=len(self.examples.negative),
            candidates_evaluated=self._evaluated,
            seconds=time.time() - started,
            incomplete_reason=reason if remaining else None,
            baseline_positive=baseline_positive,
            stripped_from_background=self.stripped,
            ties=hypothesis_ties,
        )

    def _find_clause(
        self, learned: Sequence[Rule], remaining: set, verbose: bool = False
    ) -> Optional[tuple[Rule, frozenset, list[Rule]]]:
        """Best clause explaining some unexplained example and no negative.

        Returns the chosen clause, what it covers, and any clauses that cover
        exactly the same examples -- which the caller reports rather than
        quietly discarding.
        """
        already = self._evaluate(learned)
        already_positive = already[0] if already else frozenset()

        # Bodies that covered nothing. Every superset of one of these covers
        # nothing either, because adding a literal only narrows a conjunction.
        useless: list[frozenset] = []

        for size in range(1, self.bias.max_body + 1):
            best: Optional[tuple[int, str, Rule, frozenset]] = None
            equal: dict[frozenset, list[Rule]] = {}
            for rule, body, _ in candidate_bodies(self.bias, size, self._literals):
                if self._evaluated >= self.budget:
                    break
                body_set = frozenset(str(part) for part in body)
                if any(pruned <= body_set for pruned in useless):
                    continue
                result = self._evaluate([*learned, rule])
                if result is None:
                    continue  # pathological candidate, discarded
                positive, negative = result
                gained = (positive - already_positive) & remaining
                if not gained:
                    useless.append(body_set)
                    continue
                if negative:
                    continue  # covers something it must not
                equal.setdefault(gained, []).append(rule)
                score = (len(gained), str(rule))
                if best is None or score > (best[0], best[1]):
                    best = (len(gained), str(rule), rule, gained)
            if best is not None:
                # Anything covering the same examples is an equally good answer.
                tied = [r for r in equal.get(best[3], []) if r is not best[2]]
                return best[2], best[3], tied
            if self._evaluated >= self.budget:
                break
        return None

    # -- testing candidates -------------------------------------------------

    def _evaluate(self, clauses: Sequence[Rule]) -> Optional[tuple[frozenset, frozenset]]:
        """Which examples this program derives. None if it is unusable.

        Judged by the same engine that will run the rules in production, so a
        hypothesis cannot pass here and behave differently later.
        """
        key = tuple(sorted(str(clause) for clause in clauses))
        if key in self._cache:
            return self._cache[key]
        self._evaluated += 1

        program = Program(
            rules=list(self.background.rules) + list(clauses),
            shown=set(self.background.shown),
            constants=dict(self.background.constants),
        )
        try:
            model = ForwardChainer(
                program, max_atoms=self.max_atoms, max_justifications=1, strict=False
            ).run()
        except Exception:
            # An unsafe or unstratified candidate is simply not a hypothesis.
            self._cache[key] = None
            return None
        if model.truncated:
            self._cache[key] = None
            return None

        result = (
            frozenset(atom for atom in self.examples.positive if atom in model),
            frozenset(atom for atom in self.examples.negative if atom in model),
        )
        self._cache[key] = result
        return result

    # -- using the result ---------------------------------------------------

    def model_for(self, hypothesis: Hypothesis) -> Model:
        """The model of background knowledge plus a learned hypothesis."""
        program = Program(rules=list(self.background.rules) + list(hypothesis.rules))
        return ForwardChainer(program).run()

    def explain(self, hypothesis: Hypothesis, example: Union[Atom, str]) -> ProofNode:
        """Proof of one example under the learned rules.

        A learned rule is a rule like any other, so everything the rest of the
        system does with rules -- proofs, constraint checking, cross-checking
        against clingo -- applies to it unchanged.
        """
        atom = parse_atom(example) if isinstance(example, str) else example
        return explain(self.model_for(hypothesis), atom)


def learn_rules(
    background: Union[KnowledgeBase, Program],
    target: Union[str, Signature],
    positive: Iterable[Union[Atom, str]],
    negative: Iterable[Union[Atom, str]] = (),
    body_predicates: Iterable[Union[str, Signature]] = (),
    **bias_options,
) -> Hypothesis:
    """Convenience wrapper: build the bias and learner, return the hypothesis."""
    bias = LanguageBias.for_target(target, body_predicates, **bias_options)
    examples = Examples(
        positive=[parse_atom(p) if isinstance(p, str) else p for p in positive],
        negative=[parse_atom(n) if isinstance(n, str) else n for n in negative],
    )
    return RuleLearner(background, bias, examples).learn()
