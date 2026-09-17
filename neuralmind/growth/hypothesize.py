"""Turning a gap into candidate definitions.

:mod:`neuralmind.induction` already searches a bounded hypothesis space and
reports when the examples do not single out an answer. What is missing for a
growth loop is the step before and the step after: deriving the search space
from a gap nobody wrote a bias for, and keeping the *competing* definitions
rather than only the winner.

Keeping the competitors is the point. A learner that returns one rule has made
a choice the evidence did not justify, and the honest response is not to pick
better -- it is to notice that a question would settle it. So this returns a
:class:`Proposal` carrying every definition that fits equally well, which is
exactly what :class:`~neuralmind.growth.questions.QuestionPicker` needs.

Examples come from the closed-world reading of what is already known: the
positives are what the knowledge base derives (or what the user supplied), and
everything else over the constants is negative. That is honest only when the
positives really are complete, which is why a proposal says how many examples
it rested on -- a rule learned from three examples is a rule that fits three
examples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.program import Rule
from ..core.terms import Atom, Const
from ..induction.bias import LanguageBias, Signature
from ..induction.learn import Examples, Hypothesis, RuleLearner
from .gaps import Gap

__all__ = ["Proposal", "Hypothesiser"]


@dataclass
class Proposal:
    """What the learner found for one gap, competitors included."""

    target: Signature
    #: Every definition that fits the examples equally well, best first.
    candidates: list[list[Rule]] = field(default_factory=list)
    hypothesis: Optional[Hypothesis] = None
    examples: Optional[Examples] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return bool(self.candidates)

    @property
    def settled(self) -> bool:
        """True when exactly one definition fits -- nothing left to ask."""
        return len(self.candidates) == 1

    @property
    def best(self) -> Optional[list[Rule]]:
        return self.candidates[0] if self.candidates else None

    def describe(self) -> str:
        if not self.candidates:
            return f"no definition of {self.target} found: {self.reason}"
        head = "\n".join(str(rule) for rule in self.candidates[0])
        if self.settled:
            return head
        return f"{head}\n  ...and {len(self.candidates) - 1} other definition(s) that fit equally well"

    def to_dict(self) -> dict:
        return {
            "target": str(self.target),
            "candidates": [[str(r) for r in c] for c in self.candidates],
            "settled": self.settled,
            "examples": len(self.examples) if self.examples else 0,
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return self.describe()


class Hypothesiser:
    """Runs induction on a gap, and keeps what competes."""

    def __init__(
        self,
        knowledge,
        max_body: int = 3,
        max_variables: int = 3,
        max_clauses: int = 2,
        max_predicates: int = 8,
        allow_comparison: bool = True,
    ) -> None:
        self.knowledge = knowledge
        self.max_body = max_body
        self.max_variables = max_variables
        self.max_clauses = max_clauses
        self.max_predicates = max_predicates
        #: Comparisons are off by default in the bias and must be on here.
        #: Half the relations worth learning need one -- "sibling" is
        #: ``parent(P, X), parent(P, Y), X != Y``, and without the
        #: disequality the true rule is not in the space at all, so the loop
        #: settles confidently on something else.
        self.allow_comparison = allow_comparison

    def constants(self) -> list[Const]:
        """Every constant the knowledge base mentions."""
        found: dict[str, Const] = {}
        for record in self.knowledge.facts:
            for argument in record.atom.args:
                if isinstance(argument, Const):
                    found.setdefault(str(argument.value), argument)
        return [found[key] for key in sorted(found)]

    def for_gap(
        self,
        gap: Gap,
        positive: Optional[Sequence[Atom]] = None,
        negative: Optional[Sequence[Atom]] = None,
        allow_recursion: bool = True,
    ) -> Proposal:
        """Propose definitions for the predicate a gap is about."""
        goal = gap.goal or gap.missing
        if goal is None:
            return Proposal(
                target=Signature("unknown", 0),
                reason="this gap is not about a predicate",
            )
        target = Signature(goal.predicate, goal.arity)
        return self.for_target(target, positive, negative, allow_recursion)

    def for_target(
        self,
        target,
        positive: Optional[Sequence[Atom]] = None,
        negative: Optional[Sequence[Atom]] = None,
        allow_recursion: bool = True,
    ) -> Proposal:
        signature = Signature.parse(target)
        constants = self.constants()
        if positive is None:
            positive = [
                record.atom
                for record in self.knowledge.facts
                if record.atom.signature == (signature.name, signature.arity)
            ]
        if not positive:
            return Proposal(
                target=signature,
                examples=Examples(),
                reason=(
                    f"nothing in the knowledge base is an example of "
                    f"{signature}; ask for one"
                ),
            )
        if negative is None:
            examples = Examples.closed_world(positive, constants, signature)
        else:
            examples = Examples(positive=list(positive), negative=list(negative))

        bias = LanguageBias.from_knowledge(
            self.knowledge,
            signature,
            max_body=self.max_body,
            max_variables=self.max_variables,
            max_clauses=self.max_clauses,
            max_predicates=self.max_predicates,
            allow_recursion=allow_recursion,
            allow_comparison=self.allow_comparison,
        )
        learner = RuleLearner(self.knowledge, bias, examples)
        hypothesis = learner.learn()
        if not hypothesis.rules:
            return Proposal(
                target=signature,
                hypothesis=hypothesis,
                examples=examples,
                reason=hypothesis.incomplete_reason or "the search space contains no fit",
            )
        return Proposal(
            target=signature,
            candidates=_competitors(hypothesis),
            hypothesis=hypothesis,
            examples=examples,
            reason="" if hypothesis.correct else (hypothesis.incomplete_reason or ""),
        )


    def version_space(
        self,
        target,
        positive: Sequence[Atom],
        negative: Sequence[Atom] = (),
        limit: int = 200,
    ) -> list[list[Rule]]:
        """Every single-clause definition consistent with the examples so far.

        This is the version space, and it is what questions are for. The
        learner's ``ties`` are the same idea seen through one search: they are
        the clauses that fit *as well as* the winner. Enumerating directly is
        better here because the loop needs the space to be wide at the start --
        with one example and no negatives almost everything fits, and that is
        the honest starting point, not a failure.

        Consistency means: derives every positive, derives no negative. A rule
        that derives extra atoms nobody has ruled on is still a candidate --
        that is exactly what the next question is about.
        """
        from ..induction.enumerate import candidate_bodies, candidate_literals
        from ..induction.enumerate import candidate_comparisons

        signature = Signature.parse(target)
        bias = LanguageBias.from_knowledge(
            self.knowledge,
            signature,
            max_body=self.max_body,
            max_variables=self.max_variables,
            max_clauses=1,
            max_predicates=self.max_predicates,
            allow_recursion=False,
            allow_comparison=self.allow_comparison,
        )
        literals = candidate_literals(bias) + candidate_comparisons(bias)
        wanted, forbidden = set(positive), set(negative)
        background = self.knowledge.program(check=False)
        found: list[list[Rule]] = []
        for size in range(1, bias.max_body + 1):
            for rule, _body, _ in candidate_bodies(bias, size, literals):
                derived = _derives(background, [rule], signature)
                if derived is None:
                    continue
                if wanted <= derived and not (forbidden & derived):
                    found.append([rule])
                    if len(found) >= limit:
                        break
            if len(found) >= limit:
                break
        # Simplest first. When several definitions agree on every case there
        # is to ask about, the evidence does not choose between them and
        # something has to; the shortest body is the standard answer and the
        # only one that does not smuggle in a preference about vocabulary.
        found.sort(key=lambda clauses: (sum(len(r.body) for r in clauses),
                                        len(clauses), str(clauses[0])))
        return found


def _derives(background, clauses, signature) -> Optional[set]:
    """What a candidate definition derives for its target, or None if unusable."""
    from ..core.program import Program
    from ..inference.forward import ForwardChainer

    program = Program(
        rules=list(background.rules) + list(clauses),
        shown=set(background.shown),
        constants=dict(background.constants),
    )
    try:
        model = ForwardChainer(program).run()
    except Exception:
        return None
    return {
        a for a in model.atoms if a.signature == (signature.name, signature.arity)
    }


def _competitors(hypothesis: Hypothesis) -> list[list[Rule]]:
    """The chosen definition, then each tied alternative substituted in.

    A tie is recorded against the clause it competes with, so the alternatives
    are whole definitions with that one clause swapped -- not loose rules. That
    matters because a probe has to be evaluated against a *definition*, and a
    clause on its own derives nothing.
    """
    chosen = list(hypothesis.rules)
    found = [chosen]
    for index, alternatives in sorted(hypothesis.ties.items()):
        for alternative in alternatives:
            variant = list(chosen)
            if index < len(variant):
                variant[index] = alternative
            else:
                variant.append(alternative)
            if variant not in found:
                found.append(variant)
    return found
