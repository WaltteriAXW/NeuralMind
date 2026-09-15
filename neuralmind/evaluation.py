"""Evaluation with failure attribution.

Phase 7 of the roadmap sets a specific bar: be able to say, in one sentence,
what fraction of failures are "perception was wrong" versus "a rule was
missing". Accuracy alone cannot tell you that, and in a two-part system it is
the only number that tells you what to fix next.

The attribution works because the benchmark ships the gold symbolic theory
next to the English. Each question is scored three ways:

* **perception** -- the theory the text layer extracted differs from the gold
  theory, or the question itself was read as the wrong goal.
* **reasoning** -- perception was right and the engine still answered wrong.
  In a closed-world Datalog engine this almost always means a missing rule.
* **engine** -- the engine answers wrong even when handed the gold theory
  directly. That is a bug in the reasoner, and it is separated out so it can
  never hide inside "reasoning".
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional, Sequence

from .core.program import Program, Rule
from .core.terms import Atom, Const, Literal, Var
from .datasets.proofwriter import Problem, Question
from .inference.forward import ForwardChainer
from .knowledge.base import KnowledgeBase
from .perception.text import TextPerceptor

__all__ = ["evaluate", "EvaluationReport", "QuestionOutcome", "PERCEPTION", "REASONING", "ENGINE"]

PERCEPTION = "perception"
REASONING = "reasoning"
ENGINE = "engine"


@dataclass
class QuestionOutcome:
    """The result for one question, with the reason if it went wrong."""

    problem: int
    question: str
    expected: bool
    predicted: bool
    depth: int
    failure: Optional[str] = None
    detail: str = ""

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted

    def to_dict(self) -> dict:
        return {
            "problem": self.problem,
            "question": self.question,
            "expected": self.expected,
            "predicted": self.predicted,
            "depth": self.depth,
            "correct": self.correct,
            "failure": self.failure,
            "detail": self.detail,
        }


@dataclass
class EvaluationReport:
    """Aggregate results, broken down by proof depth and failure cause."""

    outcomes: list[QuestionOutcome] = field(default_factory=list)
    problems: int = 0
    unparsed_sentences: int = 0
    theory_mismatches: int = 0

    @property
    def total(self) -> int:
        return len(self.outcomes)

    @property
    def correct(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def failures(self) -> list[QuestionOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.correct]

    def failure_breakdown(self) -> dict[str, int]:
        return dict(Counter(outcome.failure or "unknown" for outcome in self.failures))

    def accuracy_by_depth(self) -> dict[int, float]:
        buckets: dict[int, list[bool]] = {}
        for outcome in self.outcomes:
            buckets.setdefault(outcome.depth, []).append(outcome.correct)
        return {
            depth: sum(results) / len(results) for depth, results in sorted(buckets.items())
        }

    def describe(self) -> str:
        """The one-sentence answer Phase 7 asks for, plus the detail behind it."""
        lines = [
            f"{self.correct}/{self.total} correct ({self.accuracy:.1%}) "
            f"over {self.problems} problems"
        ]
        breakdown = self.failure_breakdown()
        if not self.failures:
            lines.append("no failures")
        else:
            counts = ", ".join(f"{count} {kind}" for kind, count in sorted(breakdown.items()))
            perception = breakdown.get(PERCEPTION, 0)
            lines.append(
                f"of {len(self.failures)} failures: {counts} "
                f"-- {perception / len(self.failures):.0%} are perception errors, "
                f"the rest are reasoning"
            )
        lines.append(
            "accuracy by proof depth: "
            + ", ".join(
                f"depth {depth}: {value:.1%}" for depth, value in self.accuracy_by_depth().items()
            )
        )
        if self.unparsed_sentences:
            lines.append(f"{self.unparsed_sentences} sentence(s) could not be parsed at all")
        if self.theory_mismatches:
            lines.append(
                f"{self.theory_mismatches} problem(s) where the extracted theory "
                "differed from the gold theory"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "problems": self.problems,
            "total": self.total,
            "correct": self.correct,
            "accuracy": round(self.accuracy, 4),
            "failure_breakdown": self.failure_breakdown(),
            "accuracy_by_depth": {
                str(depth): round(value, 4) for depth, value in self.accuracy_by_depth().items()
            },
            "unparsed_sentences": self.unparsed_sentences,
            "theory_mismatches": self.theory_mismatches,
            "failures": [outcome.to_dict() for outcome in self.failures],
        }

    def __str__(self) -> str:
        return self.describe()


def evaluate(
    problems: Sequence[Problem],
    perceptor=None,
    extra_rules: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> EvaluationReport:
    """Run the full pipeline over a benchmark and attribute every failure."""
    perceptor = perceptor or TextPerceptor()
    report = EvaluationReport(problems=len(problems))

    for index, problem in enumerate(problems):
        perception = perceptor.perceive(problem.theory)
        report.unparsed_sentences += len(perception.unparsed)

        kb = KnowledgeBase(f"problem-{index}")
        if extra_rules:
            kb.add_rules(extra_rules)
        perception.into(kb)

        extracted_facts = {record.atom.as_tuple() for record in kb.facts}
        extracted_rules = {_normalise_rule(rule) for rule in kb.rules.derivation_rules}
        gold_rules = {_normalise_gold_rule(rule) for rule in problem.gold_rules}
        theory_matches = extracted_facts == problem.gold_facts and extracted_rules == gold_rules
        if not theory_matches:
            report.theory_mismatches += 1

        try:
            engine = kb.engine()
            engine.solve()
        except Exception as exc:  # a malformed extracted theory is a perception failure
            for question in problem.questions:
                report.outcomes.append(
                    QuestionOutcome(
                        problem=index,
                        question=question.text,
                        expected=question.answer,
                        predicted=False,
                        depth=question.depth,
                        failure=PERCEPTION,
                        detail=f"extracted theory did not compile: {exc}",
                    )
                )
            continue

        gold_engine = _gold_engine(problem)
        for question in problem.questions:
            goal_atom = _goal_atom(question.goal)
            detail = ""
            try:
                asked = perceptor.parse_question(question.text)
            except Exception as exc:
                asked = None
                detail = f"question not parsed: {exc}"
            question_matches = asked == goal_atom
            predicted = bool(asked is not None and engine.model.holds(asked))

            failure: Optional[str] = None
            if predicted != question.answer:
                gold_predicted = gold_engine.holds(goal_atom)
                if gold_predicted != question.answer:
                    failure = ENGINE
                    detail = detail or "engine disagrees with the gold closure"
                elif not theory_matches or not question_matches:
                    failure = PERCEPTION
                    detail = detail or _theory_diff(
                        extracted_facts, extracted_rules, problem.gold_facts, gold_rules
                    )
                else:
                    failure = REASONING
                    detail = detail or "theory matched but the conclusion was not derived"
            report.outcomes.append(
                QuestionOutcome(
                    problem=index,
                    question=question.text,
                    expected=question.answer,
                    predicted=predicted,
                    depth=question.depth,
                    failure=failure,
                    detail=detail,
                )
            )
        if progress is not None:
            progress(index + 1, len(problems))
    return report


# -- gold-theory helpers --------------------------------------------------


def _goal_atom(goal: tuple) -> Atom:
    return Atom(goal[0], tuple(Const(part) for part in goal[1:]))


def _gold_engine(problem: Problem):
    """Run the engine on the gold theory, to separate engine bugs from the rest."""
    rules = [Rule(_goal_atom(fact), ()) for fact in problem.gold_facts]
    for head, body in problem.gold_rules:
        rules.append(
            Rule(
                head=_template_atom(head),
                body=tuple(Literal(_template_atom(literal)) for literal in body),
            )
        )
    return ForwardChainer(Program(rules=rules)).run()


def _template_atom(literal: tuple) -> Atom:
    return Atom(
        literal[0],
        tuple(Var("X") if part == "?" else Const(part) for part in literal[1:]),
    )


def _normalise_rule(rule: Rule) -> tuple:
    """Put an extracted rule into the gold form: ``(head, frozenset(body))``."""
    assert rule.head is not None
    head = _atom_template(rule.head)
    body = frozenset(
        _atom_template(part.atom)
        for part in rule.body
        if isinstance(part, Literal) and not part.negated
    )
    return (head, body)


def _normalise_gold_rule(rule: tuple) -> tuple:
    head, body = rule
    return (tuple(head), frozenset(tuple(literal) for literal in body))


def _atom_template(atom: Atom) -> tuple:
    parts = [atom.predicate]
    for argument in atom.args:
        if isinstance(argument, Var):
            parts.append("?")
        elif isinstance(argument, Const):
            parts.append(str(argument.value))
        else:
            parts.append(str(argument))
    return tuple(parts)


def _theory_diff(facts, rules, gold_facts, gold_rules) -> str:
    missing_facts = gold_facts - facts
    extra_facts = facts - gold_facts
    missing_rules = gold_rules - rules
    parts = []
    if missing_facts:
        parts.append(f"missed fact(s): {sorted(missing_facts)[:3]}")
    if extra_facts:
        parts.append(f"invented fact(s): {sorted(extra_facts)[:3]}")
    if missing_rules:
        parts.append(f"missed {len(missing_rules)} rule(s)")
    return "; ".join(parts) or "extracted theory differed from gold"
