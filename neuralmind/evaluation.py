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
            asked, asked_negated, detail = _read_query(perceptor, question.text)
            question_matches = asked == goal_atom and asked_negated == question.negated
            derived = bool(asked is not None and engine.model.holds(asked))
            # A negative question ("The mouse is not blue.") is true exactly
            # when the goal is not derivable. The *perceived* polarity is used
            # here, not the gold one: misreading the polarity is a real
            # end-to-end failure and must show up as one.
            predicted = (not derived) if asked_negated else derived

            failure: Optional[str] = None
            if predicted != question.answer:
                gold_derived = gold_engine.holds(goal_atom)
                gold_predicted = (not gold_derived) if question.negated else gold_derived
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


def _read_query(perceptor, text: str) -> tuple[Optional[Atom], bool, str]:
    """Read a question, whether it is phrased as one or as a statement.

    Generated benchmarks ask "Is Bob green?"; the ProofWriter corpus asserts
    "The mouse is not blue." and expects a true/false judgement. Both go
    through the perception layer, so a question the system cannot read counts
    against it exactly as a theory sentence would.

    Returns ``(goal atom, negated, detail)``.
    """
    stripped = text.strip()
    if stripped.endswith("?"):
        parser = getattr(perceptor, "parse_question", None)
        if parser is not None:
            try:
                return parser(stripped), False, ""
            except Exception as exc:
                return None, False, f"question not parsed: {exc}"
    perception = perceptor.perceive(stripped)
    if len(perception.facts) != 1:
        return (
            None,
            False,
            f"statement produced {len(perception.facts)} facts, expected exactly one",
        )
    atom = perception.facts[0].atom
    if atom.predicate.startswith("not_"):
        return Atom(atom.predicate[4:], atom.args), True, ""
    return atom, False, ""


def _gold_engine(problem: Problem):
    """Run the engine on the gold theory, to separate engine bugs from the rest."""
    rules = [Rule(_goal_atom(fact), ()) for fact in problem.gold_facts]
    for head, body in problem.gold_rules:
        rules.append(
            Rule(
                head=_template_atom(head),
                body=tuple(_template_literal(literal) for literal in body),
            )
        )
    return ForwardChainer(Program(rules=rules)).run()


def _template_atom(literal: tuple) -> Atom:
    """Build an atom from a gold template, stripping any negation marker."""
    predicate = literal[0]
    if predicate.startswith("not_"):
        predicate = predicate[4:]
    return Atom(
        predicate,
        tuple(Var("X") if part == "?" else Const(part) for part in literal[1:]),
    )


def _template_literal(literal: tuple) -> Literal:
    """A body literal, negated when the gold template marks it so.

    A negated condition has to become negation-as-failure, not a positive
    literal over a predicate named ``not_p``: the latter is never derived, so
    the rule would simply never fire.
    """
    return Literal(_template_atom(literal), negated=literal[0].startswith("not_"))


def _normalise_rule(rule: Rule) -> tuple:
    """Put an extracted rule into the gold form: ``(head, frozenset(body))``."""
    assert rule.head is not None
    head = _atom_template(rule.head)
    body = frozenset(
        _atom_template(part.atom, negated=part.negated)
        for part in rule.body
        if isinstance(part, Literal)
    )
    return (head, body)


def _normalise_gold_rule(rule: tuple) -> tuple:
    head, body = rule
    return (tuple(head), frozenset(tuple(literal) for literal in body))


def _atom_template(atom: Atom, negated: bool = False) -> tuple:
    predicate = atom.predicate
    if negated and not predicate.startswith("not_"):
        predicate = f"not_{predicate}"
    parts = [predicate]
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
