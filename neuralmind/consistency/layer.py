"""The Type 5 layer: hard rules checked against uncertain perception.

Two capabilities, both from the same rule set the crisp engine uses.

:meth:`ConsistencyLayer.check` evaluates every rule fuzzily over the perceived
confidences and returns a satisfaction degree per rule. A rule that holds only
because of a fact the classifier was 51% sure of scores low, which is the
warning the crisp engine cannot give.

:meth:`ConsistencyLayer.resolve` goes further and does what the blueprint's
Phase 4 asks for: when the most likely perception violates a hard rule, search
the classifier's other candidates for the most probable reading that does not.
This is exact weighted model counting over a small candidate set -- the same
job Scallop and DeepProbLog do with gradients, done by enumeration because for
a handful of slots with a handful of candidates each, exact is both feasible
and easier to trust.

Scaling is honest: ``resolve`` is ``O(candidates ** slots)`` solver calls.
The default of 3 candidates over 2 slots is 9. Past a few hundred, either
narrow the candidates or move to a differentiable framework.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.program import Program, Rule
from ..core.terms import Atom, Const
from ..inference.forward import ForwardChainer, match_body
from ..inference.model import Model, Violation
from ..knowledge.base import FactRecord, KnowledgeBase
from .fuzzy import FuzzySemantics

__all__ = ["ConsistencyLayer", "ConsistencyReport", "RuleSatisfaction", "Interpretation"]


@dataclass
class RuleSatisfaction:
    """How well one rule holds over the whole model, and where it holds worst."""

    rule: Rule
    satisfaction: float
    groundings: int
    worst_grounding: Optional[tuple[Atom, ...]] = None
    worst_value: float = 1.0

    @property
    def is_constraint(self) -> bool:
        return self.rule.is_constraint

    def to_dict(self) -> dict:
        payload = {
            "rule": str(self.rule),
            "source": self.rule.origin(),
            "satisfaction": round(self.satisfaction, 4),
            "groundings": self.groundings,
        }
        if self.worst_grounding:
            payload["worst_grounding"] = [str(a) for a in self.worst_grounding]
            payload["worst_value"] = round(self.worst_value, 4)
        return payload


@dataclass
class ConsistencyReport:
    """The Type 5 verdict on one interpretation."""

    satisfaction: float
    rules: list[RuleSatisfaction] = field(default_factory=list)
    hard_violations: list[Violation] = field(default_factory=list)
    #: Rules that hold crisply but only just, in fuzzy terms.
    weak_rules: list[RuleSatisfaction] = field(default_factory=list)
    threshold: float = 0.8
    #: Propagated truth value of every atom in the model.
    truth: dict = field(default_factory=dict)

    def confidence(self, atom: Atom) -> float:
        """How well supported a conclusion is, after propagation.

        A conclusion is only as good as the chain under it: derive something
        from a fact the classifier was 55% sure of, and this reports 0.55, not
        the 1.0 the crisp model shows.
        """
        return float(self.truth.get(atom, 1.0 if atom in self.truth else 0.0))

    def uncertain_conclusions(self, threshold: Optional[float] = None) -> list[tuple]:
        """Derived atoms whose support is weaker than the threshold."""
        limit = self.threshold if threshold is None else threshold
        return sorted(
            ((atom, value) for atom, value in self.truth.items() if value < limit),
            key=lambda pair: pair[1],
        )

    @property
    def consistent(self) -> bool:
        """No hard rule is broken and nothing is only weakly satisfied."""
        return not self.hard_violations and not self.weak_rules

    @property
    def crisp_consistent(self) -> bool:
        return not self.hard_violations

    def describe(self) -> str:
        lines = [f"overall satisfaction {self.satisfaction:.3f} (threshold {self.threshold:.2f})"]
        if self.hard_violations:
            lines.append("hard violations:")
            lines.extend(f"  - {v.describe()}" for v in self.hard_violations)
        if self.weak_rules:
            lines.append("weakly satisfied rules (true, but resting on uncertain facts):")
            for rule in self.weak_rules:
                support = (
                    ", ".join(str(a) for a in rule.worst_grounding)
                    if rule.worst_grounding
                    else ""
                )
                lines.append(
                    f"  - {rule.rule.origin()} at {rule.worst_value:.3f}"
                    + (f" for {support}" if support else "")
                )
        if self.consistent:
            lines.append("no violations")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "satisfaction": round(self.satisfaction, 4),
            "consistent": self.consistent,
            "crisp_consistent": self.crisp_consistent,
            "threshold": self.threshold,
            "hard_violations": [v.describe() for v in self.hard_violations],
            "weak_rules": [r.to_dict() for r in self.weak_rules],
            "uncertain_conclusions": [
                {"atom": str(atom), "confidence": round(value, 4)}
                for atom, value in self.uncertain_conclusions()
            ],
            "rules": [r.to_dict() for r in self.rules],
        }

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Interpretation:
    """One candidate reading of the uncertain input."""

    facts: list[FactRecord]
    probability: float
    model: Optional[Model] = None
    report: Optional[ConsistencyReport] = None
    rank: int = 0
    #: True if this differs from the classifier's top prediction.
    revised: bool = False

    @property
    def consistent(self) -> bool:
        return self.report.crisp_consistent if self.report else False

    def atoms(self) -> list[Atom]:
        return [record.atom for record in self.facts]

    def to_dict(self) -> dict:
        return {
            "facts": [record.to_dict() for record in self.facts],
            "probability": round(self.probability, 6),
            "consistent": self.consistent,
            "revised": self.revised,
            "rank": self.rank,
        }

    def __str__(self) -> str:
        facts = ", ".join(str(record.atom) for record in self.facts)
        flag = " [revised]" if self.revised else ""
        return f"{facts}  p={self.probability:.4f}{flag}"


class ConsistencyLayer:
    """Checks and repairs uncertain perception against hard rules."""

    def __init__(
        self,
        knowledge: KnowledgeBase | Program,
        semantics: Optional[FuzzySemantics] = None,
        threshold: float = 0.8,
    ) -> None:
        self.knowledge = knowledge
        self.semantics = semantics or FuzzySemantics()
        self.threshold = threshold

    # -- checking --------------------------------------------------------

    def propagate(
        self, model: Model, confidences: Optional[dict[Atom, float]] = None
    ) -> dict[Atom, float]:
        """Push input confidences through the derivations.

        An input fact keeps its own confidence. A derived atom takes the fuzzy
        conjunction of its supports, combined across alternative derivations
        with the t-conorm -- two independent reasons to believe something are
        better than one. Atoms are visited in derivation-depth order, so every
        support is settled before the atom that rests on it.
        """
        confidences = confidences or {}
        truth: dict[Atom, float] = {}
        for atom in sorted(model.atoms, key=lambda a: (model.depth.get(a, 0), str(a))):
            if atom in confidences:
                truth[atom] = float(confidences[atom])
                continue
            values: list[float] = []
            for justification in model.justifications.get(atom, ()):
                if justification.rule.is_fact and not justification.support:
                    values.append(1.0)
                    continue
                if any(support not in truth for support in justification.support):
                    continue  # rests on something not yet settled; a shallower
                              # derivation of the same atom will cover it
                body = [truth[support] for support in justification.support]
                body.extend(
                    self.semantics.negate(truth.get(absent, 0.0))
                    for absent in justification.negative_support
                )
                values.append(self.semantics.conjoin(body))
            truth[atom] = self.semantics.disjoin(values) if values else 1.0
        return truth

    def check(
        self,
        model: Model,
        confidences: Optional[dict[Atom, float]] = None,
        program: Optional[Program] = None,
    ) -> ConsistencyReport:
        """Evaluate every rule fuzzily over a model and its confidences."""
        program = program or self._program()
        propagated = self.propagate(model, confidences)

        def truth(atom: Atom) -> float:
            return propagated.get(atom, 0.0)

        satisfactions: list[RuleSatisfaction] = []
        for rule in program.rules:
            if rule.is_fact:
                continue
            satisfactions.append(self._rule_satisfaction(rule, model, truth))

        overall = self.semantics.forall([s.satisfaction for s in satisfactions]) if satisfactions else 1.0
        weak = [
            s
            for s in satisfactions
            if s.satisfaction < self.threshold and s.groundings > 0
        ]
        return ConsistencyReport(
            satisfaction=overall,
            rules=satisfactions,
            hard_violations=list(model.violations),
            weak_rules=weak,
            threshold=self.threshold,
            truth=propagated,
        )

    def _rule_satisfaction(self, rule: Rule, model: Model, truth) -> RuleSatisfaction:
        """Satisfaction of one rule, aggregated over all its groundings."""
        try:
            steps = tuple(rule.plan())
        except Exception:
            return RuleSatisfaction(rule, 1.0, 0)

        values: list[float] = []
        worst_value = 1.0
        worst_grounding: Optional[tuple[Atom, ...]] = None
        for bindings, support, negative in match_body(steps, model):
            body_truths = [truth(atom) for atom in support]
            body_truths.extend(self.semantics.negate(truth(atom)) for atom in negative)
            body = self.semantics.conjoin(body_truths)
            if rule.is_constraint:
                # ":- body" is satisfied exactly to the degree the body is false.
                value = self.semantics.negate(body)
            else:
                assert rule.head is not None
                head = truth(rule.head.ground(bindings))
                value = self.semantics.implies(body, head)
            values.append(value)
            if value < worst_value:
                worst_value = value
                worst_grounding = support + negative
        return RuleSatisfaction(
            rule=rule,
            satisfaction=self.semantics.forall(values) if values else 1.0,
            groundings=len(values),
            worst_grounding=worst_grounding,
            worst_value=worst_value,
        )

    # -- repair ----------------------------------------------------------

    def resolve(
        self,
        distributions: Sequence,
        evidence: Iterable[FactRecord | Atom] = (),
        candidates_per_slot: int = 3,
        top_k: int = 3,
        require_consistent: bool = True,
    ) -> list[Interpretation]:
        """Find the most probable reading of the input that the rules allow.

        ``distributions`` is a sequence of
        :class:`~neuralmind.perception.vision.SlotDistribution` -- one per
        uncertain slot. Every combination of the top ``candidates_per_slot``
        values is scored by joint probability and checked against the hard
        rules; the consistent ones are returned best-first.

        With ``require_consistent=False`` inconsistent readings are kept too,
        which is useful for showing *why* the top prediction was rejected.
        """
        base_facts = [
            record if isinstance(record, FactRecord) else FactRecord(record)
            for record in evidence
        ]
        slot_options = [
            [(dist, value, probability) for value, probability in dist.top(candidates_per_slot)]
            for dist in distributions
        ]
        if not slot_options:
            raise ValueError("resolve() needs at least one distribution")

        best_choice = {
            id(dist): dist.best[0] for dist in distributions
        }
        interpretations: list[Interpretation] = []
        for combination in itertools.product(*slot_options):
            facts = list(base_facts)
            probability = 1.0
            revised = False
            for dist, value, slot_probability in combination:
                facts.append(
                    FactRecord(
                        atom=dist.atom(value),
                        confidence=slot_probability,
                        provenance="consistency:candidate",
                    )
                )
                probability *= slot_probability
                if best_choice[id(dist)] != value:
                    revised = True
            model, report = self._evaluate(facts)
            if require_consistent and not report.crisp_consistent:
                continue
            interpretations.append(
                Interpretation(
                    facts=facts,
                    probability=probability,
                    model=model,
                    report=report,
                    revised=revised,
                )
            )

        interpretations.sort(key=lambda i: -i.probability)
        for rank, interpretation in enumerate(interpretations):
            interpretation.rank = rank
        return interpretations[:top_k]

    def _evaluate(self, facts: Sequence[FactRecord]) -> tuple[Model, ConsistencyReport]:
        program = self._program()
        combined = Program(
            rules=[Rule(record.atom, (), source="candidate", label="given") for record in facts]
            + list(program.rules),
            shown=set(program.shown),
            constants=dict(program.constants),
        )
        model = ForwardChainer(combined).run()
        confidences = {record.atom: record.confidence for record in facts}
        report = self.check(model, confidences, combined)
        return model, report

    def _program(self) -> Program:
        if isinstance(self.knowledge, Program):
            return self.knowledge
        return self.knowledge.program(check=False)
