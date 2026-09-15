"""The Type 3 pipeline, end to end.

This is the blueprint's diagram as code, read left to right::

    raw input -> perception -> symbolic facts -> inference -> Type 5 check
              -> proof tree -> output (JSON, prose, or both)

The pipeline owns the wiring and the policy: what confidence is too low to
accept, what to do when a hard rule is violated, and whether to render prose.
Each layer stays replaceable -- swap the perceptor, keep everything else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

from .consistency.layer import ConsistencyLayer, ConsistencyReport, Interpretation
from .core.parser import ParseError, parse_atom
from .core.terms import Atom
from .inference.engine import Answer, ReasoningEngine
from .inference.model import Model
from .inference.proof import ProofNode
from .knowledge.base import FactRecord, KnowledgeBase
from .output.nlg import Realiser
from .output.serialize import to_json, violations_document
from .perception.base import Perception

__all__ = ["NeuralMindPipeline", "PipelineResult", "ViolationPolicy"]


class ViolationPolicy:
    """What to do when perception produces facts that break a hard rule."""

    #: Report the violation and answer anyway.
    FLAG = "flag"
    #: Refuse to answer; the result carries the violations instead.
    REJECT = "reject"
    #: Search the classifier's other candidates for a consistent reading.
    RESOLVE = "resolve"


@dataclass
class PipelineResult:
    """Everything one run produced, from raw input to rendered answer."""

    question: Optional[str] = None
    query: Optional[Atom] = None
    answer: Optional[Answer] = None
    perception: Optional[Perception] = None
    model: Optional[Model] = None
    consistency: Optional[ConsistencyReport] = None
    explanation: Optional[str] = None
    interpretations: list[Interpretation] = field(default_factory=list)
    rejected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def holds(self) -> bool:
        return bool(self.answer and self.answer.holds)

    @property
    def proof(self) -> Optional[ProofNode]:
        return self.answer.proof if self.answer else None

    @property
    def consistent(self) -> bool:
        return self.consistency.crisp_consistent if self.consistency else True

    def to_dict(self, include_facts: bool = True) -> dict:
        document: dict[str, Any] = {}
        if self.question:
            document["question"] = self.question
        if self.query is not None:
            document["query"] = str(self.query)
        if self.answer is not None:
            document["answer"] = self.answer.to_dict()
        if self.explanation:
            document["explanation"] = self.explanation
        if include_facts and self.perception is not None:
            document["perception"] = self.perception.to_dict()
        if self.consistency is not None:
            document["consistency"] = self.consistency.to_dict()
        if self.model is not None:
            document["violations"] = violations_document(self.model.violations)
            document["model_summary"] = self.model.summary()
        if self.interpretations:
            document["interpretations"] = [i.to_dict() for i in self.interpretations]
        if self.rejected:
            document["rejected"] = True
        if self.notes:
            document["notes"] = list(self.notes)
        return document

    def to_json(self, indent: Optional[int] = 2, include_facts: bool = True) -> str:
        return to_json(self.to_dict(include_facts=include_facts), indent=indent)

    def __str__(self) -> str:
        lines: list[str] = []
        if self.question:
            lines.append(f"Q: {self.question}")
        if self.rejected:
            lines.append("REJECTED: the input violates a hard rule")
        if self.answer is not None:
            lines.append(str(self.answer))
        if self.explanation:
            lines.append("")
            lines.append(self.explanation)
        if self.consistency is not None and not self.consistency.consistent:
            lines.append("")
            lines.append(self.consistency.describe())
        return "\n".join(lines)


class NeuralMindPipeline:
    """Perception, reasoning, consistency checking and rendering in one object.

    Parameters
    ----------
    knowledge:
        The knowledge base. Rules should be loaded before running -- the
        blueprint's Phase 2 point is that the rules are the hard part, and the
        pipeline does not invent any.
    perceptor:
        Turns raw input into facts. Defaults to the text perceptor.
    minimum_confidence:
        Perceived facts below this are dropped before reasoning, and recorded
        in the result's notes so the discard is visible.
    on_violation:
        One of :class:`ViolationPolicy`.
    prose:
        Whether to render an English explanation alongside the JSON.
    """

    def __init__(
        self,
        knowledge: Optional[KnowledgeBase] = None,
        perceptor=None,
        minimum_confidence: float = 0.0,
        on_violation: str = ViolationPolicy.FLAG,
        prose: bool = True,
        realiser: Optional[Realiser] = None,
        consistency: Optional[ConsistencyLayer] = None,
    ) -> None:
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase("pipeline")
        self._perceptor = perceptor
        self.minimum_confidence = minimum_confidence
        self.on_violation = on_violation
        self.prose = prose
        self.realiser = realiser or Realiser()
        self._consistency = consistency

    # -- lazily built parts ----------------------------------------------

    @property
    def perceptor(self):
        if self._perceptor is None:
            from .perception.text import TextPerceptor

            self._perceptor = TextPerceptor()
        return self._perceptor

    @property
    def consistency(self) -> ConsistencyLayer:
        if self._consistency is None:
            self._consistency = ConsistencyLayer(self.knowledge)
        return self._consistency

    # -- stages ------------------------------------------------------------

    def observe(self, raw: Any, **kwargs) -> Perception:
        """Run perception and write the results into the knowledge base."""
        perception = self.perceptor.perceive(raw, **kwargs)
        kept = perception.filter(self.minimum_confidence)
        kept.into(self.knowledge)
        self.realiser.learn_names(perception.diagnostics.get("proper_names", ()))
        return perception

    def reason(self) -> tuple[ReasoningEngine, Model]:
        """Compute the model over everything currently known."""
        engine = self.knowledge.engine()
        return engine, engine.solve()

    def check(self, model: Optional[Model] = None) -> ConsistencyReport:
        """Run the Type 5 check over the current model."""
        if model is None:
            _, model = self.reason()
        confidences = {record.atom: record.confidence for record in self.knowledge.facts}
        return self.consistency.check(model, confidences)

    # -- the whole run -----------------------------------------------------

    def run(
        self,
        raw: Optional[Any] = None,
        question: Optional[Union[str, Atom]] = None,
        distributions: Optional[Sequence] = None,
        **perceive_kwargs,
    ) -> PipelineResult:
        """Perceive, reason, check, and answer.

        ``raw`` is optional: with a knowledge base already populated, calling
        with only a ``question`` queries what is already known.
        """
        result = PipelineResult(question=question if isinstance(question, str) else None)

        if raw is not None:
            perception = self.observe(raw, **perceive_kwargs)
            result.perception = perception
            dropped = [r for r in perception.facts if r.confidence < self.minimum_confidence]
            if dropped:
                result.notes.append(
                    f"dropped {len(dropped)} fact(s) below confidence "
                    f"{self.minimum_confidence}: "
                    + ", ".join(str(r.atom) for r in dropped[:5])
                )
            if perception.unparsed:
                result.notes.append(
                    f"{len(perception.unparsed)} input fragment(s) could not be parsed"
                )

        engine, model = self.reason()
        result.model = model
        result.consistency = self.check(model)

        if model.violations:
            result.notes.append(
                f"{len(model.violations)} hard rule violation(s) in the input"
            )
            if self.on_violation == ViolationPolicy.REJECT:
                result.rejected = True
                return result
            if self.on_violation == ViolationPolicy.RESOLVE and distributions:
                result.interpretations = self.resolve(distributions)
                if result.interpretations:
                    best = result.interpretations[0]
                    result.notes.append(
                        "consistency layer revised the reading to: "
                        + ", ".join(str(r.atom) for r in best.facts if r.confidence < 1.0)
                    )
                    for record in best.facts:
                        self.knowledge.add_fact(
                            record.atom,
                            confidence=record.confidence,
                            provenance="consistency:resolved",
                        )
                    engine, model = self.reason()
                    result.model = model
                    result.consistency = self.check(model)

        if question is not None:
            query = self._as_query(question)
            result.query = query
            result.answer = engine.ask(query)
            if self.prose:
                result.explanation = self.realiser.realise_answer(result.answer)
        return result

    def resolve(self, distributions: Sequence, **kwargs) -> list[Interpretation]:
        """Ask the Type 5 layer for the most probable consistent reading."""
        certain = [record for record in self.knowledge.facts if record.certain]
        return self.consistency.resolve(distributions, evidence=certain, **kwargs)

    def ask(self, question: Union[str, Atom]) -> PipelineResult:
        """Query what is already known, without perceiving anything new."""
        return self.run(raw=None, question=question)

    # -- helpers -----------------------------------------------------------

    def _as_query(self, question: Union[str, Atom]) -> Atom:
        """Accept either a logic atom or an English question.

        Logic syntax wins when the text parses as an atom -- ``ancestor(a, c)``
        contains a space but is not a question, so the presence of whitespace
        cannot be the test.
        """
        if isinstance(question, Atom):
            return question
        text = question.strip()
        parser = getattr(self.perceptor, "parse_question", None)
        if not text.endswith("?"):
            try:
                return parse_atom(text)
            except ParseError:
                pass
        if parser is not None:
            try:
                return parser(text)
            except Exception:
                pass
        return parse_atom(text.rstrip("?").strip())

    @classmethod
    def from_rules(
        cls, *rule_sources: Union[str, Path], name: str = "pipeline", **kwargs
    ) -> "NeuralMindPipeline":
        """Build a pipeline from rule files or inline ASP source."""
        kb = KnowledgeBase(name)
        for source in rule_sources:
            path = Path(source)
            if path.suffix == ".lp" and path.exists():
                kb.load_rules(path)
            else:
                kb.add_rules(str(source))
        return cls(kb, **kwargs)
