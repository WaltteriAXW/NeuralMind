"""``Mind`` -- one entry point a host can hold, that is never told where it is.

Phase One's :class:`~neuralmind.pipeline.NeuralMindPipeline` is a pipeline: you
hand it text and a question and it runs the stages in order. ``Mind`` is the
object a host keeps. It takes observations as they come, answers when asked,
and says what it does not know.

The constructor takes no domain, no pack and no mode argument, and that is a
design rule rather than an oversight. A host that has to declare "you are a
bank assistant" has already made the mind's hardest decision for it, and
whatever it says will be wrong in the places that matter -- the training
simulator that is half game and half engineering tool, the bank app with a shop
inside it. Later phases work the situation out from the observations; this one
establishes the surface that makes that possible, and keeps it honest in the
meantime by reporting an unresolved context rather than assuming one.

What it adds over the pipeline:

**Three-valued answers.** ``yes``, ``no``, ``unknown``. A predicate the program
does not fully describe answers ``unknown`` when nothing derives it, instead of
silently answering ``no``; see :class:`~neuralmind.inference.engine.Answer`.

**Observations are kept raw.** :meth:`observe` accepts any JSON, text or array
and stores it unchanged alongside whatever perception made of it. Nothing here
reads the structure yet -- context discovery does, later -- but the record has
to exist from the start or there is nothing to look back at.

**Short answers that are still true.** :meth:`ask` returns an ``Answer`` whose
``brief`` is one line under a word budget, with every clause traceable to a
node of the proof. See :mod:`neuralmind.output.brief`.

Usage::

    mind = Mind()
    mind.tell("Bob is a cat. All cats are mammals.")
    answer = mind.ask("Is Bob a mammal?")
    answer.status          # "yes"
    mind.brief(answer)     # "Yes — bob is a mammal, because bob is a cat."
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence, Union

from .core.terms import Atom
from .inference.engine import Answer, ReasoningEngine
from .knowledge.base import FactRecord, KnowledgeBase
from .output.brief import BRIEF_WORDS, Brief
from .output.brief import brief as render_brief
from .output.nlg import Realiser
from .perception.base import Perception

__all__ = ["Mind", "Observation"]


@dataclass
class Observation:
    """One thing the host sent, kept exactly as it arrived.

    ``payload`` is untouched. ``perception`` is what the perception layer made
    of it, or ``None`` for a payload no perceptor claimed -- which is not a
    failure. A structured record the mind cannot yet interpret is still
    evidence about where it is, and discarding it would throw that away.
    """

    payload: Any
    kind: str
    perception: Optional[Perception] = None
    facts: tuple[Atom, ...] = ()

    @property
    def understood(self) -> bool:
        return bool(self.facts)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "understood": self.understood,
            "facts": [str(a) for a in self.facts],
            "payload": _jsonable(self.payload),
        }


class Mind:
    """A reasoning mind behind one stable interface.

    Parameters
    ----------
    knowledge:
        An existing knowledge base to build on. A fresh one by default.
    perceptor:
        Overrides text perception. Built on first use otherwise, so importing
        ``Mind`` does not load spaCy.
    budget_ms:
        Advisory time budget per call, recorded in the self-report. Phase Two's
        controller enforces it; here it is stored so the host can set it once.
    seed:
        Kept for determinism. Nothing in the core is random, so this exists to
        be threaded through the parts that will be.
    """

    def __init__(
        self,
        knowledge: Optional[KnowledgeBase] = None,
        perceptor=None,
        budget_ms: int = 20,
        seed: int = 0,
        realiser: Optional[Realiser] = None,
    ) -> None:
        self.knowledge = knowledge if knowledge is not None else KnowledgeBase("mind")
        self.budget_ms = budget_ms
        self.seed = seed
        self.realiser = realiser or Realiser()
        self.observations: list[Observation] = []
        self._perceptor = perceptor
        self._engine: Optional[ReasoningEngine] = None

    # -- parts built on demand ---------------------------------------------

    @property
    def perceptor(self):
        if self._perceptor is None:
            from .perception.text import TextPerceptor

            self._perceptor = TextPerceptor()
        return self._perceptor

    @property
    def engine(self) -> ReasoningEngine:
        """The engine over everything currently known, rebuilt when it changes."""
        if self._engine is None:
            self._engine = self.knowledge.engine()
        return self._engine

    def _invalidate(self) -> None:
        self._engine = None

    # -- input --------------------------------------------------------------

    def observe(self, payload: Any) -> Observation:
        """Take anything the host sends and keep it.

        Text goes through perception and becomes facts. A mapping or sequence
        is stored as-is: reading structure is context discovery's job, and
        guessing at it here would be the "you are in a bank" assumption by
        another route.
        """
        kind = _classify(payload)
        if kind == "text":
            perception = self.tell(str(payload))
            observation = Observation(
                payload=payload,
                kind=kind,
                perception=perception,
                facts=tuple(record.atom for record in perception.facts),
            )
        else:
            observation = Observation(payload=payload, kind=kind)
        self.observations.append(observation)
        return observation

    def tell(self, text: str) -> Perception:
        """Read English into the knowledge base."""
        perception = self.perceptor.perceive(text)
        perception.into(self.knowledge)
        self.realiser.learn_names(perception.diagnostics.get("proper_names", ()))
        self._invalidate()
        return perception

    def add_rules(self, source: str) -> "Mind":
        """Add rules in the logic language, including ``#open`` declarations."""
        self.knowledge.add_rules(source)
        self._invalidate()
        return self

    def add_facts(self, facts: Iterable[Union[Atom, str, FactRecord]], **kwargs) -> "Mind":
        self.knowledge.add_facts(facts, **kwargs)
        self._invalidate()
        return self

    # -- output -------------------------------------------------------------

    def ask(self, question: Union[str, Atom], **kwargs) -> Answer:
        """Answer a question in English or in logic syntax.

        The answer is three-valued. ``unknown`` means the knowledge base does
        not settle the question, which is different from settling it as ``no``.
        """
        goal = self._as_goal(question)
        return self.engine.ask(goal, **kwargs)

    def brief(self, answer: Answer, length: str = "brief", max_words: int = BRIEF_WORDS) -> str:
        """One line for a status bar, an alert or a speech bubble."""
        return render_brief(answer, self.realiser, length, max_words).text

    def explain(self, answer_or_atom) -> Brief:
        """The full brief object, clauses and all, for a host that wants the parts."""
        answer = (
            answer_or_atom
            if isinstance(answer_or_atom, Answer)
            else self.ask(answer_or_atom)
        )
        return render_brief(answer, self.realiser)

    def self_report(self) -> str:
        """What the mind can say about itself, in at most three sentences.

        Everything here is read off state, never estimated. Phase Two's
        self-model turns these into facts the mind can reason over; until then
        the honest version is a short status line that says the context is
        unresolved rather than implying one has been worked out.
        """
        facts = len(self.knowledge)
        rules = len(self.knowledge.rules.derivation_rules)
        seen = len(self.observations)
        unread = sum(1 for o in self.observations if not o.understood)
        lines = [
            f"Context unresolved — {seen} observation(s) in, "
            f"{unread} not yet interpretable."
        ]
        lines.append(f"Knowledge: {facts} fact(s), {rules} rule(s).")
        open_predicates = sorted(
            f"{name}/{arity}" for name, arity in self.knowledge.rules.open_predicates
        )
        if open_predicates:
            lines.append(
                "Open (silence means unknown): " + ", ".join(open_predicates) + "."
            )
        return " ".join(lines)

    def questions(self) -> list[str]:
        """What the mind would need to be told. Empty until the growth loop lands."""
        return []

    # -- helpers ------------------------------------------------------------

    def _as_goal(self, question: Union[str, Atom]) -> Atom:
        if isinstance(question, Atom):
            return question
        from .core.parser import ParseError, parse_atom

        text = question.strip()
        if text.endswith("?"):
            return self.perceptor.parse_question(text)
        try:
            return parse_atom(text)
        except (ParseError, SyntaxError):
            return self.perceptor.parse_question(text)

    def __repr__(self) -> str:
        return (
            f"Mind(facts={len(self.knowledge)}, "
            f"rules={len(self.knowledge.rules.derivation_rules)}, "
            f"observations={len(self.observations)})"
        )


def _classify(payload: Any) -> str:
    """What kind of thing the host just sent, by shape alone."""
    if isinstance(payload, str):
        return "text"
    if isinstance(payload, dict):
        return "record"
    if isinstance(payload, (list, tuple)):
        return "records" if payload and isinstance(payload[0], dict) else "sequence"
    if hasattr(payload, "shape"):  # a NumPy array or a tensor
        return "array"
    return "value"


def _jsonable(payload: Any) -> Any:
    """A JSON-safe echo of a payload, for reports. Never used for reasoning."""
    try:
        json.dumps(payload)
        return payload
    except (TypeError, ValueError):
        return repr(payload)
