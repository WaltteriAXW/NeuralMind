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

**It works out where it is.** :meth:`observe` feeds the context layer
(:mod:`neuralmind.self`), which reads the *shape* of what arrives and derives
which facets are in play -- money, inventory, conversation -- with the ordinary
engine, so every hypothesis has a proof. Stakes follow the facets and reach the
autonomy gate immediately, before any domain is recognised: money plus an
action list is dangerous whether or not the mind has worked out it is in a
bank.

**A way back.** Everything the mind learns goes through the safety kernel
(:mod:`neuralmind.kernel`): layered knowledge with a read-only core, a firewall
on every rule, canaries checked after every change, and a transaction that
leaves no trace when one fails. :meth:`learn` is the guarded path and there is
no shorter one. :meth:`decide` puts every action through the autonomy gate,
where caution rises on a guess and autonomy rises only on a grant.

**More than one kind of reasoning.** :meth:`solve` goes through the workspace
(:mod:`neuralmind.workspace`), where a logic engine, a constraint solver, a
unit converter and a graph search meet on a blackboard. Each posts atoms with
their own proofs, so one tree spans all of them:

    safe(beam1)  (by a design rule)
    |-- beam(beam1)  [given]
    `-- leq(beam1_load, beam1_rating)  [by arithmetic: 3200 <= 5000]
        |-- value(beam1_load, 3200)  [by units: 3200 N = 3200 kg*m/s^2]
        `-- value(beam1_rating, 5000)  [by units: 5 kN = 5000 kg*m/s^2]

Every specialist but the logic one is optional, and a missing backend costs
exactly the questions that needed it -- they come back ``unknown`` naming what
is missing.

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
        tenant: Optional[str] = None,
        granted: int = 1,
    ) -> None:
        from .kernel import CONFIRMED, Kernel

        #: The safety kernel. Its confirmed layer *is* :attr:`knowledge`, so
        #: the ordinary paths keep working while everything guarded -- the
        #: read-only core, the firewall, the canaries, the autonomy gate --
        #: sits underneath rather than beside them.
        self.kernel = Kernel("mind", tenant=tenant, granted=granted)
        if knowledge is not None:
            self.kernel.layers[CONFIRMED].knowledge = knowledge
        self.knowledge = self.kernel.layers[CONFIRMED].knowledge
        self.budget_ms = budget_ms
        self.seed = seed
        self.realiser = realiser or Realiser()
        self.observations: list[Observation] = []
        self._perceptor = perceptor
        self._engine: Optional[ReasoningEngine] = None
        self._workspace = None
        self._controller = None
        self._context = None
        self._self_model = None

    # -- parts built on demand ---------------------------------------------

    @property
    def perceptor(self):
        if self._perceptor is None:
            from .perception.text import TextPerceptor

            self._perceptor = TextPerceptor()
        return self._perceptor

    @property
    def engine(self) -> ReasoningEngine:
        """The engine over everything currently known, rebuilt when it changes.

        "Everything known" is what the current mode can see: the sandbox is
        excluded always, and a degraded mode sees less still.
        """
        if self._engine is None:
            self._engine = self.kernel.layers.engine(self.kernel.mode_layers)
        return self._engine

    @property
    def context(self):
        """The context layer. Built on first use, so importing is cheap."""
        if self._context is None:
            from .self import ContextDiscovery

            self._context = ContextDiscovery()
        return self._context

    @property
    def model(self):
        """The mind's facts about itself, refreshed from the parts that hold them."""
        if self._self_model is None:
            from .self import SelfModel

            self._self_model = SelfModel("mind")
        self._self_model.sync(
            kernel=self.kernel,
            reading=self._context.reading if self._context is not None else None,
            specialists=self.specialists(),
        )
        return self._self_model

    @property
    def workspace(self):
        """The blackboard every specialist reads and writes."""
        if self._workspace is None:
            from .workspace import Workspace

            self._workspace = Workspace()
            self._seed_workspace()
        return self._workspace

    @property
    def controller(self):
        """The controller over the specialists, warmed on first use.

        Warming happens here rather than inside a query because importing z3
        and building pint's unit registry together cost about 300ms -- more
        than ten times a realistic per-query budget, and paid exactly once.
        Charging that to whichever question came first would make the budget
        meaningless.
        """
        if self._controller is None:
            from .workspace import Controller
            from .workspace.specialists import default_specialists

            self._controller = Controller(
                self.workspace,
                default_specialists(self.knowledge),
                budget_ms=self.budget_ms,
            )
        return self._controller

    def _seed_workspace(self) -> None:
        """Put what is already known on the blackboard as given facts."""
        from .inference.proof import FACT, ProofNode

        for record in self.kernel.layers.facts(self.kernel.mode_layers):
            self._workspace.post(
                record.atom,
                ProofNode(record.atom, FACT, confidence=None),
                source="given",
                confidence=record.confidence,
            )

    def _invalidate(self) -> None:
        self._engine = None
        if self._workspace is not None:
            self._seed_workspace()
        if self._controller is not None:
            for specialist in self._controller.router.specialists:
                invalidate = getattr(specialist, "invalidate", None)
                if invalidate is not None:
                    invalidate()

    # -- input --------------------------------------------------------------

    def observe(self, payload: Any) -> Observation:
        """Take anything the host sends and keep it.

        Text goes through perception and becomes facts. A mapping or sequence
        is stored as-is: reading structure is context discovery's job, and
        guessing at it here would be the "you are in a bank" assumption by
        another route.
        """
        # Every observation is evidence about where this is, whatever else it
        # is. Reading it costs a rule evaluation and is the only way the mind
        # ever finds out.
        self.context.observe(payload)
        self.context.apply(self.kernel.autonomy)
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
        """Add rules directly. Use :meth:`learn` for anything the mind inferred.

        This is the host's own channel: a rule written by a person who knows
        the domain is not the same object as one a learner proposed, and making
        the host pass the firewall to state its own policy would be theatre.
        Anything the *mind* produced goes through :meth:`learn` instead.
        """
        self.knowledge.add_rules(source)
        self._invalidate()
        return self

    def learn(self, source: str, name: str = "learned"):
        """Add a rule the guarded way: firewall, snapshot, canaries, rollback.

        Returns an :class:`~neuralmind.kernel.Outcome`, which is falsey when
        the rule was refused and says which check refused it.
        """
        from .kernel import CONFIRMED

        outcome = self.kernel.learn(source, CONFIRMED, name)
        self._invalidate()
        return outcome

    def watch(self, goals: Iterable) -> "Mind":
        """Record what the mind answers now, and defend it from future changes."""
        self.kernel.watch(goals)
        return self

    def decide(self, action: str, needs: int = 2, confirmed: bool = False):
        """Put an action through the autonomy gate. Nothing reaches a host unchecked."""
        return self.kernel.decide(action, needs, confirmed)

    def grant(self, level: int) -> "Mind":
        """Set the autonomy ceiling. Only a host does this."""
        self.kernel.autonomy.grant(level)
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

    def solve(self, question: Union[str, Atom], budget_ms: Optional[float] = None):
        """Answer through the workspace, using every specialist that applies.

        Use this where :meth:`ask` is not enough: a question that needs numbers,
        units or paths as well as rules. The answer is a
        :class:`~neuralmind.workspace.controller.Conclusion`, carrying one proof
        that spans whichever specialists contributed.
        """
        return self.controller.solve(self._as_goal(question), budget_ms)

    def specialists(self) -> dict:
        """Which specialists are installed. Part of the self-report."""
        from .workspace.specialists import installed

        return installed()

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
        reading = self._context.reading if self._context is not None else None
        # Three sentences, each about one thing: where it is, what it knows,
        # what it cannot do. Every clause is read off state -- nothing here is
        # estimated, which is what makes the report safe to show a user.
        open_predicates = sorted(
            f"{name}/{arity}" for name, arity in self.knowledge.rules.open_predicates
        )
        knowledge = f"Knowledge: {facts} fact(s), {rules} rule(s)"
        if self.knowledge.rules.open_world:
            knowledge += ", open-world"
        elif open_predicates:
            knowledge += ", open on " + ", ".join(open_predicates)
        if reading is None or reading.unresolved:
            where = (
                f"Context unresolved — {seen} observation(s) in, "
                f"{unread} not yet interpretable."
            )
        else:
            where = (
                f"Looks like {reading.domain} "
                f"({reading.domain_confidence:.2f}) from {seen} observation(s): "
                + ", ".join(f.name for f in reading.facets[:3])
                + f"; stakes {reading.stakes}."
            )
        lines = [where, knowledge + "."]
        missing = sorted(name for name, ready in self.specialists().items() if not ready)
        limits = []
        if self.kernel.watchdog.mode.name != "full":
            limits.append(self.kernel.watchdog.describe())
        if missing:
            limits.append(", ".join(missing) + " not installed")
        if len(self.kernel.quarantine):
            limits.append(f"{len(self.kernel.quarantine)} rule(s) quarantined")
        if limits:
            lines.append("Cannot: " + "; ".join(limits) + ".")
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
