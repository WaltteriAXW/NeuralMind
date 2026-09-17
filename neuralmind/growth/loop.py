"""The growth cycle: notice a gap, propose, ask, confirm.

One turn of the loop:

1. **Propose.** Run induction over the examples gathered so far.
2. **Check.** If exactly one definition fits, there is nothing to ask.
3. **Ask.** Otherwise pick the probe that splits the competitors most evenly
   and put it to the teacher.
4. **Record.** Fold the answer in as a new example and go round again.

It stops when the definition is settled, when the question budget runs out, or
when the teacher says it does not know -- which is a real answer and not a
failure, though it does mean this probe cannot narrow anything.

Nothing learned here affects an answer until it is confirmed. A settled
definition is a *proposal*; it goes through the safety kernel like anything
else, and the kernel is what decides whether it enters the confirmed layer.
That separation is deliberate: the loop's job is to work out what the evidence
supports, and the kernel's job is to refuse it anyway if it breaks something.

The teacher is a callable ``(atom) -> bool | None``. In a session it is the
user; in a test it is an oracle built from the true definition, which is what
makes "how many questions did this take?" a number rather than an impression.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..core.program import Rule
from ..core.terms import Atom
from ..induction.bias import Signature
from ..induction.learn import Examples
from .hypothesize import Hypothesiser, Proposal
from .questions import (
    ASK_EXAMPLE,
    ASK_MEMBERSHIP,
    ASK_REFUTATION,
    Question,
    QuestionPicker,
)

__all__ = ["GrowthLoop", "Session", "Teacher"]

#: Answers a membership question. ``None`` means "I don't know", which is
#: allowed and simply narrows nothing.
Teacher = Callable[[Atom], Optional[bool]]


@dataclass
class Session:
    """What one run of the loop did, and what it cost."""

    target: Signature
    proposal: Optional[Proposal] = None
    asked: list[Question] = field(default_factory=list)
    answers: list[tuple[Atom, Optional[bool]]] = field(default_factory=list)
    positive: list[Atom] = field(default_factory=list)
    negative: list[Atom] = field(default_factory=list)
    stopped: str = ""

    @property
    def questions(self) -> int:
        return len(self.asked)

    #: How many definitions still fit when the loop stopped.
    survivors: int = 0

    @property
    def settled(self) -> bool:
        """True when no further question would narrow the answer.

        Not the same as "one candidate left". Several definitions can agree on
        every case there is to ask about -- ``parent(A,C), parent(C,B)`` and
        the same with a redundant guard that happens to hold everywhere -- and
        when that is so, asking more is not learning more. Reporting those as
        unsettled would make the loop look like it failed at the moment it
        succeeded.
        """
        return self.stopped == "settled"

    @property
    def rules(self) -> list[Rule]:
        return list(self.proposal.best or []) if self.proposal else []

    def describe(self) -> str:
        if not self.proposal or not self.proposal.candidates:
            return f"{self.target}: nothing found after {self.questions} question(s)"
        state = "settled" if self.settled else "still undetermined"
        survivors = (
            f", {self.survivors} equivalent definition(s)"
            if self.settled and self.survivors > 1
            else ""
        )
        return (
            f"{self.target}: {state} after {self.questions} question(s)"
            f"{survivors}\n" + self.proposal.describe()
        )

    def to_dict(self) -> dict:
        return {
            "target": str(self.target),
            "questions": self.questions,
            "settled": self.settled,
            "survivors": self.survivors,
            "stopped": self.stopped,
            "asked": [q.to_dict() for q in self.asked],
            "rules": [str(r) for r in self.rules],
        }

    def __str__(self) -> str:
        return self.describe()


class GrowthLoop:
    """Learns a definition by asking the fewest questions it can."""

    def __init__(
        self,
        knowledge,
        hypothesiser: Optional[Hypothesiser] = None,
        max_questions: int = 30,
        refutation_budget: int = 3,
    ) -> None:
        self.knowledge = knowledge
        self.hypothesiser = hypothesiser or Hypothesiser(knowledge)
        self.max_questions = max_questions
        #: How many consecutive "no"s to a refutation probe before settling.
        #:
        #: A refutation probe asks about something no candidate predicts, so a
        #: "no" confirms what they all already said and shrinks nothing --
        #: which means without a budget the loop asks them until it runs out.
        #: This is a *sample* of the region every candidate agrees is false,
        #: and a small one: three noes are weak evidence that the space
        #: contains the answer. What it buys is catching the case where
        #: nothing fits, not proving the definition right.
        self.refutation_budget = refutation_budget

    def learn(
        self,
        target,
        teacher: Teacher,
        positive: Sequence[Atom] = (),
        negative: Sequence[Atom] = (),
        strategy: str = "active",
    ) -> Session:
        """Ask until the definition is settled, or the budget runs out.

        ``strategy`` is ``"active"`` or ``"random"``. The second exists to be
        measured against: a claim that active selection saves questions is only
        worth making next to the alternative, run on the same task.
        """
        signature = Signature.parse(target)
        seeded = list(positive)
        refuted = list(negative)
        if not seeded and not refuted:
            # Whatever the knowledge base already holds for the target. A
            # caller who says nothing means "use what you know", not "start
            # from nothing" -- starting from nothing would make the loop ask
            # for an example it already has.
            #
            # Strong negation is how a "no" is stated: "-grandparent(b, c)."
            # is a claim that it is false, which is exactly a negative
            # example. Failure-to-derive is not, since that is true of
            # everything not yet learned.
            for record in self.knowledge.facts:
                atom = record.atom
                if atom.signature == (signature.name, signature.arity):
                    seeded.append(atom)
                elif (
                    atom.is_negated
                    and atom.positive.signature == (signature.name, signature.arity)
                ):
                    refuted.append(atom.positive)
        session = Session(target=signature, positive=seeded, negative=refuted)
        constants = self.hypothesiser.constants()
        universe = None
        refuted_tries = 0
        # Probes the teacher could not answer. They have to be excluded from
        # the picker's universe, not just from the random walk: the active
        # strategy re-ranks from scratch each round and would otherwise ask
        # the same unanswerable question until the budget ran out.
        unanswerable: set[Atom] = set()

        while session.questions < self.max_questions:
            if not session.positive:
                # Nothing to generalise from. A membership probe here is a
                # shot in the dark over the whole universe; "show me one" is
                # cheaper for the teacher and worth more.
                question = Question(
                    kind=ASK_EXAMPLE,
                    text=f"Show me something that is {signature.name}?",
                )
                session.asked.append(question)
                session.stopped = "no examples to generalise from"
                return session
            # The version space, not the learner's pick. With one example and
            # no negatives almost everything fits, and that wide start is the
            # honest one: it is what the questions are for. Asking the learner
            # instead would hand back a single rule chosen on a tie-break and
            # leave nothing to ask about.
            candidates = self.hypothesiser.version_space(
                signature, session.positive, session.negative
            )
            if candidates:
                # The learner's own answer goes in too. It can use recursion
                # and several clauses, which the single-clause space cannot --
                # without it, "ancestor" settles on its base case and the loop
                # never asks the question that would reveal the recursion.
                learned = self.hypothesiser.for_target(
                    signature,
                    positive=session.positive,
                    # The answered negatives, and *only* those. Letting this
                    # fall through to the closed-world completion marks every
                    # unasked pair false -- including the true ones nobody has
                    # been asked about yet -- which is exactly the evidence a
                    # recursive clause needs and would be rejected for using.
                    negative=list(session.negative),
                )
                for extra in learned.candidates:
                    if extra not in candidates:
                        candidates.append(extra)
                proposal = Proposal(
                    target=signature,
                    candidates=candidates,
                    examples=Examples(
                        positive=list(session.positive),
                        negative=list(session.negative),
                    ),
                )
            else:
                # Nothing single-clause fits. Recursion and multi-clause
                # definitions need the learner's sequential covering.
                proposal = self.hypothesiser.for_target(
                    signature,
                    positive=session.positive,
                    negative=list(session.negative),
                )
            session.proposal = proposal

            if not proposal.candidates:
                # Nothing in the space fits. More negatives will not help; an
                # example might.
                question = Question(
                    kind=ASK_EXAMPLE,
                    text=f"Show me something that is {signature.name}?",
                )
                session.asked.append(question)
                session.stopped = proposal.reason or "no definition fits"
                return session

            picker = QuestionPicker(
                self.knowledge,
                proposal.candidates,
                (signature.name, signature.arity),
                constants,
                known=set(session.positive) | set(session.negative) | unanswerable,
                positive=set(session.positive),
            )
            if universe is None:
                universe = picker.universe()

            question = self._pick(picker, strategy, universe, session, refuted_tries)
            if question is None:
                session.stopped = "settled"
                session.survivors = len(picker.candidates)
                return session

            session.asked.append(question)
            verdict = teacher(question.atom)
            session.answers.append((question.atom, verdict))
            if verdict is None:
                # "I don't know" is a real answer. It narrows nothing, so the
                # probe is retired rather than retried.
                unanswerable.add(question.atom)
                universe = [a for a in universe if a != question.atom]
                continue
            if question.kind == ASK_REFUTATION and not verdict:
                # Confirms what every candidate already said: spends a
                # question and shrinks nothing.
                refuted_tries += 1
            else:
                refuted_tries = 0
            (session.positive if verdict else session.negative).append(question.atom)

        session.stopped = f"reached the {self.max_questions}-question limit"
        return session

    def _pick(
        self,
        picker: QuestionPicker,
        strategy: str,
        universe: Sequence[Atom],
        session: Session,
        refuted_tries: int = 0,
    ) -> Optional[Question]:
        if strategy == "active":
            splitter = picker.rank()
            if splitter:
                return splitter[0]
            if refuted_tries >= self.refutation_budget:
                return None
            asked = {a for a, _ in session.answers}
            for probe in picker.refutations():
                if probe.atom not in asked:
                    return probe
            return None
        if strategy != "random":
            raise ValueError(f"strategy must be 'active' or 'random', not {strategy!r}")
        if picker.settled():
            return None
        # Deterministic "random": walk the universe in order, skipping what has
        # been answered. Seeding a generator would make the comparison depend
        # on the seed, which is not the thing being measured.
        answered = {a for a, _ in session.answers}
        for atom in universe:
            if atom in answered:
                continue
            return Question(kind=ASK_MEMBERSHIP, atom=atom, text=f"Is {atom} true?")
        return None
