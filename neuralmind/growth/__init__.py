"""The growth loop: noticing what is missing, and closing it by asking.

Phase One could learn a rule when you handed it examples and named the target.
That is a tool, not growth. What was missing is the part that notices a gap
without being pointed at one, works out which question would settle it, and
then *does not use the answer* until someone confirms it.

Five pieces:

:mod:`~neuralmind.growth.gaps`
    Gaps collected from diagnoses the system already produces -- a stalled
    rule, a refused sentence, a broken constraint. Never inferred, so a gap
    always arrives with its evidence.
:mod:`~neuralmind.growth.hypothesize`
    Induction over a gap, keeping the definitions that compete rather than
    only the winner. The competitors are what a question is for.
:mod:`~neuralmind.growth.questions`
    The probe that splits the candidates most evenly -- and, when they all
    agree, one that would refute them all at once.
:mod:`~neuralmind.growth.loop`
    Propose, ask, fold in, repeat.
:mod:`~neuralmind.growth.consolidate`
    Predicate invention: naming the conjunction that keeps coming back, as a
    rewriting that is checked to change no answer.
:mod:`~neuralmind.growth.memory`
    What survives a restart, with a belief state on everything: proposed,
    confirmed, retired. Only confirmed things answer questions.

Nothing here writes to the knowledge base. A settled definition is a
*proposal*; :class:`Grower` hands it to the safety kernel, which decides. That
split is the reason the kernel came first: the loop's job is to work out what
the evidence supports, and the kernel's is to refuse it anyway when it breaks
something.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

from .consolidate import Consolidator, Invention
from .gaps import (
    INCONSISTENT,
    MISSING_FACT,
    MISSING_RULE,
    UNREADABLE,
    Gap,
    GapCollector,
)
from .hypothesize import Hypothesiser, Proposal
from .loop import GrowthLoop, Session, Teacher
from .memory import CONFIRMED, PROPOSED, RETIRED, Belief, Memory
from .questions import (
    ASK_EXAMPLE,
    ASK_MEMBERSHIP,
    ASK_REFUTATION,
    Question,
    QuestionPicker,
    split_score,
)

__all__ = [
    "Grower",
    "GapCollector", "Gap",
    "MISSING_RULE", "MISSING_FACT", "UNREADABLE", "INCONSISTENT",
    "Hypothesiser", "Proposal",
    "QuestionPicker", "Question", "split_score",
    "ASK_MEMBERSHIP", "ASK_REFUTATION", "ASK_EXAMPLE",
    "GrowthLoop", "Session", "Teacher",
    "Consolidator", "Invention",
    "Memory", "Belief", "PROPOSED", "CONFIRMED", "RETIRED",
]


class Grower:
    """The loop, wired to a safety kernel so nothing it learns goes unchecked.

    The division of labour is the point. :class:`GrowthLoop` decides what the
    evidence supports; :class:`~neuralmind.kernel.Kernel` decides whether it
    may be believed. Neither can do the other's job, and a proposal that
    survives the first and fails the second is the normal, healthy case.
    """

    def __init__(
        self,
        kernel,
        memory: Optional[Memory] = None,
        max_questions: int = 30,
    ) -> None:
        self.kernel = kernel
        self.memory = memory or Memory()
        self.gaps = GapCollector()
        self.max_questions = max_questions

    # -- noticing -----------------------------------------------------------

    def notice(self, answer) -> Optional[Gap]:
        """Record what an unsuccessful answer says is missing."""
        return self.gaps.from_answer(answer)

    def ask(self, goal, **kwargs):
        """Answer, and notice the gap if there is one."""
        answer = self.kernel.ask(goal, **kwargs)
        if answer is not None:
            self.notice(answer)
        return answer

    # -- growing ------------------------------------------------------------

    def grow(
        self,
        target,
        teacher,
        positive: Sequence = (),
        negative: Sequence = (),
        strategy: str = "active",
    ) -> Session:
        """Run one growth cycle and store the result as a *proposal*.

        Stored, not believed. Design rule 5, and a test asserts it: nothing
        learned changes an answer until a person confirms it.
        """
        knowledge = self.kernel.layers.view(self.kernel.mode_layers)
        loop = GrowthLoop(knowledge, max_questions=self.max_questions)
        session = loop.learn(target, teacher, positive, negative, strategy)
        for rule in session.rules:
            self.memory.remember(
                str(rule),
                kind="rule",
                state=PROPOSED,
                provenance=f"growth:{session.target}",
                evidence=f"{session.questions} question(s)",
            )
        return session

    def confirm(self, belief, by: str = "") -> tuple:
        """Put a proposal through the kernel. Only the kernel may accept it."""
        if not by:
            raise ValueError("confirming a proposal requires naming who did it")
        record = (
            belief
            if isinstance(belief, Belief)
            else self.memory.find(str(belief), "rule")
        )
        if record is None:
            raise KeyError(f"no proposal matching {belief}")
        outcome = self.kernel.learn(record.body, name=record.provenance or "growth")
        if outcome:
            self.memory.confirm(record, f"confirmed by {by}")
        else:
            self.memory.retire(record, f"refused by the kernel: {outcome.reason}")
        return record, outcome

    # -- tidying ------------------------------------------------------------

    def consolidate(self, limit: int = 3) -> list[Invention]:
        """Propose names for conjunctions that keep recurring.

        Only inventions whose rewrite is verified to change no answer are
        returned; the rest are dropped, because a "simplification" that moves
        an answer is not one.
        """
        knowledge = self.kernel.layers.view(self.kernel.mode_layers)
        consolidator = Consolidator(knowledge)
        kept = []
        for invention in consolidator.propose(limit=limit * 3):
            rewritten = consolidator.apply(invention)
            identical, _why = consolidator.verify(rewritten)
            if identical:
                kept.append(invention)
            if len(kept) >= limit:
                break
        return kept

    # -- reporting ----------------------------------------------------------

    def questions(self) -> list[str]:
        """What the mind would need to be told, worst gap first."""
        return [gap.describe() for gap in self.gaps.ranked()]

    def summary(self) -> dict:
        return {
            "gaps": self.gaps.summary(),
            "memory": self.memory.summary(),
        }
