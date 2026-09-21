"""Asking a person, one question at a time -- and checking the answers.

Two ideas here, and the second is the one that matters.

**Ask in the order that unlocks most.** A developer's attention is the
scarcest thing the builder consumes, so questions are ranked by how much the
answer would settle: a unit confirms one field, an effect makes an action
usable by the planner, a rule explains a flag nobody could derive. Questions
whose answers can be inferred from an earlier answer are dropped rather than
asked, which is why the count of questions asked is a number worth reporting.

**An answer is a claim, and claims are checked.** This is the roadmap's third
acceptance test and the reason this module is not just a prompt list. When
somebody says "yes, ``close_valve_b`` sets ``flow`` to 0", that becomes a
statement about every observation the builder will ever see. If a later record
shows ``close_valve_b`` leaving flow at 12, the claim has a counterexample --
and the builder reopens the question with the record attached, rather than
carrying a wrong answer into the pack where it will quietly break something.

That asymmetry is deliberate: a person can be wrong, the log cannot be argued
with, and the builder trusts the log. What it does *not* do is silently
override the person -- being contradicted reopens a question, it does not
decide one. A developer who meant it can say so again, with the
counterexample in front of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

__all__ = [
    "Question",
    "Answer",
    "Claim",
    "Counterexample",
    "Interview",
    "YES",
    "NO",
    "UNSURE",
]

YES, NO, UNSURE = "yes", "no", "unsure"

#: What each kind of answer settles, used to order the asking. Higher first.
WORTH = {
    "effect": 40,     # makes an action usable by the planner
    "rule": 30,       # explains something nothing could derive
    "unit": 20,       # settles one field, and every quantity check on it
    "state": 15,
    "name": 5,
    "licence": 5,
}


@dataclass(frozen=True)
class Counterexample:
    """One record that contradicts a claim."""

    index: int
    record: dict
    why: str

    def describe(self) -> str:
        return f"record {self.index}: {self.why}"


@dataclass
class Claim:
    """What an answer committed to, and how to tell if it was wrong."""

    key: str
    statement: str
    #: ``(log) -> list[Counterexample]``. Never mutates anything.
    check: Callable[[Sequence[dict]], list]
    answer: str = YES

    def test(self, log: Sequence[dict]) -> list[Counterexample]:
        if self.answer != YES:
            return []
        return list(self.check(log))

    def __str__(self) -> str:
        return self.statement


@dataclass
class Question:
    """One thing to ask, and what a yes would commit to."""

    key: str
    kind: str
    text: str
    options: tuple = (YES, NO)
    #: What is proposed, for whoever has to act on the answer.
    detail: str = ""
    #: Built when the question is answered yes.
    claim: Optional[Claim] = None
    #: Records that reopened this question, if it was answered before.
    contradicted_by: tuple = ()

    @property
    def worth(self) -> int:
        return WORTH.get(self.kind, 10)

    @property
    def reopened(self) -> bool:
        return bool(self.contradicted_by)

    def describe(self) -> str:
        choices = " / ".join(self.options)
        head = f"{self.text} [{choices}]"
        if self.reopened:
            evidence = self.contradicted_by[0].describe()
            return (
                f"{head}\n    (you said yes before; {evidence} — "
                f"{len(self.contradicted_by)} record(s) disagree)"
            )
        return head

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "kind": self.kind,
            "text": self.text,
            "options": list(self.options),
            "detail": self.detail,
            "reopened": self.reopened,
            "contradicted_by": [c.describe() for c in self.contradicted_by],
        }

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Answer:
    """What somebody said, and when."""

    key: str
    reply: str
    note: str = ""

    @property
    def agreed(self) -> bool:
        return self.reply == YES


class Interview:
    """The questions outstanding, the answers given, and the claims standing.

    Holds no opinion about *what* to ask -- the builder supplies questions.
    What it owns is the order, the record of what was settled, and the
    re-checking that turns a wrong answer into a reopened question rather
    than a quiet mistake.
    """

    def __init__(self) -> None:
        self.pending: list[Question] = []
        self.answers: dict[str, Answer] = {}
        self.claims: dict[str, Claim] = {}
        self.asked: int = 0
        #: Every time a claim was contradicted, in order.
        self.reopenings: list[tuple] = []

    # -- what to ask --------------------------------------------------------

    def offer(self, questions: Iterable[Question]) -> "Interview":
        """Add questions, skipping any already settled and not reopened."""
        known = {q.key for q in self.pending}
        for question in questions:
            if question.key in known:
                continue
            if question.key in self.answers and not question.reopened:
                continue
            self.pending.append(question)
            known.add(question.key)
        self.pending.sort(key=lambda q: (q.reopened, q.worth), reverse=True)
        return self

    @property
    def next(self) -> Optional[Question]:
        return self.pending[0] if self.pending else None

    def __len__(self) -> int:
        return len(self.pending)

    # -- answering ----------------------------------------------------------

    def answer(self, key: str, reply: str, note: str = "") -> Answer:
        """Record a reply. A yes stands its claim up to be checked later."""
        question = next((q for q in self.pending if q.key == key), None)
        if question is None:
            raise KeyError(f"nothing outstanding with key {key!r}")
        self.pending.remove(question)
        self.asked += 1

        given = Answer(key, reply, note)
        self.answers[key] = given
        if question.claim is not None:
            claim = question.claim
            claim.answer = reply
            if reply == YES:
                self.claims[key] = claim
            else:
                self.claims.pop(key, None)
        return given

    def settled(self, key: str) -> bool:
        return key in self.answers

    def agreed(self, key: str) -> bool:
        given = self.answers.get(key)
        return given is not None and given.agreed

    # -- checking what was said --------------------------------------------

    def recheck(
        self, log: Sequence[dict], questions: Optional[dict] = None
    ) -> list[Question]:
        """Test every standing claim against the log; reopen what fails.

        Returns the questions that came back. A claim with a counterexample
        is not overruled -- it is *asked again*, with the record that
        disagrees attached, because the person may know something the log
        does not show and the builder does not get to decide that.
        """
        questions = questions or {}
        reopened = []
        for key, claim in list(self.claims.items()):
            failures = claim.test(log)
            if not failures:
                continue
            original = questions.get(key)
            question = Question(
                key=key,
                kind=original.kind if original else "effect",
                text=original.text if original else claim.statement,
                options=original.options if original else (YES, NO),
                detail=claim.statement,
                claim=claim,
                contradicted_by=tuple(failures[:3]),
            )
            self.answers.pop(key, None)
            self.claims.pop(key, None)
            self.reopenings.append((key, tuple(failures[:3])))
            reopened.append(question)
        if reopened:
            self.offer(reopened)
        return reopened

    # -- reporting ----------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"{self.asked} question(s) asked, {len(self.claims)} claim(s) "
            f"standing, {len(self.pending)} outstanding"
        ]
        if self.reopenings:
            lines.append(f"{len(self.reopenings)} answer(s) contradicted later:")
            for key, failures in self.reopenings:
                lines.append(f"  {key}: {failures[0].describe()}")
        if self.next is not None:
            lines.append(f"next: {self.next.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "asked": self.asked,
            "outstanding": [q.to_dict() for q in self.pending],
            "answers": {k: v.reply for k, v in self.answers.items()},
            "claims": [c.statement for c in self.claims.values()],
            "reopenings": [
                {"key": key, "because": [f.describe() for f in failures]}
                for key, failures in self.reopenings
            ],
        }

    def __repr__(self) -> str:
        return (
            f"Interview({self.asked} asked, {len(self.pending)} outstanding, "
            f"{len(self.reopenings)} reopened)"
        )
