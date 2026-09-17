"""Why an answer was wrong -- exactly one reason, chosen deterministically.

Phase One had three categories and the roadmap asks for six. The extra ones
only became meaningful as the mind grew parts that could fail in new ways: a
specialist that is not installed (P2.1), a context read wrongly (P2.4),
knowledge nobody ever stated (P2.6).

The hard requirement is "every failure has **exactly one** attribution", and
that is a statement about *priority*, not about taxonomy. A single wrong answer
usually has several true things to say about it -- the budget ran out *and* the
theory was misread *and* no rule covered the case -- and a breakdown that
counts one failure three times cannot be used to decide what to fix. So the
categories are ordered, most upstream first, and the first that applies is the
one recorded.

The order, and why each sits where it does:

1. ``engine`` -- the reasoner gets it wrong even handed the *gold* theory.
   First because it invalidates everything downstream: if the engine is wrong
   then "perception was wrong" is not established, it is merely consistent.
   This is the category Phase One exists to keep at zero.
2. ``budget`` -- it ran out of time. Second because an unfinished answer tells
   you nothing about whether the knowledge was there; attributing it to a
   missing rule would be inventing a finding.
3. ``specialist`` -- the question needed a specialist that is absent or failed.
   A known, structural gap, and one a user can close by installing something.
4. ``wrong_context`` -- the mind decided it was somewhere it is not. Above
   perception because a misread context selects the wrong vocabulary, so the
   perception error is downstream of it and fixing perception would not help.
5. ``perception`` -- the symbols extracted differ from what the text meant.
6. ``missing_knowledge`` -- the reasoning is right and needs a fact nobody
   stated. Distinct from a missing rule: the fix is an import, not a rule.
7. ``missing_rule`` -- everything was read correctly, everything needed was
   present, and no rule covers the case. Last because it is what remains when
   nothing else applies, and a category of last resort must be checked last or
   it swallows the others.

Nothing here guesses. Each category is decided by evidence the system already
produces -- a gold closure to compare against, a controller's ``cause``, a
diagnosis's missing literal -- and :func:`attribute` refuses rather than
choosing when it has none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = [
    "Attribution",
    "Evidence",
    "attribute",
    "breakdown",
    "REMEDY",
    "CATEGORIES",
    "ENGINE",
    "BUDGET",
    "SPECIALIST",
    "WRONG_CONTEXT",
    "PERCEPTION",
    "MISSING_KNOWLEDGE",
    "MISSING_RULE",
    "UNATTRIBUTED",
]

ENGINE = "engine"
BUDGET = "budget"
SPECIALIST = "specialist"
WRONG_CONTEXT = "wrong_context"
PERCEPTION = "perception"
MISSING_KNOWLEDGE = "missing_knowledge"
MISSING_RULE = "missing_rule"

#: When there is no evidence for any category. Counted, never hidden: a
#: benchmark that cannot say why it failed has a gap in its own instrumentation
#: and should show it rather than pick the most plausible story.
UNATTRIBUTED = "unattributed"

#: Most upstream first. The order *is* the design; see the module docstring.
CATEGORIES = (
    ENGINE,
    BUDGET,
    SPECIALIST,
    WRONG_CONTEXT,
    PERCEPTION,
    MISSING_KNOWLEDGE,
    MISSING_RULE,
)

#: What a user would do about each one.
REMEDY = {
    ENGINE: "fix the reasoner -- this one is a bug, not a gap",
    BUDGET: "raise the time budget, or make the question cheaper",
    SPECIALIST: "install the specialist the question needs",
    WRONG_CONTEXT: "the facet rules read this host wrongly",
    PERCEPTION: "the text layer extracted the wrong symbols",
    MISSING_KNOWLEDGE: "a fact nobody stated -- import it or be told it",
    MISSING_RULE: "no rule covers this case -- learn one or write one",
    UNATTRIBUTED: "the benchmark could not say; instrument it",
}


@dataclass(frozen=True)
class Evidence:
    """What is known about one wrong answer. Absent fields mean "not checked".

    Every field is something the system produces in the course of answering,
    not something computed for the benchmark's benefit -- which is what keeps
    the attribution honest rather than circular.
    """

    #: The gold theory gives the wrong answer too.
    gold_disagrees: bool = False
    #: The controller stopped for time.
    out_of_budget: bool = False
    #: A specialist was needed and was missing or broken.
    specialist_missing: Optional[str] = None
    #: The context reading disagreed with what the host actually is.
    context_wrong: Optional[str] = None
    #: The extracted theory differs from the gold theory.
    theory_mismatch: bool = False
    #: The question itself was read as a different goal.
    question_mismatch: bool = False
    #: A ground literal nothing states and no rule derives.
    missing_fact: Optional[str] = None
    #: Rules exist for the goal but none fired.
    rules_stalled: bool = False

    def to_dict(self) -> dict:
        return {
            key: value
            for key, value in {
                "gold_disagrees": self.gold_disagrees,
                "out_of_budget": self.out_of_budget,
                "specialist_missing": self.specialist_missing,
                "context_wrong": self.context_wrong,
                "theory_mismatch": self.theory_mismatch,
                "question_mismatch": self.question_mismatch,
                "missing_fact": self.missing_fact,
                "rules_stalled": self.rules_stalled,
            }.items()
            if value
        }


@dataclass(frozen=True)
class Attribution:
    """The one category a failure is recorded under, and why."""

    category: str
    detail: str = ""
    evidence: Optional[Evidence] = None

    @property
    def remedy(self) -> str:
        return REMEDY.get(self.category, "")

    def describe(self) -> str:
        return f"{self.category}: {self.detail}" if self.detail else self.category

    def to_dict(self) -> dict:
        payload = {"category": self.category, "detail": self.detail}
        if self.evidence is not None:
            payload["evidence"] = self.evidence.to_dict()
        return payload

    def __str__(self) -> str:
        return self.describe()


def attribute(evidence: Evidence) -> Attribution:
    """The first category that applies, in the order that matters.

    Deliberately a chain of ``if``s rather than a score: a failure belongs to
    the most upstream thing that went wrong, and the point of the ordering is
    that it is inspectable. A weighted combination would be harder to argue
    with and no more correct.
    """
    if evidence.gold_disagrees:
        return Attribution(
            ENGINE,
            "the engine disagrees with the gold closure, so nothing downstream "
            "of it is established",
            evidence,
        )
    if evidence.out_of_budget:
        return Attribution(
            BUDGET, "stopped for time, so what it would have said is unknown", evidence
        )
    if evidence.specialist_missing:
        return Attribution(
            SPECIALIST,
            f"needed the {evidence.specialist_missing} specialist",
            evidence,
        )
    if evidence.context_wrong:
        return Attribution(
            WRONG_CONTEXT,
            f"read the host as {evidence.context_wrong}, which selects the "
            f"wrong vocabulary",
            evidence,
        )
    if evidence.theory_mismatch or evidence.question_mismatch:
        what = "the question" if evidence.question_mismatch else "the theory"
        return Attribution(PERCEPTION, f"{what} was read as different symbols", evidence)
    if evidence.missing_fact:
        return Attribution(
            MISSING_KNOWLEDGE,
            f"nothing states {evidence.missing_fact}",
            evidence,
        )
    if evidence.rules_stalled:
        return Attribution(
            MISSING_RULE, "rules exist for the goal and none of them fired", evidence
        )
    return Attribution(
        UNATTRIBUTED, "no evidence pointed at a cause", evidence
    )


def breakdown(attributions: Sequence[Attribution]) -> dict:
    """Counts by category, in the priority order rather than by size.

    Ordered by priority on purpose: the most upstream non-zero row is the one
    worth working on, and sorting by count would bury an engine failure under
    a hundred perception ones.
    """
    counts = {category: 0 for category in CATEGORIES}
    counts[UNATTRIBUTED] = 0
    for attribution in attributions:
        counts[attribution.category] = counts.get(attribution.category, 0) + 1
    return {category: n for category, n in counts.items() if n}
