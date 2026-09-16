"""The controller: an agenda, a time budget, and an answer either way.

It runs the loop the blackboard implies. Take the most wanted goal, ask the
router who might have something to say about it, run them in order, post what
they find, repeat until the question is answered or the budget runs out.

Two properties matter more than speed.

**It always answers.** Running out of time produces ``unknown`` with the reason
"budget", not a hang and not an exception. A host with a frame to render cannot
use a reasoner that sometimes takes a second, and "I did not finish" is a
usable answer where silence is not.

**It degrades one specialist at a time.** A missing optional dependency takes
out exactly the questions that needed it, and those questions come back
``unknown`` naming what is missing. Everything else is unaffected, because
nothing in the core ever depended on the specialist being there.

The budget is cooperative -- see :class:`~neuralmind.workspace.specialist.Budget`.
Nothing can interrupt a specialist mid-call, so the controller checks between
calls and specialists check within them. In exchange the overshoot is bounded
by one specialist call rather than being unbounded, and the tests hold it to
10%.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..core.terms import Atom
from ..inference.proof import ProofNode
from .blackboard import Workspace
from .router import Router
from .specialist import Budget, Result

__all__ = ["Controller", "Conclusion"]

#: Why a goal came back unanswered.
NO_SPECIALIST = "no specialist"
BUDGET = "budget"
NOT_ESTABLISHED = "not established"


@dataclass
class Conclusion:
    """The controller's answer: a status, a proof if there is one, and why not."""

    goal: Atom
    status: str
    atom: Optional[Atom] = None
    proof: Optional[ProofNode] = None
    #: Which specialists ran, in order.
    consulted: list[str] = field(default_factory=list)
    #: What each one said when it found nothing, for the brief line.
    reasons: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    rounds: int = 0
    #: Set when the budget ran out: ``budget``, ``no specialist``, or
    #: ``not established``.
    cause: str = ""

    def __bool__(self) -> bool:
        return self.status == "yes"

    @property
    def reason(self) -> str:
        """One line saying why there is no answer, in a person's terms.

        The first reason, not all of them. A query that walks a chain of
        subgoals collects one reason per link, and the one that matters is the
        first -- what the goal itself was missing. The rest are the detail
        behind that, and are kept in :attr:`reasons` for anyone who wants them.
        """
        if self.status == "yes":
            return ""
        if self.cause == BUDGET:
            return f"ran out of time after {self.elapsed_ms:.0f}ms"
        if self.cause == NO_SPECIALIST:
            return "no specialist here handles that kind of question"
        return self.reasons[0] if self.reasons else "nothing established it"

    def to_dict(self) -> dict:
        payload = {
            "goal": str(self.goal),
            "status": self.status,
            "consulted": list(self.consulted),
            "elapsed_ms": round(self.elapsed_ms, 2),
            "rounds": self.rounds,
        }
        if self.atom is not None:
            payload["atom"] = str(self.atom)
        if self.proof is not None:
            payload["proof"] = self.proof.to_dict()
        if self.status != "yes":
            payload["cause"] = self.cause
            payload["reason"] = self.reason
        return payload

    def __str__(self) -> str:
        if self.proof is not None:
            return str(self.proof)
        return f"{self.goal}: {self.status} -- {self.reason}"


class Controller:
    """Runs specialists against an agenda until the goal is settled."""

    def __init__(
        self,
        workspace: Workspace,
        specialists: Sequence[object],
        budget_ms: float = 20.0,
        max_rounds: int = 200,
    ) -> None:
        self.workspace = workspace
        self.router = Router(specialists)
        self.budget_ms = float(budget_ms)
        #: A stop against an agenda that never settles, not a performance
        #: knob -- the budget is what actually bounds a query. A chain across
        #: three specialists can easily need twenty rounds, and cutting it off
        #: there would look like "unknown" when it was really "not finished".
        self.max_rounds = max_rounds
        self._warm = False

    def warm(self) -> "Controller":
        """Pay every specialist's one-time setup now, outside any budget.

        Importing z3 costs about 30ms and building pint's unit registry about
        260ms -- each several times a realistic per-query budget, and each paid
        exactly once. Charging that to whichever query happened to be first
        would make the budget meaningless and the first answer look like a
        reasoning failure. So it is paid up front, deliberately and visibly.
        """
        if self._warm:
            return self
        self._warm = True
        for specialist in self.router.specialists:
            prepare = getattr(specialist, "prepare", None)
            if prepare is None:
                continue
            try:
                prepare()
            except Exception:  # warming is best-effort; a failure surfaces later
                pass
        return self

    def solve(self, goal: Atom, budget_ms: Optional[float] = None) -> Conclusion:
        """Work on ``goal`` until it holds, nothing is left to try, or time runs out."""
        self.warm()
        budget = Budget(self.budget_ms if budget_ms is None else budget_ms)
        consulted: list[str] = []
        reasons: list[str] = []
        # Facts persist between queries -- that is what a blackboard is for --
        # but goals do not. A subgoal left unsatisfied by an earlier question
        # would otherwise be picked up here and answered instead of this one,
        # and the reason reported would belong to a different query entirely.
        self.workspace.clear_goals()
        self.workspace.want(goal, priority=1.0)
        rounds = 0
        cause = NOT_ESTABLISHED
        answered: Optional[Atom] = None
        # Goals every specialist has already had a go at and got nowhere on.
        # Without this the controller livelocks on a goal whose subgoal cannot
        # be established: it re-posts the parent, re-derives the same subgoal,
        # fails it again, and does that until the budget runs out -- reporting
        # "out of time" for something it had actually settled on the first try.
        attempted: set[Atom] = set()

        while rounds < self.max_rounds:
            if self.workspace.holds(goal):
                answered = goal
                break
            match = self._matching(goal)
            if match is not None:
                answered = match
                break
            if budget.exhausted:
                cause = BUDGET
                break
            taken = self.workspace.take_goal()
            current = taken.atom if taken is not None else goal
            if current in attempted:
                if taken is None:
                    break  # the query's own goal, already tried: nothing left
                continue
            candidates = self.router.rank(current, self.workspace)
            if not candidates:
                if current == goal:
                    cause = NO_SPECIALIST
                    break
                reasons.append(f"nothing handles {current}")
                continue
            rounds += 1
            progressed = False
            for candidate in candidates:
                if budget.exhausted:
                    cause = BUDGET
                    break
                consulted.append(candidate.name)
                result = self._run(candidate, current, budget)
                for finding in result.findings:
                    if self.workspace.post(
                        finding.atom, finding.proof, candidate.name, finding.confidence
                    ):
                        progressed = True
                if result.subgoals:
                    # A subgoal outranks the goal that raised it: finish what
                    # you started before widening. The goal itself goes back on
                    # the agenda underneath them, because it is exactly what
                    # should be retried once they land -- dropping it is how a
                    # chain of specialists stops one link short of the answer.
                    #
                    # Only for subgoals nothing has tried yet, though. Re-posting
                    # a parent whose subgoal already failed is the livelock.
                    depth = (taken.priority if taken is not None else 1.0) + 1.0
                    fresh = [
                        subgoal
                        for subgoal in result.subgoals
                        if not self.workspace.holds(subgoal) and subgoal not in attempted
                    ]
                    for subgoal in fresh:
                        self.workspace.want(subgoal, priority=depth, parent=current)
                        progressed = True
                    if fresh:
                        self.workspace.want(
                            current,
                            priority=depth - 0.5,
                            parent=taken.parent if taken is not None else None,
                        )
                if result.exhausted:
                    cause = BUDGET
                if result.reason and not result.findings:
                    # Only the query's own goal explains the query. A reason
                    # from three subgoals down is detail, not the answer to
                    # "why don't you know?".
                    label = "" if current == goal else f"({current}) "
                    reasons.append(f"{label}{candidate.name}: {result.reason}")
                if result.findings:
                    break
            if not progressed:
                # Everyone had a turn and nothing moved. Do not come back to
                # this goal; whatever raised it has to look elsewhere.
                attempted.add(current)
                if not self.workspace.goals:
                    break

        if answered is None and self.workspace.holds(goal):
            answered = goal
        if answered is None:
            answered = self._matching(goal)

        if answered is not None:
            return Conclusion(
                goal=goal,
                status="yes",
                atom=answered,
                proof=self.workspace.proof(answered),
                consulted=consulted,
                elapsed_ms=budget.elapsed_ms,
                rounds=rounds,
            )
        if budget.exhausted:
            cause = BUDGET
        self.workspace.ask(goal, _question_reason(cause, reasons), "controller")
        return Conclusion(
            goal=goal,
            status="unknown",
            consulted=consulted,
            # The goal's own reasons first: they are what the query was about.
            reasons=_ordered(reasons),
            elapsed_ms=budget.elapsed_ms,
            rounds=rounds,
            cause=cause,
        )

    # -- helpers ------------------------------------------------------------

    def _matching(self, goal: Atom) -> Optional[Atom]:
        """A posted atom that answers a goal with variables in it."""
        if goal.is_ground:
            return goal if self.workspace.holds(goal) else None
        from ..inference.model import match_atom

        for atom in self.workspace:
            if atom.signature == goal.signature and match_atom(goal, atom, {}) is not None:
                return atom
        return None

    def _run(self, candidate, goal: Atom, budget: Budget) -> Result:
        """Call a specialist, and never let it take the query down with it."""
        try:
            result = candidate.specialist.run(goal, self.workspace, budget)
        except Exception as exc:  # a specialist fault is that specialist's problem
            return Result.nothing(f"failed: {_brief(exc)}")
        return result if isinstance(result, Result) else Result.nothing("returned nothing")


def _question_reason(cause: str, reasons: Sequence[str]) -> str:
    if cause == BUDGET:
        return "the time budget ran out"
    if cause == NO_SPECIALIST:
        return "no specialist accepted it"
    return "; ".join(_dedupe(reasons)) or "nothing established it"


def _dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(items))


def _ordered(reasons: Sequence[str]) -> list[str]:
    """Reasons about the query's own goal first, then the subgoal detail."""
    unique = _dedupe(reasons)
    direct = [r for r in unique if not r.startswith("(")]
    return direct + [r for r in unique if r.startswith("(")]


def _brief(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc).splitlines()[0]}" if str(exc) else type(exc).__name__
