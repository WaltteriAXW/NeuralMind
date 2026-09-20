"""Carrying a plan out, and noticing when it has stopped making sense.

The world does not hold still while a plan runs. A door locks, a valve someone
else opened is open already, a key is not where it was. The question this
module answers is not "did something change" -- something always changes --
but **does this plan still work**, which is a much narrower question with a
much more useful answer.

The narrow version is answerable from the causal links. A surprise matters if
it breaks a precondition of a step still to come, or undoes a fluent that a
later step or the goal depends on. Anything else is news, not a problem. A
mind that replans on every difference throws away a good plan because a light
turned on somewhere; one that replans on none of them walks into a locked
door. The links are what let it tell the two apart, and they are the same
links that explain the plan in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from ..core.terms import Atom
from .domain import Domain, Goal, State, as_atoms
from .plan import GOAL, Plan, Step

__all__ = ["Execution", "Surprise", "Progress"]


@dataclass(frozen=True)
class Surprise:
    """The difference between what was expected and what was observed."""

    appeared: frozenset[Atom] = frozenset()
    vanished: frozenset[Atom] = frozenset()
    #: Links the surprise broke. Empty means the plan survives it.
    broken: tuple = ()
    #: Preconditions of remaining steps that no longer hold.
    unmet: tuple[Atom, ...] = ()

    @property
    def any_change(self) -> bool:
        return bool(self.appeared or self.vanished)

    @property
    def matters(self) -> bool:
        """Whether the plan has to be rebuilt, rather than merely noted."""
        return bool(self.broken or self.unmet)

    def describe(self) -> str:
        if not self.any_change:
            return "nothing unexpected"
        parts = []
        if self.appeared:
            parts.append("now true: " + ", ".join(sorted(str(a) for a in self.appeared)))
        if self.vanished:
            parts.append(
                "no longer true: " + ", ".join(sorted(str(a) for a in self.vanished))
            )
        verdict = "the plan no longer works" if self.matters else "the plan still works"
        return f"{'; '.join(parts)} — {verdict}"

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Progress:
    """One move: what was done, what happened, and what follows."""

    step: Optional[Step] = None
    surprise: Optional[Surprise] = None
    replanned: bool = False
    done: bool = False
    stuck: str = ""

    def describe(self) -> str:
        if self.stuck:
            return f"stuck: {self.stuck}"
        if self.done:
            return "done"
        line = f"did {self.step.action}" if self.step else "nothing to do"
        if self.surprise is not None and self.surprise.any_change:
            line += f" — {self.surprise.describe()}"
        if self.replanned:
            line += " — replanned"
        return line

    def __str__(self) -> str:
        return self.describe()


class Execution:
    """A plan being carried out against a world that keeps moving.

    Holds the plan, where it has got to, and what it expects to be true. Feed
    it observations; it tells you the next thing to do, and replans by itself
    when -- and only when -- the surprise actually broke something.
    """

    def __init__(
        self,
        domain: Domain,
        planner=None,
        goal: Optional[Goal] = None,
    ) -> None:
        from .planner import Planner

        self.domain = domain
        self.planner = planner if planner is not None else Planner()
        self.goal = goal if goal is not None else domain.goal
        self.plan: Plan = self.planner.plan(domain, self.goal)
        self.position = 0
        self.history: list[Progress] = []
        #: Every replan, with what forced it -- the record of a plan's life.
        self.replans: list[Surprise] = []

    # -- where it is --------------------------------------------------------

    @property
    def state(self) -> State:
        return self.domain.state

    @property
    def finished(self) -> bool:
        return self.goal.satisfied_by(self.state)

    @property
    def remaining(self) -> list[Step]:
        return self.plan.steps[self.position :]

    @property
    def next_step(self) -> Optional[Step]:
        return self.remaining[0] if self.remaining else None

    # -- taking one step ----------------------------------------------------

    def observe(self, facts: Iterable[Union[str, Atom]]) -> Surprise:
        """Replace what is believed to hold, and say whether that matters.

        Takes the whole observed state rather than a delta. A host reporting
        what it sees is reliable; a host reporting what changed has to have
        got the bookkeeping right, and when it has not, the mind's picture
        drifts from the world silently.
        """
        observed = State.of(facts, self.domain.library)
        expected = self.state
        appeared, vanished = expected.difference(observed)
        self.domain = Domain(
            self.domain.library, observed, self.goal, self.domain.rules
        )

        if not (appeared or vanished):
            return Surprise()

        broken = tuple(
            link
            for link in self.plan.links
            if link.producer < self.position
            and (link.consumer >= self.position or link.consumer == GOAL)
            and self._link_broken(link, observed)
        )
        unmet = tuple(
            need
            for step in self.remaining
            for need in step.needs
            if need not in observed.fluents
        ) + tuple(
            absent
            for step in self.remaining
            for absent in step.needs_absent
            if absent in observed.fluents
        )
        # A precondition that a step still to come was going to establish is
        # not unmet -- it was never expected to hold yet.
        pending = {a for step in self.remaining for a in step.adds}
        unmet = tuple(u for u in unmet if u not in pending)

        return Surprise(appeared, vanished, broken, unmet)

    def _link_broken(self, link, observed: State) -> bool:
        if link.removes:
            return link.fluent in observed.fluents
        return link.fluent not in observed.fluents

    def step(
        self, facts: Optional[Iterable[Union[str, Atom]]] = None
    ) -> Progress:
        """Advance one action, replanning first if the world made that necessary."""
        surprise = self.observe(facts) if facts is not None else None

        if self.finished:
            progress = Progress(surprise=surprise, done=True)
            self.history.append(progress)
            return progress

        if surprise is not None and surprise.matters:
            self.replans.append(surprise)
            self.plan = self.planner.plan(self.domain, self.goal)
            self.position = 0
            if not self.plan.found:
                progress = Progress(
                    surprise=surprise, replanned=True, stuck=self.plan.reason
                )
                self.history.append(progress)
                return progress
            progress = Progress(
                step=self.next_step, surprise=surprise, replanned=True
            )
        else:
            progress = Progress(step=self.next_step, surprise=surprise)

        if progress.step is None:
            progress = Progress(surprise=surprise, stuck="the plan is empty")
            self.history.append(progress)
            return progress

        self._apply(progress.step)
        self.position += 1
        self.history.append(progress)
        return progress

    def run(self, world=None, limit: int = 100) -> list[Progress]:
        """Carry the plan out to the end, or until nothing more can be done.

        ``world`` is an optional callable taking the step just performed and
        returning what is observed afterwards -- that is the whole environment
        interface, and it is deliberately one function, because anything
        wider is a commitment to a particular kind of environment.
        """
        moves: list[Progress] = []
        for _ in range(limit):
            if self.finished:
                moves.append(Progress(done=True))
                break
            facts = None
            last = moves[-1].step if moves and moves[-1].step is not None else None
            if world is not None and last is not None:
                facts = world(last)
            progress = self.step(facts)
            moves.append(progress)
            if progress.stuck or progress.done:
                break
        return moves

    def _apply(self, step: Step) -> None:
        """Believe the action worked, until an observation says otherwise."""
        fluents = (set(self.state.fluents) - set(step.cancels)) | set(step.adds)
        self.domain = Domain(
            self.domain.library,
            self.state.with_fluents(fluents),
            self.goal,
            self.domain.rules,
        )

    # -- reporting ----------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"step {self.position} of {len(self.plan)}; "
            f"{'goal met' if self.finished else 'goal not met yet'}"
        ]
        if self.replans:
            lines.append(f"replanned {len(self.replans)} time(s):")
            lines += [f"  {s.describe()}" for s in self.replans]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "position": self.position,
            "finished": self.finished,
            "plan": self.plan.to_dict(),
            "replans": [s.describe() for s in self.replans],
        }

    def __repr__(self) -> str:
        return (
            f"Execution(step {self.position}/{len(self.plan)}, "
            f"{len(self.replans)} replan(s))"
        )
