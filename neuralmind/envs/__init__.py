"""Environments the mind can act in.

An adapter's only job is translation: the environment becomes facts, an
instruction becomes a rule, and a planned step becomes whatever the
environment's own interface wants. Deciding stays in
:mod:`neuralmind.agency`, which is why these files are short and why the
next environment does not need any of this rewritten.

:func:`run_episode` is the loop that closes it: read, plan, act, read again,
and replan when -- and only when -- what came back broke the plan.

**Exploring is planning, not a special case.** An agent that cannot see the
red key cannot plan to stand in front of it, and the usual answer is a
hand-written wander. Here the fallback is another goal: stand somewhere that
borders a cell nobody has looked at. That is a rule

    frontier(X, Y) :- seen(X, Y), ahead(X, Y, D, X2, Y2), not seen(X2, Y2).

and the planner handles it with the machinery it already has, so exploring
produces a plan with reasons like everything else does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..agency import Agency, Domain, Goal, Planner, Plan
from ..agency.learn import ActionLearner
from .grid import (
    ACTIONS,
    STATE_RULES,
    GridReader,
    GridWorld,
    minigrid_available,
)
from .missions import GOAL_ATOM, Mission, MissionError, read_mission

__all__ = [
    "Episode",
    "run_episode",
    "TextEpisode",
    "run_text_episode",
    "GridReader",
    "GridWorld",
    "minigrid_available",
    "read_mission",
    "Mission",
    "MissionError",
    "EXPLORE_RULES",
    "EXPLORE_GOAL",
]

#: The frontier: somewhere known that borders somewhere unknown.
EXPLORE_RULES = """
frontier(X, Y) :- seen(X, Y), ahead(X, Y, D, X2, Y2), not seen(X2, Y2).
at_frontier :- at(agent, X, Y), frontier(X, Y).
"""
EXPLORE_GOAL = "at_frontier"


@dataclass
class Episode:
    """What happened in one run of one environment."""

    mission: Optional[Mission] = None
    solved: bool = False
    steps: int = 0
    replans: int = 0
    explorations: int = 0
    reward: float = 0.0
    seconds: float = 0.0
    #: The first plan that aimed at the instruction rather than at exploring.
    plan: Optional[Plan] = None
    #: Why it stopped, when it did not succeed.
    reason: str = ""
    #: One line per action taken, with the reason it was taken.
    log: list = field(default_factory=list)
    #: Actions that did not do what the model said they would.
    surprises: int = 0
    #: Preconditions worked out from those, and adopted.
    learned: list = field(default_factory=list)

    def describe(self) -> str:
        head = "solved" if self.solved else f"not solved: {self.reason}"
        return (
            f"{self.mission.describe() if self.mission else '(no mission)'} — "
            f"{head} in {self.steps} step(s), {self.replans} replan(s), "
            f"{self.explorations} exploration(s)"
        )

    def to_dict(self) -> dict:
        return {
            "mission": self.mission.to_dict() if self.mission else None,
            "solved": self.solved,
            "steps": self.steps,
            "replans": self.replans,
            "explorations": self.explorations,
            "surprises": self.surprises,
            "learned": list(self.learned),
            "reason": self.reason,
            "plan": self.plan.to_dict() if self.plan is not None else None,
        }

    def __str__(self) -> str:
        return self.describe()


def run_episode(
    env,
    seed: Optional[int] = None,
    view: str = "agent",
    max_steps: int = 120,
    budget_ms: float = 2_000.0,
    max_horizon: int = 30,
    learner: Optional[ActionLearner] = None,
) -> Episode:
    """Carry out one instruction in one environment.

    Replans whenever the world stops matching the plan, and falls back to
    exploring when the instruction cannot yet be planned for. Returns what
    happened, including the plan that finally did it and why each of its
    steps was there.
    """
    import time

    started = time.perf_counter()
    obs, _ = env.reset(seed=seed)
    reader = GridReader(view=view)
    planner = Planner(max_horizon=max_horizon, budget_ms=budget_ms)

    try:
        world = reader.read(env, obs)
    except MissionError as exc:
        return Episode(reason=f"could not read the instruction: {exc}", seconds=0.0)

    episode = Episode(mission=world.mission)
    agency = Agency(planner=planner)
    agency.learn_actions(ACTIONS)
    if learner is not None:
        # Share the library, so a lesson learned here is a lesson the next
        # plan is made under rather than a note in a log.
        learner.library = agency.library

    plan: Optional[Plan] = None
    position = 0
    exploring = False

    for _ in range(max_steps):
        if episode.steps >= max_steps:
            break

        agency.rules = ()
        agency.learn_state_rules(world.rules() + EXPLORE_RULES)
        agency.observe(world.facts)

        needs_new_plan = (
            plan is None
            or position >= len(plan.steps)
            or not _still_valid(plan, position, agency, world)
        )
        if needs_new_plan:
            if plan is not None:
                episode.replans += 1
            plan = planner.plan(agency.domain(), Goal.of(GOAL_ATOM))
            exploring = False
            if plan.already:
                episode.solved = True
                break
            if not plan.found:
                plan = planner.plan(agency.domain(), Goal.of(EXPLORE_GOAL))
                exploring = True
                episode.explorations += 1
                if not plan.found:
                    episode.reason = (
                        "nowhere left to look and no way to carry out the "
                        f"instruction: {plan.reason}"
                    )
                    break
            elif episode.plan is None:
                episode.plan = plan
            position = 0

        step = plan.steps[position]
        episode.log.append(
            f"{step.action} — "
            f"{'exploring' if exploring else plan.why(position)}"
        )
        before = agency.domain().state
        obs, reward, terminated, truncated, _ = env.step(
            reader.to_env_action(env, step)
        )
        episode.steps += 1
        position += 1
        world = reader.read(env, obs)

        if learner is not None:
            agency.observe(world.facts)
            attempt = learner.record(step.action, before, agency.domain().state)
            if not attempt.worked:
                episode.surprises += 1
                lesson = learner.best(step.name)
                if lesson is not None and lesson.complete:
                    learner.apply(lesson)
                    episode.learned.append(lesson.describe())
                    plan = None  # the plan was made under a model now known wrong

        if terminated:
            episode.solved = reward > 0
            episode.reward = float(reward)
            if not episode.solved:
                episode.reason = "the environment ended the episode without reward"
            break
        if truncated:
            episode.reason = "out of environment steps"
            break
    else:
        episode.reason = "out of steps"

    if not episode.solved and not episode.reason:
        episode.reason = "out of steps"
    episode.seconds = time.perf_counter() - started
    return episode


def _still_valid(plan: Plan, position: int, agency: Agency, world: GridWorld) -> bool:
    """Whether the rest of the plan can still be carried out.

    Cheaper than replanning and, more to the point, it is what makes the
    replan count mean something: a mind that rebuilds its plan every step
    has a replan count that measures the loop rather than the world.
    """
    state = agency.domain().state
    for step in plan.steps[position:]:
        if any(need not in state.fluents for need in step.needs):
            return False
        if any(absent in state.fluents for absent in step.needs_absent):
            return False
        fluents = (set(state.fluents) - set(step.cancels)) | set(step.adds)
        state = state.with_fluents(fluents)
    return True


# -- TextWorld --------------------------------------------------------------


@dataclass
class TextEpisode:
    """What happened in one TextWorld game."""

    objective: str = ""
    won: bool = False
    steps: int = 0
    replans: int = 0
    rejected: int = 0
    #: Actions tried to find out what happens, rather than to reach the goal.
    experiments: int = 0
    learned: list = field(default_factory=list)
    commands: list = field(default_factory=list)
    reason: str = ""
    plan: Optional[Plan] = None

    def describe(self) -> str:
        head = "won" if self.won else f"not won: {self.reason}"
        lesson = f", learned {len(self.learned)}" if self.learned else ""
        return (
            f"{head} in {self.steps} command(s), {self.replans} replan(s), "
            f"{self.rejected} rejected{lesson}"
        )

    def to_dict(self) -> dict:
        return {
            "won": self.won,
            "steps": self.steps,
            "replans": self.replans,
            "rejected": self.rejected,
            "experiments": self.experiments,
            "learned": list(self.learned),
            "commands": list(self.commands),
            "reason": self.reason,
        }

    def __str__(self) -> str:
        return self.describe()


def run_text_episode(
    seed: int = 7,
    rooms: int = 3,
    objects: int = 4,
    quest_length: int = 3,
    naive: bool = False,
    max_steps: int = 40,
    budget_ms: float = 4_000.0,
    max_horizon: int = 20,
    learn: bool = True,
) -> "TextEpisode":
    """Play one generated 'simple' game.

    With ``naive=True`` the mind starts with an action model that is missing
    a condition, tries a command the game refuses, and has to work out what
    it left out from the transition where nothing changed.
    """
    import textworld
    from textworld import EnvInfos

    from ..agency.learn import ActionLearner
    from .text import ACTIONS as TEXT_ACTIONS
    from .text import NAIVE_ACTIONS, TextReader, make_simple_game

    path, game = make_simple_game(seed, rooms, objects, quest_length)
    infos = EnvInfos(facts=True, admissible_commands=True, won=True, lost=True,
                     objective=True)
    env = textworld.start(path, infos)
    state = env.reset()

    reader = TextReader()
    world = reader.read(state, game)
    episode = TextEpisode(objective=world.objective)

    agency = Agency(planner=Planner(max_horizon=max_horizon, budget_ms=budget_ms))
    agency.learn_actions(NAIVE_ACTIONS if naive else TEXT_ACTIONS)
    learner = ActionLearner(agency.library) if learn else None

    plan: Optional[Plan] = None
    position = 0
    tried: set = set()

    for _ in range(max_steps):
        agency.observe(world.facts)
        if not world.goal:
            episode.reason = "the game states no win condition"
            break

        goal = Goal.of(world.goal)
        if goal.satisfied_by(agency.domain().state):
            episode.won = True
            break

        if plan is None or position >= len(plan.steps):
            if plan is not None:
                episode.replans += 1
            plan = agency.planner.plan(agency.domain(), goal)
            position = 0
            if not plan.found:
                episode.reason = plan.reason
                break
            if episode.plan is None:
                episode.plan = plan

        step = plan.steps[position]
        command = reader.to_command(step)
        before = agency.domain().state
        state, _, done = env.step(command)
        episode.steps += 1
        episode.commands.append(command)
        position += 1

        world = reader.read(state, game)
        agency.observe(world.facts)

        if learner is not None:
            attempt = learner.record(step.action, before, agency.domain().state)
            # Asked after every attempt, not only after a failure. The
            # evidence usually completes on a *success*: it is the working
            # case that says which of the things that differ was the one
            # that mattered.
            lesson = learner.best(step.name)
            if lesson is not None and lesson.complete:
                learner.apply(lesson)
                episode.learned.append(lesson.describe())
                plan = None
            if not attempt.worked:
                episode.rejected += 1
                plan = None
                # Before anything else: this exact command was refused, so it
                # is not a candidate for the next experiment. Leaving it in
                # the pool is how "try something else" picks the thing that
                # just failed -- it is, after all, the action most similar to
                # the action that just failed.
                tried.add(str(step.action))
                if lesson is None or not lesson.complete:
                    # Nothing to conclude yet, and the model still says this
                    # action is possible -- so the next plan would be the
                    # same plan. Repeating a refused command is not
                    # persistence, it is a loop. Try something the model says
                    # is possible and that has not been tried, because that
                    # is what produces the success the lesson needs.
                    experiment = _untried(agency, tried, step.action)
                    if experiment is not None:
                        tried.add(str(experiment))
                        episode.experiments += 1
                        before = agency.domain().state
                        state, _, done = env.step(reader.to_command(
                            _as_step(agency, experiment)
                        ))
                        episode.steps += 1
                        episode.commands.append(reader.to_command(
                            _as_step(agency, experiment)
                        ))
                        world = reader.read(state, game)
                        agency.observe(world.facts)
                        learner.record(
                            experiment, before, agency.domain().state
                        )

        if state.get("won"):
            episode.won = True
            break
        if state.get("lost"):
            episode.reason = "the game was lost"
            break
    else:
        episode.reason = "out of commands"

    if not episode.won and not episode.reason:
        episode.reason = "out of commands"
    return episode


def _untried(agency: Agency, tried: set, after=None):
    """Something possible that has not been tried, nearest the failure first.

    "Nearest" means: mentions one of the things the refused action mentioned.
    Wandering into the next room might eventually turn up the answer, but an
    action about the refrigerator is far likelier to explain why a command
    about the refrigerator was refused -- and the point of an experiment is
    the lesson, not the motion.
    """
    candidates = [a for a in agency.possible() if str(a) not in tried]
    if not candidates:
        return None
    if after is None:
        return candidates[0]
    involved = {str(arg) for arg in after.args}
    candidates.sort(
        key=lambda a: -len(involved & {str(arg) for arg in a.args})
    )
    return candidates[0]


def _as_step(agency: Agency, action):
    """Wrap a bare action atom as a step, so the adapters take one type."""
    from ..agency.plan import Step

    schema = agency.library.get(action.predicate, len(action.args))
    return Step(index=0, action=action, schema=schema)
