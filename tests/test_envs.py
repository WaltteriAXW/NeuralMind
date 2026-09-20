"""P2.7's environments: instructions in, actions out.

The planning machinery is tested in ``test_agency.py``. What is tested here
is the translation on either side of it -- an instruction becoming a goal, a
grid becoming facts, a planned step becoming something the environment
accepts -- plus the two properties that make the translation trustworthy:
nothing is guessed at, and what cannot be read is refused rather than
approximated.
"""

import pytest

from neuralmind.envs.missions import (
    GOAL_ATOM,
    Mission,
    MissionError,
    read_mission,
)

def _has(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


needs_minigrid = pytest.mark.skipif(
    not _has("minigrid"), reason="minigrid is not installed"
)
needs_textworld = pytest.mark.skipif(
    not _has("textworld"), reason="textworld is not installed"
)
needs_clingo = pytest.mark.skipif(
    not _has("clingo"), reason="planning searches, and clingo is the searcher"
)


# -- reading an instruction -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("go to the red key",
         "reached :- in_front(O), kind(O, key), colour(O, red)."),
        ("pick up a ball",
         "reached :- carrying(O), kind(O, ball)."),
        ("open the blue door",
         "reached :- door_open(O), kind(O, door), colour(O, blue)."),
        ("pick up a box behind you",
         "reached :- carrying(O), kind(O, box), behind_start(O)."),
        ("open a door on your left",
         "reached :- door_open(O), kind(O, door), left_of_start(O)."),
    ],
)
def test_an_instruction_becomes_a_rule(text, expected):
    """A rule, not a target: 'a red key' means any red key, and only a rule
    can say that. A ground goal would have to pick one and commit."""
    assert read_mission(text).rule() == expected


def test_an_instruction_outside_the_grammar_is_refused():
    """Half-reading an instruction produces a goal that looks reasonable and
    is not what was asked, and nothing downstream can notice."""
    with pytest.raises(MissionError, match="no verb I know"):
        read_mission("dance with the red key")


def test_a_word_left_over_is_refused_and_named():
    with pytest.raises(MissionError, match="submarine"):
        read_mission("go to the red submarine")


def test_an_instruction_naming_no_object_is_refused():
    with pytest.raises(MissionError, match="nothing says which object"):
        read_mission("go to")


# -- reading a grid ---------------------------------------------------------


@needs_minigrid
def test_the_grid_becomes_facts_about_shape_not_about_the_game():
    from minigrid.envs.babyai import GoToLocal

    from neuralmind.envs.grid import GridReader

    env = GoToLocal(room_size=6, num_dists=2)
    obs, _ = env.reset(seed=3)
    world = GridReader(view="agent").read(env, obs)

    predicates = {f.predicate for f in world.facts}
    assert {"at", "facing", "seen", "ahead", "kind", "colour"} <= predicates
    assert world.mission is not None
    # Geometry arrives as facts because the action language has no arithmetic.
    assert any(f.predicate == "ahead" and len(f.args) == 5 for f in world.facts)


@needs_minigrid
def test_the_partial_view_sees_less_than_the_whole_grid():
    """If these matched, the 'agent view' setting would not be doing
    anything, and the replanning would never be exercised."""
    from minigrid.envs.babyai import GoToLocal

    from neuralmind.envs.grid import GridReader

    env = GoToLocal(room_size=8, num_dists=4)
    obs, _ = env.reset(seed=5)
    partial = GridReader(view="agent").read(env, obs)
    full = GridReader(view="full").read(env, obs)
    assert len(partial.seen) < len(full.seen)


@needs_minigrid
def test_a_planned_step_becomes_an_action_the_environment_accepts():
    from minigrid.envs.babyai import GoToLocal

    from neuralmind.agency.plan import Step
    from neuralmind.core.parser import parse_atom
    from neuralmind.envs.grid import GridReader

    env = GoToLocal(room_size=6)
    env.reset(seed=1)
    reader = GridReader()
    step = Step(index=0, action=parse_atom("turn_left(east, north)"))
    assert reader.to_env_action(env, step) == int(env.unwrapped.actions.left)

    with pytest.raises(KeyError, match="not something this environment can do"):
        reader.to_env_action(env, Step(index=0, action=parse_atom("fly(up)")))


@needs_minigrid
@needs_clingo
def test_an_instruction_is_carried_out_and_every_step_has_a_reason():
    from minigrid.envs.babyai import GoToLocal

    from neuralmind.envs import run_episode

    env = GoToLocal(room_size=6, num_dists=2)
    episode = run_episode(env, seed=3)
    assert episode.solved, episode.reason
    assert episode.log, "an episode that did something should say what"
    assert all("—" in line for line in episode.log)


@needs_minigrid
@needs_clingo
def test_exploring_is_a_goal_rather_than_a_special_case():
    """The fallback when the target is not in view is another plan, with
    reasons, not a hand-written wander."""
    from minigrid.envs.babyai import GoToLocal

    from neuralmind.envs import EXPLORE_GOAL, run_episode

    env = GoToLocal(room_size=8, num_dists=4)
    solved = sum(run_episode(env, seed=s).solved for s in range(8))
    assert solved == 8


# -- reading a text world ---------------------------------------------------


@needs_textworld
def test_textworlds_propositions_become_atoms():
    import textworld
    from textworld import EnvInfos

    from neuralmind.envs.text import TextReader, make_simple_game

    path, game = make_simple_game(seed=7)
    env = textworld.start(path, EnvInfos(facts=True, won=True, objective=True))
    world = TextReader().read(env.reset(), game)

    predicates = {f.predicate for f in world.facts}
    assert "at" in predicates and "exit" in predicates
    # The player is named the same way in the facts and in the goal, which is
    # the whole reason the goal can be checked against the state at all.
    assert any(str(f) == "at(player, dish_pit)" for f in world.facts)
    assert world.goal


@needs_textworld
@needs_clingo
def test_a_generated_game_is_won_with_the_walkthroughs_own_moves():
    from neuralmind.envs import run_text_episode

    episode = run_text_episode(seed=7)
    assert episode.won, episode.reason
    assert episode.commands == [
        "open refrigerator",
        "take teapot from refrigerator",
        "put teapot on chair",
    ]


@needs_textworld
@needs_clingo
def test_a_missing_precondition_is_learned_from_being_refused():
    """The roadmap's clause, in a real game: the model says you can take
    something out of a closed container, the game disagrees, and the mind
    comes out of the episode knowing which condition it was missing."""
    from neuralmind.envs import run_text_episode

    episode = run_text_episode(seed=7, naive=True)
    assert episode.rejected >= 1, "the wrong model should have been refused"
    assert episode.learned, "and being refused should have taught it something"
    assert "opened(C)" in episode.learned[0]
    assert episode.won, episode.reason
