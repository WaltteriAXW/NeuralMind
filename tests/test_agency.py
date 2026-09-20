"""P2.7: action, time and goals -- the mind decides, not just answers.

Its bars, from the roadmap:

* plans are found and every step carries a proof for why it is needed;
* replanning after an unexpected observation works, including the case where
  a door locks mid-plan;
* an action rule can be learned from failed attempts.

What is tested here is the machinery: the action language and what it infers,
the planner's three outcomes, the causal links that justify each step, when a
surprise does and does not force a replan, and the gate that stands between a
plan and it happening. The environment stages are measured by the school.
"""

import pytest

from neuralmind.agency import (
    Agency,
    ActionError,
    ActionLibrary,
    Domain,
    Goal,
    L0,
    L1,
    L2,
    L3,
    Planner,
    PlanningError,
    State,
    parse_actions,
    to_asp,
)
from neuralmind.agency.execute import Execution
from neuralmind.core.parser import parse_atom
from neuralmind.inference.clingo_backend import clingo_available
from neuralmind.inference.proof import ACTION, FACT

needs_clingo = pytest.mark.skipif(
    not clingo_available(), reason="planning searches, and clingo is the searcher"
)

VALVES = """
% Opening a valve needs it closed, and makes it open.
action open_valve(V):
    needs  valve(V), closed(V), not locked(V)
    causes open(V), -closed(V)

action close_valve(V):
    needs  valve(V), open(V)
    causes closed(V), -open(V)

action unlock(V):
    needs  valve(V), locked(V)
    causes -locked(V)
    note   the key is on the panel

action vent(V):
    needs  valve(V), open(V)
    causes vented(V)
"""

DOORS = """
action unlock(D):
    needs  door(D), locked(D), carrying(key)
    causes -locked(D)

action open_door(D):
    needs  door(D), shut(D), not locked(D)
    causes open(D), -shut(D)

action walk_through(D):
    needs  door(D), open(D)
    causes through(D)
"""


def library():
    return parse_actions(VALVES)


# -- the action language ----------------------------------------------------


def test_an_action_says_what_it_needs_and_what_it_changes():
    open_valve = library().get("open_valve", 1)
    assert [str(n) for n in open_valve.needs] == [
        "valve(V)", "closed(V)", "not locked(V)"
    ]
    assert [str(a) for a in open_valve.adds] == ["open(V)"]
    assert [str(c) for c in open_valve.cancels] == ["closed(V)"]


def test_fluents_are_inferred_from_the_effects_not_declared():
    """Nobody should have to keep a list of what changes in step with the
    rules that change it."""
    lib = library()
    assert lib.fluents == {"open", "closed", "locked", "vented"}
    assert lib.statics() == {"valve"}


def test_a_fluent_nothing_causes_can_still_be_declared():
    """For what the environment does on its own. Without it the planner
    carries a stale value forward for ever."""
    lib = library().declare_fluent("raining")
    assert "raining" in lib.fluents


def test_an_effect_on_something_the_action_does_not_name_is_refused():
    """An effect is attributed to the action happening, so the action has to
    say what it affects. ``W`` here is bound by nothing and named by
    nothing."""
    with pytest.raises(ActionError, match="not among its parameters"):
        parse_actions("""
action wish(V):
    needs  valve(V)
    causes connected(V, W)
""")


def test_an_effect_bound_only_by_a_precondition_is_still_refused():
    """The subtler case: ``W`` *is* bound, so the rule is safe in the
    ordinary sense -- but the action term carries no record of which ``W``,
    so the effect could not be attributed to it."""
    with pytest.raises(ActionError, match="write link"):
        parse_actions("""
action link(V):
    needs  valve(V), near(V, W)
    causes connected(V, W)
""")


def test_a_negative_condition_on_an_unbound_variable_is_refused():
    with pytest.raises(ActionError, match="unsafe"):
        parse_actions("""
action shut(V):
    needs  valve(V), not blocking(W, V)
    causes closed(V)
""")


def test_an_action_that_changes_nothing_is_refused():
    with pytest.raises(ActionError, match="causes nothing"):
        parse_actions("""
action stare(V):
    needs  valve(V)
    causes
""")


def test_an_action_cannot_both_cause_and_cancel_the_same_thing():
    with pytest.raises(ActionError, match="both causes and cancels"):
        parse_actions("""
action flicker(V):
    needs  valve(V)
    causes open(V), -open(V)
""")


def test_two_actions_with_the_same_signature_are_refused():
    with pytest.raises(ActionError, match="two actions named"):
        parse_actions("""
action go(V):
    needs  valve(V)
    causes open(V)

action go(W):
    needs  valve(W)
    causes closed(W)
""")


def test_a_clause_outside_an_action_says_so_rather_than_being_ignored():
    with pytest.raises(ActionError, match="expected an 'action"):
        parse_actions("needs valve(V)")


# -- what the library can work out about itself -----------------------------


def test_reversibility_is_read_off_the_library():
    """An action is reversible when something else can put back what it
    changed. Nothing declares this."""
    lib = library()
    assert lib.reversible(lib.get("open_valve", 1))
    assert lib.reversible(lib.get("close_valve", 1))
    assert not lib.reversible(lib.get("vent", 1))
    assert [a.name for a in lib.irreversible()] == ["unlock", "vent"]


def test_an_unreachable_goal_is_reported_before_any_search():
    """"No action can bring this about" and "not within 20 steps" are
    different answers, and only one of them means try a longer horizon."""
    domain = Domain.build(VALVES, facts=["valve(v1)", "closed(v1)"], goal=["flying(v1)"])
    assert [str(a) for a in domain.unreachable()] == ["flying(v1)"]


def test_the_state_sorts_itself_into_fixed_and_changing():
    state = State.of(["valve(v1)", "closed(v1)"], library())
    assert {str(a) for a in state.statics} == {"valve(v1)"}
    assert {str(a) for a in state.fluents} == {"closed(v1)"}


def test_a_state_must_be_ground():
    with pytest.raises(ActionError, match="ground"):
        State.of(["valve(V)"], library())


# -- planning ---------------------------------------------------------------


@needs_clingo
def test_a_plan_is_found_and_every_step_says_what_it_is_for():
    domain = Domain.build(
        VALVES,
        facts=["valve(v1)", "valve(v2)", "closed(v1)", "closed(v2)"],
        goal=["open(v1)", "open(v2)"],
    )
    plan = Planner().plan(domain)
    assert plan.found
    assert len(plan) == 2
    assert plan.check() == []
    for index in range(len(plan)):
        assert plan.reasons(index), "a step with no reason is a step with no place"


@needs_clingo
def test_a_step_that_only_clears_the_way_still_has_a_reason():
    """The case a positive-only causal link misses: unlock produces nothing,
    it removes something, and it is still the reason the plan works."""
    domain = Domain.build(
        VALVES,
        facts=["valve(v1)", "closed(v1)", "locked(v1)"],
        goal=["open(v1)"],
    )
    plan = Planner().plan(domain)
    assert [s.name for s in plan] == ["unlock", "open_valve"]
    assert plan.check() == []
    assert "out of the way" in plan.why(0)


@needs_clingo
def test_the_proof_for_a_step_bottoms_out_in_what_was_observed():
    domain = Domain.build(
        VALVES, facts=["valve(v1)", "closed(v1)", "locked(v1)"], goal=["open(v1)"]
    )
    plan = Planner().plan(domain)
    proof = plan.proof(1)
    assert proof.kind == ACTION
    assert str(proof.conclusion) == "open_valve(v1)"
    leaves = {str(leaf.conclusion) for leaf in proof.leaves()}
    assert leaves == {"closed(v1)", "locked(v1)"}
    assert all(leaf.kind == FACT for leaf in proof.leaves())


@needs_clingo
def test_the_shortest_plan_is_found_because_the_horizon_grows():
    domain = Domain.build(
        DOORS,
        facts=["door(d1)", "shut(d1)", "locked(d1)", "carrying(key)"],
        goal=["through(d1)"],
    )
    plan = Planner().plan(domain)
    assert len(plan) == 3
    assert plan.horizon == 3, "a longer horizon would mean it skipped a length"


@needs_clingo
def test_a_goal_already_true_asks_for_nothing():
    domain = Domain.build(VALVES, facts=["valve(v1)", "open(v1)"], goal=["open(v1)"])
    plan = Planner().plan(domain)
    assert plan.already and not plan.steps
    assert "already" in plan.brief()


@needs_clingo
def test_an_impossible_goal_says_so_instead_of_searching_for_ever():
    domain = Domain.build(VALVES, facts=["valve(v1)", "closed(v1)"], goal=["flying(v1)"])
    plan = Planner().plan(domain)
    assert not plan.found
    assert "no action can bring about" in plan.reason


@needs_clingo
def test_running_out_of_horizon_is_not_the_same_as_running_out_of_time():
    """The distinction the planner exists to keep: one means stop looking,
    the other means look longer."""
    domain = Domain.build(
        DOORS,
        facts=["door(d1)", "shut(d1)", "locked(d1)", "carrying(key)"],
        goal=["through(d1)"],
    )
    exhausted = Planner(max_horizon=1).plan(domain)
    assert not exhausted.found
    assert "no plan exists within" in exhausted.reason
    assert "ruled out, not skipped" in exhausted.reason

    starved = Planner(budget_ms=0.0).plan(domain)
    assert not starved.found
    assert "budget" in starved.reason


@needs_clingo
def test_a_goal_can_ask_for_something_to_stop_being_true():
    domain = Domain.build(VALVES, facts=["valve(v1)", "open(v1)"], goal=["-open(v1)"])
    plan = Planner().plan(domain)
    assert [s.name for s in plan] == ["close_valve"]
    assert "asks to be false" in plan.why(0)


def test_the_encoding_can_be_read():
    """A planner whose program you cannot look at is one you cannot debug."""
    domain = Domain.build(VALVES, facts=["valve(v1)", "closed(v1)"], goal=["open(v1)"])
    base, step, check = to_asp(domain)
    assert "holds(closed(v1), 0)." in base
    assert "goal(open(v1))." in base
    assert "1 { occurs(A, t) : poss(A, t) } 1." in step
    assert "holds(open(A), t) :- holds(open(A), t-1), not cancelled(open(A), t)." in step
    assert "#external query(t)." in check


# -- carrying it out --------------------------------------------------------


@needs_clingo
def test_a_plan_runs_to_the_end_when_nothing_interferes():
    domain = Domain.build(
        DOORS, facts=["door(d1)", "shut(d1)", "carrying(key)"], goal=["through(d1)"]
    )
    run = Execution(domain)
    run.run()
    assert run.finished
    assert not run.replans


@needs_clingo
def test_replanning_when_a_door_locks_mid_plan():
    """The roadmap's own test. The plan was sound when it was made and is not
    any more, and the mind has to notice from an observation rather than from
    being told."""
    domain = Domain.build(
        DOORS, facts=["door(d1)", "shut(d1)", "carrying(key)"], goal=["through(d1)"]
    )
    run = Execution(domain)
    assert [s.name for s in run.plan] == ["open_door", "walk_through"]

    run.step()  # open the door
    progress = run.step(["door(d1)", "shut(d1)", "locked(d1)", "carrying(key)"])

    assert progress.replanned
    assert progress.surprise.matters
    assert [s.name for s in run.plan] == ["unlock", "open_door", "walk_through"]

    run.run()
    assert run.finished
    assert len(run.replans) == 1


@needs_clingo
def test_a_surprise_that_breaks_nothing_does_not_throw_the_plan_away():
    """The other half of the claim, and the harder one. A mind that replans
    on every difference replans constantly."""
    source = DOORS + """
action switch_on(L):
    needs  lamp(L)
    causes lit(L)
"""
    domain = Domain.build(
        source, facts=["door(d1)", "shut(d1)", "lamp(l1)"], goal=["through(d1)"]
    )
    run = Execution(domain)
    run.step()
    progress = run.step(["door(d1)", "open(d1)", "lamp(l1)", "lit(l1)"])

    assert progress.surprise.any_change, "something did change"
    assert not progress.surprise.matters, "but nothing the plan relied on"
    assert not progress.replanned
    assert run.finished


@needs_clingo
def test_a_surprise_that_helps_shortens_the_plan_rather_than_stopping_it():
    """Someone else opened the door.

    This *does* replan, and should: the first step needed the door shut and
    the door is not shut, so the plan as written cannot run. What matters is
    that the mind comes back with less to do rather than declaring itself
    stuck -- the same mechanism that handles a setback handles a gift.
    """
    domain = Domain.build(
        DOORS, facts=["door(d1)", "shut(d1)", "carrying(key)"], goal=["through(d1)"]
    )
    run = Execution(domain)
    before = len(run.plan)
    progress = run.step(["door(d1)", "open(d1)", "carrying(key)"])

    assert progress.replanned
    assert not progress.stuck
    assert len(run.plan) < before
    assert [s.name for s in run.plan] == ["walk_through"]


@needs_clingo
def test_an_execution_that_cannot_recover_says_what_stopped_it():
    domain = Domain.build(
        DOORS, facts=["door(d1)", "shut(d1)", "carrying(key)"], goal=["through(d1)"]
    )
    run = Execution(domain)
    run.step()
    # The key is gone and the door has locked: there is no way on.
    progress = run.step(["door(d1)", "shut(d1)", "locked(d1)"])
    assert progress.stuck
    assert not run.finished


# -- the gate between a plan and it happening -------------------------------


@needs_clingo
def test_an_irreversible_step_waits_for_a_person():
    agency = Agency().learn_actions(VALVES)
    agency.observe(["valve(v1)", "closed(v1)"])
    choice = agency.decide("vented(v1)")

    assert choice.found
    assert choice.needs_person
    step, _ = choice.blocked
    assert step.name == "vent"
    assert [s.name for s in choice.allowed] == ["open_valve"]
    assert "needs you" in choice.brief()


@needs_clingo
def test_a_grant_is_what_lifts_it_and_nothing_else_is():
    agency = Agency().learn_actions(VALVES)
    agency.observe(["valve(v1)", "closed(v1)"])
    assert agency.decide("vented(v1)").needs_person

    agency.gate.grant(L3)
    assert not agency.decide("vented(v1)").needs_person


@needs_clingo
def test_confirming_an_action_gets_past_it_once_the_host_allows_acting():
    agency = Agency().learn_actions(VALVES)
    agency.observe(["valve(v1)", "closed(v1)"])
    agency.gate.grant(L2)
    assert not agency.decide("vented(v1)", confirmed=["vent"]).needs_person


@needs_clingo
def test_confirmation_is_not_a_way_around_a_low_grant():
    """At L1 the host has not authorised acting at all, so a confirmation
    has nothing to confirm. A person saying yes cannot grant authority the
    host withheld."""
    agency = Agency().learn_actions(VALVES)
    agency.observe(["valve(v1)", "closed(v1)"])
    agency.gate.grant(L1)
    assert agency.decide("vented(v1)", confirmed=["vent"]).needs_person


@needs_clingo
def test_permission_stops_at_the_first_blocked_step_because_a_plan_is_ordered():
    """Permission for step 3 without permission for step 2 is permission to
    do nothing."""
    agency = Agency().learn_actions(VALVES)
    agency.observe(["valve(v1)", "closed(v1)", "locked(v1)"])
    agency.gate.grant(L1)
    choice = agency.decide("vented(v1)")
    # unlock is irreversible, so it blocks -- and everything after it too.
    assert choice.allowed == []
    assert choice.blocked[0].name == "unlock"


def test_a_declared_authority_overrides_the_inference():
    agency = Agency().learn_actions("""
action nudge(V):
    needs  valve(V)
    causes nudged(V)
    authority 0
""")
    assert agency.authority_for(agency.library.get("nudge", 1)) == 0


@needs_clingo
def test_the_mind_plans_from_what_it_already_knows():
    """Not from a second private world model that can drift from the facts."""
    from neuralmind import Mind

    mind = Mind()
    mind.learn_actions(VALVES)
    mind.add_facts(["valve(v1)", "closed(v1)"])
    choice = mind.plan("open(v1)")
    assert [s.name for s in choice.plan] == ["open_valve"]
    assert "Open valve v1" in choice.brief()


@needs_clingo
def test_a_grant_on_the_mind_reaches_planning():
    """The agency shares the kernel's gate rather than keeping its own."""
    from neuralmind import Mind

    mind = Mind()
    mind.learn_actions(VALVES)
    mind.add_facts(["valve(v1)", "closed(v1)"])
    assert mind.plan("vented(v1)").needs_person
    mind.grant(L3)
    assert not mind.plan("vented(v1)").needs_person


# -- learning what an action really needs -----------------------------------


DOORS_NAIVE = """
action toggle(D):
    needs  door(D), door_shut(D)
    causes door_open(D), -door_shut(D)
"""


def _attempt(learner, lib, name, facts_before, facts_after):
    from neuralmind.agency.domain import State

    return learner.record(
        parse_atom(name), State.of(facts_before, lib), State.of(facts_after, lib)
    )


def test_a_failed_action_teaches_the_precondition_nobody_wrote_down():
    """The model says a shut door opens. A locked one does not, and the
    difference is visible in the transition: the action happened and the
    state did not move."""
    from neuralmind.agency.learn import ActionLearner

    lib = parse_actions(DOORS_NAIVE)
    learner = ActionLearner(lib)
    for door in ("d1", "d2"):
        _attempt(learner, lib, f"toggle({door})",
                 [f"door({door})", f"door_shut({door})"],
                 [f"door({door})", f"door_open({door})"])
    attempt = _attempt(
        learner, lib, "toggle(d3)",
        ["door(d3)", "door_shut(d3)", "door_locked(d3)"],
        ["door(d3)", "door_shut(d3)", "door_locked(d3)"],
    )

    assert not attempt.worked
    lesson = learner.best()
    assert str(lesson.literal) == "not door_locked(D)"
    assert lesson.complete

    learner.apply(lesson)
    assert "not door_locked(D)" in lib.get("toggle", 1).describe()


def test_the_lesson_is_about_the_action_not_about_the_room():
    """A fact whose arguments are nothing to do with the action cannot be
    lifted onto it, so it is never proposed however well it correlates."""
    from neuralmind.agency.learn import ActionLearner

    lib = parse_actions(DOORS_NAIVE)
    learner = ActionLearner(lib)
    for door in ("d1", "d2"):
        _attempt(learner, lib, f"toggle({door})",
                 [f"door({door})", f"door_shut({door})", "lamp_on(l1)"],
                 [f"door({door})", f"door_open({door})", "lamp_on(l1)"])
    _attempt(learner, lib, "toggle(d3)",
             ["door(d3)", "door_shut(d3)", "door_locked(d3)"],
             ["door(d3)", "door_shut(d3)", "door_locked(d3)"])

    proposed = {str(p.literal) for p in learner.propose()}
    assert "not door_locked(D)" in proposed
    assert not any("lamp_on" in p for p in proposed)


def test_one_failure_and_no_successes_teaches_nothing():
    """An action that has only ever failed has told you it does not work,
    which is not the same as telling you why."""
    from neuralmind.agency.learn import ActionLearner

    lib = parse_actions(DOORS_NAIVE)
    learner = ActionLearner(lib)
    _attempt(learner, lib, "toggle(d1)",
             ["door(d1)", "door_shut(d1)", "door_locked(d1)"],
             ["door(d1)", "door_shut(d1)", "door_locked(d1)"])
    assert learner.propose() == []


def test_a_missing_positive_precondition_is_learned_too():
    """The other shape: something that has to be there, not absent."""
    from neuralmind.agency.learn import ActionLearner

    lib = parse_actions("""
action unlock(D):
    needs  door(D), door_locked(D)
    causes -door_locked(D)
""")
    learner = ActionLearner(lib)
    for door in ("d1", "d2"):
        _attempt(learner, lib, f"unlock({door})",
                 [f"door({door})", f"door_locked({door})", f"key_for({door})"],
                 [f"door({door})", f"key_for({door})"])
    _attempt(learner, lib, "unlock(d3)",
             ["door(d3)", "door_locked(d3)"],
             ["door(d3)", "door_locked(d3)"])

    lesson = learner.best()
    assert str(lesson.literal) == "key_for(D)"
    assert lesson.complete


@needs_clingo
def test_a_learned_precondition_changes_the_next_plan():
    """The point of learning it. Before, the planner happily plans to open a
    locked door; after, it does not."""
    from neuralmind.agency.learn import ActionLearner

    lib = parse_actions(DOORS_NAIVE)
    facts = ["door(d1)", "door_shut(d1)", "door_locked(d1)"]
    domain = Domain.build(lib, facts=facts, goal=["door_open(d1)"])
    assert Planner().plan(domain).found, "the naive model sees no problem"

    learner = ActionLearner(lib)
    for door in ("d2", "d3"):
        _attempt(learner, lib, f"toggle({door})",
                 [f"door({door})", f"door_shut({door})"],
                 [f"door({door})", f"door_open({door})"])
    _attempt(learner, lib, "toggle(d1)", facts, facts)
    learner.apply(learner.best())

    after = Planner().plan(Domain.build(lib, facts=facts, goal=["door_open(d1)"]))
    assert not after.found
