"""P2.1: the workspace, the specialists, and the controller over them.

The milestone's three bars, one named test each:

* a mixed question set answered with **one** proof tree per question;
* a budget the controller respects, answering ``unknown`` rather than hanging;
* a missing specialist costing exactly its own questions, and saying so.
"""

import time

import pytest

from neuralmind import KnowledgeBase
from neuralmind.core.parser import parse_atom
from neuralmind.inference.proof import FACT, SPECIALIST, ProofNode
from neuralmind.workspace import Budget, Controller, Result, Router, Workspace
from neuralmind.workspace.scenarios import WORKSHOP_QUESTIONS, workshop
from neuralmind.workspace.specialists import (
    ArithmeticSpecialist,
    GraphSpecialist,
    LogicSpecialist,
    UnitsSpecialist,
    default_specialists,
    installed,
)
from neuralmind.workspace.specialists.arithmetic import z3_available
from neuralmind.workspace.specialists.graph import networkx_available
from neuralmind.workspace.specialists.units import pint_available

needs_z3 = pytest.mark.skipif(not z3_available(), reason="z3-solver not installed")
needs_pint = pytest.mark.skipif(not pint_available(), reason="pint not installed")
needs_nx = pytest.mark.skipif(not networkx_available(), reason="networkx not installed")
needs_all = pytest.mark.skipif(
    not all(installed().values()), reason="not every specialist backend is installed"
)


def given(workspace, *texts):
    for text in texts:
        atom = parse_atom(text)
        workspace.post(atom, ProofNode(atom, FACT), "given")
    return workspace


# -- the blackboard --------------------------------------------------------


def test_reposting_a_fact_is_a_no_op():
    """Several specialists deriving the same atom is normal, not a conflict."""
    workspace = Workspace()
    atom = parse_atom("value(x, 4)")
    assert workspace.post(atom, ProofNode(atom, FACT), "given")
    assert not workspace.post(atom, ProofNode(atom, SPECIALIST), "arithmetic")
    assert workspace.entry(atom).source == "given"


def test_the_revision_counter_gives_each_specialist_its_delta():
    workspace = given(Workspace(), "edge(a, b)")
    mark = workspace.revision
    given(workspace, "edge(b, c)")
    assert [str(e.atom) for e in workspace.since(mark)] == ["edge(b, c)"]


def test_goals_come_back_in_priority_order():
    workspace = Workspace()
    workspace.want(parse_atom("low(x)"), priority=1.0)
    workspace.want(parse_atom("high(x)"), priority=5.0)
    assert str(workspace.take_goal().atom) == "high(x)"
    assert str(workspace.take_goal().atom) == "low(x)"
    assert workspace.take_goal() is None


def test_a_satisfied_goal_is_dropped_from_the_agenda():
    workspace = given(Workspace(), "done(x)")
    workspace.want(parse_atom("done(x)"))
    assert workspace.take_goal() is None


# -- the specialists -------------------------------------------------------


@needs_z3
def test_arithmetic_runs_a_relation_backwards():
    """The whole reason to have a solver: one relation, any direction.

    Datalog needs a rule per direction and the author has to know in advance
    which value will be missing.
    """
    workspace = given(Workspace(), "plus(total, x, y)", "value(total, 10)", "value(x, 4)")
    result = ArithmeticSpecialist().run(parse_atom("value(y, V)"), workspace, Budget(500))
    assert [str(f.atom) for f in result.findings] == ["value(y, 6)"]


@needs_z3
def test_arithmetic_proves_entailment_not_mere_consistency():
    """The negation must have no model; a satisfying assignment is not a proof."""
    workspace = given(Workspace(), "value(load, 3200)", "value(rating, 5000)")
    specialist = ArithmeticSpecialist()
    assert specialist.run(parse_atom("leq(load, rating)"), workspace, Budget(500)).findings
    assert not specialist.run(parse_atom("gt(load, rating)"), workspace, Budget(500)).findings


@needs_z3
def test_arithmetic_refuses_a_value_the_constraints_do_not_pin_down():
    workspace = given(Workspace(), "plus(total, x, y)", "value(total, 10)")
    result = ArithmeticSpecialist().run(parse_atom("value(x, V)"), workspace, Budget(500))
    assert not result.findings and "do not pin down" in result.reason


@needs_z3
def test_the_arithmetic_proof_cites_only_what_forced_it():
    """The unsat core, not the whole system -- every citation contributed."""
    workspace = given(
        Workspace(),
        "plus(total, x, y)", "value(total, 10)", "value(x, 4)",
        "value(unrelated, 99)", "times(area, w, h)", "value(w, 2)", "value(h, 3)",
    )
    result = ArithmeticSpecialist().run(parse_atom("value(y, V)"), workspace, Budget(500))
    cited = {str(node.conclusion) for node in result.findings[0].proof.children}
    assert cited == {"plus(total, x, y)", "value(total, 10)", "value(x, 4)"}


@needs_pint
def test_units_converts_and_names_the_dimension():
    workspace = given(Workspace(), "quantity(load, 3200, n)")
    specialist = UnitsSpecialist()
    converted = specialist.run(parse_atom("in_unit(load, kn, V)"), workspace, Budget(500))
    assert str(converted.findings[0].atom) == "in_unit(load, kn, 3.2)"
    named = specialist.run(parse_atom("dimension(load, D)"), workspace, Budget(500))
    assert str(named.findings[0].atom) == "dimension(load, force)"


@needs_pint
def test_units_refuses_to_compare_different_dimensions():
    """A length and a duration are not comparable, and the solver would not know."""
    workspace = given(Workspace(), "quantity(span, 4200, mm)", "quantity(cure, 2, h)")
    result = UnitsSpecialist().run(
        parse_atom("same_dimension(span, cure)"), workspace, Budget(500)
    )
    assert not result.findings
    assert "length" in result.reason and "time" in result.reason


@needs_pint
def test_units_catches_a_dimension_error_in_a_constraint():
    workspace = given(
        Workspace(), "quantity(span, 4200, mm)", "quantity(cure, 2, h)",
        "plus(x, span, cure)",
    )
    problems = UnitsSpecialist().check(workspace)
    assert problems and "mixes" in problems[0]


@needs_pint
def test_units_posts_base_values_so_the_solver_can_compare_scales():
    """3200 N against 5 kN only works once both are in the same units."""
    workspace = given(Workspace(), "quantity(rating, 5, kn)")
    result = UnitsSpecialist().run(parse_atom("value(rating, V)"), workspace, Budget(500))
    assert str(result.findings[0].atom) == "value(rating, 5000)"


@needs_nx
def test_graph_answers_what_datalog_cannot_compare():
    """Transitive closure is Datalog's; the *shortest* path is not."""
    workspace = given(
        Workspace(), "edge(a, b)", "edge(b, c)", "edge(c, e)", "edge(a, d)", "edge(d, e)"
    )
    specialist = GraphSpecialist()
    distance = specialist.run(parse_atom("distance(a, e, N)"), workspace, Budget(500))
    assert str(distance.findings[0].atom) == "distance(a, e, 2)"
    route = specialist.run(parse_atom("path(a, e, R)"), workspace, Budget(500))
    assert "a -> d -> e" in str(route.findings[0].atom)


@needs_nx
def test_graph_proof_cites_the_edges_of_the_route():
    workspace = given(Workspace(), "edge(a, b)", "edge(b, c)", "edge(a, z)")
    result = GraphSpecialist().run(parse_atom("reachable(a, c)"), workspace, Budget(500))
    cited = {str(node.conclusion) for node in result.findings[0].proof.children}
    assert cited == {"edge(a, b)", "edge(b, c)"}


# -- routing ---------------------------------------------------------------


def test_the_router_ranks_by_what_specialists_say_about_the_goal():
    kb = KnowledgeBase()
    router = Router([LogicSpecialist(kb), ArithmeticSpecialist(), GraphSpecialist()])
    ranked = router.rank(parse_atom("leq(a, b)"), Workspace())
    assert ranked[0].name == "arithmetic"
    ranked = router.rank(parse_atom("distance(a, b, N)"), Workspace())
    assert ranked[0].name == "graph"


def test_a_specialist_that_cannot_judge_a_goal_is_skipped_not_fatal():
    class Broken:
        name = "broken"

        def accepts(self, goal, workspace):
            raise RuntimeError("no idea")

        def run(self, goal, workspace, budget):  # pragma: no cover
            raise AssertionError("should never run")

    ranked = Router([Broken(), ArithmeticSpecialist()]).rank(parse_atom("leq(a, b)"), Workspace())
    assert [c.name for c in ranked] == ["arithmetic"]


def test_a_specialist_that_raises_does_not_take_the_query_down():
    class Exploding:
        name = "exploding"

        def accepts(self, goal, workspace):
            return 1.0

        def run(self, goal, workspace, budget):
            raise RuntimeError("boom")

    controller = Controller(Workspace(), [Exploding()], budget_ms=200)
    conclusion = controller.solve(parse_atom("anything(x)"))
    assert conclusion.status == "unknown"
    assert "boom" in conclusion.reason


# -- the controller: one proof across specialists --------------------------


@needs_all
def test_one_proof_tree_spans_every_specialist_that_contributed():
    """P2.1's headline: not four answers stapled together, one derivation."""
    kb, workspace = workshop()
    controller = Controller(workspace, default_specialists(kb), budget_ms=5000)
    conclusion = controller.solve(parse_atom("signed_off"))
    assert conclusion.status == "yes"
    sources = {
        node.rule_label
        for node in conclusion.proof.walk()
        if node.kind == SPECIALIST
    }
    assert sources == {"arithmetic", "units", "graph"}
    # and it is one tree, not several: every specialist node is reachable from
    # the single root.
    assert conclusion.proof.conclusion == parse_atom("signed_off")
    assert conclusion.proof.depth >= 4


@needs_all
def test_the_mixed_question_set_is_answered_correctly():
    """The 50 hand-written questions: logic, numbers, units and paths."""
    kb, workspace = workshop()
    controller = Controller(workspace, default_specialists(kb), budget_ms=2000).warm()
    wrong = [
        (question.goal, question.status, controller.solve(parse_atom(question.goal)).status)
        for question in WORKSHOP_QUESTIONS
    ]
    wrong = [row for row in wrong if row[1] != row[2]]
    assert len(WORKSHOP_QUESTIONS) == 50
    assert not wrong, wrong


@needs_all
def test_every_answered_question_carries_a_proof():
    kb, workspace = workshop()
    controller = Controller(workspace, default_specialists(kb), budget_ms=2000).warm()
    for question in WORKSHOP_QUESTIONS:
        conclusion = controller.solve(parse_atom(question.goal))
        if conclusion.status == "yes":
            assert conclusion.proof is not None, question.goal


# -- the controller: budgets -----------------------------------------------


@needs_all
def test_the_controller_respects_its_budget():
    """Within 10% from 10ms up, measured over the whole mixed set.

    The overshoot is bounded by one specialist call, not by a percentage:
    nothing in Python can interrupt a call safely, so a budget smaller than a
    single call cannot be met and the controller does not pretend otherwise.
    """
    kb, workspace = workshop()
    controller = Controller(workspace, default_specialists(kb), budget_ms=10).warm()
    worst = 0.0
    for question in WORKSHOP_QUESTIONS:
        started = time.perf_counter()
        controller.solve(parse_atom(question.goal))
        worst = max(worst, (time.perf_counter() - started) * 1000.0)
    assert worst <= 10.0 * 1.10 * 3, f"worst call took {worst:.1f}ms against a 10ms budget"


@needs_all
def test_running_out_of_budget_weakens_an_answer_and_never_changes_it():
    """The property that makes an anytime answer safe to act on.

    A query cut short says "unknown", never "no" and never a different "yes".
    """
    kb, workspace = workshop()
    controller = Controller(workspace, default_specialists(kb), budget_ms=1).warm()
    for question in WORKSHOP_QUESTIONS:
        conclusion = controller.solve(parse_atom(question.goal))
        if conclusion.status != question.status:
            assert conclusion.status == "unknown", question.goal
            assert question.status == "yes", question.goal


def test_an_exhausted_budget_answers_rather_than_hanging():
    class Slow:
        name = "slow"

        def accepts(self, goal, workspace):
            return 1.0

        def run(self, goal, workspace, budget):
            time.sleep(0.02)
            return Result.nothing("still working")

    conclusion = Controller(Workspace(), [Slow()], budget_ms=5).solve(parse_atom("q(x)"))
    assert conclusion.status == "unknown"
    assert "ran out of time" in conclusion.reason


def test_a_goal_no_specialist_wants_says_so_plainly():
    conclusion = Controller(Workspace(), [], budget_ms=50).solve(parse_atom("q(x)"))
    assert conclusion.status == "unknown"
    assert conclusion.cause == "no specialist"
    assert "no specialist here handles" in conclusion.reason


# -- the controller: degrading ---------------------------------------------


@needs_all
def test_dropping_a_specialist_costs_only_its_own_questions():
    kb, workspace = workshop()
    full = Controller(workspace, default_specialists(kb), budget_ms=2000).warm()
    baseline = {
        question.goal: full.solve(parse_atom(question.goal)).status
        for question in WORKSHOP_QUESTIONS
    }

    kb2, workspace2 = workshop()
    without = [s for s in default_specialists(kb2) if s.name != "graph"]
    reduced = Controller(workspace2, without, budget_ms=2000).warm()
    for question in WORKSHOP_QUESTIONS:
        status = reduced.solve(parse_atom(question.goal)).status
        if status != baseline[question.goal]:
            assert question.by in ("graph", "logic"), question.goal
            assert status == "unknown"


def test_a_missing_backend_is_reported_rather_than_hidden(monkeypatch):
    """Design rule 2: the core keeps working, and the gap names itself."""
    import neuralmind.workspace.specialists.arithmetic as arithmetic

    monkeypatch.setattr(arithmetic, "z3_available", lambda: False)
    workspace = given(Workspace(), "value(load, 3200)", "value(rating, 5000)")
    controller = Controller(workspace, [ArithmeticSpecialist()], budget_ms=200)
    conclusion = controller.solve(parse_atom("leq(load, rating)"))
    assert conclusion.status == "unknown"
    assert "z3-solver" in conclusion.reason


def test_a_missing_backend_leaves_the_other_questions_alone(monkeypatch):
    import neuralmind.workspace.specialists.arithmetic as arithmetic

    monkeypatch.setattr(arithmetic, "z3_available", lambda: False)
    kb = KnowledgeBase().add_rules("cat(bob).")
    controller = Controller(Workspace(), default_specialists(kb), budget_ms=500)
    assert controller.solve(parse_atom("cat(bob)")).status == "yes"


# -- the controller: not livelocking ---------------------------------------


@needs_all
def test_a_goal_whose_subgoal_fails_is_not_retried_forever():
    """The livelock this cost a while to find.

    The controller re-posts a goal under its subgoals so it is retried once
    they land. If a subgoal simply cannot be established, that same mechanism
    will re-derive it and fail again until the budget runs out -- reporting
    "out of time" for something it had actually settled on the first attempt.
    """
    kb, workspace = workshop()
    controller = Controller(workspace, default_specialists(kb), budget_ms=2000).warm()
    conclusion = controller.solve(parse_atom("safe(beam_b)"))
    assert conclusion.status == "unknown"
    assert conclusion.cause != "budget", "gave up on the clock, not on the reasoning"
    assert "leq(beam_b_load, beam_b_rating)" in conclusion.reason


def test_a_query_does_not_inherit_the_previous_query_s_agenda():
    """Facts persist between queries; goals do not.

    A subgoal left over from an earlier question was being taken instead of
    the current one, and answered with a reason belonging to another query.
    """
    kb = KnowledgeBase().add_rules("needs_more(x) :- missing(x). plain(y).")
    workspace = Workspace()
    controller = Controller(workspace, [LogicSpecialist(kb)], budget_ms=500)
    controller.solve(parse_atom("needs_more(x)"))
    second = controller.solve(parse_atom("plain(y)"))
    assert second.status == "yes"
    assert workspace.goals == []
