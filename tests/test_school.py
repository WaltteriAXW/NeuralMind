"""P2.13: the school.

Its two bars:

* one command reproduces every Phase Two number;
* every failure in every stage has exactly one attribution.

The stages themselves are measured by the stages; what is tested here is the
machinery around them -- the priority order of the causes, the gating, the
honesty about what cannot run, and the dashboard that makes a regression fail
by name.
"""

import json
from pathlib import Path

import pytest

from neuralmind.school import (
    BLOCKED,
    CURRICULUM,
    FAILED,
    PASSED,
    UNAVAILABLE,
    Report,
    Result,
    School,
    Stage,
)
from neuralmind.school.attribution import (
    BUDGET,
    CATEGORIES,
    ENGINE,
    MISSING_KNOWLEDGE,
    MISSING_RULE,
    PERCEPTION,
    REMEDY,
    SPECIALIST,
    UNATTRIBUTED,
    WRONG_CONTEXT,
    Evidence,
    attribute,
    breakdown,
)


# -- attribution: exactly one cause ----------------------------------------


@pytest.mark.parametrize(
    "evidence,expected",
    [
        (Evidence(gold_disagrees=True), ENGINE),
        (Evidence(out_of_budget=True), BUDGET),
        (Evidence(specialist_missing="arithmetic"), SPECIALIST),
        (Evidence(context_wrong="grocery"), WRONG_CONTEXT),
        (Evidence(theory_mismatch=True), PERCEPTION),
        (Evidence(question_mismatch=True), PERCEPTION),
        (Evidence(missing_fact="furry(bob)"), MISSING_KNOWLEDGE),
        (Evidence(rules_stalled=True), MISSING_RULE),
        (Evidence(), UNATTRIBUTED),
    ],
)
def test_each_kind_of_evidence_lands_in_its_own_category(evidence, expected):
    assert attribute(evidence).category == expected


def test_a_failure_with_several_true_causes_gets_the_most_upstream_one():
    """The hard requirement is *exactly one*, and that is about priority.

    A wrong answer usually has several true things to say about it. A breakdown
    that counts one failure three times cannot be used to decide what to fix.
    """
    everything = Evidence(
        gold_disagrees=True,
        out_of_budget=True,
        specialist_missing="units",
        context_wrong="game",
        theory_mismatch=True,
        missing_fact="p(a)",
        rules_stalled=True,
    )
    assert attribute(everything).category == ENGINE


def test_an_engine_failure_outranks_everything_because_it_invalidates_it():
    """If the engine is wrong then "perception was wrong" is not established,
    it is merely consistent."""
    assert attribute(Evidence(gold_disagrees=True, theory_mismatch=True)).category == ENGINE


def test_a_missing_rule_is_checked_last_so_it_cannot_swallow_the_others():
    """A category of last resort has to be last or it absorbs the rest."""
    assert CATEGORIES[-1] == MISSING_RULE
    assert attribute(Evidence(rules_stalled=True, missing_fact="p(a)")).category != MISSING_RULE


def test_every_category_says_what_to_do_about_it():
    for category in list(CATEGORIES) + [UNATTRIBUTED]:
        assert REMEDY[category]


def test_a_breakdown_is_ordered_by_priority_not_by_size():
    """Sorting by count would bury one engine failure under a hundred
    perception ones."""
    counts = breakdown(
        [attribute(Evidence(theory_mismatch=True)) for _ in range(100)]
        + [attribute(Evidence(gold_disagrees=True))]
    )
    assert list(counts) == [ENGINE, PERCEPTION]


def test_an_unattributed_failure_is_counted_rather_than_guessed_at():
    """A benchmark that cannot say why it failed has a gap in its own
    instrumentation and should show it."""
    assert attribute(Evidence()).category == UNATTRIBUTED
    assert "instrument it" in attribute(Evidence()).remedy


# -- the curriculum: gating and honesty ------------------------------------


def _stage(name, status, after=(), metric=None, target=None):
    return Stage(
        name=name,
        about="a test stage",
        check=lambda: "" if status != UNAVAILABLE else "not available here",
        run=lambda quick: Result(
            stage=name,
            status=status,
            metric=metric,
            target=target,
        ),
        after=tuple(after),
    )


def test_a_stage_whose_prerequisite_failed_is_blocked_not_run():
    """There is no information in a later score from a mind that failed the
    earlier stage."""
    ran = []
    later = Stage(
        name="later",
        about="",
        check=lambda: "",
        run=lambda quick: ran.append(1) or Result("later", PASSED),
        after=("first",),
    )
    report = School([_stage("first", FAILED), later]).run()
    assert [r.status for r in report.results] == [FAILED, BLOCKED]
    assert not ran, "the blocked stage ran anyway"
    assert "first did not pass" in report.results[1].detail


def test_a_prerequisite_that_could_not_run_does_not_block_what_follows():
    """Unavailable is not failure. Treating it as one would make the whole
    curriculum red for reasons nobody can act on."""
    report = School(
        [_stage("first", UNAVAILABLE), _stage("second", PASSED, after=("first",))]
    ).run()
    assert [r.status for r in report.results] == [UNAVAILABLE, PASSED]


def test_a_stage_that_cannot_run_says_why():
    report = School([_stage("first", UNAVAILABLE)]).run()
    assert report.results[0].detail == "not available here"


def test_a_stage_that_raises_is_a_failed_stage_not_a_crashed_run():
    def explode(quick):
        raise RuntimeError("boom")

    stage = Stage(name="boom", about="", check=lambda: "", run=explode)
    with pytest.raises(RuntimeError):
        School([stage]).run()


def test_health_ignores_what_could_not_run():
    report = School(
        [_stage("a", PASSED), _stage("b", UNAVAILABLE)]
    ).run()
    assert report.healthy


def test_health_is_false_when_something_that_ran_failed():
    report = School([_stage("a", PASSED), _stage("b", FAILED)]).run()
    assert not report.healthy


def test_only_runs_the_stages_asked_for():
    report = School([_stage("a", PASSED), _stage("b", PASSED)]).run(only=["b"])
    assert [r.stage for r in report.results] == ["b"]


def test_an_unknown_stage_is_refused_rather_than_ignored():
    with pytest.raises(KeyError, match="unknown stage"):
        School([_stage("a", PASSED)]).run(only=["nope"])


# -- the real curriculum ----------------------------------------------------


def test_the_curriculum_lists_what_it_cannot_reach_rather_than_omitting_it():
    """A curriculum that hides what it cannot run reports a smaller, better-
    looking mind than the one that exists."""
    unreachable = [s for s in CURRICULUM if s.why_not()]
    assert unreachable, "nothing is unavailable, which is suspicious"
    for stage in unreachable:
        assert stage.why_not(), stage.name
        assert stage.about, stage.name


def test_a_check_that_raises_becomes_a_reason_and_not_a_crash():
    """The likeliest place in the curriculum to raise is an availability check.

    Checks find out whether a dependency is there by importing something, so
    an environment missing that dependency is exactly where they blow up -- and
    a check that takes the process down destroys the listing of what *else*
    could have run, which is the one thing the school is for. This is not
    hypothetical: without it, a bare environment with no NumPy could not print
    the curriculum at all.
    """

    def explodes() -> str:
        raise ModuleNotFoundError("No module named 'pretend'")

    stage = Stage(
        name="boom", about="a stage whose check is broken",
        check=explodes, run=lambda quick: Result(stage="boom", status=PASSED),
    )
    assert "ModuleNotFoundError" in stage.why_not()
    assert "pretend" in stage.why_not()

    report = School([stage]).run()
    assert report.results[0].status == UNAVAILABLE
    # And it does not count as a failure: nothing was measured.
    assert report.healthy


def test_a_stage_that_names_an_extra_names_one_that_exists():
    """An install hint pointing at an extra that is not declared is worse than
    no hint: it sends someone to a command that fails."""
    import re

    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    declared = set(re.findall(r"^(\w+) = \[", pyproject, re.MULTILINE))
    assert "specialists" in declared, "the extras section moved"

    for stage in CURRICULUM:
        for extra in re.findall(r"neuralmind\[(\w+)\]", stage.why_not() or ""):
            assert extra in declared, f"{stage.name} points at a missing extra"


def test_every_stage_names_the_milestone_it_defends():
    for stage in CURRICULUM:
        assert stage.milestone, stage.name


def test_every_stage_has_a_distinct_name():
    names = [stage.name for stage in CURRICULUM]
    assert len(names) == len(set(names))


def test_prerequisites_name_stages_that_exist_and_come_earlier():
    """A curriculum is ordered; a stage depending on a later one is a cycle."""
    seen = set()
    for stage in CURRICULUM:
        for prerequisite in stage.after:
            assert prerequisite in seen, f"{stage.name} depends on {prerequisite}"
        seen.add(stage.name)


@pytest.mark.parametrize(
    "name", ["mixed-specialists", "where-am-i", "context-drift"]
)
def test_the_fast_stages_pass(name):
    report = School().run(quick=True, only=[name])
    result = report.results[0]
    assert result.status in (PASSED, UNAVAILABLE), result.describe()


# -- the dashboard ----------------------------------------------------------


def test_the_dashboard_keeps_history_so_a_regression_is_visible(tmp_path):
    path = tmp_path / "school.json"
    School.write(
        School([_stage("a", PASSED, metric=100.0, target=99.0)]).run(), path
    )
    School.write(
        School([_stage("a", FAILED, metric=80.0, target=99.0)]).run(), path
    )
    payload = json.loads(path.read_text())
    assert len(payload["runs"]) == 2
    assert payload["regressions"][0]["stage"] == "a"
    assert payload["regressions"][0]["was"] == 100.0
    assert payload["regressions"][0]["now"] == 80.0


def test_a_stage_that_slipped_but_still_passes_is_still_reported(tmp_path):
    """100% to 97% is a number nobody notices in a table."""
    path = tmp_path / "school.json"
    School.write(School([_stage("a", PASSED, metric=100.0, target=90.0)]).run(), path)
    School.write(School([_stage("a", PASSED, metric=97.0, target=90.0)]).run(), path)
    payload = json.loads(path.read_text())
    assert payload["regressions"][0]["now"] == 97.0
    assert "still passing" in payload["regressions"][0]["detail"]


def test_the_first_run_reports_no_regressions(tmp_path):
    path = tmp_path / "school.json"
    School.write(School([_stage("a", PASSED, metric=100.0)]).run(), path)
    assert json.loads(path.read_text())["regressions"] == []


def test_the_dashboard_survives_a_corrupt_file(tmp_path):
    """A dashboard that cannot be read must not stop the run that writes it."""
    path = tmp_path / "school.json"
    path.write_text("{ not json")
    School.write(School([_stage("a", PASSED, metric=1.0)]).run(), path)
    assert json.loads(path.read_text())["latest"]["counts"]["passed"] == 1


def test_the_dashboard_keeps_only_the_recent_runs(tmp_path):
    path = tmp_path / "school.json"
    for _ in range(5):
        School.write(School([_stage("a", PASSED, metric=1.0)]).run(), path, keep=3)
    assert len(json.loads(path.read_text())["runs"]) == 3


def test_a_report_serialises_its_failure_causes(tmp_path):
    stage = Stage(
        name="a",
        about="",
        check=lambda: "",
        run=lambda quick: Result(
            stage="a",
            status=FAILED,
            attributions=[attribute(Evidence(theory_mismatch=True))],
        ),
    )
    payload = School([stage]).run().to_dict()
    assert payload["failures_by_cause"] == {PERCEPTION: 1}
