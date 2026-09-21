"""P2.8: no pack fits, so draft one.

The roadmap's three bars:

* from an observation log of a domain no pack covers, produce a pack that
  passes its own tests and answers a held-out question set;
* the same process on a small game produces a playable action model;
* a wrong developer answer is caught later by a contradiction with the
  observations, and the question is asked again rather than the answer being
  quietly kept.

What is tested here is the reasoning, not the fixtures: that a reading is
made from behaviour rather than from a type, that nothing is proposed where
the log does not settle it, and that the evidence behind every proposal is
the evidence it claims.
"""

import json

import pytest

from neuralmind.builder import (
    NO,
    YES,
    Builder,
    Threshold,
    learn_effects,
    learn_thresholds,
    survey,
    triggered_by,
)
from neuralmind.builder.pack import answer, score, write_pack
from neuralmind.builder.stations import (
    HELD_OUT,
    VAULT_COMMANDS,
    PumpStation,
    Vault,
    pump_station_log,
    read_log,
    vault_log,
    write_log,
)
from neuralmind.builder.survey import COUNTER, FLAG, QUANTITY, STATE


def _has(module: str) -> bool:
    try:
        __import__(module)
    except ImportError:
        return False
    return True


needs_clingo = pytest.mark.skipif(
    not _has("clingo"), reason="planning searches, and clingo is the searcher"
)

PUMP = pump_station_log(steps=300)


# -- when to start building -------------------------------------------------


class _Reading:
    def __init__(self, confidence, unexplained, covered=0.5):
        self.domain_confidence = confidence
        self.unexplained = unexplained
        self.covered = covered


def test_an_unrecognised_host_with_unplaced_words_starts_a_build():
    why = triggered_by(_Reading(0.5, ["pump_a", "valve_b", "alarm"]))
    assert why and "unplaced" in why


def test_a_host_read_confidently_does_not():
    assert triggered_by(_Reading(0.95, [], covered=1.0)) is None


def test_one_stray_word_is_not_a_domain():
    assert triggered_by(_Reading(0.9, ["widget"], covered=0.95)) is None


# -- reading the log --------------------------------------------------------


def test_each_field_is_read_from_how_its_values_behave():
    reading = survey(PUMP)
    assert reading.of("pump_a").kind == STATE
    assert set(reading.of("pump_a").values) == {"idle", "running"}
    assert reading.of("pressure").kind == QUANTITY
    assert reading.of("pressure").unit == "bar"
    assert reading.of("alarm").kind == FLAG


def test_a_tick_is_a_counter_and_not_a_quantity():
    """Both are numbers. A schema that calls a tick a measurement invites
    questions about its units, and there are none."""
    assert survey(PUMP).of("tick").kind == COUNTER


def test_the_command_list_is_recognised_by_what_was_acted_on():
    """A list of strings is only a command list because its entries turn up
    as the actions that were taken."""
    reading = survey(PUMP)
    assert reading.of("commands").kind == "commands"
    assert set(reading.commands) >= {"start_pump_a", "open_valve_b"}


def test_a_field_that_never_differs_is_not_part_of_the_state():
    reading = survey(PUMP)
    assert not reading.of("commands").changing
    assert reading.of("pump_a").changing


def test_what_a_facet_already_explains_is_left_alone():
    reading = survey(PUMP, explained={"flow": "quantities"})
    assert reading.of("flow").explained_by == "quantities"
    assert "flow" not in [f.name for f in reading.new]


def test_every_finding_says_what_it_was_read_from():
    for finding in survey(PUMP).findings:
        assert finding.because, finding.name


# -- what each command does -------------------------------------------------


def test_a_commands_effect_is_counted_over_the_cases_where_it_could_happen():
    """Not over every case. A command that ran fifty times while the alarm
    was already off did not switch it off fifty times."""
    reading = survey(PUMP)
    drafts = {d.name: d for d in learn_effects(PUMP, reading)}
    effect = next(
        e for e in drafts["start_pump_a"].effects if e.field == "pump_a"
    )
    assert effect.to_value == "running"
    assert effect.always
    assert effect.cases < drafts["start_pump_a"].cases


def test_a_conditional_effect_names_its_condition():
    """The roadmap's own example, and a thing no log of a competent operator
    would ever show: starting the pump moves fluid only if the valve is open."""
    reading = survey(PUMP)
    drafts = {d.name: d for d in learn_effects(PUMP, reading)}
    effect = next(e for e in drafts["start_pump_a"].effects if e.field == "flow")
    assert effect.condition is not None
    assert ("valve_b", "open") in effect.conditions
    assert effect.always, "counted inside the condition, it should be exact"


def test_a_drifting_quantity_is_not_proposed_as_an_effect():
    """Pressure moves under every command and lands somewhere different each
    time. 'Sets pressure to 3.2 bar' would be the log's regularity mistaken
    for the world's."""
    reading = survey(PUMP)
    for draft in learn_effects(PUMP, reading):
        assert all(e.field != "pressure" for e in draft.effects), draft.name


def test_an_effect_carries_the_arithmetic_that_produced_it():
    reading = survey(PUMP)
    for draft in learn_effects(PUMP, reading):
        for effect in draft.effects:
            assert effect.cases > 0
            assert 0 < effect.happened <= effect.cases
            assert "of the" in effect.describe()


# -- rules for what nothing can derive --------------------------------------


def test_a_flag_gets_a_threshold_with_no_counterexamples():
    reading = survey(PUMP)
    found = learn_thresholds(PUMP, reading)
    assert found
    threshold = found[0]
    assert threshold.flag == "alarm"
    assert threshold.quantity == "pressure"
    assert threshold.sound


def test_a_threshold_says_what_it_leaves_unexplained():
    """The alarm stays on after the pressure falls, so no threshold explains
    all of it -- and a learner that rounded past those cases would produce a
    rule that is wrong in exactly the situation an alarm exists for."""
    threshold = learn_thresholds(PUMP, survey(PUMP))[0]
    assert not threshold.complete
    assert 0 < threshold.coverage < 1
    assert "explains only" in threshold.describe()


def test_a_learned_threshold_is_written_with_its_scaling_shown():
    threshold = learn_thresholds(PUMP, survey(PUMP))[0]
    asp = threshold.to_asp()
    assert "high_pressure :-" in asp
    assert "10ths" in asp, "the scaling has to be written down beside the number"


def test_the_invented_concept_earns_its_place_on_a_second_thing():
    """reset_alarm's condition could not be said before high_pressure existed.
    A learned concept explaining a second thing is the strongest evidence a
    log can give that it is real."""
    builder = Builder.from_log(PUMP, name="pump_station").step()
    draft = next(d for d in builder.actions if d.name == "reset_alarm")
    effect = next(e for e in draft.effects if e.field == "alarm")
    assert any("high_pressure" == name for name, _ in effect.conditions)


# -- the pack ---------------------------------------------------------------


def test_the_pack_answers_a_held_out_question_set():
    """The roadmap's first bar. The questions were never in the log."""
    builder = Builder.from_log(PUMP, name="pump_station").step()
    right, total, wrong = score(builder.pack(), HELD_OUT)
    assert right == total, wrong


def test_a_draft_never_files_itself_as_finished():
    pack = Builder.from_log(PUMP, name="pump_station").step().pack()
    assert 'status = "proposed"' in pack.pack_toml()
    assert "confirmed" not in pack.pack_toml()


def test_the_pack_says_what_is_still_outstanding():
    pack = Builder.from_log(PUMP, name="pump_station").step().pack()
    done, missing = pack.checklist()
    assert done and missing
    assert any("licence" in line for line in missing)
    assert 0 < pack.completeness() < 1


def test_the_pack_is_written_as_files_somebody_can_read(tmp_path):
    builder = Builder.from_log(PUMP, name="pump_station").step()
    folder = write_pack(builder.pack(), tmp_path)
    for name in ("pack.toml", "schema.lp", "rules.lp", "actions.lp",
                 "BUILD_NOTES.md", "lexicon.json"):
        assert (folder / name).exists(), name
    assert (folder / "tests" / "test_confirmed.py").exists()
    json.loads((folder / "lexicon.json").read_text())


@needs_clingo
def test_the_drafted_action_model_is_one_the_planner_accepts():
    from neuralmind.agency import parse_actions

    builder = Builder.from_log(PUMP, name="pump_station").step()
    library = parse_actions(builder.pack().actions_lp())
    assert len(library) >= 5
    assert "start_pump_a" in library


@needs_clingo
def test_a_drafted_model_plans_the_thing_the_log_never_showed():
    """Nothing in the log ever started the pump *and then* opened the valve.
    The model says what each does; the planner puts them in order."""
    from neuralmind.agency import Agency, parse_actions

    builder = Builder.from_log(PUMP, name="pump_station").step()
    agency = Agency().learn_actions(parse_actions(builder.pack().actions_lp()))
    agency.gate.grant(3)
    agency.observe(["pump_a(idle)", "valve_b(closed)", "flow(0)"])
    choice = agency.decide("flow(12)")
    assert choice.found
    assert len(choice.plan) == 2


# -- the second bar: a small game ------------------------------------------


@needs_clingo
def test_a_game_no_pack_covers_produces_a_model_that_actually_plays():
    """The roadmap's second bar, taken literally: draft the model from a log
    of random button-pressing, plan with it, then run the plan in the room
    and see whether the door opens."""
    from neuralmind.agency import Agency, parse_actions

    builder = Builder.from_log(vault_log(steps=600), name="vault").step()
    library = parse_actions(builder.pack().actions_lp())

    agency = Agency().learn_actions(library)
    agency.gate.grant(3)
    agency.observe(["holding(nothing)", "door(locked)", "lamp(off)"])
    choice = agency.decide("inside")
    assert choice.found, "the drafted model should be playable"

    game = Vault(99)
    for step in choice.plan:
        game.apply(step.name.split("_when_")[0])
    assert game.inside, "and the plan should work in the actual room"


def test_the_three_way_condition_is_found():
    """unlock_door needs the key, the light, and a locked door. Two of those
    are real conditions; the third is what 'changing the door' means."""
    builder = Builder.from_log(vault_log(steps=600), name="vault").step()
    draft = next(d for d in builder.actions if d.name == "unlock_door")
    effect = next(e for e in draft.effects if e.field == "door")
    named = {name for name, _ in effect.conditions}
    assert {"holding", "lamp"} <= named


# -- the third bar: a wrong answer, caught later ---------------------------


def _drive(station, commands, log):
    for command in commands:
        before = station.snapshot()
        station.apply(command)
        record = dict(before)
        record["action"] = command
        log.append(record)
    return log


CALM = [
    "close_valve_b", "start_pump_a", "start_pump_a", "start_pump_a",
    "start_pump_a", "open_valve_b", "open_valve_b", "open_valve_b",
    "open_valve_b", "reset_alarm",
]


def test_a_wrong_answer_is_caught_by_a_later_observation():
    """The roadmap's third bar.

    From a log where reset_alarm only ever happened while the pressure was
    low, it looks unconditional. Somebody confirms it. Then the world shows
    it failing -- and the builder reopens the question with the record
    attached, rather than carrying a wrong answer into the pack.
    """
    station = PumpStation(0)
    log = []
    for _ in range(4):
        _drive(station, CALM, log)
    log.append(station.snapshot())

    builder = Builder.from_log(log, name="pump_station").step()
    question = next(
        q for q in builder.interview.pending
        if q.key == "effect:reset_alarm:alarm"
    )
    assert "4 of the 4" in question.text
    builder.answer(question.key, YES)
    assert len(builder.interview.claims) == 1

    later = []
    _drive(
        station,
        ["close_valve_b", "start_pump_a", "start_pump_a", "start_pump_a",
         "start_pump_a", "reset_alarm", "reset_alarm"],
        later,
    )
    later.append(station.snapshot())
    reopened = builder.observe(later)

    assert reopened, "the contradiction should have come back as a question"
    back = reopened[0]
    assert back.key == "effect:reset_alarm:alarm"
    assert back.reopened
    assert back.contradicted_by
    assert "not False" in back.contradicted_by[0].why
    assert not builder.interview.agreed(back.key), "the answer no longer stands"


def test_being_contradicted_reopens_a_question_rather_than_deciding_it():
    """The builder does not overrule the person. It puts the question back
    with the evidence, because they may know something the log does not show."""
    station = PumpStation(0)
    log = []
    for _ in range(4):
        _drive(station, CALM, log)
    log.append(station.snapshot())
    builder = Builder.from_log(log, name="pump_station").step()
    builder.answer("effect:reset_alarm:alarm", YES)

    later = []
    _drive(station, ["close_valve_b"] + ["start_pump_a"] * 4 + ["reset_alarm"] * 2, later)
    later.append(station.snapshot())
    builder.observe(later)

    assert builder.interview.reopenings
    assert builder.interview.next is not None
    assert "you said yes before" in builder.interview.next.describe()


def test_an_answer_of_no_makes_no_claim_to_contradict():
    builder = Builder.from_log(PUMP, name="pump_station").step()
    key = builder.interview.next.key
    builder.answer(key, NO)
    assert key not in builder.interview.claims
    assert not builder.interview.agreed(key)


# -- asking well ------------------------------------------------------------

def test_the_most_unlocking_question_comes_first():
    """An effect makes an action usable by the planner; a unit settles one
    field. Attention is the scarcest thing the builder spends."""
    builder = Builder.from_log(PUMP, name="pump_station").step()
    assert builder.interview.next.kind == "effect"


def test_a_settled_question_is_not_asked_again():
    builder = Builder.from_log(PUMP, name="pump_station").step()
    key = builder.interview.next.key
    builder.answer(key, YES)
    builder.step()
    assert all(q.key != key for q in builder.interview.pending)


def test_the_developers_view_is_a_checklist_not_an_essay():
    builder = Builder.from_log(PUMP, name="pump_station").step()
    lines = builder.progress().splitlines()
    assert lines[0].startswith('New module "pump_station"')
    assert any(line.strip().startswith("✔") for line in lines)
    assert any(line.strip().startswith("✘") for line in lines)
    assert len(lines) < 20, "a checklist, not an essay"


# -- the log on disk --------------------------------------------------------


def test_a_log_survives_a_round_trip(tmp_path):
    path = write_log(PUMP[:20], tmp_path / "log.jsonl")
    assert read_log(path) == PUMP[:20]


# -- the commands a person actually types -----------------------------------


def test_the_shell_drafts_from_a_log_and_answers_its_questions(tmp_path):
    from neuralmind.shell import Shell

    path = write_log(PUMP, tmp_path / "pump.jsonl")
    shell = Shell()

    assert "bootstrap a log first" in shell.handle(":pack status")
    out = shell.handle(f":bootstrap {path} pump_station")
    assert 'New module "pump_station"' in out
    assert "✔" in out and "✘" in out

    asked = shell.handle(":ask")
    assert "Is that its effect?" in asked
    answered = shell.handle(":ask yes")
    assert "noted:" in answered and "next:" in answered

    written = shell.handle(f":pack write {tmp_path / 'packs'}")
    assert "written to" in written
    assert (tmp_path / "packs" / "pump_station" / "pack.toml").exists()


def test_the_shell_refuses_an_answer_outside_the_options(tmp_path):
    from neuralmind.shell import Shell

    path = write_log(PUMP, tmp_path / "pump.jsonl")
    shell = Shell()
    shell.handle(f":bootstrap {path}")
    assert "answer one of" in shell.handle(":ask maybe")


def test_promoting_a_pack_is_signed(tmp_path):
    """A drafted pack is a proposal about somebody else's domain."""
    from neuralmind.shell import Shell

    path = write_log(PUMP, tmp_path / "pump.jsonl")
    shell = Shell()
    shell.handle(f":bootstrap {path}")
    assert "signed" in shell.handle(":pack promote")


def test_a_pack_with_things_outstanding_is_not_promoted(tmp_path):
    from neuralmind.builder.shadow import Shadow

    builder = Builder.from_log(PUMP, name="pump_station").step()
    shadow = Shadow(builder.pack())
    shadow.run()
    allowed, why = shadow.may_promote()
    assert not allowed
    assert "outstanding" in why
    assert "not promoted" in shadow.promote(approved_by="someone")


def test_the_draft_goes_through_the_firewall_like_anything_else():
    """A pack drafted by counting gets no shortcut past the checks a
    hand-written rule faces."""
    from neuralmind.builder.shadow import Shadow

    builder = Builder.from_log(PUMP, name="pump_station").step()
    shadow = Shadow(builder.pack())
    trial = shadow.run()
    assert trial.loaded, trial.refused_because
    assert not trial.canaries_failed


def test_the_cli_drafts_a_pack_from_a_log(tmp_path):
    from neuralmind.cli import main

    path = write_log(PUMP, tmp_path / "pump.jsonl")
    code = main([
        "pack", "new",
        "--from-observations", str(path),
        "--name", "pump_station",
        "--out", str(tmp_path / "packs"),
    ])
    assert code == 0
    assert (tmp_path / "packs" / "pump_station" / "actions.lp").exists()


def test_the_cli_says_so_when_the_log_is_not_there(tmp_path):
    from neuralmind.cli import main

    assert main([
        "pack", "new", "--from-observations", str(tmp_path / "nope.jsonl")
    ]) == 1
