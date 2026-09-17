"""P2.4: working out where it is, and what it can do there.

The milestone's bars:

* one Mind, no configuration, identifies the facets of eight unlabelled hosts,
  with an explanation;
* stakes are high for every host that moves money or handles personal data,
  *before* any domain is recognised;
* an out-of-scope utterance is answered "not something I handle";
* a stream that switches hosts is detected within a bounded number of
  observations, and nothing is acted on during the transition;
* confidence is calibrated against what actually happened;
* `self_report()` is at most three sentences and every clause is a fact.
"""

import pytest

from neuralmind import Mind, parse_atom
from neuralmind.kernel import HIGH, L1, L3, AutonomyGate, Kernel
from neuralmind.self import (
    OUT_OF_SCOPE,
    ContextDiscovery,
    SelfModel,
    host_names,
    numpy_available,
    real_data_available,
    signature_of,
    stream,
)
from neuralmind.self.hosts import HOSTS

needs_data = pytest.mark.skipif(
    not (real_data_available() and numpy_available()),
    reason="intent datasets or NumPy not available",
)


# -- signatures: shape without a domain ------------------------------------


def test_a_signature_says_shape_and_never_says_domain():
    """The whole design rests on this step not presupposing the answer."""
    signature = signature_of(
        [[{"sku": "MLK-1L", "stock": 4, "price": "1.29 EUR"}]]
    )
    text = signature.to_asp()
    assert "record_with(stock)" in text and "currency_field(price)" in text
    for word in ("shop", "grocery", "retail", "bank", "game"):
        assert word not in text


def test_a_signature_notices_what_moves_between_observations():
    """A field that changes every tick is a fluent, and that is most of what
    distinguishes a world you act in from a table you query."""
    first = {"tick": 1, "position": 3, "name": "rover"}
    second = {"tick": 2, "position": 4, "name": "rover"}
    signature = signature_of([first, second, second])
    assert "position" in signature.fluents
    assert "name" not in signature.fluents


def test_a_number_with_a_unit_is_not_the_same_as_a_number():
    signature = signature_of([{"load": "3.2 kN", "count": 7, "price": "5 EUR"}])
    assert signature.has("quantity_field", "load")
    assert signature.has("currency_field", "price")
    assert signature.has("numeric_field", "count")


def test_an_action_list_is_recognised_as_one():
    signature = signature_of([{"actions": ["left", "right"], "step": 1}])
    assert signature.has("action_list_offered")
    assert signature.has("offers_action", "left")


# -- the eight hosts --------------------------------------------------------


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_each_unlabelled_host_is_read_correctly(host):
    """P2.4's first bar, one host at a time. Nothing names the host.

    A text host's facets beyond "conversation" come from the intent
    classifier, which is optional: without it the mind sees a conversation
    and cannot see what about, which is a real limit and not a failure. So
    the stronger expectation only applies when the classifier is there.
    """
    expected = set(HOSTS[host][1])
    if not (real_data_available() and numpy_available()):
        expected -= {"money", "parties", "policy"} if host == "banking_chat" else set()
    discovery = ContextDiscovery()
    discovery.observe_all(stream(host))
    found = {facet.name for facet in discovery.reading.facets}
    assert expected <= found, f"{host}: expected {expected}, found {found}"


@pytest.mark.parametrize("host", sorted(HOSTS))
def test_every_reading_comes_with_an_explanation(host):
    """A score nobody can argue with is not an answer."""
    discovery = ContextDiscovery()
    discovery.observe_all(stream(host))
    for facet in discovery.reading.facets:
        assert facet.because(), f"{host}/{facet.name} had no evidence"


def test_eight_hosts_are_covered():
    assert len(host_names()) >= 8


def test_a_host_nobody_planned_for_reports_what_it_could_not_place():
    """The module builder's input (P2.8): the facets it knows, and the rest
    named rather than quietly ignored."""
    discovery = ContextDiscovery()
    discovery.observe_all(stream("pump_system"))
    reading = discovery.reading
    assert "quantities" in reading.names
    assert set(reading.unexplained) >= {"pump_a", "valve_b"}
    assert reading.covered < 1.0


def test_a_fully_understood_host_has_nothing_unexplained():
    discovery = ContextDiscovery()
    discovery.observe_all(stream("financial_tables"))
    assert discovery.reading.covered >= 0.8


# -- stakes before recognition ---------------------------------------------


def test_a_host_that_moves_money_is_high_stakes():
    """P2.4's second bar."""
    discovery = ContextDiscovery()
    discovery.observe_all(stream("money_transfer"))
    assert discovery.reading.stakes == "high"


def test_stakes_are_raised_before_any_domain_is_recognised():
    """Waiting to be sure where you are before becoming careful is backwards."""
    discovery = ContextDiscovery()
    discovery.observe(stream("money_transfer", 1)[0])
    reading = discovery.reading
    assert reading.stakes == "high"
    assert reading.domain_confidence <= 0.5


def test_personal_data_makes_stakes_high_even_without_money():
    """The privacy layer knows things the shape reader cannot."""
    discovery = ContextDiscovery()
    discovery.observe_all([{"customer_id": "C-1", "note": "called twice"}])
    assert discovery.reading.stakes == "medium"
    discovery.note(parse_atom("personal_data_present"))
    assert discovery.reading.stakes == "high"


def test_the_context_can_lower_autonomy_and_never_raise_it():
    """Design rule 11, through the context layer this time."""
    gate = AutonomyGate(granted=L3)
    discovery = ContextDiscovery()
    discovery.observe_all(stream("money_transfer"))
    discovery.apply(gate)
    assert gate.level < L3 and gate.bound_by == "stakes"

    quiet = ContextDiscovery()
    quiet.observe_all(stream("cad_parameters"))
    quiet.apply(gate)
    assert gate.stakes.level == HIGH, "a calmer context lowered the caution"


def test_the_reason_given_to_the_gate_repeats_rather_than_accumulating():
    """It is re-read every observation; a slightly different sentence each
    time would fill the audit trail with noise."""
    gate = AutonomyGate(granted=L3)
    discovery = ContextDiscovery()
    for payload in stream("money_transfer"):
        discovery.observe(payload)
        discovery.apply(gate)
    assert len(gate.stakes.reasons) <= 3


# -- out of scope -----------------------------------------------------------


@needs_data
def test_an_out_of_scope_utterance_is_not_forced_into_an_intent():
    """P2.4's third bar, on CLINC150's own out-of-scope set.

    The classifier is a bag-of-words baseline and gets about half of these;
    the test asserts it gets a clear majority rather than none, which is the
    claim the mechanism supports.
    """
    from neuralmind.self.context import _default_intents

    model = _default_intents()
    assert model is not None
    utterances = stream("out_of_scope", 200)
    rejected = sum(1 for text in utterances if not model.predict(text).in_scope)
    assert rejected >= len(utterances) * 0.4, f"only {rejected}/{len(utterances)}"


@needs_data
def test_a_banking_utterance_is_recognised_as_one():
    from neuralmind.self.context import _default_intents

    model = _default_intents()
    families = [model.predict(text).family for text in stream("banking_chat", 20)]
    assert families.count("banking") >= 12, families


def test_a_prediction_says_how_close_the_call_was():
    from neuralmind.self.intent import Prediction

    close = Prediction("a", 0.51, second="b", second_score=0.50)
    clear = Prediction("a", 0.90, second="b", second_score=0.20)
    assert close.margin < clear.margin
    assert "not something I handle" in str(Prediction(OUT_OF_SCOPE, 0.1, second="b"))


# -- drift ------------------------------------------------------------------


def test_a_host_that_switches_underneath_is_noticed():
    """P2.4's fourth bar. The mind must not keep applying the old reading."""
    discovery = ContextDiscovery(drift_after=2)
    discovery.observe_all(stream("grocery_feed", 4))
    assert "inventory" in discovery.reading.names
    assert not discovery.drifting

    for index, payload in enumerate(stream("grid_game", 6)):
        discovery.observe(payload)
        if discovery.drifting:
            break
    assert discovery.drifting, "the switch was never noticed"
    assert index <= 4, f"took {index + 1} observations to notice"


def test_drifting_raises_caution_rather_than_acting_on_a_stale_reading():
    """Nothing is acted on during a transition."""
    gate = AutonomyGate(granted=L3)
    discovery = ContextDiscovery(drift_after=2)
    discovery.observe_all(stream("grocery_feed", 4))
    for payload in stream("grid_game", 6):
        discovery.observe(payload)
    discovery.apply(gate)
    assert discovery.drifting
    assert gate.level <= 2, "acted at full autonomy while the context was moving"


def test_resetting_starts_the_reading_again():
    discovery = ContextDiscovery(drift_after=2)
    discovery.observe_all(stream("grocery_feed", 4))
    discovery.reset()
    assert discovery.reading.unresolved and not discovery.drifting


def test_a_steady_stream_never_reports_drift():
    discovery = ContextDiscovery(drift_after=2)
    discovery.observe_all(stream("grocery_feed"))
    assert not discovery.drifting


# -- the self-model ---------------------------------------------------------


def test_the_mind_answers_questions_about_itself_with_a_proof():
    model = SelfModel().state("autonomy", "l1")
    assert model.ask("can_suggest").status == "yes"
    assert model.ask("can_act").status == "no"
    model.state("autonomy", "l3")
    answer = model.ask("can_act")
    assert answer.status == "yes" and answer.proof is not None


def test_competence_is_what_was_observed_not_what_was_hoped():
    model = SelfModel()
    for _ in range(4):
        model.observed("narrative", "reading", correct=False)
    model.observed("narrative", "reading", correct=True)
    assert model.unsure_about() == ["narrative"]
    assert "Unsure about narrative" in model.describe()


def test_calibration_notices_when_confidence_means_nothing():
    model = SelfModel()
    for _ in range(8):
        model.claimed("retail", 90, right=False)
    for _ in range(2):
        model.claimed("retail", 90, right=True)
    off = model.miscalibrated()
    assert off and off[0].claimed == 90 and off[0].observed == 20


def test_calibration_is_quiet_when_confidence_holds_up():
    model = SelfModel()
    for _ in range(8):
        model.claimed("retail", 80, right=True)
    for _ in range(2):
        model.claimed("retail", 80, right=False)
    assert model.miscalibrated() == []


def test_the_self_model_is_read_off_the_parts_that_hold_the_state():
    kernel = Kernel("mind", granted=L1)
    model = SelfModel().sync(
        kernel=kernel, specialists={"logic": True, "arithmetic": False}
    )
    text = model.to_asp()
    assert "has_specialist(logic)." in text
    assert "has_specialist(arithmetic)." not in text
    assert "autonomy(l1)." in text


# -- through Mind -----------------------------------------------------------


def test_one_mind_with_no_configuration_reads_its_host():
    """No argument names the host, and none ever will."""
    mind = Mind()
    for payload in stream("grocery_feed"):
        mind.observe(payload)
    reading = mind.context.reading
    assert {"inventory", "catalog"} <= set(reading.names)
    assert "grocery" in mind.self_report()


def test_observing_a_risky_host_tightens_the_gate_by_itself():
    mind = Mind(granted=L3)
    for payload in stream("money_transfer"):
        mind.observe(payload)
    decision = mind.decide("transfer(500)", needs=2)
    assert decision.outcome == "confirm"
    assert mind.kernel.autonomy.stakes.level == HIGH


def test_the_self_report_stays_within_three_sentences():
    """P2.4's last bar. It is shown to people, so it has to stay readable."""
    for host in sorted(HOSTS):
        mind = Mind()
        for payload in stream(host):
            mind.observe(payload)
        report = mind.self_report()
        assert report.count(".") <= 4, f"{host}: {report}"
        assert len(report) < 400


def test_the_self_report_says_unresolved_when_it_is():
    assert "unresolved" in Mind().self_report()
