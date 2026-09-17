"""P2.2: the safety kernel.

The milestone's four bars, plus the mechanisms behind them:

* 10,000 random, malformed and contradictory inputs leave the core's canaries
  passing (a slice here; the full run is in the docs);
* an injected bad pack is quarantined and the mind keeps answering;
* every degrade-mode change appears in the self-report with its reason;
* tenants do not leak, and no action above the granted level reaches the host.
"""

import random
import string

import pytest

from neuralmind import parse_atom
from neuralmind.kernel import (
    CONFIRMED,
    CORE,
    CORE_ONLY,
    DEFAULT_LAYERS,
    FULL,
    HIGH,
    L0,
    L1,
    L2,
    L3,
    LAYER_ORDER,
    OBSERVE_ONLY,
    SANDBOX,
    SESSION,
    TENANT,
    AutonomyGate,
    CanarySet,
    Firewall,
    Kernel,
    LayerStack,
    PrivacyPolicy,
    Quarantine,
    ReadOnlyLayer,
    Retention,
    Stakes,
    Watchdog,
    transaction,
)

CORE_RULES = """
mammal(X) :- cat(X).
mammal(X) :- dog(X).
warm_blooded(X) :- mammal(X).
:- cat(X), dog(X).
#open flies/1.
"""


@pytest.fixture
def kernel():
    k = Kernel("test", granted=L2)
    k.load_core(CORE_RULES)
    k.layers.add_fact("cat(bob)", CONFIRMED)
    k.layers.add_fact("dog(rex)", CONFIRMED)
    k.watch(["mammal(bob)", "warm_blooded(bob)", "flies(bob)", "dog(bob)"])
    return k


# -- layers ----------------------------------------------------------------


def test_the_core_is_read_only_at_runtime():
    """Nothing the mind learns may modify what it shipped with."""
    stack = LayerStack("t")
    stack.load_core("mammal(X) :- cat(X).")
    with pytest.raises(ReadOnlyLayer, match="read-only"):
        stack.add_rules("anything(X) :- cat(X).", CORE)
    with pytest.raises(ReadOnlyLayer):
        stack.add_fact("cat(bob)", CORE)


def test_the_sandbox_is_excluded_from_queries_by_default():
    """Tentative knowledge cannot answer a question by accident.

    The sandbox is dangerous only if you read it, so reading it is the thing
    you opt into rather than the thing you opt out of.
    """
    stack = LayerStack("t")
    stack.add_fact("cat(bob)", CONFIRMED)
    stack.add_rules("weird(X) :- cat(X).", SANDBOX)
    assert not stack.holds(parse_atom("weird(bob)"))
    assert stack.holds(parse_atom("weird(bob)"), LAYER_ORDER)


def test_a_later_layer_overrides_an_earlier_one_for_facts():
    stack = LayerStack("t")
    stack.add_fact("rate(standard)", CONFIRMED, confidence=0.5)
    stack.add_fact("rate(standard)", SESSION, confidence=1.0)
    assert stack.view().confidence_of(parse_atom("rate(standard)")) == 1.0


def test_rules_accumulate_rather_than_override():
    """A rule is a licence to conclude; removing one silently changes answers."""
    stack = LayerStack("t")
    stack.load_core("mammal(X) :- cat(X).")
    stack.add_rules("mammal(X) :- dog(X).", CONFIRMED)
    stack.add_fact("dog(rex)", CONFIRMED)
    stack.add_fact("cat(bob)", CONFIRMED)
    assert stack.holds(parse_atom("mammal(bob)"))
    assert stack.holds(parse_atom("mammal(rex)"))


# -- transactions ----------------------------------------------------------


def test_a_rolled_back_transaction_leaves_no_trace():
    stack = LayerStack("t")
    stack.add_fact("cat(bob)", CONFIRMED)
    before = stack.view().to_asp()
    with transaction(stack, "something risky") as change:
        stack.add_rules("flies(X) :- cat(X).", CONFIRMED)
        stack.add_fact("dog(rex)", SESSION)
        change.rollback("changed my mind")
    assert change.rolled_back
    assert stack.view().to_asp() == before


def test_an_exception_inside_a_transaction_rolls_back_and_re_raises():
    """A step that crashed half-way is the worst case to leave in place."""
    stack = LayerStack("t")
    stack.add_fact("cat(bob)", CONFIRMED)
    before = stack.view().to_asp()
    with pytest.raises(RuntimeError):
        with transaction(stack, "crashing") as change:
            stack.add_fact("dog(rex)", CONFIRMED)
            raise RuntimeError("boom")
    assert change.rolled_back and "boom" in change.reason
    assert stack.view().to_asp() == before


# -- canaries --------------------------------------------------------------


def test_a_canary_catches_an_over_general_rule(kernel):
    """unknown → yes is the change an over-general induced rule makes."""
    outcome = kernel.learn("flies(X) :- cat(X).")
    assert not outcome
    assert "flies(bob)" in outcome.reason and "unknown" in outcome.reason


def test_canaries_record_all_three_answers():
    """Silence counts. An answer that was unknown and is now yes is a change."""
    stack = LayerStack("t")
    stack.load_core("mammal(X) :- cat(X). #open flies/1.")
    stack.add_fact("cat(bob)", CONFIRMED)
    captured = CanarySet.capture(stack, ["mammal(bob)", "flies(bob)", "dog(bob)"])
    assert {c.status for c in captured} == {"yes", "unknown", "no"}


def test_a_theory_that_will_not_compile_fails_every_canary():
    stack = LayerStack("t")
    stack.load_core("mammal(X) :- cat(X).")
    stack.add_fact("cat(bob)", CONFIRMED)
    canaries = CanarySet.capture(stack, ["mammal(bob)"])
    stack.add_rules("broken(X, Y) :- cat(X).", CONFIRMED)
    assert canaries.check(stack)


# -- firewall --------------------------------------------------------------


@pytest.mark.parametrize(
    "source,check",
    [
        ("blocked(E, R) :- suspended(E).", "safety"),
        ("p(X) :- cat(X), not q(X). q(X) :- cat(X), not p(X).", "stratification"),
        ("dog(X) :- cat(X).", "consistency"),
        ("-mammal(X) :- cat(X).", "consistency"),
        ("this is not valid syntax(", "parse"),
    ],
)
def test_the_firewall_rejects_and_says_which_check(source, check):
    stack = LayerStack("t")
    stack.load_core(CORE_RULES)
    stack.add_fact("cat(bob)", CONFIRMED)
    verdict = Firewall(stack).check(source)
    assert not verdict
    assert verdict.check == check


def test_the_firewall_admits_a_rule_that_is_merely_new():
    stack = LayerStack("t")
    stack.load_core(CORE_RULES)
    stack.add_fact("cat(bob)", CONFIRMED)
    assert Firewall(stack).check("furry(X) :- cat(X).")


def test_the_firewall_checks_against_a_copy_and_never_the_live_stack():
    """A firewall that modified what it checked would not be one."""
    stack = LayerStack("t")
    stack.load_core(CORE_RULES)
    stack.add_fact("cat(bob)", CONFIRMED)
    before = stack.view().to_asp()
    Firewall(stack).check("furry(X) :- cat(X).")
    Firewall(stack).check("dog(X) :- cat(X).")
    assert stack.view().to_asp() == before


def test_the_firewall_allows_what_the_engine_can_actually_run():
    """A firewall stricter than the engine is a bug this project shipped once.

    Predicate-level stratification rejects theories the engine runs fine via
    local (atom-level) stratification, so the check has to use the same test.
    """
    stack = LayerStack("t")
    stack.load_core("p(a). q(b).")
    verdict = Firewall(stack).check("r(X) :- p(X), not s(X). s(X) :- q(X).")
    assert verdict, verdict.describe()


# -- quarantine ------------------------------------------------------------


def test_quarantine_counts_repeats_rather_than_forgetting():
    """One bad rule is an accident; the same one four times has a cause."""
    quarantine = Quarantine()
    for _ in range(3):
        quarantine.add("flies(X) :- cat(X).", reason="broke a canary")
    assert quarantine.repeat_offenders[0].occurrences == 3


def test_releasing_from_quarantine_requires_naming_who_did_it():
    quarantine = Quarantine()
    quarantine.add("p(X) :- q(X).", reason="broke a canary")
    with pytest.raises(ValueError, match="naming who"):
        quarantine.release("p(X) :- q(X).", by="")
    assert quarantine.release("p(X) :- q(X).", by="a reviewer").released_by


def test_a_quarantined_rule_is_not_tried_again(kernel):
    kernel.learn("flies(X) :- cat(X).")
    second = kernel.learn("flies(X) :- cat(X).")
    assert not second and "quarantined" in second.reason


# -- degrade modes ---------------------------------------------------------


def test_repeated_trouble_steps_the_mode_down():
    watchdog = Watchdog(tolerance=3)
    assert watchdog.report("budget") is None
    assert watchdog.report("budget") is None
    change = watchdog.report("budget", "a slow query")
    assert change is not None and change.to == CORE_ONLY


def test_one_success_clears_the_strikes():
    """A budget overrun on an unusually hard question is normal, not a trend."""
    watchdog = Watchdog(tolerance=3)
    watchdog.report("budget")
    watchdog.report("budget")
    watchdog.healthy()
    assert watchdog.report("budget") is None
    assert watchdog.mode.name == FULL


def test_core_only_reads_only_the_core():
    watchdog = Watchdog(tolerance=1)
    watchdog.report("budget")
    assert tuple(watchdog.mode.layers) == (CORE,)
    assert watchdog.mode.answers and not watchdog.mode.learns


def test_observe_only_answers_nothing(kernel):
    kernel.watchdog.step_down("first")
    kernel.watchdog.step_down("second")
    assert kernel.watchdog.mode.name == OBSERVE_ONLY
    assert kernel.ask("mammal(bob)") is None


def test_stepping_back_up_requires_evidence():
    """A mode that recovers because time passed recovers into the same fault."""
    watchdog = Watchdog(tolerance=1)
    watchdog.report("budget")
    with pytest.raises(ValueError, match="requires evidence"):
        watchdog.step_up("")


def test_recovering_checks_the_canaries_first(kernel):
    kernel.watchdog.step_down("something went wrong")
    assert kernel.watchdog.mode.name == CORE_ONLY
    change = kernel.recover()
    assert change is not None and change.to == FULL


def test_recovering_refuses_while_a_canary_is_still_failing(kernel):
    kernel.watchdog.step_down("something went wrong")
    # Break a canary in a way no rollback undid.
    kernel.layers.add_rules("mammal(X) :- cat(X).", CONFIRMED)
    kernel.canaries.add("mammal(bob)", "no", "deliberately wrong")
    assert kernel.recover() is None


def test_every_mode_change_appears_in_the_self_report(kernel):
    """P2.2's third bar."""
    kernel.watchdog.step_down("the canaries disagreed after a pack landed")
    report = kernel.self_report()
    assert CORE_ONLY in report
    assert "the canaries disagreed after a pack landed" in report


def test_a_rolled_back_change_does_not_degrade_the_mode(kernel):
    """The system working is not the system failing.

    A canary failure the rollback already fixed is a rule that was caught.
    Stepping down is for a failure still present after the way back was taken.
    """
    kernel.learn("flies(X) :- cat(X).")
    assert kernel.watchdog.mode.name == FULL
    assert kernel.canaries.healthy(kernel.layers)


# -- autonomy --------------------------------------------------------------


def test_stakes_lower_autonomy_and_a_grant_raises_it():
    """Design rule 11, as arithmetic: min(grant, stakes ceiling)."""
    gate = AutonomyGate(granted=L3)
    assert gate.check("transfer(500)", needs=L2).outcome == "act"
    gate.raise_stakes(HIGH, "money moves")
    decision = gate.check("transfer(500)", needs=L2)
    assert decision.outcome == "confirm" and decision.level == L2
    assert gate.bound_by == "stakes"


def test_stakes_can_never_be_lowered():
    """A guess that lowers caution is a way for a misread to authorise a transfer."""
    stakes = Stakes().raised_to(HIGH, "money moves")
    assert stakes.raised_to("low", "looks fine now").level == HIGH


def test_high_stakes_cap_autonomy_at_confirmation_whatever_was_granted():
    gate = AutonomyGate(granted=L3).raise_stakes(HIGH, "decides about a person")
    assert gate.level == L2
    assert not gate.check("decline(application)", needs=L3).allowed


def test_no_action_above_the_granted_level_reaches_the_host(kernel):
    """P2.2's fourth bar, second half."""
    kernel.autonomy.grant(L1)
    decision = kernel.decide("transfer(500)", needs=L2)
    assert not decision.allowed and decision.outcome == "suggest"
    assert all(not d.allowed for d in kernel.autonomy.log if d.action == "transfer(500)")


def test_observe_only_autonomy_refuses_even_to_answer():
    gate = AutonomyGate(granted=L0)
    assert gate.check("answer(anything)", needs=L1).outcome == "refuse"


def test_every_decision_is_logged_with_what_bound_it(kernel):
    kernel.decide("a", needs=L1)
    kernel.decide("b", needs=L3)
    assert len(kernel.autonomy.log) == 2
    assert all(d.to_dict()["granted"] for d in kernel.autonomy.log)


# -- privacy ---------------------------------------------------------------


def test_personal_facts_are_routed_into_a_confined_layer_whatever_was_asked():
    """The caller asking for the wrong layer is the mistake this stops."""
    stack = LayerStack("acme", tenant="acme")
    policy = PrivacyPolicy().declare("salary", "pay")
    policy.admit(["salary(bob, 50000)", "cat(felix)"], stack, layer=CONFIRMED)
    assert stack.layer_of(parse_atom("salary(bob, 50000)")) == SESSION
    assert stack.layer_of(parse_atom("cat(felix)")) == CONFIRMED


@pytest.mark.parametrize(
    "fact",
    [
        'email(carol, "c@example.io")',
        'phone(dan, "+358 40 1234567")',
        'iban(eve, "FI2112345600000785")',
    ],
)
def test_patterns_catch_what_the_host_forgot_to_declare(fact):
    stack = LayerStack("acme", tenant="acme")
    policy = PrivacyPolicy()
    policy.admit([fact], stack, layer=CONFIRMED)
    assert len(policy.tags) == 1


def test_a_confined_fact_cannot_be_promoted_out():
    stack = LayerStack("acme", tenant="acme")
    stack.add_fact("salary(bob, 50000)", SESSION)
    with pytest.raises(ReadOnlyLayer, match="never become shared"):
        stack.promote(parse_atom("salary(bob, 50000)"), SESSION, CONFIRMED)


def test_an_ordinary_fact_can_be_promoted():
    stack = LayerStack("t")
    stack.add_rules("cat(felix).", SANDBOX)
    stack.add_fact("dog(rex)", SANDBOX)
    stack.promote(parse_atom("dog(rex)"), SANDBOX, CONFIRMED)
    assert stack.layer_of(parse_atom("dog(rex)")) == CONFIRMED


def test_a_rule_induced_from_personal_data_waits_for_a_person():
    policy = PrivacyPolicy().declare("salary", "pay")
    policy.tag(parse_atom("salary(bob, 50000)"))
    held = policy.review(
        "well_paid(X) :- salary(X, S), S > 40000.",
        [parse_atom("salary(bob, 50000)")],
    )
    assert held is not None and policy.held == [held]
    with pytest.raises(ValueError, match="naming who"):
        policy.clear(held, by="")
    policy.clear(held, by="a reviewer")
    assert policy.held == []


def test_a_rule_induced_from_nothing_personal_is_not_held():
    policy = PrivacyPolicy()
    assert policy.review("mammal(X) :- cat(X).", [parse_atom("cat(bob)")]) is None


def test_retention_deletes_on_the_sweep_and_not_before():
    stack = LayerStack("acme", tenant="acme")
    policy = PrivacyPolicy().declare("salary", "pay").keep_for("pay", 60.0)
    policy.admit(["salary(bob, 50000)"], stack, layer=SESSION)
    assert policy.sweep(stack, now=0) == []
    removed = policy.sweep(stack, now=policy.tags[0].at + 61)
    assert [str(a) for a in removed] == ["salary(bob, 50000)"]
    assert stack.layer_of(parse_atom("salary(bob, 50000)")) is None


def test_ending_a_session_forgets_it():
    stack = LayerStack("acme", tenant="acme")
    policy = PrivacyPolicy().declare("salary", "pay")
    policy.admit(["salary(bob, 50000)"], stack, layer=SESSION)
    assert policy.forget_session(stack) == 1
    assert stack.layer_of(parse_atom("salary(bob, 50000)")) is None
    assert policy.tags == []


def test_a_failing_detector_does_not_open_the_gate():
    """If the classifier breaks, the fact is personal, not public."""
    def broken(atom):
        raise RuntimeError("model unavailable")

    policy = PrivacyPolicy(detector=broken)
    assert policy.classify(parse_atom("anything(x)")) == "personal"


# -- tenants ---------------------------------------------------------------


def test_what_one_tenant_knows_never_reaches_another():
    """Design rule 12, and P2.2's fourth bar.

    Marker records are planted in one tenant and looked for everywhere in the
    other: its answers, its rules, and a snapshot of it.
    """
    from neuralmind.kernel import Snapshot

    acme = Kernel("acme", tenant="acme")
    beta = Kernel("beta", tenant="beta")
    for kernel in (acme, beta):
        kernel.load_core(CORE_RULES)

    marker = parse_atom("secret(acme_only_marker)")
    acme.layers.add_fact(marker, TENANT)
    acme.layers.add_rules("derived(X) :- secret(X).", CONFIRMED)

    assert acme.layers.holds(marker)
    assert not beta.layers.holds(marker)
    assert not beta.layers.holds(parse_atom("derived(acme_only_marker)"))

    text = beta.layers.view(LAYER_ORDER).to_asp()
    assert "acme_only_marker" not in text

    snapshot = Snapshot.of(beta.layers)
    for knowledge in snapshot.layers.values():
        assert "acme_only_marker" not in knowledge.to_asp()


def test_a_snapshot_of_one_tenant_restores_only_that_tenant():
    from neuralmind.kernel import Snapshot

    acme = LayerStack("acme", tenant="acme")
    acme.add_fact("secret(a)", TENANT)
    taken = Snapshot.of(acme)
    beta = LayerStack("beta", tenant="beta")
    beta.add_fact("secret(b)", TENANT)
    taken.restore(acme)
    assert acme.holds(parse_atom("secret(a)"))
    assert not acme.holds(parse_atom("secret(b)"))


# -- the whole thing under pressure ----------------------------------------


def _junk(rng) -> str:
    kind = rng.randrange(8)
    word = lambda: "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 6)))
    var = lambda: rng.choice("XYZW")
    if kind == 0:
        return "".join(rng.choices(string.printable, k=rng.randint(1, 40)))
    if kind == 1:
        return f"{word()}({var()} :- {word()}({var()})."
    if kind == 2:
        return f"{word()}({var()}, {var()}) :- {word()}({var()})."
    if kind == 3:
        a, b = word(), word()
        return f"{a}(X) :- cat(X), not {b}(X). {b}(X) :- cat(X), not {a}(X)."
    if kind == 4:
        return rng.choice(["dog(X) :- cat(X).", "-mammal(X) :- cat(X)."])
    if kind == 5:
        return f"{rng.choice(['flies', 'mammal', 'warm_blooded'])}(X) :- {word()}(X)."
    if kind == 6:
        return "big(X) :- cat(X). big(f(X)) :- big(X)."
    return f"{word()}({var()}) :- cat({var()})."


def test_junk_never_reaches_the_core(kernel):
    """A slice of P2.2's fuzz bar; the 10,000-input run is in the docs."""
    rng = random.Random(7)
    core_text = kernel.layers.core.to_asp()
    for _ in range(400):
        kernel.learn(_junk(rng), CONFIRMED)
        assert kernel.layers.core.to_asp() == core_text
        assert kernel.canaries.healthy(kernel.layers, kernel.mode_layers)


def test_a_bad_pack_is_quarantined_and_the_mind_keeps_answering(kernel):
    """P2.2's second bar, with all three kinds of bad at once.

    The runaway case is arithmetic rather than a function term: this language
    has no function symbols, so ``deep(g(X))`` is rejected at the parser and
    never reaches the budget check it was meant to exercise.
    """
    kernel.layers.add_fact("n(1)", CONFIRMED)
    pack = {
        "unsafe": "unsafe(A, B) :- cat(A).",
        "contradiction": "dog(X) :- cat(X).",
        "runaway": "n(Y) :- n(X), Y = X + 1.",
    }
    checks = {}
    for label, rule in pack.items():
        outcome = kernel.learn(rule, CONFIRMED)
        assert not outcome, label
        # `is not None`, not truthiness: a rejected Verdict is deliberately
        # falsey, so `if outcome.verdict` asks a different question.
        checks[label] = (
            outcome.verdict.check if outcome.verdict is not None else "canary"
        )
    assert checks == {
        "unsafe": "safety",
        "contradiction": "consistency",
        "runaway": "budget",
    }
    assert len(kernel.quarantine) == 3
    assert kernel.ask("mammal(bob)").status == "yes"
    assert kernel.ask("warm_blooded(bob)").status == "yes"
    assert kernel.canaries.healthy(kernel.layers)





def test_a_rejected_verdict_is_falsey_but_still_present():
    """`if verdict` asks "did it pass?", not "is there one?".

    Conflating the two silently drops the reason from every rejection, which
    is the most useful part of it.
    """
    from neuralmind.kernel import REJECTED, Verdict

    verdict = Verdict(REJECTED, "p(X, Y) :- q(X).", "safety", "unbound Y")
    assert not verdict
    assert verdict is not None and verdict.check == "safety"


def test_the_canaries_can_read_a_model_someone_already_solved():
    """The firewall's dry run is exactly the program the canaries need.

    Solving it twice per guarded change was pure waste, and the saving grows
    with the knowledge base -- which is where it matters.
    """
    stack = LayerStack("t")
    stack.load_core(CORE_RULES)
    stack.add_fact("cat(bob)", CONFIRMED)
    canaries = CanarySet.capture(stack, ["mammal(bob)", "flies(bob)", "dog(bob)"])

    verdict = Firewall(stack).check("furry(X) :- cat(X).")
    assert verdict and verdict.model is not None
    assert canaries.check(stack, model=verdict.model) == canaries.check(stack)


def test_reading_a_model_keeps_the_three_answers_apart():
    """An absence means different things per predicate, and the model alone
    cannot say which -- so it carries the program that can."""
    stack = LayerStack("t")
    stack.load_core(CORE_RULES)
    stack.add_fact("cat(bob)", CONFIRMED)
    canaries = CanarySet.capture(stack, ["mammal(bob)", "flies(bob)", "dog(bob)"])
    verdict = Firewall(stack).check("furry(X) :- cat(X).")
    # flies/1 is open, dog/1 is not: both are underivable and they differ.
    statuses = {str(c.goal): c.status for c in canaries}
    assert statuses["flies(bob)"] == "unknown"
    assert statuses["dog(bob)"] == "no"
    assert not canaries.check(stack, model=verdict.model)


def test_a_sandbox_change_does_not_reuse_the_firewall_model(kernel):
    """The firewall's view excludes the sandbox, so its dry-run model is of a
    different program than the one that exists afterwards.

    Reusing it there would check the canaries against a theory that includes
    the sandbox rule as though it were confirmed -- which is both the wrong
    answer and the wrong direction, since the sandbox is meant to be invisible.
    """
    from neuralmind.kernel import LAYER_ORDER, SANDBOX

    assert SANDBOX not in kernel.firewall.layers

    # A rule that would break a canary if it were visible.
    outcome = kernel.learn("flies(X) :- cat(X).", SANDBOX)
    assert outcome, outcome.reason
    # Accepted, because the sandbox is not in the default view...
    assert kernel.canaries.healthy(kernel.layers)
    assert kernel.ask("flies(bob)").status == "unknown"
    # ...and visible only to someone who asks for it.
    assert kernel.layers.engine(LAYER_ORDER).ask("flies(bob)").status == "yes"
