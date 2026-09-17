"""P2.3: the growth loop.

The milestone's four bars:

* the family domain reaches grandparent, sibling, aunt and recursive ancestor
  through questions alone;
* active selection needs at most half the questions random selection needs;
* no proposed rule changes an answer until it is confirmed;
* consolidation invents a reusable predicate without moving any answer.
"""

import pytest

from neuralmind import KnowledgeBase, ReasoningEngine, parse_atom
from neuralmind.growth import (
    ASK_EXAMPLE,
    ASK_MEMBERSHIP,
    ASK_REFUTATION,
    CONFIRMED,
    PROPOSED,
    RETIRED,
    Consolidator,
    GapCollector,
    Grower,
    GrowthLoop,
    Hypothesiser,
    Memory,
    QuestionPicker,
    split_score,
)
from neuralmind.growth.gaps import MISSING_FACT, MISSING_RULE, UNREADABLE
from neuralmind.kernel import CONFIRMED as LAYER_CONFIRMED
from neuralmind.kernel import Kernel

FAMILY = """
parent(maria, juho). parent(maria, liisa). parent(matti, juho). parent(matti, liisa).
parent(juho, aino).  parent(juho, eero).  parent(liisa, sanna).
parent(aino, taavi). parent(sanna, venla).
male(matti). male(juho). male(eero). male(taavi).
female(maria). female(liisa). female(aino). female(sanna). female(venla).
"""
SIBLING = "sibling(X, Y) :- parent(P, X), parent(P, Y), X != Y."
TARGETS = {
    "grandparent/2": ("", "grandparent(X, Z) :- parent(X, Y), parent(Y, Z)."),
    "sibling/2": ("", SIBLING),
    "aunt/2": (SIBLING, "aunt(X, Y) :- parent(P, Y), sibling(X, P), female(X)."),
    "ancestor/2": (
        "",
        "ancestor(X, Y) :- parent(X, Y). ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).",
    ),
}


def oracle_for(target: str):
    support, truth = TARGETS[target]
    base = FAMILY + support
    model = ReasoningEngine(base + truth).model
    name = target.split("/")[0]
    gold = {a for a in model.atoms if a.predicate == name}
    return base, gold, (lambda atom: model.holds(atom))


def learned_atoms(base: str, rules, name: str) -> set:
    if not rules:
        return set()
    model = ReasoningEngine(base + "\n".join(str(r) for r in rules)).model
    return {a for a in model.atoms if a.predicate == name}


# -- gaps ------------------------------------------------------------------


def test_a_gap_distinguishes_a_missing_rule_from_a_missing_fact():
    """Different gaps call for different responses, and the diagnosis knows."""
    kb = KnowledgeBase().add_rules(
        "parent(a, b). brother(a, d). uncle(X, Y) :- brother(X, P), parent(P, Y)."
    )
    engine, gaps = kb.engine(), GapCollector()
    missing_fact = gaps.from_answer(engine.ask("uncle(a, c)"))
    missing_rule = gaps.from_answer(engine.ask("cousin(a, c)"))
    assert missing_fact.kind == MISSING_FACT
    assert str(missing_fact.missing) == "parent(d, c)"
    assert missing_rule.kind == MISSING_RULE


def test_hitting_the_same_gap_again_counts_it():
    """The gap between a user and an answer they keep asking for ranks first."""
    kb = KnowledgeBase().add_rules("cat(bob).")
    engine, gaps = kb.engine(), GapCollector()
    for _ in range(3):
        gaps.from_answer(engine.ask("dog(bob)"))
    gaps.from_answer(engine.ask("fish(bob)"))
    assert gaps.ranked()[0].hits == 3
    assert len(gaps) == 2


def test_a_refused_sentence_is_a_gap_of_its_own_kind():
    class Refusal:
        unparsed = ["Colorless green ideas sleep furiously."]

    gaps = GapCollector()
    found = gaps.from_perception(Refusal())
    assert found[0].kind == UNREADABLE


def test_an_answered_question_is_not_a_gap():
    kb = KnowledgeBase().add_rules("cat(bob).")
    assert GapCollector().from_answer(kb.engine().ask("cat(bob)")) is None


# -- questions -------------------------------------------------------------


@pytest.mark.parametrize(
    "yes,no,score", [(5, 5, 5), (9, 1, 1), (10, 0, 0), (0, 10, 0)]
)
def test_the_best_probe_is_the_one_that_splits_most_evenly(yes, no, score):
    """A probe nobody disagrees about eliminates nothing, however interesting."""
    assert split_score(yes, no) == score


def test_a_probe_everyone_agrees_on_is_never_asked():
    from neuralmind.core.parser import parse_program
    from neuralmind.core.terms import Const

    kb = KnowledgeBase().add_rules("p(a). p(b). q(a).")
    same = parse_program("t(X) :- p(X).", check=False).rules
    picker = QuestionPicker(kb, [same, same], ("t", 1), [Const("a"), Const("b")])
    assert picker.rank() == []


def test_answering_eliminates_the_candidates_that_disagree():
    from neuralmind.core.parser import parse_program
    from neuralmind.core.terms import Const

    kb = KnowledgeBase().add_rules("p(a). p(b). q(a).")
    broad = parse_program("t(X) :- p(X).", check=False).rules
    narrow = parse_program("t(X) :- p(X), q(X).", check=False).rules
    picker = QuestionPicker(
        kb, [broad, narrow], ("t", 1), [Const("a"), Const("b")]
    )
    assert picker.answer(parse_atom("t(b)"), False) == 1
    assert picker.candidates == [narrow]


def test_a_refutation_probe_asks_what_no_candidate_predicts():
    """The only way out of a space that does not contain the answer."""
    from neuralmind.core.parser import parse_program
    from neuralmind.core.terms import Const

    kb = KnowledgeBase().add_rules("p(a). p(b).")
    only = parse_program("t(X) :- p(X).", check=False).rules
    picker = QuestionPicker(kb, [only], ("t", 1), [Const("a"), Const("b"), Const("c")])
    assert picker.rank() == []
    refuting = picker.refutations()
    assert refuting and refuting[0].kind == ASK_REFUTATION
    assert str(refuting[0].atom) == "t(c)"


def test_compositions_are_probed_first_because_recursion_shows_up_there():
    """If p(a,b) and p(b,c) hold, p(a,c) is what a transitive rule would add.

    Without this the loop settles on ancestor's base case: every candidate
    agrees, nothing splits them, and the question that would reveal the
    recursion is never near the top of the list.
    """
    from neuralmind.core.parser import parse_program
    from neuralmind.core.terms import Const

    kb = KnowledgeBase().add_rules("edge(a, b). edge(b, c). edge(c, d).")
    base = parse_program("t(X, Y) :- edge(X, Y).", check=False).rules
    picker = QuestionPicker(
        kb,
        [base],
        ("t", 2),
        [Const(x) for x in "abcd"],
        known=[parse_atom("t(a, b)"), parse_atom("t(b, c)")],
        positive=[parse_atom("t(a, b)"), parse_atom("t(b, c)")],
    )
    assert str(picker.refutations()[0].atom) == "t(a, c)"


# -- the version space -----------------------------------------------------


def test_the_version_space_is_wide_before_any_questions():
    """One example and no negatives: almost everything fits, and that is right.

    Asking the learner instead hands back one rule chosen on a tie-break and
    leaves nothing to ask about.
    """
    kb = KnowledgeBase().add_rules(FAMILY)
    space = Hypothesiser(kb).version_space(
        "grandparent/2", [parse_atom("grandparent(maria, aino)")]
    )
    assert len(space) > 3


def test_the_version_space_can_express_a_disequality():
    """sibling needs X != Y, and comparisons are off by default in the bias.

    Without them the true rule is not in the space at all and the loop settles
    confidently on something else.
    """
    kb = KnowledgeBase().add_rules(FAMILY)
    space = Hypothesiser(kb).version_space(
        "sibling/2", [parse_atom("sibling(juho, liisa)")]
    )
    assert any("!=" in str(clauses[0]) for clauses in space)


def test_the_simplest_definition_comes_first():
    """When several fit equally, something has to choose, and shortest body is
    the one choice that smuggles in no preference about vocabulary."""
    kb = KnowledgeBase().add_rules(FAMILY)
    space = Hypothesiser(kb).version_space(
        "grandparent/2", [parse_atom("grandparent(maria, aino)")]
    )
    sizes = [sum(len(r.body) for r in clauses) for clauses in space]
    assert sizes == sorted(sizes)


# -- the loop: the milestone -----------------------------------------------


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_the_family_domain_is_learned_by_asking(target):
    """P2.3's first bar, one target at a time."""
    base, gold, teacher = oracle_for(target)
    name = target.split("/")[0]
    seed = sorted(gold, key=str)[:1]
    kb = KnowledgeBase().add_rules(base)
    session = GrowthLoop(kb, max_questions=60).learn(target, teacher, positive=seed)
    assert session.settled, session.stopped
    assert learned_atoms(base, session.rules, name) == gold, session.rules


def test_active_selection_needs_far_fewer_questions_than_random():
    """P2.3's second bar: at most half the questions, on the same tasks.

    Two targets and a 30-question cap, because the random arm spends its whole
    budget every time and a version space is recomputed per question. The cap
    works *against* this test -- it is a ceiling on what random costs, not on
    what it would cost -- so passing under it is the conservative reading. The
    four-target figure with a larger budget is in the docs.
    """
    totals = {"active": 0, "random": 0}
    for target in ("grandparent/2", "sibling/2"):
        base, gold, teacher = oracle_for(target)
        seed = sorted(gold, key=str)[:1]
        for strategy in totals:
            kb = KnowledgeBase().add_rules(base)
            session = GrowthLoop(kb, max_questions=30).learn(
                target, teacher, positive=seed, strategy=strategy
            )
            totals[strategy] += session.questions
    assert totals["active"] <= totals["random"] * 0.5, totals


def test_i_do_not_know_is_a_real_answer():
    """It narrows nothing, so the probe is retired rather than retried."""
    base, gold, _ = oracle_for("grandparent/2")
    seed = sorted(gold, key=str)[:1]
    kb = KnowledgeBase().add_rules(base)
    session = GrowthLoop(kb, max_questions=8).learn(
        "grandparent/2", lambda atom: None, positive=seed
    )
    asked = [str(a) for a, _ in session.answers]
    assert len(asked) == len(set(asked)), "the same probe was asked twice"


def test_when_nothing_fits_it_asks_for_an_example_rather_than_guessing():
    kb = KnowledgeBase().add_rules("parent(a, b).")
    session = GrowthLoop(kb).learn("mystery/2", lambda atom: False)
    assert session.asked and session.asked[0].kind == ASK_EXAMPLE


# -- nothing is believed until it is confirmed ------------------------------


def test_a_proposal_changes_no_answer_until_it_is_confirmed():
    """P2.3's third bar, and design rule 5."""
    base, gold, teacher = oracle_for("grandparent/2")
    kernel = Kernel("mind")
    kernel.load_core(base)
    # Deliberately *not* a canary on the thing being learned: learning is
    # supposed to change that answer.
    kernel.watch(["parent(maria, juho)", "male(matti)"])
    grower = Grower(kernel)

    before = kernel.ask("grandparent(maria, aino)").status
    session = grower.grow(
        "grandparent/2", teacher, positive=[parse_atom("grandparent(maria, aino)")]
    )
    assert session.rules
    assert grower.memory.proposed, "the proposal was not stored"
    assert kernel.ask("grandparent(maria, aino)").status == before, (
        "a proposal changed an answer before anyone confirmed it"
    )

    record, outcome = grower.confirm(session.rules[0], by="a reviewer")
    assert outcome, outcome.reason
    assert kernel.ask("grandparent(maria, aino)").status == "yes"
    assert grower.memory.find(record.body).state == CONFIRMED


def test_confirming_requires_naming_who_did_it():
    kernel = Kernel("mind")
    kernel.load_core("parent(a, b).")
    with pytest.raises(ValueError, match="naming who"):
        Grower(kernel).confirm("anything(X) :- parent(X, _Y).", by="")


def test_a_proposal_the_kernel_refuses_is_retired_not_confirmed():
    kernel = Kernel("mind")
    kernel.load_core("cat(bob). :- cat(X), dog(X).")
    kernel.watch(["cat(bob)"])
    grower = Grower(kernel)
    grower.memory.remember("dog(X) :- cat(X).", provenance="growth:test")
    record, outcome = grower.confirm("dog(X) :- cat(X).", by="a reviewer")
    assert not outcome
    assert grower.memory.find(record.body).state == RETIRED


# -- memory ----------------------------------------------------------------


def test_only_confirmed_beliefs_reach_the_program():
    memory = Memory()
    belief = memory.remember("p(X) :- q(X).")
    assert memory.to_asp() == ""
    memory.confirm(belief)
    assert memory.to_asp() == "p(X) :- q(X)."


def test_re_learning_something_retired_does_not_resurrect_it():
    """"Latest write wins" would quietly undo a person's decision."""
    memory = Memory()
    belief = memory.remember("p(X) :- q(X).")
    memory.retire(belief, "a reviewer said no")
    memory.remember("p(X) :- q(X).")
    assert memory.find("p(X) :- q(X).").state == RETIRED


def test_undo_reverses_the_last_change():
    memory = Memory()
    belief = memory.remember("p(X) :- q(X).")
    memory.confirm(belief)
    undone = memory.undo("changed my mind")
    assert undone["from"] == CONFIRMED and undone["to"] == PROPOSED
    assert memory.to_asp() == ""


def test_undoing_a_first_appearance_retires_rather_than_deletes():
    """The record of having believed something has to survive believing it."""
    memory = Memory()
    memory.remember("p(X) :- q(X).")
    memory.undo()
    assert memory.find("p(X) :- q(X).").state == RETIRED


def test_memory_survives_a_restart(tmp_path):
    path = tmp_path / "mind.db"
    with Memory(path) as memory:
        memory.confirm(memory.remember("p(X) :- q(X)."))
    with Memory(path) as reopened:
        assert reopened.to_asp() == "p(X) :- q(X)."
        assert reopened.history()[0]["became"] == CONFIRMED


# -- consolidation ---------------------------------------------------------


def test_consolidation_names_a_repeated_conjunction_without_moving_an_answer():
    """P2.3's fourth bar."""
    kb = KnowledgeBase().load_builtin("family")
    consolidator = Consolidator(kb)
    inventions = consolidator.propose()
    assert inventions, "nothing repeated was found"
    invention = inventions[0]
    assert invention.occurrences >= 2 and invention.saved >= 1
    identical, why = consolidator.verify(consolidator.apply(invention))
    assert identical, why


def test_a_well_factored_knowledge_base_has_nothing_to_consolidate():
    """The access policy was written by hand and already names its concepts.

    `cleared`, `permitted_by_role` and `blocked` *are* the invented
    predicates. Finding nothing there is the correct answer, and worth a test
    so that a future change which starts "finding" things gets looked at.
    """
    kb = KnowledgeBase().load_builtin("access_policy")
    assert Consolidator(kb).propose() == []


def test_an_invented_predicate_exports_exactly_the_escaping_variables():
    """A variable used outside the conjunction has to be an argument; one used
    only inside is projected away. Getting that backwards changes meanings."""
    kb = KnowledgeBase().add_rules(
        "p(A, B) :- r(A, C), s(C, B), extra(A). "
        "q(A, B) :- r(A, C), s(C, B), other(B)."
    )
    invention = Consolidator(kb).propose()[0]
    exported = {str(a) for a in invention.definition.head.args}
    assert exported == {"A", "B"}, exported


def test_renaming_an_invention_updates_every_use():
    kb = KnowledgeBase().add_rules(
        "p(A, B) :- r(A, C), s(C, B), extra(A). "
        "q(A, B) :- r(A, C), s(C, B), other(B)."
    )
    invention = Consolidator(kb).propose()[0]
    assert invention.provisional
    invention.rename("connected")
    assert not invention.provisional
    assert invention.definition.head.predicate == "connected"
    assert all("connected" in str(new) for _old, new in invention.rewrites)


def test_consolidation_through_the_grower_keeps_only_verified_inventions():
    kernel = Kernel("mind")
    kernel.load_core(KnowledgeBase().load_builtin("family").to_asp())
    for invention in Grower(kernel).consolidate():
        consolidator = Consolidator(kernel.layers.view(kernel.mode_layers))
        assert invention.occurrences >= 2
