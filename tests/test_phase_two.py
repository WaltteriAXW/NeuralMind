"""Phase Two milestones, one named test per "done when".

Mirrors tests/test_roadmap_phases.py, which does the same for Phase One. A
milestone that has no test here is not done, whatever the code says.
"""

import pytest

from neuralmind import KnowledgeBase, ReasoningEngine, parse_atom, parse_program
from neuralmind.core.terms import Atom, Const
from neuralmind.mind import Mind
from neuralmind.output.brief import BRIEF_WORDS, brief
from neuralmind.output.nlg import Realiser


# -- P2.0: strong negation -------------------------------------------------


def test_strong_negation_parses_and_survives_a_round_trip():
    program = parse_program("-flies(pingu). penguin(X) :- -flies(X).")
    assert str(program.rules[0]) == "-flies(pingu)."
    # The printed form is what clingo reads, so it has to stay exactly this.
    assert "-flies(X)" in str(program.rules[1])


def test_strong_negation_is_not_the_same_as_failure_to_derive():
    """"not p" means p was not derived; "-p" claims p is false.

    Conflating them is what makes an open-world answer wrong: silence about a
    predicate the program only partly describes is not evidence of anything.
    """
    engine = ReasoningEngine("#open flies/1. bird(tweety). penguin(pingu). -flies(X) :- penguin(X).")
    assert engine.ask("flies(pingu)").status == "no"
    assert engine.ask("flies(tweety)").status == "unknown"


def test_a_negative_answer_carries_a_proof():
    engine = ReasoningEngine("#open flies/1. penguin(pingu). -flies(X) :- penguin(X).")
    answer = engine.ask("flies(pingu)")
    assert answer.status == "no"
    assert answer.refutation is not None
    assert str(answer.refutation.conclusion) == "-flies(pingu)"
    assert parse_atom("penguin(pingu)") in answer.refutation.premises()


def test_deriving_both_polarities_is_a_contradiction():
    """clingo rejects such a program; the engine reports which pair clashed."""
    model = ReasoningEngine("flies(tweety). -flies(tweety). cold(bob).").model
    assert [str(a) for a in model.contradictions] == ["flies(tweety)"]
    assert not model.coherent and not model.consistent


def test_the_complement_of_a_complement_is_the_atom():
    atom = Atom("flies", (Const("tweety"),))
    assert atom.complement().complement() == atom
    assert atom.complement().is_negated and not atom.is_negated
    assert atom.complement().positive == atom


# -- P2.0: open and closed predicates -------------------------------------


def test_closed_world_is_still_the_default():
    """Phase One's benchmarks are graded under CWA and must not move."""
    engine = ReasoningEngine("cat(bob).")
    assert engine.ask("cat(zoe)").status == "no"
    assert engine.ask("dog(bob)").status == "no"


@pytest.mark.parametrize(
    "declaration,expected",
    [("#open flies/1.", "unknown"), ("#closed flies/1.", "no"), ("", "no")],
)
def test_the_declaration_decides_what_silence_means(declaration, expected):
    engine = ReasoningEngine(f"{declaration} bird(tweety).")
    assert engine.ask("flies(tweety)").status == expected


def test_a_later_closed_declaration_overrides_an_earlier_open_one():
    kb = KnowledgeBase().add_rules("#open flies/1. bird(tweety).")
    assert kb.engine().ask("flies(tweety)").status == "unknown"
    kb.add_rules("#closed flies/1.")
    assert kb.engine().ask("flies(tweety)").status == "no"


def test_declarations_survive_the_knowledge_base():
    """They live on the Program, which the KB rebuilds on every query."""
    kb = KnowledgeBase().add_rules("#open flies/1, swims/1. bird(tweety).")
    assert kb.engine().ask("swims(tweety)").status == "unknown"
    assert kb.copy().engine().ask("swims(tweety)").status == "unknown"


def test_unknown_is_falsey_so_phase_one_code_still_reads_it_correctly():
    answer = ReasoningEngine("#open flies/1. bird(tweety).").ask("flies(tweety)")
    assert not answer and not answer.holds and answer.unknown


# -- P2.0: the gap behind an unknown ---------------------------------------


def test_an_unknown_answer_names_what_would_settle_it():
    """"I don't know" is useless; "I don't know whether Bob is furry" is a question."""
    engine = ReasoningEngine(
        "#open warm/1. isa(bob, cat). isa(X, mammal) :- isa(X, cat). "
        "warm(X) :- isa(X, mammal), furry(X)."
    )
    answer = engine.ask("warm(bob)")
    assert answer.status == "unknown"
    attempt = max(answer.diagnosis.attempts, key=lambda a: a.progress)
    assert [str(a) for a in attempt.established] == ["isa(bob, mammal)"]
    assert str(attempt.missing) == "furry(bob)"


# -- P2.0: brief answers ---------------------------------------------------


@pytest.fixture(scope="module")
def zoo():
    return ReasoningEngine(
        "#open furry/1, warm_blooded/1. "
        "isa(bob, cat). isa(X, mammal) :- isa(X, cat). "
        "attr(X, warm_blooded) :- isa(X, mammal). "
        "-attr(X, cold) :- attr(X, warm_blooded)."
    )


@pytest.mark.parametrize(
    "query,status,contains",
    [
        ("attr(bob, warm_blooded)", "yes", "because"),
        ("attr(bob, cold)", "no", "not cold"),
        ("isa(bob, fish)", "no", "nothing shows"),
        ("furry(bob)", "unknown", "nothing here says"),
    ],
)
def test_a_brief_line_leads_with_the_answer(zoo, query, status, contains):
    line = brief(zoo.ask(query))
    assert line.status == status
    assert line.text.startswith(("Yes", "No", "Unknown"))
    assert contains in line.text


def test_every_clause_of_a_brief_line_comes_from_the_proof(zoo):
    """The compression rule: shorter than the proof, never different from it."""
    answer = zoo.ask("attr(bob, warm_blooded)")
    line = brief(answer)
    derived = {str(node.conclusion) for node in answer.proof.walk()}
    for clause in line.clauses:
        assert str(clause.atom) in derived, f"{clause.text!r} is not in the proof"


def test_a_brief_line_stays_inside_its_word_budget():
    engine = ReasoningEngine(
        "attr(bob, "
        + ").  attr(bob, ".join(f"p{i}" for i in range(20))
        + "). big(X) :- attr(X, p0)."
    )
    line = brief(engine.ask("big(bob)"))
    assert line.words <= BRIEF_WORDS


def test_dropped_clauses_are_counted_rather_than_hidden():
    facts = " ".join(f"likes(bob, thing{i})." for i in range(12))
    engine = ReasoningEngine(
        facts + " social(X) :- likes(X, thing0), likes(X, thing1), likes(X, thing2), "
        "likes(X, thing3), likes(X, thing4), likes(X, thing5), likes(X, thing6)."
    )
    line = brief(engine.ask("social(bob)"))
    assert line.words <= BRIEF_WORDS
    assert line.dropped, "clauses were cut but the line did not say so"
    assert "more" in line.text


def test_the_short_form_gets_three_sentences(zoo):
    line = brief(zoo.ask("attr(bob, warm_blooded)"), length="short")
    assert 1 <= line.text.count(".") <= 3


def test_an_unknown_brief_names_the_missing_literal():
    engine = ReasoningEngine(
        "#open warm/1. isa(bob, cat). isa(X, mammal) :- isa(X, cat). "
        "warm(X) :- isa(X, mammal), furry(X)."
    )
    line = brief(engine.ask("warm(bob)"), Realiser())
    assert line.status == "unknown"
    assert "but" in line.text and "furry" in line.text


def test_an_unknown_line_uses_but_and_a_yes_line_uses_because(zoo):
    assert "because" in brief(zoo.ask("attr(bob, warm_blooded)")).text


def test_an_unknown_brief_reports_what_is_already_known():
    engine = ReasoningEngine(
        "#open warm/1. isa(bob, cat). isa(X, mammal) :- isa(X, cat). "
        "warm(X) :- isa(X, mammal), furry(X)."
    )
    line = brief(engine.ask("warm(bob)"))
    assert any(clause.role == "established" for clause in line.clauses)
    assert any(clause.role == "gap" for clause in line.clauses)


def test_brief_rejects_an_unknown_length(zoo):
    with pytest.raises(ValueError, match="length must be"):
        brief(zoo.ask("isa(bob, cat)"), length="medium")


# -- P2.0: the Mind facade -------------------------------------------------


def test_the_mind_takes_no_domain_or_mode_argument():
    """Design rule 8: no API argument names the host type.

    A host that has to declare what it is has already made the mind's hardest
    decision for it, and later phases depend on that decision never existing.
    """
    import inspect

    parameters = set(inspect.signature(Mind.__init__).parameters)
    forbidden = {"domain", "mode", "pack", "packs", "context", "host", "profile"}
    assert not parameters & forbidden


def test_the_mind_answers_and_explains_in_one_line():
    mind = Mind()
    mind.tell("Bob is a cat. All cats are mammals. If something is a mammal then it is warm blooded.")
    answer = mind.ask("Is Bob warm blooded?")
    assert answer.status == "yes"
    assert mind.brief(answer).startswith("Yes")


def test_the_mind_accepts_a_question_in_logic_syntax():
    mind = Mind()
    mind.tell("Bob is a cat.")
    assert mind.ask("isa(bob, cat)").status == "yes"


def test_a_structured_observation_is_kept_whole_and_not_guessed_at():
    """Reading structure is context discovery's job; storing it starts now."""
    mind = Mind()
    payload = {"sku": "MLK-1L", "stock": 4, "price": "1.29 EUR"}
    observation = mind.observe(payload)
    assert observation.kind == "record"
    assert observation.payload is payload
    assert not observation.understood
    assert mind.observations == [observation]


@pytest.mark.parametrize(
    "payload,kind",
    [
        ("Bob is a cat.", "text"),
        ({"a": 1}, "record"),
        ([{"a": 1}], "records"),
        ([1, 2, 3], "sequence"),
        (42, "value"),
    ],
)
def test_observations_are_classified_by_shape_alone(payload, kind):
    assert Mind().observe(payload).kind == kind


def test_the_self_report_says_what_it_has_worked_out_and_no_more():
    """Before P2.4 this asserted "unresolved" unconditionally, because
    claiming to know where it was would have been a lie. Now there is a
    context layer, so the honest assertion is the weaker one: it reports a
    reading when it has one and says unresolved when it does not.
    """
    empty = Mind()
    assert "unresolved" in empty.self_report()

    mind = Mind()
    mind.observe({"sku": "MLK-1L", "stock": 4, "price": "1.29 EUR"})
    report = mind.self_report()
    assert report.count(".") <= 4
    reading = mind.context.reading
    if reading.unresolved:
        assert "unresolved" in report
    else:
        assert reading.facets[0].name in report or reading.domain in report


def test_the_self_report_is_read_off_state_not_estimated():
    mind = Mind()
    mind.tell("Bob is a cat.")
    mind.add_rules("#open furry/1.")
    report = mind.self_report()
    assert "1 fact(s)" in report
    assert "furry/1" in report


def test_the_mind_reports_an_unknown_rather_than_guessing_no():
    mind = Mind()
    mind.tell("Bob is a cat.")
    mind.add_rules("#open furry/1.")
    answer = mind.ask("furry(bob)")
    assert answer.status == "unknown"
    assert mind.brief(answer).startswith("Unknown")


# -- P2.0: the open-world corpus -------------------------------------------

from neuralmind.datasets import proofwriter_corpus as corpus  # noqa: E402

owa = pytest.mark.skipif(
    not corpus.available("depth-2", world="owa"),
    reason="ProofWriter OWA splits not downloaded",
)


def test_the_word_unknown_is_not_read_as_a_yes():
    """bool("Unknown") is True, and that would score 46% of a split wrong.

    It would also look like a reasoning failure rather than a parsing one,
    which is the kind of bug a benchmark hides instead of catching.
    """
    from neuralmind.datasets.proofwriter_corpus import _questions_of

    record = {
        "questions": {
            "Q1": {"question": "a", "answer": "Unknown", "QDep": 0,
                   "representation": '("bob" "is" "blue" "+")'},
            "Q2": {"question": "b", "answer": True, "QDep": 0,
                   "representation": '("bob" "is" "red" "+")'},
            "Q3": {"question": "c", "answer": False, "QDep": 0,
                   "representation": '("bob" "is" "green" "-")'},
        }
    }
    statuses = [q.status for q in _questions_of(record, "direct", "owa")]
    assert statuses == ["unknown", "yes", "no"]


def test_the_two_readings_mark_negatives_differently():
    """Under CWA a negative is a renamed predicate; under OWA it is a claim.

    ``not_blue`` never interacts with ``blue``; ``-blue`` contradicts it. Using
    the closed-world marker in an open world would make every "no" underivable.
    """
    from neuralmind.datasets.proofwriter_corpus import _mark

    assert _mark(("blue", "bob"), False, "cwa") == ("not_blue", "bob")
    assert _mark(("blue", "bob"), False, "owa") == ("-blue", "bob")
    assert _mark(("blue", "bob"), True, "owa") == ("blue", "bob")


def test_a_bare_open_declaration_opens_everything():
    """OWA means silence about *any* predicate is silence, named or not.

    Most of the Unknown questions ask about predicates the theory never
    mentions, so opening only what appears in it answers them all "no".
    """
    engine = ReasoningEngine("#open. bird(tweety).")
    assert engine.ask("flies(tweety)").status == "unknown"
    assert engine.ask("bird(zoe)").status == "unknown"


def test_an_explicit_closed_declaration_beats_the_bare_open():
    engine = ReasoningEngine("#open. #closed roster/1. roster(anne).")
    assert engine.ask("roster(bob)").status == "no"
    assert engine.ask("anything(bob)").status == "unknown"


def test_the_strong_negation_schema_reads_conditions_as_derivable_not_absent():
    """In an open world, not having proved p is not the same as proving not-p.

    So "if it is not quiet" is a positive literal over -quiet, which something
    has to derive -- not negation-as-failure, which would fire on silence.
    """
    from neuralmind.perception.controlled import ControlledEnglishParser, TripleSchema

    strong = ControlledEnglishParser(TripleSchema("direct", negation="-"))
    rules = {str(r) for r in strong.perceive(
        "If someone is kind and not quiet then they are not blue."
    ).rules}
    assert "-blue(X) :- kind(X), -quiet(X)." in rules

    closed = ControlledEnglishParser(TripleSchema("direct"))
    rules = {str(r) for r in closed.perceive(
        "If someone is kind and not quiet then they are not blue."
    ).rules}
    assert "not_blue(X) :- kind(X), not quiet(X)." in rules


@owa
@pytest.mark.parametrize("split", ["depth-0", "depth-2", "depth-5"])
def test_the_open_world_splits_are_exact_on_the_generated_register(split):
    """P2.0's bar: OWA matches CWA's exactness on the splits it was built for.

    These splits are ~46% Unknown, so a system that answered "no" to whatever
    it could not derive would score around 54% here while scoring 100% on the
    CWA splits. The gap is the whole point of the milestone.
    """
    from neuralmind.evaluation import evaluate
    from neuralmind.perception.controlled import TripleSchema
    from neuralmind.perception.text import TextPerceptor

    problems = corpus.load(split, limit=25, world="owa", schema="direct")
    perceptor = TextPerceptor(TripleSchema("direct", negation="-"), prefer="grammar")
    report = evaluate(problems, perceptor)
    assert report.accuracy == 1.0, report.failure_breakdown()
    assert any(q.status == "unknown" for p in problems for q in p.questions)


@owa
def test_reading_a_negative_as_a_renamed_predicate_is_scored_wrong():
    """The benchmark has to be able to catch this specific mistake.

    Read with the closed-world marker, "The rabbit is not green" becomes
    ``not_green(rabbit)`` -- a predicate that stands apart from ``green`` and
    is happily derived, so the system confidently answers "yes" where the gold
    answer is "no". Getting 100% above means that never happens, which is only
    meaningful if the wrong reading visibly fails.
    """
    from neuralmind.evaluation import evaluate
    from neuralmind.perception.controlled import TripleSchema
    from neuralmind.perception.text import TextPerceptor

    problems = corpus.load("depth-2", limit=10, world="owa", schema="direct")
    closed = TextPerceptor(TripleSchema("direct"), prefer="grammar")
    report = evaluate(problems, closed)
    assert report.accuracy < 0.8
    assert any(o.expected == "no" and o.predicted == "yes" for o in report.failures)


@owa
def test_the_three_answers_are_distinct_cells_in_the_report():
    """If "unknown" collapsed into "no" the split above would pass regardless."""
    from neuralmind.evaluation import evaluate
    from neuralmind.perception.controlled import TripleSchema
    from neuralmind.perception.text import TextPerceptor

    problems = corpus.load("depth-2", limit=10, world="owa", schema="direct")
    report = evaluate(
        problems, TextPerceptor(TripleSchema("direct", negation="-"), prefer="grammar")
    )
    assert {o.predicted for o in report.outcomes} == {"yes", "no", "unknown"}
    assert {o.expected for o in report.outcomes} == {"yes", "no", "unknown"}


@owa
def test_loading_a_world_that_does_not_exist_says_so():
    with pytest.raises(ValueError, match="unknown world"):
        corpus.load("depth-2", world="possible")


@owa
def test_brief_answers_stay_short_and_stay_traceable_at_scale():
    """P2.0's other bar, measured rather than asserted.

    The target is 95% of brief lines within the word budget and every clause
    readable back to a proof node. Run over a slice of the corpus here; the
    full-corpus figures are in the README.
    """
    from neuralmind.knowledge.base import KnowledgeBase
    from neuralmind.perception.controlled import TripleSchema
    from neuralmind.perception.text import TextPerceptor

    perceptor = TextPerceptor(TripleSchema("direct", negation="-"), prefer="grammar")
    lengths, untraced, statuses = [], 0, set()
    for split in ("depth-2", "birds-electricity"):
        for problem in corpus.load(split, limit=6, world="owa", schema="direct"):
            kb = KnowledgeBase("brief")
            perceptor.perceive(problem.theory).into(kb)
            kb.rules.open_world = True
            engine, realiser = kb.engine(), Realiser()
            for question in problem.questions:
                facts = perceptor.perceive(question.text).facts
                if not facts:
                    continue
                answer = engine.ask(facts[0].atom)
                line = brief(answer, realiser)
                lengths.append(line.words)
                statuses.add(line.status)
                for clause in line.clauses:
                    if str(clause.atom) not in _sources(answer):
                        untraced += 1

    assert statuses == {"yes", "no", "unknown"}, "the slice did not exercise all three"
    within = sum(1 for n in lengths if n <= BRIEF_WORDS) / len(lengths)
    assert within >= 0.95, f"only {within:.1%} of brief lines fit the budget"
    assert untraced == 0, f"{untraced} clause(s) were not in any proof"


def _sources(answer) -> set:
    """Every atom the answer's own evidence mentions."""
    found = {str(answer.query)}
    if answer.evidence is not None:
        found |= {str(node.conclusion) for node in answer.evidence.walk()}
    if answer.diagnosis is not None:
        for attempt in answer.diagnosis.attempts:
            found |= {str(atom) for atom in attempt.established}
            if attempt.missing is not None:
                found.add(str(attempt.missing))
    return found


# -- P2.1: the workspace, reached through Mind -----------------------------


def test_the_self_report_says_what_it_cannot_do():
    """A specialist that is not installed is a limit, and limits are reported."""
    mind = Mind()
    report = mind.self_report()
    missing = [name for name, ready in mind.specialists().items() if not ready]
    assert report.count(".") <= 3
    if missing:
        assert "Cannot" in report and missing[0] in report
    else:
        assert "Cannot" not in report


def test_the_mind_solves_through_the_workspace():
    """Mind.solve goes through the specialists; Mind.ask stays deductive."""
    from neuralmind.workspace.specialists import installed

    if not all(installed().values()):
        pytest.skip("not every specialist backend is installed")
    from neuralmind.workspace.scenarios import WORKSHOP_FACTS, WORKSHOP_RULES

    mind = Mind(budget_ms=2000)
    mind.add_rules(WORKSHOP_RULES)
    mind.add_rules(WORKSHOP_FACTS)
    conclusion = mind.solve("safe(beam_a)")
    assert conclusion.status == "yes"
    assert "arithmetic" in conclusion.consulted

    # The plain deductive path cannot reach it: the inequality is not a fact
    # and no rule derives it, so ask() is right to say it does not hold.
    assert not mind.ask("safe(beam_a)").holds


def test_telling_the_mind_something_new_reaches_the_specialists():
    """A blackboard that went stale would answer from a world that moved on."""
    from neuralmind.workspace.specialists import installed

    if not installed()["graph"]:
        pytest.skip("networkx not installed")
    mind = Mind(budget_ms=1000)
    mind.add_rules("edge(a, b).")
    assert mind.solve("reachable(a, c)").status == "unknown"
    mind.add_rules("edge(b, c).")
    assert mind.solve("reachable(a, c)").status == "yes"
