"""The interactive session.

``Shell.handle`` is a pure function from a line to a response, which is why
none of this needs a terminal.
"""

import io

import pytest

from neuralmind.core.parser import parse_atom
from neuralmind.shell import Shell
from neuralmind.perception.text import spacy_available


@pytest.fixture
def shell():
    return Shell()


def converse(shell, *lines):
    """Feed lines in order and return the responses joined."""
    return "\n".join(filter(None, (shell.handle(line) for line in lines)))


# -- reading input ---------------------------------------------------------


def test_english_facts_and_rules_are_shown_as_symbols(shell):
    assert "+ isa(bob, cat)" in shell.handle("Bob is a cat.")
    assert "isa(X, mammal) :- isa(X, cat)." in shell.handle("All cats are mammals.")


def test_logic_is_accepted_directly(shell):
    assert "+ parent(alice, bob)" in shell.handle("parent(alice, bob).")
    assert "ancestor(X, Y) :- parent(X, Y)." in shell.handle(
        "ancestor(X, Y) :- parent(X, Y)."
    )


def test_a_missing_full_stop_is_forgiven(shell):
    assert "+ parent(alice, bob)" in shell.handle("parent(alice, bob)")


def test_blank_lines_and_comments_are_ignored(shell):
    assert shell.handle("") == ""
    assert shell.handle("   ") == ""
    assert shell.handle("% a note") == ""


def test_unreadable_input_is_refused_not_absorbed(shell):
    response = shell.handle("?!?!")
    assert "could not read" in response
    assert not shell.knowledge.facts


def test_an_unanchored_reading_is_marked_as_a_guess(shell):
    """"wibble wobble" has no copula or determiner to locate a verb.

    It is still read -- refusing outright would lose real sentences like
    "Alice likes Bob" -- but it is never absorbed silently.
    """
    response = shell.handle("wibble wobble")
    assert "guess" in response and "0.50" in response
    assert shell.knowledge.confidence_of(parse_atom("act(wibble, wobble)")) == 0.5


def test_an_anchored_reading_is_certain(shell):
    assert "guess" not in shell.handle("The cat chases the mouse.")


@pytest.mark.skipif(not spacy_available(), reason="spaCy model not installed")
def test_free_text_is_read_and_says_which_reader_guessed(shell):
    """The session reads prose the controlled grammar refuses outright.

    The two readers doubt for different reasons, so the note says which one
    is speaking rather than giving the grammar's reason for a parse result.
    """
    response = shell.handle(
        "Charlie is green, but often kind, even when he is blue and cold."
    )
    assert "attr(charlie, kind)" in response
    assert "dependency parse" in response and "0.70" in response


# -- answering -------------------------------------------------------------


def test_a_true_question_answers_yes_with_a_derivation(shell):
    converse(shell, "Bob is a cat.", "All cats are mammals.")
    response = shell.handle("Is Bob a mammal?")
    assert response.startswith("yes")
    assert "isa(bob, cat)" in response and "[given]" in response


def test_a_false_question_answers_no_and_says_why_not(shell):
    converse(shell, "Bob is a cat.")
    response = shell.handle("Is Bob green?")
    assert response.startswith("no")
    assert "false" in response


def test_a_logic_query_solves_for_variables(shell):
    converse(shell, ":load family", "parent(a, b).", "parent(b, c).")
    response = shell.handle("ancestor(a, X)?")
    assert "X = b" in response and "X = c" in response


def test_proofs_can_be_switched_off(shell):
    converse(shell, "Bob is a cat.", ":proof off")
    response = shell.handle("Is Bob a cat?")
    assert response.strip() == "yes"


def test_prose_can_be_switched_on(shell):
    converse(shell, "Bob is a cat.", "All cats are mammals.", ":prose on", ":proof off")
    assert "Bob is a cat" in shell.handle("Is Bob a mammal?")


def test_an_unreadable_question_is_refused(shell):
    assert "could not read that question" in shell.handle("?")


# -- contradictions --------------------------------------------------------


def test_a_contradiction_is_reported_the_moment_it_appears(shell):
    converse(shell, ":load triples", "Bob is blue.")
    response = shell.handle("Bob is not blue.")
    assert "Contradictory attribute" in response


def test_the_same_violation_is_not_repeated_on_every_line(shell):
    converse(shell, ":load triples", "Bob is blue.", "Bob is not blue.")
    assert "Contradictory" not in shell.handle("Alice is red.")
    # ...but :check still lists it on demand.
    assert "Contradictory" in shell.handle(":check")


# -- commands --------------------------------------------------------------


def test_facts_and_rules_listings(shell):
    converse(shell, "Bob is a cat.", "All cats are mammals.")
    assert "isa(bob, cat)" in shell.handle(":facts")
    assert "isa(X, mammal)" in shell.handle(":rules")
    assert "isa(bob, cat)" not in shell.handle(":facts nothing")


def test_why_explains_a_derived_atom(shell):
    converse(shell, "Bob is a cat.", "All cats are mammals.")
    assert "isa(bob, cat)" in shell.handle(":why isa(bob, mammal)")


def test_why_on_something_false_says_so(shell):
    assert "not in the model" in shell.handle(":why isa(bob, mammal)")


def test_retract_removes_a_fact(shell):
    shell.handle("Bob is a cat.")
    assert "- isa(bob, cat)" in shell.handle(":retract isa(bob, cat)")
    assert "(no facts)" in shell.handle(":facts")


def test_retracting_something_absent_is_reported(shell):
    assert "was not asserted" in shell.handle(":retract isa(bob, cat)")


def test_clear_forgets_everything(shell):
    converse(shell, "Bob is a cat.", ":clear")
    assert "(no facts)" in shell.handle(":facts")


def test_load_and_sets(shell):
    assert "family" in shell.handle(":sets")
    assert "rule(s)" in shell.handle(":load family")
    assert "?" in shell.handle(":load no_such_ruleset")


def test_model_lists_derivable_atoms(shell):
    converse(shell, "Bob is a cat.", "All cats are mammals.")
    assert "isa(bob, mammal)" in shell.handle(":model")


def test_save_writes_asp(shell, tmp_path):
    converse(shell, "Bob is a cat.", "All cats are mammals.")
    target = tmp_path / "session.lp"
    assert "wrote" in shell.handle(f":save {target}")
    written = target.read_text()
    assert "isa(bob, cat)." in written and ":-" in written


def test_schema_switching(shell):
    assert "triple" in shell.handle(":schema")
    assert "now 'direct'" in shell.handle(":schema direct")
    assert "+ blue(bob)" in shell.handle("Bob is blue.")
    assert "?" in shell.handle(":schema nonsense")


def test_learn_induces_a_rule_from_what_is_known(shell):
    converse(
        shell,
        "parent(a, b).", "parent(b, c).", "parent(a, d).", "parent(d, e).",
        "grandparent(a, c).", "grandparent(a, e).",
    )
    response = shell.handle(":learn grandparent/2 from parent/2")
    assert "grandparent(A, B) :- parent(A, C), parent(C, B)." in response


def test_learn_needs_examples(shell):
    shell.handle("parent(a, b).")
    assert "nothing to generalise" in shell.handle(":learn grandparent/2 from parent/2")


def test_learn_rejects_a_malformed_request(shell):
    assert "name/arity" in shell.handle(":learn grandparent")


def test_learn_works_out_the_vocabulary_for_itself(shell):
    """No 'from' clause: the session's own predicates are the search space."""
    converse(
        shell,
        "parent(m, j).", "parent(m, l).", "parent(j, a).",
        "sibling(j, l).", "sibling(l, j).",
    )
    response = shell.handle(":learn sibling/2")
    assert "sibling(A, B) :- parent(C, A), parent(C, B), A != B." in response


def test_unknown_commands_are_reported(shell):
    assert "unknown command" in shell.handle(":nonsense")


def test_help_mentions_the_commands(shell):
    text = shell.handle(":help")
    assert ":why" in text and ":check" in text


def test_quit_stops_the_loop(shell):
    assert shell.handle(":quit") == "bye"
    assert not shell.running


# -- the loop --------------------------------------------------------------


def test_run_reads_a_script_and_echoes_it():
    script = io.StringIO(
        "Bob is a cat.\nAll cats are mammals.\nIs Bob a mammal?\n:quit\n"
    )
    out = io.StringIO()
    assert Shell().run(stream=script, out=out, banner=False) == 0
    text = out.getvalue()
    assert "> Bob is a cat." in text  # non-interactive input is echoed
    assert "yes" in text and "bye" in text


def test_run_stops_at_end_of_input():
    out = io.StringIO()
    Shell().run(stream=io.StringIO("Bob is a cat.\n"), out=out, banner=False)
    assert "+ isa(bob, cat)" in out.getvalue()


# -- surviving bad input ---------------------------------------------------


def test_an_unsafe_rule_is_rejected_not_accepted(shell):
    """One bad rule must not wedge the session.

    Accepting it would break every later command, leaving :clear -- which
    throws away the work -- as the only way out.
    """
    shell.handle("parent(a, b).")
    response = shell.handle("p(X, Y) :- q(X).")
    assert "rejected" in response and "head variable" in response
    # ...and everything still works.
    assert "+ parent(b, c)" in shell.handle("parent(b, c).")
    assert shell.handle("parent(a, X)?").startswith("yes")


def test_a_rule_creating_recursive_negation_is_rejected(shell):
    converse(shell, "d(1).", "a(X) :- d(X), not b(X).")
    assert "rejected" in shell.handle("b(X) :- d(X), not a(X).")
    assert shell.handle("a(1)?").startswith("yes")


def test_a_rejected_rule_leaves_no_trace(shell):
    shell.handle("p(X, Y) :- q(X).")
    assert "(no rules)" in shell.handle(":rules")
    assert "(no facts)" in shell.handle(":facts")


# -- teaching by correction ------------------------------------------------


@pytest.fixture
def birds(shell):
    converse(
        shell,
        "bird(tweety).", "bird(pingu).", "bird(eagle).", "penguin(pingu).",
        "flies(X) :- bird(X).",
    )
    return shell


def test_wrong_proposes_ranked_repairs(birds):
    response = birds.handle(":wrong flies(pingu)")
    assert "should not hold" in response
    assert "not penguin(X)" in response
    assert "[1]" in response and "accept" in response


def test_accept_applies_the_chosen_repair(birds):
    birds.handle(":wrong flies(pingu)")
    assert "applied" in birds.handle(":accept 1")
    assert birds.handle("flies(pingu)?").startswith("no")
    assert birds.handle("flies(eagle)?").startswith("yes")


def test_undo_reverses_an_accepted_repair(birds):
    birds.handle(":wrong flies(pingu)")
    birds.handle(":accept 1")
    assert "undid" in birds.handle(":undo")
    assert birds.handle("flies(pingu)?").startswith("yes")


def test_accept_needs_a_proposal_first(shell):
    assert "nothing proposed" in shell.handle(":accept 1")


def test_accept_validates_its_argument(birds):
    birds.handle(":wrong flies(pingu)")
    assert "needs a number" in birds.handle(":accept x")
    assert "choose between" in birds.handle(":accept 99")


def test_expect_proposes_a_rule(shell):
    converse(shell, "parent(m, j).", "parent(j, a).", "parent(m, l).", "parent(l, o).")
    response = shell.handle(":expect grandparent(m, a)")
    assert "grandparent(A, B) :- parent(A, C), parent(C, B)." in response


def test_complaining_about_something_already_right_says_so(birds):
    assert "nothing to fix" in birds.handle(":wrong flies(nobody)")
    assert "nothing to fix" in birds.handle(":expect flies(tweety)")


def test_correction_needs_an_atom(shell):
    assert "give an atom" in shell.handle(":wrong")


# -- persistence and history -----------------------------------------------


def test_undo_reverses_an_assertion(shell):
    shell.handle("Bob is a cat.")
    assert "undid" in shell.handle(":undo")
    assert "(no facts)" in shell.handle(":facts")


def test_undo_on_an_empty_session(shell):
    assert "nothing to undo" in shell.handle(":undo")


def test_history_records_what_was_asserted(shell):
    converse(shell, "Bob is a cat.", "parent(a, b).")
    history = shell.handle(":history")
    assert "Bob is a cat." in history and "parent(a, b)." in history


def test_save_then_open_round_trips(shell, tmp_path):
    converse(shell, "Bob is a cat.", "All cats are mammals.")
    target = tmp_path / "kb.lp"
    shell.handle(f":save {target}")
    shell.handle(":clear")
    assert "(no facts)" in shell.handle(":facts")
    assert "opened" in shell.handle(f":open {target}")
    assert shell.handle("Is Bob a mammal?").startswith("yes")


def test_open_reports_a_missing_file(shell):
    assert "no such file" in shell.handle(":open /nowhere/kb.lp")


def test_open_refuses_a_file_that_would_not_compile(shell, tmp_path):
    broken = tmp_path / "broken.lp"
    broken.write_text("p(X, Y) :- q(X).\n")
    converse(shell, "Bob is a cat.")
    assert "would not load" in shell.handle(f":open {broken}")
    # ...and the session is untouched.
    assert "isa(bob, cat)" in shell.handle(":facts")


# -- Phase Two: growth in the session --------------------------------------


def test_a_question_it_cannot_answer_becomes_a_gap(shell):
    """The diagnosis is produced anyway; recording it is free."""
    shell.handle("parent(a, b).")
    shell.handle("grandparent(a, c)?")
    shell.handle("grandparent(a, c)?")
    listed = shell.handle(":gaps")
    assert "grandparent(a, c)" in listed and "2" in listed


def test_gaps_become_questions_a_person_can_answer(shell):
    shell.handle("parent(a, b).")
    shell.handle("brother(a, d).")
    shell.handle("uncle(X, Y) :- brother(X, P), parent(P, Y).")
    shell.handle("uncle(a, c)?")
    assert "Is parent(d, c) true?" in shell.handle(":questions")


def test_gaps_are_empty_before_anything_is_asked(shell):
    assert "no gaps yet" in shell.handle(":gaps")


def test_the_shell_reads_strong_negation_as_logic(shell):
    """"-flies(pingu)." is a claim, not a sentence to be parsed as English.

    Without the leading "-" in the logic pattern it went to the text reader
    and came back as two nonsense facts about the predicate name.
    """
    assert "-grandparent(b, c)" in shell.handle("-grandparent(b, c).")
    assert "act(" not in shell.handle("-flies(pingu).")


def test_grow_asks_rather_than_assuming(shell):
    for fact in ("parent(a, b).", "parent(b, c).", "parent(a, d).", "parent(d, e)."):
        shell.handle(fact)
    shell.handle("grandparent(a, c).")
    response = shell.handle(":grow grandparent/2")
    assert "I need to know" in response or "proposed" in response


def test_grow_needs_something_to_generalise_from(shell):
    shell.handle("parent(a, b).")
    assert "nothing found" in shell.handle(":grow mystery/2")


def test_beliefs_starts_empty(shell):
    assert "nothing remembered" in shell.handle(":beliefs")
