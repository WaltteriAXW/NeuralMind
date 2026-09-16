"""The interactive session.

``Shell.handle`` is a pure function from a line to a response, which is why
none of this needs a terminal.
"""

import io

import pytest

from neuralmind.core.parser import parse_atom
from neuralmind.shell import Shell


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
    assert "usage:" in shell.handle(":learn grandparent")


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
