"""The command-line interface."""

import json

import pytest

from neuralmind.cli import main

RULES = "ancestor(X, Y) :- parent(X, Y). ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z)."


def run(capsys, *argv):
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_ask_prints_a_proof(capsys):
    code, out, _ = run(
        capsys, "ask", "-r", RULES, "-f", "parent(a, b)", "-f", "parent(b, c)", "ancestor(a, c)"
    )
    assert code == 0
    assert "ancestor(a, c)" in out and "[given]" in out


def test_ask_returns_nonzero_when_the_answer_is_no(capsys):
    code, out, _ = run(capsys, "ask", "-r", RULES, "-f", "parent(a, b)", "ancestor(c, a)")
    assert code == 1 and "false" in out


def test_ask_emits_json(capsys):
    code, out, _ = run(
        capsys, "ask", "--json", "-r", RULES, "-f", "parent(a, b)", "ancestor(a, b)"
    )
    assert json.loads(out)["answer"]["holds"] is True


def test_ask_accepts_an_english_question_over_perceived_text(capsys):
    code, out, _ = run(
        capsys,
        "ask",
        "--text",
        "Bob is a cat. All cats are mammals.",
        "Is Bob a mammal?",
    )
    assert code == 0 and "isa(bob, mammal)" in out


def test_read_lists_extracted_facts_and_rules(capsys):
    code, out, _ = run(capsys, "read", "Bob is a cat. All cats are mammals.")
    assert code == 0
    assert "isa(bob, cat)" in out and "isa(X, mammal) :- isa(X, cat)." in out


def test_check_reports_violations_and_exits_nonzero(capsys):
    code, out, _ = run(
        capsys, "check", "-r", "triples", "-f", "attr(bob, blue)", "-f", "not_attr(bob, blue)"
    )
    assert code == 1 and "Contradictory attribute" in out


def test_check_is_clean_when_nothing_is_broken(capsys):
    code, out, _ = run(capsys, "check", "-r", "triples", "-f", "attr(bob, blue)")
    assert code == 0 and "no violations" in out


def test_solve_prints_the_model(capsys):
    code, out, _ = run(capsys, "solve", "-r", RULES, "-f", "parent(a, b)", "-p", "ancestor")
    assert code == 0 and "ancestor(a, b)" in out and "parent" not in out


def test_verify_cross_checks_against_clingo(capsys):
    pytest.importorskip("clingo")
    code, out, _ = run(capsys, "verify", "-r", RULES, "-f", "parent(a, b)")
    assert code == 0 and "cross-check passed" in out


def test_eval_reports_accuracy(capsys):
    code, out, _ = run(capsys, "eval", "-n", "5", "--json")
    assert code == 0 and json.loads(out)["accuracy"] >= 0.0


def test_demo_family_runs(capsys):
    code, out, _ = run(capsys, "demo", "family")
    assert code == 0 and "Kinship reasoning" in out


def test_unknown_demo_is_reported(capsys):
    code, _, err = run(capsys, "demo", "nope")
    assert code == 2 and "unknown demo" in err


def test_bad_rule_file_gives_a_clean_error(capsys):
    code, _, err = run(capsys, "ask", "-r", "p(X) :- q(Y).", "p(a)")
    assert code == 2 and "error:" in err
