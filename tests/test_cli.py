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


FAMILY_FACTS = [
    "parent(maria,juho)", "parent(maria,liisa)", "parent(juho,aino)",
    "parent(liisa,onni)", "parent(aino,elias)",
]


def _fact_args():
    return [arg for fact in FAMILY_FACTS for arg in ("-f", fact)]


def test_induce_learns_a_rule(capsys):
    code, out, _ = run(
        capsys, "induce", "--target", "grandparent/2", "--body", "parent/2",
        "--no-recursion", "--max-vars", "3", "--max-body", "2", "--closed-world",
        *_fact_args(),
        "--positive", "grandparent(maria,aino)",
        "--positive", "grandparent(maria,onni)",
        "--positive", "grandparent(juho,elias)",
    )
    assert code == 0
    assert "grandparent(A, B) :- parent(A, C), parent(C, B)." in out
    assert "complete and consistent" in out


def test_induce_emits_json(capsys):
    code, out, _ = run(
        capsys, "induce", "--json", "--target", "grandparent/2", "--body", "parent/2",
        "--no-recursion", "--max-vars", "3", "--max-body", "2", "--closed-world",
        *_fact_args(),
        "--positive", "grandparent(maria,aino)",
        "--positive", "grandparent(maria,onni)",
        "--positive", "grandparent(juho,elias)",
    )
    document = json.loads(out)
    assert document["rules"] and document["consistent"] is True


def test_induce_requires_an_example(capsys):
    code, _, err = run(capsys, "induce", "--target", "p/1", "--body", "q/1")
    assert code == 2 and "at least one --positive" in err


def test_incomplete_positives_under_closed_world_get_a_hint(capsys):
    """The classic trap: one positive listed, the rest silently become negatives."""
    code, out, _ = run(
        capsys, "induce", "--target", "grandparent/2", "--body", "parent/2",
        "--no-recursion", "--max-vars", "3", "--max-body", "2", "--closed-world",
        *_fact_args(), "--positive", "grandparent(maria,aino)",
    )
    assert code == 1
    assert "no rule found" in out and "--closed-world made every unlisted atom" in out


def test_induce_reports_failure_with_a_nonzero_exit(capsys):
    code, out, _ = run(
        capsys, "induce", "--target", "mystery/2", "--body", "parent/2",
        "--no-recursion", "--max-vars", "2", "--max-body", "1",
        *_fact_args(),
        "--positive", "mystery(maria,elias)", "--negative", "mystery(elias,maria)",
    )
    assert code == 1 and "no rule found" in out


def test_induce_reads_examples_from_a_file(capsys, tmp_path):
    path = tmp_path / "positives.lp"
    path.write_text(
        "% grandparents\n"
        "grandparent(maria,aino).\n"
        "grandparent(maria,onni).\n"
        "grandparent(juho,elias).\n\n"
    )
    code, out, _ = run(
        capsys, "induce", "--target", "grandparent/2", "--body", "parent/2",
        "--no-recursion", "--max-vars", "3", "--max-body", "2", "--closed-world",
        *_fact_args(), "--positives-file", str(path),
    )
    assert code == 0 and "parent(A, C), parent(C, B)" in out


def test_demo_induce_runs(capsys):
    code, out, _ = run(capsys, "demo", "induce")
    assert code == 0 and "Learning rules from examples" in out


def test_eval_rejects_an_unknown_corpus_split(capsys):
    code, _, err = run(capsys, "eval", "--corpus", "depth-9")
    assert code == 2 and "unknown split" in err


def test_eval_on_the_real_corpus(capsys):
    from neuralmind.datasets import proofwriter_corpus

    if not proofwriter_corpus.available("depth-2"):
        pytest.skip("ProofWriter corpus not downloaded")
    code, out, _ = run(capsys, "eval", "--corpus", "depth-2", "-n", "10", "--json")
    document = json.loads(out)
    assert document["accuracy"] == 1.0
    assert document["failure_breakdown"] == {}


# -- Phase Two: reasoning with specialists ---------------------------------


def _all_specialists() -> bool:
    from neuralmind.workspace.specialists import installed

    return all(installed().values())


@pytest.mark.skipif(not _all_specialists(), reason="not every specialist is installed")
def test_reason_prints_one_proof_across_specialists(capsys):
    code, out, _ = run(capsys, "reason", "--scenario", "workshop", "signed_off")
    assert code == 0
    assert "by arithmetic" in out and "by units" in out and "by graph" in out


@pytest.mark.skipif(not _all_specialists(), reason="not every specialist is installed")
def test_reason_says_why_when_it_cannot_answer(capsys):
    code, out, _ = run(capsys, "reason", "--scenario", "workshop", "safe(beam_b)")
    assert code == 1
    assert "unknown" in out and "leq(beam_b_load, beam_b_rating)" in out


@pytest.mark.skipif(not _all_specialists(), reason="not every specialist is installed")
def test_reason_reports_who_ran(capsys):
    code, out, _ = run(capsys, "reason", "--scenario", "workshop", "-v", "fits")
    assert code == 0
    assert "consulted:" in out and "round(s)" in out


def test_reason_works_on_plain_rules_without_a_scenario(capsys):
    code, out, _ = run(capsys, "reason", "-f", "cat(bob)", "cat(bob)")
    assert code == 0 and "cat(bob)" in out


@pytest.mark.skipif(not _all_specialists(), reason="not every specialist is installed")
def test_reason_emits_json(capsys):
    code, out, _ = run(capsys, "reason", "--json", "--scenario", "workshop", "fits")
    payload = json.loads(out)
    assert code == 0
    assert payload["status"] == "yes" and "proof" in payload
