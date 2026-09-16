"""Learning rules from examples."""

import pytest

from neuralmind.core.parser import parse_atom
from neuralmind.induction import (
    Examples, Hypothesis, LanguageBias, RuleLearner, Signature, learn_rules,
)
from neuralmind.induction.enumerate import (
    canonical_key, candidate_bodies, candidate_literals, head_atom, is_connected,
)
from neuralmind.knowledge.base import KnowledgeBase

EDGES = [
    ("maria", "juho"), ("maria", "liisa"), ("juho", "aino"),
    ("liisa", "onni"), ("aino", "elias"),
]
PEOPLE = ["maria", "juho", "liisa", "aino", "onni", "elias"]


@pytest.fixture
def family():
    kb = KnowledgeBase("family")
    kb.add_facts([f"parent({a}, {b})" for a, b in EDGES])
    return kb


def transitive_closure(edges):
    """Ground truth, computed without the inference engine."""
    closure = set(edges)
    changed = True
    while changed:
        changed = False
        for a, b in list(closure):
            for c, d in edges:
                if d == a and (c, b) not in closure:
                    closure.add((c, b))
                    changed = True
                if b == c and (a, d) not in closure:
                    closure.add((a, d))
                    changed = True
    return closure


# -- the bias --------------------------------------------------------------


def test_signature_parsing():
    assert Signature.parse("parent/2") == Signature("parent", 2)
    assert Signature.parse(("parent", 2)) == Signature("parent", 2)
    with pytest.raises(ValueError, match="name/arity"):
        Signature.parse("parent")


def test_bias_rejects_too_few_variables():
    with pytest.raises(ValueError, match="at least the target's arity"):
        LanguageBias.for_target("p/3", ["q/2"], max_variables=2)


def test_recursion_adds_the_target_to_the_body_predicates():
    with_recursion = LanguageBias.for_target("ancestor/2", ["parent/2"])
    without = LanguageBias.for_target("ancestor/2", ["parent/2"], allow_recursion=False)
    assert Signature("ancestor", 2) in with_recursion.usable_predicates
    assert Signature("ancestor", 2) not in without.usable_predicates
    assert without.space_size() < with_recursion.space_size()


def test_space_size_grows_with_the_bias():
    small = LanguageBias.for_target("p/2", ["q/2"], max_variables=3, max_body=2)
    large = LanguageBias.for_target("p/2", ["q/2"], max_variables=4, max_body=3)
    assert large.space_size() > small.space_size()


# -- enumeration -----------------------------------------------------------


def test_the_head_is_never_a_body_literal():
    bias = LanguageBias.for_target("p/2", ["q/2"])
    head = head_atom(bias)
    assert all(literal.atom != head for literal in candidate_literals(bias))


def test_negation_doubles_the_literals_but_never_negates_the_target():
    positive = LanguageBias.for_target("p/2", ["q/2"], allow_recursion=True)
    negated = LanguageBias.for_target(
        "p/2", ["q/2"], allow_recursion=True, allow_negation=True
    )
    literals = candidate_literals(negated)
    assert len(literals) > len(candidate_literals(positive))
    assert not any(l.negated and l.atom.predicate == "p" for l in literals)


def test_connectivity_filter():
    bias = LanguageBias.for_target("p/2", ["q/2"], max_variables=4)
    head = head_atom(bias)
    linked = [l for l in candidate_literals(bias) if str(l.atom) == "q(A, C)"]
    floating = [l for l in candidate_literals(bias) if str(l.atom) == "q(C, D)"]
    assert is_connected(head, linked)
    assert not is_connected(head, floating)


def test_canonical_key_ignores_variable_renaming():
    from neuralmind.core.terms import Atom, Literal, Var

    head = Atom("p", (Var("A"), Var("B")))
    first = [Literal(Atom("q", (Var("A"), Var("C")))), Literal(Atom("q", (Var("C"), Var("B"))))]
    second = [Literal(Atom("q", (Var("A"), Var("D")))), Literal(Atom("q", (Var("D"), Var("B"))))]
    assert canonical_key(head, first) == canonical_key(head, second)


def test_enumeration_yields_each_clause_once():
    bias = LanguageBias.for_target("p/2", ["q/2"], max_variables=3, allow_recursion=False)
    clauses = [str(rule) for rule, _, _ in candidate_bodies(bias, 2)]
    assert len(clauses) == len(set(clauses))


def test_unsafe_clauses_are_never_generated():
    bias = LanguageBias.for_target("p/2", ["q/1"], max_variables=3, allow_recursion=False)
    for rule, _, _ in candidate_bodies(bias, 1):
        rule.plan()  # would raise SafetyError on an unsafe clause


# -- learning --------------------------------------------------------------


def test_learns_grandparent(family):
    positives = [
        f"grandparent({a}, {c})"
        for a, b in EDGES
        for c in [d for e, d in EDGES if e == b]
    ]
    examples = Examples.closed_world(positives, PEOPLE)
    hypothesis = learn_rules(
        family, "grandparent/2", examples.positive, examples.negative, ["parent/2"],
        allow_recursion=False, max_variables=3, max_body=2,
    )
    assert hypothesis.correct
    assert [str(r) for r in hypothesis.rules] == [
        "grandparent(A, B) :- parent(A, C), parent(C, B)."
    ]


def test_learns_a_recursive_definition(family):
    truth = transitive_closure(EDGES)
    examples = Examples.closed_world(
        [f"ancestor({a}, {b})" for a, b in sorted(truth)], PEOPLE
    )
    bias = LanguageBias.for_target(
        "ancestor/2", ["parent/2"], max_variables=3, max_body=2, max_clauses=3
    )
    hypothesis = RuleLearner(family, bias, examples).learn()
    assert hypothesis.correct
    assert len(hypothesis.rules) == 2
    # One base clause and one that calls the target again.
    recursive = [r for r in hypothesis.rules if any(
        l.atom.predicate == "ancestor" for l in r.body
    )]
    assert len(recursive) == 1


def test_learns_an_exception_using_negation():
    kb = KnowledgeBase("birds")
    birds = ["tweety", "robin", "pingu", "skipper", "eagle"]
    kb.add_facts([f"bird({b})" for b in birds] + ["penguin(pingu)", "penguin(skipper)"])
    examples = Examples.closed_world(
        ["flies(tweety)", "flies(robin)", "flies(eagle)"], birds
    )
    bias = LanguageBias.for_target(
        "flies/1", ["bird/1", "penguin/1"],
        max_variables=1, max_body=2, allow_negation=True, allow_recursion=False,
    )
    hypothesis = RuleLearner(kb, bias, examples).learn()
    assert hypothesis.correct
    assert str(hypothesis.rules[0]) == "flies(A) :- bird(A), not penguin(A)."


def test_learns_a_two_clause_definition(family):
    kb = KnowledgeBase("family")
    kb.add_facts([f"parent({a}, {b})" for a, b in EDGES])
    kb.add_facts(["male(juho)", "male(onni)", "male(elias)", "female(maria)",
                  "female(liisa)", "female(aino)"])
    # "child of a parent, of either sex" needs two clauses under this bias.
    positives = [f"offspring({b})" for _, b in EDGES]
    examples = Examples.closed_world(positives, PEOPLE, Signature("offspring", 1))
    bias = LanguageBias.for_target(
        "offspring/1", ["parent/2"], max_variables=2, max_body=1, allow_recursion=False
    )
    hypothesis = RuleLearner(kb, bias, examples).learn()
    assert hypothesis.correct


def test_a_learned_rule_is_an_ordinary_rule(family):
    """Proofs, constraints and the clingo cross-check all still apply."""
    truth = transitive_closure(EDGES)
    examples = Examples.closed_world(
        [f"ancestor({a}, {b})" for a, b in sorted(truth)], PEOPLE
    )
    bias = LanguageBias.for_target("ancestor/2", ["parent/2"], max_variables=3, max_body=2)
    learner = RuleLearner(family, bias, examples)
    hypothesis = learner.learn()
    proof = learner.explain(hypothesis, "ancestor(maria, elias)")
    assert proof.depth >= 2
    assert all(str(a).startswith("parent(") for a in proof.premises())


# -- honesty about the result ---------------------------------------------


def test_a_definition_left_in_the_background_is_stripped(family):
    family.add_rules("grandparent(X, Z) :- parent(X, Y), parent(Y, Z).")
    positives = [
        f"grandparent({a}, {c})" for a, b in EDGES for c in [d for e, d in EDGES if e == b]
    ]
    examples = Examples.closed_world(positives, PEOPLE)
    bias = LanguageBias.for_target(
        "grandparent/2", ["parent/2"], allow_recursion=False, max_variables=3, max_body=2
    )
    hypothesis = RuleLearner(family, bias, examples).learn()
    assert hypothesis.stripped_from_background
    assert not hypothesis.leaked
    assert hypothesis.correct


def test_leakage_is_reported_when_stripping_is_disabled(family):
    family.add_rules("grandparent(X, Z) :- parent(X, Y), parent(Y, Z).")
    positives = [
        f"grandparent({a}, {c})" for a, b in EDGES for c in [d for e, d in EDGES if e == b]
    ]
    examples = Examples.closed_world(positives, PEOPLE)
    bias = LanguageBias.for_target(
        "grandparent/2", ["parent/2"], allow_recursion=False, max_variables=3, max_body=2
    )
    hypothesis = RuleLearner(family, bias, examples, strip_target=False).learn()
    assert hypothesis.leaked
    assert "not evidence" in hypothesis.describe()


def test_an_unlearnable_target_reports_incompleteness(family):
    # Nothing in the bias can express this, so the learner must say so.
    examples = Examples.from_strings(
        ["mystery(maria, elias)"], ["mystery(elias, maria)"]
    )
    bias = LanguageBias.for_target(
        "mystery/2", ["male/1"], allow_recursion=False, max_variables=2, max_body=1
    )
    hypothesis = RuleLearner(family, bias, examples).learn()
    assert not hypothesis.complete
    assert hypothesis.incomplete_reason
    assert "no rule found" in hypothesis.describe()


def test_the_budget_is_respected(family):
    examples = Examples.from_strings(["grandparent(maria, aino)"], [])
    bias = LanguageBias.for_target("grandparent/2", ["parent/2"], max_variables=4, max_body=3)
    hypothesis = RuleLearner(family, bias, examples, budget=5).learn()
    assert hypothesis.candidates_evaluated <= 6  # the in-flight candidate may finish


def test_closed_world_examples_need_something_to_work_from():
    with pytest.raises(ValueError, match="examples or a target"):
        Examples.closed_world([], ["a", "b"])


def test_hypothesis_reports_precision_and_recall():
    hypothesis = Hypothesis(
        covered_positive=frozenset({parse_atom("p(a)")}),
        covered_negative=frozenset({parse_atom("p(b)")}),
        total_positive=2,
        total_negative=5,
    )
    assert hypothesis.recall == 0.5
    assert hypothesis.precision == 0.5
    assert not hypothesis.correct
