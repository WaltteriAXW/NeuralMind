"""Guards on how the fixpoint is computed, not just what it computes.

These tests exist because a silent editing mistake once left the semi-naive
delta being passed into the matcher and then ignored. Everything still gave
the right answer -- only much slower -- so the whole test suite stayed green.
Correctness tests cannot catch that; these can.
"""

import pytest

from neuralmind.core.parser import parse_program
from neuralmind.inference.clingo_backend import clingo_available, cross_check
from neuralmind.inference.forward import ForwardChainer
from neuralmind.inference.model import AtomIndex

TRANSITIVE = """
path(X, Y) :- edge(X, Y).
path(X, Z) :- edge(X, Y), path(Y, Z).
"""


def chain(size: int) -> str:
    return "\n".join(f"edge(n{i}, n{i + 1})." for i in range(size)) + TRANSITIVE


def cycle(size: int) -> str:
    links = "\n".join(f"edge(n{i}, n{(i + 1) % size})." for i in range(size))
    return links + TRANSITIVE


@pytest.fixture
def scan_counter(monkeypatch):
    """Count how many atoms the matcher looks at in total."""
    counter = {"scanned": 0}
    original = AtomIndex.candidates

    def counting(self, pattern):
        found = original(self, pattern)
        counter["scanned"] += len(found)
        return found

    monkeypatch.setattr(AtomIndex, "candidates", counting)
    return counter


def test_work_is_proportional_to_the_result(scan_counter):
    """Semi-naive evaluation never re-derives, so work tracks output size.

    Naive iteration re-derives every known atom on every round, so the atoms
    scanned per atom produced grows with the chain. Semi-naive keeps it flat.
    """
    ratios = []
    for size in (30, 60, 120):
        scan_counter["scanned"] = 0
        model = ForwardChainer(parse_program(chain(size), "c"), max_justifications=1).run()
        ratios.append(scan_counter["scanned"] / len(model))

    # Flat, not growing. Under naive iteration this ratio roughly doubles each
    # time the chain doubles; here it must stay put.
    assert max(ratios) < 2 * min(ratios), f"work per atom grew: {ratios}"
    assert max(ratios) < 10, f"scanning {max(ratios):.1f} atoms per atom derived"


def test_a_recursive_program_is_solved_in_one_round_per_layer():
    """Each round extends every path by one edge, so rounds track depth."""
    model = ForwardChainer(parse_program(chain(25), "c"), max_justifications=1).run()
    # 25 edges give paths of length 1..25; the last round finds nothing new.
    assert model.iterations <= 27


def test_derivations_are_not_repeated_across_rounds(scan_counter):
    """A single long chain is the case naive evaluation handles worst."""
    scan_counter["scanned"] = 0
    model = ForwardChainer(parse_program(chain(100), "c"), max_justifications=1).run()
    assert len(model) == 100 + 100 * 101 // 2
    # Naive evaluation needs well over a million; semi-naive needs a few tens
    # of thousands. The gap is wide enough that the bound is unambiguous.
    assert scan_counter["scanned"] < 100_000


# -- correctness alongside the speed --------------------------------------


def test_transitive_closure_is_exactly_right():
    size = 12
    model = ForwardChainer(parse_program(chain(size), "c")).run()
    expected = {
        f"path(n{i}, n{j})" for i in range(size + 1) for j in range(i + 1, size + 1)
    }
    assert {str(a) for a in model.by_predicate("path")} == expected


def test_a_cycle_reaches_everything():
    size = 8
    model = ForwardChainer(parse_program(cycle(size), "c")).run()
    assert len(model.by_predicate("path")) == size * size


@pytest.mark.skipif(not clingo_available(), reason="clingo is not installed")
def test_recursive_programs_still_agree_with_clingo():
    """The optimisation must not change the model, only the time taken."""
    for source in (chain(20), cycle(7)):
        assert cross_check(parse_program(source, "c")).agree


def test_negation_still_sees_a_completed_lower_stratum():
    """Stratification must survive the delta: 'not' reads the finished relation."""
    source = chain(6) + """
node(X) :- edge(X, _).
node(Y) :- edge(_, Y).
unreachable(X) :- node(X), not path(n0, X).
"""
    model = ForwardChainer(parse_program(source, "c")).run()
    # Everything is reachable from n0 except n0 itself.
    assert {str(a) for a in model.by_predicate("unreachable")} == {"unreachable(n0)"}


def test_alternative_justifications_survive_semi_naive():
    """Two ways to derive the same atom must both still be recorded."""
    from neuralmind.core.parser import parse_atom

    model = ForwardChainer(
        parse_program("p(a). q(a). r(X) :- p(X). r(X) :- q(X).", "t")
    ).run()
    assert len(model.alternatives(parse_atom("r(a)"))) == 2


# -- join ordering ---------------------------------------------------------

CROSS_PRODUCT = "cousin(A, B) :- parent(P, A), parent(Q, B), sibling(P, Q)."


def relations(size: int) -> str:
    lines = []
    for i in range(size):
        lines.append(f"parent(p{i}, c{i}).")
        lines.append(f"sibling(p{i}, p{(i + 1) % size}).")
    return "\n".join(lines)


def test_the_planner_avoids_a_cartesian_product(scan_counter):
    """As written, the first two literals of this rule share no variable.

    Taken in source order that enumerates every pair of parents before the
    sibling check throws almost all of them away. The planner reorders so each
    step is an indexed lookup instead.
    """
    size = 300
    scan_counter["scanned"] = 0
    model = ForwardChainer(
        parse_program(relations(size) + "\n" + CROSS_PRODUCT, "j"), max_justifications=1
    ).run()
    assert len(model.by_predicate("cousin")) == size
    # The cross product alone would be size * size = 90,000 scans.
    assert scan_counter["scanned"] < 10 * size


def test_the_planner_reorders_only_when_it_helps():
    from neuralmind.core.parser import parse_rule

    # A body that already joins cleanly is left exactly as written.
    steps = parse_rule("path(X, Z) :- edge(X, Y), path(Y, Z).").plan()
    assert [str(s.literal.atom) for s in steps] == ["edge(X, Y)", "path(Y, Z)"]

    # One that does not is rearranged so every step shares a variable.
    steps = parse_rule(CROSS_PRODUCT).plan()
    assert [str(s.literal.atom) for s in steps] == [
        "parent(P, A)", "sibling(P, Q)", "parent(Q, B)"
    ]


def test_proofs_are_rendered_in_the_order_the_rule_was_written():
    """Reordering is an execution detail and must not leak into explanations."""
    from neuralmind.inference.engine import ReasoningEngine

    engine = ReasoningEngine(relations(4) + "\n" + CROSS_PRODUCT)
    proof = engine.ask("cousin(c0, c1)").proof
    assert proof is not None
    # Source order is parent, parent, sibling -- not the execution order.
    assert [child.conclusion.predicate for child in proof.children] == [
        "parent", "parent", "sibling"
    ]


def test_reordering_does_not_change_the_model():
    """Whatever order the planner picks, the answer is the same."""
    source = relations(20) + "\n" + CROSS_PRODUCT
    reordered = (
        relations(20) + "\ncousin(A, B) :- parent(P, A), sibling(P, Q), parent(Q, B)."
    )
    first = ForwardChainer(parse_program(source, "a")).run()
    second = ForwardChainer(parse_program(reordered, "b")).run()
    assert {str(a) for a in first.atoms} == {str(a) for a in second.atoms}
