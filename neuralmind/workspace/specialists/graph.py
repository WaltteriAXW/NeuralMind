"""The graph specialist: reachability, distance, cycles and ordering.

Datalog computes transitive closure well -- ``reachable(A, B) :- edge(A, B).``
plus one recursive clause, and the semi-naive chainer handles it efficiently.
What it computes badly is anything that has to *compare* paths. Shortest path
needs a minimum over derivations, which a least model has no way to express; a
Datalog program can tell you a path of length 7 exists and cannot tell you it
is the shortest one.

So this specialist takes the questions that are about the shape of the graph
rather than about membership in its closure:

=============================  =========================================
``reachable(a, e)``            is there a path at all?
``distance(a, e, N)``          how long is the shortest one?
``path(a, e, Route)``          what is that path? (as a printable route)
``cycle(a)``                   is this node on a cycle?
``before(a, e)``               does a dependency order put a first?
=============================  =========================================

Edges come off the blackboard as ``edge(a, b)`` or ``edge(a, b, weight)``, so
they can be given, derived by a rule, or posted by another specialist. The
proof cites the route, which is the only honest explanation a path question
has: "a -> c -> e" is the reason, and any summary of it is less.

Optional. Without ``networkx`` installed the goals it would take come back
``unknown`` with the reason.
"""

from __future__ import annotations

from typing import Optional

from ...core.terms import Atom, Const
from ..specialist import Budget, Finding, Result, specialist_proof

__all__ = ["GraphSpecialist", "networkx_available"]

_MISSING = "the graph specialist needs networkx: `pip install neuralmind[graph]`"

#: Goals this specialist answers, and whether the answer is a yes/no or a value.
GOALS = {
    "reachable": 2,
    "distance": 3,
    "path": 3,
    "cycle": 1,
    "before": 2,
}


def networkx_available() -> bool:
    try:
        import networkx  # noqa: F401
    except ImportError:
        return False
    return True


class GraphSpecialist:
    """Path and ordering questions over ``edge/2`` and ``edge/3`` facts."""

    name = "graph"

    def __init__(self, edge_predicate: str = "edge") -> None:
        self.edge_predicate = edge_predicate

    def prepare(self) -> None:
        """Import networkx and touch the algorithms the questions use."""
        if not networkx_available():
            return
        import networkx as nx

        warm = nx.DiGraph()
        warm.add_edge("a", "b")
        nx.shortest_path(warm, "a", "b")
        nx.is_directed_acyclic_graph(warm)

    def accepts(self, goal: Atom, workspace) -> float:
        if GOALS.get(goal.predicate) == goal.arity:
            return 0.95
        return 0.0

    def run(self, goal: Atom, workspace, budget: Budget) -> Result:
        if not networkx_available():
            return Result.nothing(_MISSING)
        import networkx as nx

        graph, sources = _build(workspace, self.edge_predicate, nx)
        if graph.number_of_edges() == 0:
            return Result.nothing(
                f"no {self.edge_predicate}/2 or /3 facts on the blackboard"
            )
        handler = {
            "reachable": self._reachable,
            "distance": self._distance,
            "path": self._route,
            "cycle": self._cycle,
            "before": self._before,
        }.get(goal.predicate)
        if handler is None:
            return Result.nothing(f"{goal.predicate} is not a graph question")
        return handler(goal, graph, sources, workspace, nx)

    # -- the questions ------------------------------------------------------

    def _reachable(self, goal, graph, sources, workspace, nx) -> Result:
        start, end = (_symbol(a) for a in goal.args)
        missing = _absent(graph, start, end)
        if missing:
            return Result.nothing(missing)
        try:
            route = nx.shortest_path(graph, start, end)
        except nx.NetworkXNoPath:
            return Result.nothing(f"no path from {start} to {end}")
        return self._found(goal, route, sources, workspace, _arrow(route))

    def _distance(self, goal, graph, sources, workspace, nx) -> Result:
        start, end = (_symbol(a) for a in goal.args[:2])
        missing = _absent(graph, start, end)
        if missing:
            return Result.nothing(missing)
        weighted = any(data.get("weight") is not None for *_, data in graph.edges(data=True))
        try:
            if weighted:
                length = nx.shortest_path_length(graph, start, end, weight="weight")
                route = nx.shortest_path(graph, start, end, weight="weight")
            else:
                route = nx.shortest_path(graph, start, end)
                length = len(route) - 1
        except nx.NetworkXNoPath:
            return Result.nothing(f"no path from {start} to {end}")
        atom = Atom("distance", goal.args[:2] + (_as_const(length),))
        return self._found(
            atom, route, sources, workspace,
            f"{_arrow(route)} ({'cost' if weighted else 'length'} {_as_const(length)})",
        )

    def _route(self, goal, graph, sources, workspace, nx) -> Result:
        start, end = (_symbol(a) for a in goal.args[:2])
        missing = _absent(graph, start, end)
        if missing:
            return Result.nothing(missing)
        try:
            route = nx.shortest_path(graph, start, end)
        except nx.NetworkXNoPath:
            return Result.nothing(f"no path from {start} to {end}")
        atom = Atom("path", goal.args[:2] + (Const(_arrow(route), quoted=True),))
        return self._found(atom, route, sources, workspace, _arrow(route))

    def _cycle(self, goal, graph, sources, workspace, nx) -> Result:
        node = _symbol(goal.args[0])
        if node not in graph:
            return Result.nothing(f"{node} is not in the graph")
        for cycle in nx.simple_cycles(graph):
            if node in cycle:
                route = cycle + [cycle[0]]
                return self._found(goal, route, sources, workspace, _arrow(route))
        return Result.nothing(f"{node} is not on any cycle")

    def _before(self, goal, graph, sources, workspace, nx) -> Result:
        """Dependency order: ``a`` must come before ``b`` and not the reverse."""
        start, end = (_symbol(a) for a in goal.args)
        missing = _absent(graph, start, end)
        if missing:
            return Result.nothing(missing)
        if not nx.is_directed_acyclic_graph(graph):
            return Result.nothing("the graph has a cycle, so it has no ordering")
        try:
            route = nx.shortest_path(graph, start, end)
        except nx.NetworkXNoPath:
            return Result.nothing(f"nothing requires {start} before {end}")
        return self._found(goal, route, sources, workspace, _arrow(route))

    # -- shared -------------------------------------------------------------

    def _found(self, atom, route, sources, workspace, step) -> Result:
        because = []
        for left, right in zip(route, route[1:]):
            edge = sources.get((left, right))
            proof = workspace.proof(edge) if edge is not None else None
            if proof is not None:
                because.append(proof)
        return Result(
            findings=[
                Finding(atom, specialist_proof(atom, self.name, step, because))
            ]
        )


# -- reading the blackboard ------------------------------------------------


def _build(workspace, predicate: str, nx):
    """A directed graph, plus the atom behind each edge for the proof."""
    graph = nx.DiGraph()
    sources: dict[tuple[str, str], Atom] = {}
    for atom in workspace:
        if atom.predicate != predicate or atom.arity not in (2, 3):
            continue
        left, right = (_symbol(a) for a in atom.args[:2])
        weight = None
        if atom.arity == 3 and isinstance(atom.args[2], Const) and atom.args[2].is_numeric:
            weight = float(atom.args[2].value)
        graph.add_edge(left, right, weight=weight)
        sources[(left, right)] = atom
    return graph, sources


def _absent(graph, *nodes) -> Optional[str]:
    for node in nodes:
        if node not in graph:
            return f"{node} is not in the graph"
    return None


def _symbol(term) -> str:
    return str(term.value) if isinstance(term, Const) else str(term)


def _arrow(route) -> str:
    return " -> ".join(route)


def _as_const(value) -> Const:
    number = float(value)
    return Const(int(number)) if number.is_integer() else Const(number)
