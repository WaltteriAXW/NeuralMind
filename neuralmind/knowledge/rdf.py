"""RDF bridge: move facts between the logic engine and a knowledge graph.

Useful when the facts live in (or should end up in) Wikidata-style RDF, and
when an OWL ontology already exists for the domain. rdflib is an optional
dependency; everything here raises a clear error when it is missing rather
than failing at import time.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..core.terms import Atom, Const
from .base import KnowledgeBase

__all__ = ["to_graph", "from_graph", "rdflib_available", "DEFAULT_NAMESPACE"]

DEFAULT_NAMESPACE = "https://neuralmind.local/kb#"


def rdflib_available() -> bool:
    try:
        import rdflib  # noqa: F401
    except ImportError:
        return False
    return True


def _require_rdflib():
    try:
        import rdflib
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "rdflib is not installed. Install it with `pip install rdflib` "
            "(BSD-3) to use the RDF bridge."
        ) from exc
    return rdflib


def to_graph(facts: Iterable[Atom], namespace: str = DEFAULT_NAMESPACE):
    """Serialise ground atoms as RDF triples.

    The mapping is the obvious one, with a documented fallback:

    * ``p(s, o)``   -> ``(ns:s, ns:p, ns:o)``
    * ``p(s)``      -> ``(ns:s, rdf:type, ns:P)``
    * ``p(a, b, c)`` -> a blank node typed ``ns:P`` with ``ns:p_arg1`` ... links,
      because RDF has no n-ary predicates.
    """
    rdflib = _require_rdflib()
    graph = rdflib.Graph()
    ns = rdflib.Namespace(namespace)
    graph.bind("kb", ns)

    for atom in facts:
        if atom.arity == 1:
            graph.add((_node(rdflib, ns, atom.args[0]), rdflib.RDF.type, ns[_camel(atom.predicate)]))
        elif atom.arity == 2:
            graph.add(
                (
                    _node(rdflib, ns, atom.args[0]),
                    ns[atom.predicate],
                    _node(rdflib, ns, atom.args[1]),
                )
            )
        else:
            statement = rdflib.BNode()
            graph.add((statement, rdflib.RDF.type, ns[_camel(atom.predicate)]))
            for index, arg in enumerate(atom.args, start=1):
                graph.add((statement, ns[f"{atom.predicate}_arg{index}"], _node(rdflib, ns, arg)))
    return graph


def from_graph(graph, namespace: str = DEFAULT_NAMESPACE) -> list[Atom]:
    """Read triples back into atoms, inverting :func:`to_graph`."""
    rdflib = _require_rdflib()
    atoms: list[Atom] = []
    for subject, predicate, obj in graph:
        if predicate == rdflib.RDF.type:
            atoms.append(Atom(_snake(_local(obj, namespace)), (_term(rdflib, subject, namespace),)))
            continue
        name = _local(predicate, namespace)
        if "_arg" in name:
            continue  # part of an n-ary reification, handled below
        atoms.append(
            Atom(
                name,
                (_term(rdflib, subject, namespace), _term(rdflib, obj, namespace)),
            )
        )
    return sorted(set(atoms), key=str)


def load_into(kb: KnowledgeBase, graph, provenance: str = "rdf") -> KnowledgeBase:
    """Add every triple in ``graph`` to a knowledge base as facts."""
    for atom in from_graph(graph):
        kb.add_fact(atom, provenance=provenance)
    return kb


def _node(rdflib, ns, term):
    if isinstance(term, Const) and term.is_number:
        return rdflib.Literal(term.value)
    if isinstance(term, Const) and term.quoted:
        return rdflib.Literal(str(term.value))
    return ns[str(term)]


def _term(rdflib, node, namespace: str):
    if isinstance(node, rdflib.Literal):
        value = node.toPython()
        if isinstance(value, int):
            return Const(value)
        return Const(str(value), quoted=True)
    local = _local(node, namespace)
    if local and local[0].islower() and local.replace("_", "").isalnum():
        return Const(local)
    return Const(local, quoted=True)


def _local(node, namespace: str) -> str:
    text = str(node)
    if text.startswith(namespace):
        return text[len(namespace) :]
    for separator in ("#", "/"):
        if separator in text:
            return text.rsplit(separator, 1)[-1]
    return text


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def _snake(name: str) -> str:
    out = []
    for index, char in enumerate(name):
        if char.isupper() and index:
            out.append("_")
        out.append(char.lower())
    return "".join(out)
