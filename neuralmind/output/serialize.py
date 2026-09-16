"""JSON output.

For machine-to-machine use this is the primary output format, and the
blueprint is right that it is often the clearest one for people too: a proof
tree is already a tree, and rendering it as JSON loses nothing. Prose is
optional and lives in :mod:`neuralmind.output.nlg`.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Optional

from ..core.terms import Atom, Const
from ..inference.model import Model, Violation
from ..inference.proof import ProofNode

__all__ = ["to_json", "answer_document", "model_document", "violations_document", "JsonEncoder"]


class JsonEncoder(json.JSONEncoder):
    """Encoder that understands the project's data types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, ProofNode):
            return obj.to_dict()
        if isinstance(obj, Atom):
            return str(obj)
        if isinstance(obj, Const):
            return obj.value
        if isinstance(obj, Violation):
            return {"rule": str(obj.rule), "support": [str(a) for a in obj.support]}
        if isinstance(obj, Model):
            return model_document(obj)
        if isinstance(obj, frozenset):
            return sorted(str(x) for x in obj)
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return super().default(obj)


def to_json(obj: Any, indent: Optional[int] = 2) -> str:
    """Serialise any of the project's result objects."""
    return json.dumps(obj, cls=JsonEncoder, indent=indent, sort_keys=False)


def answer_document(answer, explanation: Optional[str] = None) -> dict:
    """A complete answer: the result, the proof, and optionally the prose."""
    document = answer.to_dict()
    if explanation:
        document["explanation"] = explanation
    return document


def model_document(model: Model, predicates: Optional[Iterable[str]] = None) -> dict:
    """The whole model as JSON, grouped by predicate."""
    wanted = set(predicates) if predicates else None
    grouped: dict[str, list[str]] = {}
    for atom in model.atoms:
        if wanted and atom.predicate not in wanted:
            continue
        grouped.setdefault(atom.predicate, []).append(str(atom))
    return {
        "summary": model.summary(),
        "atoms": {name: sorted(atoms) for name, atoms in sorted(grouped.items())},
        "violations": violations_document(model.violations),
    }


def violations_document(violations: list[Violation]) -> list[dict]:
    """Constraint violations, each with the facts that caused it."""
    return [
        {
            "rule": str(violation.rule),
            "source": violation.rule.origin(),
            "label": violation.rule.label,
            "description": violation.describe(),
            "support": [str(atom) for atom in violation.support],
            "bindings": {
                name: (value.value if isinstance(value, Const) else str(value))
                for name, value in violation.bindings
            },
        }
        for violation in violations
    ]
