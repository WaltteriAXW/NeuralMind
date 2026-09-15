"""Human-readable rendering of proof trees and models.

Plain text on purpose: a proof tree printed in a terminal is the fastest way to
see whether the reasoning is right, and it is the format a developer debugging
a knowledge base actually reads.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..inference.model import Model, Violation
from ..inference.proof import CLOSED_WORLD, FACT, ProofNode

__all__ = ["render_proof", "render_model", "render_violations"]

_MARKERS = {
    FACT: "[given]",
    CLOSED_WORLD: "[not derivable]",
}


def render_proof(node: ProofNode, indent: str = "", is_last: bool = True, top: bool = True) -> str:
    """Render a proof tree as an indented ASCII tree.

    The root line is the conclusion; each level below it is the reason for the
    line above. Leaves marked ``[given]`` are inputs, everything else was
    derived by the rule shown in parentheses.
    """
    lines: list[str] = []
    if top:
        lines.append(_label(node))
        child_indent = ""
    else:
        connector = "└── " if is_last else "├── "
        lines.append(f"{indent}{connector}{_label(node)}")
        child_indent = indent + ("    " if is_last else "│   ")
    for index, child in enumerate(node.children):
        lines.append(
            render_proof(
                child,
                indent=child_indent,
                is_last=index == len(node.children) - 1,
                top=False,
            )
        )
    return "\n".join(lines)


def _label(node: ProofNode) -> str:
    text = ("not " if node.negated else "") + str(node.conclusion)
    marker = _MARKERS.get(node.kind)
    if marker:
        text = f"{text}  {marker}"
    if node.confidence is not None:
        text = f"{text}  (confidence {node.confidence:.2f})"
    if node.kind not in (FACT, CLOSED_WORLD):
        source = node.rule_label or node.rule_source
        if source:
            text = f"{text}  (by {source})"
    return text


def render_model(model: Model, predicates: Optional[Iterable[str]] = None, limit: int = 200) -> str:
    """List the atoms of a model, grouped by predicate."""
    wanted = set(predicates) if predicates else None
    grouped: dict[str, list[str]] = {}
    for atom in model.atoms:
        if wanted and atom.predicate not in wanted:
            continue
        grouped.setdefault(atom.predicate, []).append(str(atom))
    if not grouped:
        return "(no atoms)"
    lines: list[str] = []
    for predicate in sorted(grouped):
        atoms = sorted(grouped[predicate])
        shown = atoms[:limit]
        lines.append(f"{predicate}/{len(atoms)}:")
        lines.extend(f"  {a}" for a in shown)
        if len(atoms) > len(shown):
            lines.append(f"  ... and {len(atoms) - len(shown)} more")
    return "\n".join(lines)


def render_violations(violations: list[Violation]) -> str:
    """Summarise integrity-constraint violations, one per line."""
    if not violations:
        return "no violations"
    return "\n".join(f"- {v.describe()}" for v in violations)
