"""Human-readable rendering of proof trees and models.

Plain text on purpose: a proof tree printed in a terminal is the fastest way to
see whether the reasoning is right, and it is the format a developer debugging
a knowledge base actually reads.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..inference.model import Model, Violation
from ..inference.proof import ACTION, CLOSED_WORLD, FACT, SPECIALIST, ProofNode

__all__ = ["render_proof", "render_model", "render_violations"]

_MARKERS = {
    FACT: "[given]",
    CLOSED_WORLD: "[not derivable]",
    ACTION: "[do]",
}


def render_proof(
    node: Optional[ProofNode], indent: str = "", is_last: bool = True, top: bool = True
) -> str:
    """Render a proof tree as an indented ASCII tree.

    The root line is the conclusion; each level below it is the reason for the
    line above. Leaves marked ``[given]`` are inputs, everything else was
    derived by the rule shown in parentheses.
    """
    if node is None:
        return "(no proof: the atom does not hold)"
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
    if node.kind == ACTION:
        # An action node is not something that is true, it is something to
        # do -- and what makes it inspectable is the definition it came from,
        # the same way a specialist's arithmetic is.
        if node.rule_label:
            text = f"{text}  ({node.rule_label})"
        return text
    if node.kind == SPECIALIST:
        # A specialist's step is the interesting part -- "3.2 kN <= 5.0 kN"
        # says why, where the specialist's name alone only says who.
        name = node.rule_label or node.rule_source or "specialist"
        step = node.rule_instance
        text = f"{text}  [by {name}: {step}]" if step else f"{text}  [by {name}]"
        return text
    if node.kind not in (FACT, CLOSED_WORLD):
        source = node.rule_label or node.rule_source
        if source:
            text = f"{text}  (by {source})"
    return text


def render_model(model: Model, predicates: Optional[Iterable[str]] = None, limit: int = 200) -> str:
    """List the atoms of a model, grouped by predicate."""
    wanted = set(predicates) if predicates else None
    grouped: dict[tuple[str, int], list[str]] = {}
    for atom in model.atoms:
        if wanted and atom.predicate not in wanted:
            continue
        grouped.setdefault(atom.signature, []).append(str(atom))
    if not grouped:
        return "(no atoms)"
    lines: list[str] = []
    for signature in sorted(grouped):
        atoms = sorted(grouped[signature])
        shown = atoms[:limit]
        # name/arity, then how many there are -- the count used to sit where
        # every reader expects the arity.
        lines.append(f"{signature[0]}/{signature[1]}  ({len(atoms)}):")
        lines.extend(f"  {a}" for a in shown)
        if len(atoms) > len(shown):
            lines.append(f"  ... and {len(atoms) - len(shown)} more")
    return "\n".join(lines)


def render_violations(violations: list[Violation]) -> str:
    """Summarise integrity-constraint violations, one per line."""
    if not violations:
        return "no violations"
    return "\n".join(f"- {v.describe()}" for v in violations)
