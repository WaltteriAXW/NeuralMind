"""Proof trees: the explanation, read straight off the justifications.

Nothing here reconstructs or guesses a reason after the fact. Every node is a
rule instance the engine actually fired, so a proof tree cannot disagree with
the answer it explains -- which is the property a generated explanation can
never offer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.terms import Atom
from .model import Justification, Model, Violation

__all__ = [
    "ProofNode", "explain", "explain_violation", "ProofError",
    "SPECIALIST", "ACTION",
]

#: How a node is established.
FACT = "fact"            # given as input
DERIVED = "derived"      # concluded by a rule
CLOSED_WORLD = "closed-world"  # assumed false because it is not derivable
UNPROVEN = "unproven"    # atom is not in the model at all
SPECIALIST = "specialist"  # concluded by a non-deductive specialist
ACTION = "action"        # made true by doing something, rather than by being true


class ProofError(Exception):
    """The requested atom cannot be explained."""


@dataclass
class ProofNode:
    """One step of a derivation, plus the steps it rests on."""

    conclusion: Atom
    kind: str
    rule_instance: Optional[str] = None
    rule_source: Optional[str] = None
    rule_label: Optional[str] = None
    negated: bool = False
    children: list["ProofNode"] = field(default_factory=list)
    #: Filled in by the perception layer when a fact came from a neural model.
    confidence: Optional[float] = None
    provenance: Optional[str] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def depth(self) -> int:
        """Height of this subtree; a leaf has depth 1."""
        return 1 + max((child.depth for child in self.children), default=0)

    @property
    def size(self) -> int:
        """Total number of nodes in this subtree."""
        return 1 + sum(child.size for child in self.children)

    def leaves(self) -> list["ProofNode"]:
        """Every leaf -- i.e. the premises the conclusion ultimately rests on."""
        if self.is_leaf:
            return [self]
        found: list[ProofNode] = []
        for child in self.children:
            found.extend(child.leaves())
        return found

    def premises(self) -> list[Atom]:
        """The input facts this conclusion depends on, in stable order."""
        seen: dict[str, Atom] = {}
        for leaf in self.leaves():
            if leaf.kind == FACT:
                seen.setdefault(str(leaf.conclusion), leaf.conclusion)
        return [seen[key] for key in sorted(seen)]

    def walk(self):
        """Yield every node in the tree, parents before children."""
        yield self
        for child in self.children:
            yield from child.walk()

    def to_dict(self) -> dict:
        """JSON-ready view of the tree."""
        node: dict = {
            "conclusion": str(self.conclusion),
            "predicate": self.conclusion.predicate,
            "arguments": [
                arg.value if hasattr(arg, "value") else str(arg)
                for arg in self.conclusion.args
            ],
            "kind": self.kind,
        }
        if self.negated:
            node["negated"] = True
        if self.rule_instance:
            node["rule"] = self.rule_instance
        if self.rule_label:
            node["rule_label"] = self.rule_label
        if self.rule_source:
            node["rule_source"] = self.rule_source
        if self.confidence is not None:
            node["confidence"] = round(float(self.confidence), 4)
        if self.provenance:
            node["provenance"] = self.provenance
        if self.children:
            node["because"] = [child.to_dict() for child in self.children]
        return node

    def __str__(self) -> str:
        from ..output.render import render_proof

        return render_proof(self)


def explain(
    model: Model,
    atom: Atom,
    max_depth: int = 64,
    justification: Optional[Justification] = None,
) -> ProofNode:
    """Build the proof tree for ``atom``.

    Raises :class:`ProofError` if the atom is not in the model: an atom that is
    false under the closed-world assumption has no positive proof, and saying
    so plainly beats inventing one.
    """
    if atom not in model:
        raise ProofError(
            f"{atom} is not in the model, so it has no proof. "
            "Under the closed-world assumption it is simply false."
        )
    return _build(model, atom, max_depth, set(), justification)


def _build(
    model: Model,
    atom: Atom,
    remaining_depth: int,
    in_progress: set[Atom],
    justification: Optional[Justification] = None,
) -> ProofNode:
    chosen = justification or model.best_justification(atom)
    if chosen is None or not chosen.support and chosen.rule.is_fact:
        rule = chosen.rule if chosen else None
        return ProofNode(
            conclusion=atom,
            kind=FACT,
            rule_instance=str(atom) + ".",
            rule_source=rule.origin() if rule else None,
            rule_label=rule.label if rule else None,
        )
    if remaining_depth <= 0 or atom in in_progress:
        # Defensive: best_justification already guarantees decreasing depth.
        return ProofNode(conclusion=atom, kind=DERIVED, rule_instance=chosen.instance())

    node = ProofNode(
        conclusion=atom,
        kind=DERIVED,
        rule_instance=chosen.instance(),
        rule_source=chosen.rule.origin(),
        rule_label=chosen.rule.label,
    )
    nested = in_progress | {atom}
    for support in chosen.support:
        node.children.append(_build(model, support, remaining_depth - 1, nested))
    for absent in chosen.negative_support:
        node.children.append(
            ProofNode(
                conclusion=absent,
                kind=CLOSED_WORLD,
                negated=True,
                rule_instance=f"not {absent}",
            )
        )
    return node


def explain_violation(model: Model, violation: Violation, max_depth: int = 64) -> ProofNode:
    """Explain why an integrity constraint fired.

    The result is a tree rooted at the constraint itself, so a rule-checking
    application can show exactly which facts combined to break the rule.
    """
    root = ProofNode(
        conclusion=Atom("violated", (_constraint_name(violation),)),
        kind=DERIVED,
        rule_instance=violation.rule.origin(),
        rule_source=violation.rule.origin(),
        rule_label=violation.rule.label,
    )
    for support in violation.support:
        root.children.append(_build(model, support, max_depth, set()))
    for absent in violation.negative_support:
        root.children.append(
            ProofNode(
                conclusion=absent,
                kind=CLOSED_WORLD,
                negated=True,
                rule_instance=f"not {absent}",
            )
        )
    return root


def _constraint_name(violation: Violation):
    from ..core.terms import Const

    label = violation.rule.label or violation.rule.origin()
    return Const(label, quoted=True)
