"""Transactions over the layer stack: try something, and get back if it fails.

Anything that changes what the mind knows is a risky step, and the roadmap is
explicit that the kernel comes before the growth loop for this reason: a mind
that changes itself needs a guaranteed way back before it is allowed to try.

The mechanism is a snapshot and a context manager::

    with transaction(layers, "adding the induced rule") as change:
        layers.add_rules(rule, SANDBOX)
        if not canaries.check(layers):
            change.rollback("a canary stopped answering")

Rolling back restores every writable layer to what it was. The core is not
snapshotted, because it cannot change -- which is also why restoring is cheap.

On "copy-on-write": :meth:`KnowledgeBase.copy` already shares the rule objects
and copies only the dictionaries holding them, and rules are immutable
dataclasses, so a snapshot costs a dict copy per layer rather than a deep copy
of the knowledge. That is cheap enough to take one before every risky step,
which is the only way a guarantee like this stays true in practice.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

from ..knowledge.base import KnowledgeBase
from .layers import CORE, LAYER_ORDER, LayerStack

__all__ = ["Snapshot", "Transaction", "transaction", "RolledBack"]


class RolledBack(Exception):
    """Raised inside a transaction to abort it with a reason."""


@dataclass
class Snapshot:
    """The writable layers as they were at one moment."""

    layers: dict[str, KnowledgeBase]
    label: str = ""

    @classmethod
    def of(cls, stack: LayerStack, label: str = "") -> "Snapshot":
        return cls(
            layers={
                name: stack[name].knowledge.copy()
                for name in LAYER_ORDER
                if name != CORE
            },
            label=label,
        )

    def restore(self, stack: LayerStack) -> None:
        for name, knowledge in self.layers.items():
            stack[name].knowledge = knowledge.copy()

    @property
    def size(self) -> int:
        return sum(len(kb) for kb in self.layers.values())


@dataclass
class Transaction:
    """One risky step: it commits, or it leaves no trace."""

    stack: LayerStack
    before: Snapshot
    label: str = ""
    committed: bool = False
    rolled_back: bool = False
    reason: str = ""

    def rollback(self, reason: str) -> None:
        """Undo everything this transaction did, and say why."""
        if self.rolled_back:
            return
        self.before.restore(self.stack)
        self.rolled_back = True
        self.reason = reason

    def commit(self) -> None:
        self.committed = True

    def describe(self) -> str:
        if self.rolled_back:
            return f"rolled back: {self.label} — {self.reason}"
        return f"committed: {self.label}"


@contextmanager
def transaction(stack: LayerStack, label: str = "") -> Iterator[Transaction]:
    """Run a block against a snapshot, undoing it on failure.

    An exception rolls back and re-raises: a step that crashed half-way is
    exactly the case where leaving the knowledge base as it now stands is
    worst. Calling :meth:`Transaction.rollback` inside the block undoes it
    without raising, which is what a failed check wants.
    """
    change = Transaction(stack=stack, before=Snapshot.of(stack, label), label=label)
    try:
        yield change
    except Exception as exc:
        change.rollback(f"{type(exc).__name__}: {exc}")
        raise
    if not change.rolled_back:
        change.commit()
