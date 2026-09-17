"""Knowledge in layers, so a mistake can only reach as far as its own layer.

A mind that explores an unfamiliar situation will be wrong. It will read a
sentence badly, induce a rule from two examples, accept a pack that contradicts
something it already knew. None of that is avoidable; what is avoidable is
letting any of it touch what the mind came with.

So knowledge is stacked, each layer trusted less than the one below:

``core``
    Shipped with the system and **read-only at runtime**. Nothing the mind
    learns, is told, or infers can modify it. It is the floor the safety kernel
    can always fall back to, and the thing every canary is checked against.
``confirmed``
    Learned and approved. A rule reaches here only by passing the firewall and
    leaving every canary intact -- and, if it was induced from personal data,
    only after a human has seen it.
``tenant``
    Facts about one organisation. Never shared, never promoted, never visible
    to another tenant. Design rule 12.
``session``
    Facts about the conversation in hand, including anything personal. Cleared
    when the session ends.
``sandbox``
    Anything tentative: a draft pack, a hypothesis under test, a rule that has
    not earned its place. **Excluded from queries by default** -- you have to
    ask for it, which means nothing tentative can answer a question by
    accident.

A :class:`LayerStack` answers a query against a chosen set of layers. That is
the whole mechanism: the sandbox is dangerous only if you read from it, so
reading from it is the thing you have to opt into.

On ordering: later layers override earlier ones for *facts* (a session fact
about this customer beats a general one), and rules accumulate rather than
override, because a rule is a licence to conclude and removing one silently
would change answers no one asked to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from ..core.program import Program
from ..core.terms import Atom
from ..knowledge.base import FactRecord, KnowledgeBase

__all__ = [
    "LayerStack",
    "Layer",
    "CORE",
    "CONFIRMED",
    "TENANT",
    "SESSION",
    "SANDBOX",
    "LAYER_ORDER",
    "DEFAULT_LAYERS",
    "ReadOnlyLayer",
]

CORE = "core"
CONFIRMED = "confirmed"
TENANT = "tenant"
SESSION = "session"
SANDBOX = "sandbox"

#: Least trusted last. A query reads them in this order.
LAYER_ORDER = (CORE, CONFIRMED, TENANT, SESSION, SANDBOX)

#: What a query sees unless it asks for more. The sandbox is not in it.
DEFAULT_LAYERS = (CORE, CONFIRMED, TENANT, SESSION)


class ReadOnlyLayer(Exception):
    """An attempt to write to a layer that does not accept writes."""


@dataclass
class Layer:
    """One layer: a knowledge base, plus who may write to it."""

    name: str
    knowledge: KnowledgeBase
    writable: bool = True
    #: Facts here may never be copied into a less private layer.
    confined: bool = False
    #: Which tenant or session this layer belongs to, when it is confined.
    owner: Optional[str] = None

    def __len__(self) -> int:
        return len(self.knowledge)

    def stats(self) -> dict:
        payload = self.knowledge.stats()
        payload["layer"] = self.name
        payload["writable"] = self.writable
        if self.confined:
            payload["confined_to"] = self.owner
        return payload


class LayerStack:
    """The four-plus-one layers, and queries over any subset of them."""

    def __init__(self, name: str = "mind", tenant: Optional[str] = None) -> None:
        self.name = name
        self.tenant = tenant
        self._layers: dict[str, Layer] = {
            CORE: Layer(CORE, KnowledgeBase(f"{name}:core"), writable=False),
            CONFIRMED: Layer(CONFIRMED, KnowledgeBase(f"{name}:confirmed")),
            TENANT: Layer(
                TENANT, KnowledgeBase(f"{name}:tenant"), confined=True, owner=tenant
            ),
            SESSION: Layer(
                SESSION, KnowledgeBase(f"{name}:session"), confined=True, owner=tenant
            ),
            SANDBOX: Layer(SANDBOX, KnowledgeBase(f"{name}:sandbox")),
        }

    # -- access -------------------------------------------------------------

    def __getitem__(self, name: str) -> Layer:
        try:
            return self._layers[name]
        except KeyError:
            raise KeyError(
                f"unknown layer {name!r}; choose from {', '.join(LAYER_ORDER)}"
            ) from None

    def __contains__(self, name: str) -> bool:
        return name in self._layers

    def __iter__(self) -> Iterator[Layer]:
        return (self._layers[name] for name in LAYER_ORDER)

    @property
    def core(self) -> KnowledgeBase:
        return self._layers[CORE].knowledge

    def knowledge(self, layer: str) -> KnowledgeBase:
        return self[layer].knowledge

    # -- writing ------------------------------------------------------------

    def load_core(self, source: str, name: str = "core") -> "LayerStack":
        """Fill the core. Only callable while it is still being built.

        The core is read-only *at runtime*, which is not the same as immutable:
        something has to ship it. This is the seam, and it is deliberately not
        reachable through the paths the mind itself uses to learn.
        """
        self._layers[CORE].knowledge.add_rules(source, name)
        return self

    def load_core_builtin(self, name: str) -> "LayerStack":
        self._layers[CORE].knowledge.load_builtin(name)
        return self

    def seal_core(self) -> "LayerStack":
        """Mark the core finished. Later writes raise rather than be ignored."""
        self._layers[CORE].writable = False
        return self

    def add_rules(self, source: str, layer: str = SANDBOX, name: str = "inline"):
        """Add rules to a layer. Anything tentative belongs in the sandbox."""
        target = self[layer]
        if not target.writable:
            raise ReadOnlyLayer(
                f"the {layer} layer is read-only at runtime. Nothing the mind "
                f"learns may modify what it shipped with -- put it in the "
                f"sandbox and promote it once the canaries agree."
            )
        target.knowledge.add_rules(source, name)
        return self

    def add_fact(self, fact, layer: str = SESSION, **kwargs) -> FactRecord:
        target = self[layer]
        if not target.writable:
            raise ReadOnlyLayer(f"the {layer} layer is read-only at runtime")
        return target.knowledge.add_fact(fact, **kwargs)

    def add_facts(self, facts: Iterable, layer: str = SESSION, **kwargs) -> "LayerStack":
        for fact in facts:
            if isinstance(fact, FactRecord):
                self.add_fact(
                    fact.atom,
                    layer,
                    confidence=fact.confidence,
                    provenance=fact.provenance,
                    evidence=fact.evidence,
                )
            else:
                self.add_fact(fact, layer, **kwargs)
        return self

    # -- reading ------------------------------------------------------------

    def view(self, layers: Sequence[str] = DEFAULT_LAYERS) -> KnowledgeBase:
        """One knowledge base combining the named layers, least trusted last.

        A fresh object each time, so a caller cannot reach through it and
        modify a layer -- in particular the core, which this is the main way
        of reading.
        """
        for name in layers:
            if name not in self._layers:
                raise KeyError(f"unknown layer {name!r}")
        ordered = [name for name in LAYER_ORDER if name in layers]
        combined = KnowledgeBase(f"{self.name}:{'+'.join(ordered)}")
        for name in ordered:
            source = self._layers[name].knowledge
            combined.rules = combined.rules.merge(source.rules)
            for record in source.facts:
                # Later layers override: a session fact about this customer
                # beats a general one, and re-adding keeps the newer record.
                combined._facts[record.atom] = record
        return combined

    def engine(self, layers: Sequence[str] = DEFAULT_LAYERS, **kwargs):
        return self.view(layers).engine(**kwargs)

    def program(self, layers: Sequence[str] = DEFAULT_LAYERS, **kwargs) -> Program:
        return self.view(layers).program(**kwargs)

    def facts(self, layers: Sequence[str] = DEFAULT_LAYERS) -> list[FactRecord]:
        return self.view(layers).facts

    def holds(self, atom: Atom, layers: Sequence[str] = DEFAULT_LAYERS) -> bool:
        return self.engine(layers).model.holds(atom)

    def layer_of(self, atom: Atom) -> Optional[str]:
        """Which layer a fact lives in, most trusted first."""
        for name in LAYER_ORDER:
            if atom in self._layers[name].knowledge:
                return name
        return None

    # -- moving between layers ----------------------------------------------

    def promote(self, atom: Atom, source: str, target: str) -> FactRecord:
        """Move a fact up the stack, refusing to carry confined data out.

        This is the one place personal data could escape, so it is the one
        place that checks. A fact in a confined layer stays there; a rule
        *induced* from such facts is a separate question, handled by the
        firewall, because a generalisation over many people is not the same
        object as a fact about one.
        """
        record = self[source].knowledge.fact_record(atom)
        if record is None:
            raise KeyError(f"{atom} is not in the {source} layer")
        if self[source].confined and not self[target].confined:
            raise ReadOnlyLayer(
                f"{atom} is confined to the {source} layer and cannot be "
                f"promoted to {target}: facts about a specific person or "
                f"company never become shared knowledge."
            )
        promoted = self.add_fact(
            record.atom,
            target,
            confidence=record.confidence,
            provenance=record.provenance,
            evidence=record.evidence,
        )
        self[source].knowledge.remove_fact(atom)
        return promoted

    def clear(self, layer: str) -> "LayerStack":
        """Empty a layer. Used to end a session and to drop the sandbox."""
        target = self[layer]
        if not target.writable:
            raise ReadOnlyLayer(f"the {layer} layer is read-only at runtime")
        target.knowledge = KnowledgeBase(f"{self.name}:{layer}")
        return self

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        return {
            "name": self.name,
            "tenant": self.tenant,
            "layers": {
                layer.name: {
                    "facts": len(layer.knowledge),
                    "rules": len(layer.knowledge.rules.derivation_rules),
                    "writable": layer.writable,
                    "confined": layer.confined,
                }
                for layer in self
            },
        }

    def __repr__(self) -> str:
        counts = ", ".join(f"{layer.name}={len(layer.knowledge)}" for layer in self)
        return f"LayerStack({counts})"
