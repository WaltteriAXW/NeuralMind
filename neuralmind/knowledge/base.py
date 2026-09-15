"""The knowledge base: rules the domain expert writes, facts the world supplies.

Facts carry provenance and (optionally) a confidence. That is the seam between
the two halves of the system: a hand-written fact is certain and a perceived
fact is not, and the consistency layer needs to tell them apart. The symbolic
engine itself ignores confidence entirely -- it reasons over what is asserted,
and the Type 5 layer decides what may be asserted.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Union

from ..core.parser import parse_atom, parse_file, parse_program
from ..core.program import Program, Rule
from ..core.terms import Atom

__all__ = ["KnowledgeBase", "FactRecord", "rule_path", "builtin_rulesets"]

RULES_DIR = Path(__file__).parent / "rules"


@dataclass(frozen=True)
class FactRecord:
    """A fact plus where it came from and how much we trust it."""

    atom: Atom
    confidence: float = 1.0
    provenance: str = "asserted"
    #: Free-form detail, e.g. the sentence a triple was extracted from.
    evidence: Optional[str] = None

    @property
    def certain(self) -> bool:
        return self.confidence >= 1.0

    def to_dict(self) -> dict:
        payload = {
            "fact": str(self.atom),
            "confidence": round(float(self.confidence), 4),
            "provenance": self.provenance,
        }
        if self.evidence:
            payload["evidence"] = self.evidence
        return payload


class KnowledgeBase:
    """Rules plus facts, assembled into a program the engine can run."""

    def __init__(self, name: str = "kb") -> None:
        self.name = name
        self.rules = Program()
        self._facts: dict[Atom, FactRecord] = {}

    # -- rules -----------------------------------------------------------

    def load_rules(self, path: Union[str, Path]) -> "KnowledgeBase":
        """Load a ``.lp`` rule file. Facts in the file count as asserted facts."""
        program = parse_file(path, check=False)
        return self._absorb(program, provenance=f"file:{Path(path).name}")

    def load_builtin(self, name: str) -> "KnowledgeBase":
        """Load one of the rule sets shipped with the package, e.g. ``family``."""
        return self.load_rules(rule_path(name))

    def add_rules(self, source: str, name: str = "inline") -> "KnowledgeBase":
        """Parse and add ASP source."""
        return self._absorb(parse_program(source, name, check=False), provenance=name)

    def _absorb(self, program: Program, provenance: str) -> "KnowledgeBase":
        for rule in program.rules:
            if rule.is_fact and rule.head is not None and rule.head.is_ground:
                self.add_fact(rule.head, provenance=provenance)
            else:
                self.rules.add(rule)
        self.rules.shown |= program.shown
        self.rules.constants.update(program.constants)
        self.rules.raw_asp.extend(program.raw_asp)
        if program.requires_asp:
            self.rules.requires_asp = tuple(
                dict.fromkeys(self.rules.requires_asp + program.requires_asp)
            )
        return self

    # -- facts -----------------------------------------------------------

    def add_fact(
        self,
        fact: Union[Atom, str],
        confidence: float = 1.0,
        provenance: str = "asserted",
        evidence: Optional[str] = None,
    ) -> FactRecord:
        """Assert a ground fact. Re-asserting keeps the highest confidence."""
        atom = parse_atom(fact) if isinstance(fact, str) else fact
        if not atom.is_ground:
            raise ValueError(f"fact {atom} must be ground; rules go through add_rules()")
        existing = self._facts.get(atom)
        if existing is not None and existing.confidence >= confidence:
            return existing
        record = FactRecord(atom, confidence, provenance, evidence)
        self._facts[atom] = record
        return record

    def add_facts(self, facts: Iterable[Union[Atom, str, FactRecord]], **kwargs) -> "KnowledgeBase":
        for fact in facts:
            if isinstance(fact, FactRecord):
                self._facts[fact.atom] = fact
            else:
                self.add_fact(fact, **kwargs)
        return self

    def remove_fact(self, fact: Union[Atom, str]) -> bool:
        atom = parse_atom(fact) if isinstance(fact, str) else fact
        return self._facts.pop(atom, None) is not None

    def retain(self, minimum_confidence: float) -> list[FactRecord]:
        """Drop facts below a confidence threshold, returning what was dropped."""
        dropped = [r for r in self._facts.values() if r.confidence < minimum_confidence]
        for record in dropped:
            del self._facts[record.atom]
        return dropped

    @property
    def facts(self) -> list[FactRecord]:
        return sorted(self._facts.values(), key=lambda r: str(r.atom))

    def fact_record(self, atom: Atom) -> Optional[FactRecord]:
        return self._facts.get(atom)

    def confidence_of(self, atom: Atom) -> float:
        """Confidence of an asserted fact; 1.0 for anything not perceived."""
        record = self._facts.get(atom)
        return record.confidence if record else 1.0

    def uncertain_facts(self, threshold: float = 1.0) -> list[FactRecord]:
        return [r for r in self.facts if r.confidence < threshold]

    def __contains__(self, fact: Union[Atom, str]) -> bool:
        atom = parse_atom(fact) if isinstance(fact, str) else fact
        return atom in self._facts

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self) -> Iterator[FactRecord]:
        return iter(self.facts)

    # -- assembly --------------------------------------------------------

    def program(self, check: bool = True) -> Program:
        """Combine facts and rules into a single program."""
        combined = Program(
            rules=[
                Rule(record.atom, (), source=record.provenance, label="given")
                for record in self.facts
            ]
            + list(self.rules.rules),
            shown=set(self.rules.shown),
            constants=dict(self.rules.constants),
            raw_asp=list(self.rules.raw_asp),
            requires_asp=self.rules.requires_asp,
        )
        if check and not combined.raw_asp:
            combined.check()
        return combined

    def engine(self, **kwargs):
        """Build a :class:`~neuralmind.inference.engine.ReasoningEngine` over this KB."""
        from ..inference.engine import ReasoningEngine

        return ReasoningEngine(self.program(), **kwargs)

    def copy(self) -> "KnowledgeBase":
        clone = KnowledgeBase(self.name)
        clone.rules = self.rules.merge(Program())
        clone._facts = dict(self._facts)
        return clone

    # -- reporting -------------------------------------------------------

    def stats(self) -> dict:
        predicates: dict[str, int] = {}
        for record in self._facts.values():
            predicates[record.atom.predicate] = predicates.get(record.atom.predicate, 0) + 1
        return {
            "name": self.name,
            "facts": len(self._facts),
            "rules": len(self.rules.derivation_rules),
            "constraints": len(self.rules.constraints),
            "fact_predicates": dict(sorted(predicates.items())),
            "uncertain_facts": len(self.uncertain_facts()),
        }

    def to_asp(self) -> str:
        return self.program(check=False).to_asp()

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"KnowledgeBase(name={self.name!r}, facts={stats['facts']}, "
            f"rules={stats['rules']}, constraints={stats['constraints']})"
        )


def rule_path(name: str) -> Path:
    """Locate a bundled rule file by name (with or without the ``.lp``)."""
    candidate = RULES_DIR / (name if name.endswith(".lp") else f"{name}.lp")
    if not candidate.exists():
        available = ", ".join(sorted(builtin_rulesets()))
        raise FileNotFoundError(f"no bundled rule set named {name!r}; available: {available}")
    return candidate


def builtin_rulesets() -> list[str]:
    """Names of the rule sets shipped with the package."""
    return sorted(p.stem for p in RULES_DIR.glob("*.lp"))
