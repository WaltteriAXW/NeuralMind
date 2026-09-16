"""Correcting a knowledge base by pointing at wrong answers.

Writing rules is hard; noticing that an answer is wrong is easy. This module
turns the second into the first. You say "this should hold" or "this should
not", and it proposes concrete changes to the rules, each with what it fixes
and what it breaks.

That closes the loop the evaluation layer opens. `neuralmind eval` already
separates failures into "perception was wrong" and "a rule was missing"; for
the second kind, :class:`Corrector` proposes the missing rule.

Nothing is applied automatically. Every proposal is a :class:`Repair` the
caller can inspect, compare and reject -- because a rule that fixes the case in
front of you and quietly breaks four others is worse than no rule at all, and
only showing both makes that visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence, Union

from ..core.parser import parse_atom
from ..core.program import Program, Rule, SafetyError
from ..core.terms import Atom, Compare, Literal
from ..inference.forward import ForwardChainer
from ..inference.model import Model
from ..knowledge.base import KnowledgeBase
from .bias import LanguageBias, Signature
from .enumerate import candidate_comparisons, candidate_literals
from .learn import Examples, RuleLearner

__all__ = ["Repair", "Corrector"]

#: What a repair does.
ADD_RULE = "add-rule"
REMOVE_RULE = "remove-rule"
SPECIALISE = "specialise-rule"
RETRACT_FACT = "retract-fact"
ADD_FACT = "add-fact"


@dataclass
class Repair:
    """One proposed change, with its consequences measured rather than assumed."""

    kind: str
    explanation: str
    add: tuple = ()
    remove: tuple = ()
    add_facts: tuple = ()
    remove_facts: tuple = ()
    #: Complaints this change resolves.
    fixes: tuple = ()
    #: Conclusions that currently hold and would stop holding.
    breaks: tuple = ()
    #: Conclusions that would start holding, beyond those asked for.
    introduces: tuple = ()

    @property
    def clean(self) -> bool:
        """True if it fixes everything asked and breaks nothing else."""
        return bool(self.fixes) and not self.breaks

    def score(self) -> tuple:
        """Sort key: most fixed, least broken, least invented, simplest."""
        return (
            -len(self.fixes),
            len(self.breaks),
            len(self.introduces),
            len(self.add) + len(self.remove),
        )

    def apply(self, knowledge: KnowledgeBase) -> KnowledgeBase:
        """Apply this repair to a knowledge base, in place."""
        for rule in self.remove:
            try:
                knowledge.rules.rules.remove(rule)
            except ValueError:
                pass
        for rule in self.add:
            knowledge.rules.add(rule)
        for atom in self.remove_facts:
            knowledge.remove_fact(atom)
        for atom in self.add_facts:
            knowledge.add_fact(atom, provenance="repair")
        return knowledge

    def describe(self) -> str:
        lines = [self.explanation]
        for rule in self.remove:
            lines.append(f"    - {rule}")
        for atom in self.remove_facts:
            lines.append(f"    - {atom}.")
        for rule in self.add:
            lines.append(f"    + {rule}")
        for atom in self.add_facts:
            lines.append(f"    + {atom}.")
        if self.breaks:
            lines.append(
                "    but it would stop these holding: "
                + ", ".join(str(a) for a in self.breaks[:4])
                + (" ..." if len(self.breaks) > 4 else "")
            )
        if self.introduces:
            lines.append(
                "    and it would newly derive: "
                + ", ".join(str(a) for a in self.introduces[:4])
                + (" ..." if len(self.introduces) > 4 else "")
            )
        if self.clean:
            lines.append("    fixes the complaint without disturbing anything else")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "explanation": self.explanation,
            "add": [str(r) for r in self.add],
            "remove": [str(r) for r in self.remove],
            "add_facts": [str(a) for a in self.add_facts],
            "remove_facts": [str(a) for a in self.remove_facts],
            "fixes": [str(a) for a in self.fixes],
            "breaks": [str(a) for a in self.breaks],
            "introduces": [str(a) for a in self.introduces],
            "clean": self.clean,
        }

    def __str__(self) -> str:
        return self.describe()


class Corrector:
    """Proposes rule changes from complaints about specific conclusions.

    Parameters
    ----------
    knowledge:
        The knowledge base to correct. It is never modified; repairs are
        returned for the caller to apply.
    bias:
        Constrains the literals a specialisation may add and the rules
        induction may propose. Derived from the knowledge base when omitted,
        so a caller need not know the domain's vocabulary in advance.
    """

    def __init__(
        self,
        knowledge: KnowledgeBase,
        bias: Optional[LanguageBias] = None,
        max_repairs: int = 6,
    ) -> None:
        self.knowledge = knowledge
        self.bias = bias
        self.max_repairs = max_repairs

    # -- entry points -------------------------------------------------------

    def diagnose(
        self,
        expected: Iterable[Union[Atom, str]] = (),
        rejected: Iterable[Union[Atom, str]] = (),
    ) -> list[Repair]:
        """Propose changes for things that should and should not follow."""
        wanted = [_atom(a) for a in expected]
        unwanted = [_atom(a) for a in rejected]
        if not wanted and not unwanted:
            return []
        model = self._model(self.knowledge)
        repairs: list[Repair] = []
        for atom in unwanted:
            repairs.extend(self._repair_unwanted(atom, model, unwanted, wanted))
        for atom in wanted:
            repairs.extend(self._repair_missing(atom, model, unwanted))
        repairs.sort(key=Repair.score)
        return repairs[: self.max_repairs]

    def expect(self, *atoms: Union[Atom, str]) -> list[Repair]:
        """Propose changes that would make these follow."""
        return self.diagnose(expected=atoms)

    def reject(self, *atoms: Union[Atom, str]) -> list[Repair]:
        """Propose changes that would stop these following."""
        return self.diagnose(rejected=atoms)

    # -- things that hold and should not ------------------------------------

    def _repair_unwanted(
        self,
        atom: Atom,
        model: Model,
        unwanted: Sequence[Atom],
        wanted: Sequence[Atom],
    ) -> list[Repair]:
        if atom not in model:
            return []
        repairs: list[Repair] = []

        if self.knowledge.fact_record(atom) is not None:
            candidate = self._clone()
            candidate.remove_fact(atom)
            repairs.append(
                self._measure(
                    candidate,
                    model,
                    unwanted,
                    wanted,
                    Repair(
                        kind=RETRACT_FACT,
                        explanation=f"{atom} was asserted directly; withdraw it",
                        remove_facts=(atom,),
                    ),
                )
            )

        for rule in self._deriving_rules(atom, model):
            candidate = self._clone()
            try:
                candidate.rules.rules.remove(rule)
            except ValueError:
                continue
            repairs.append(
                self._measure(
                    candidate,
                    model,
                    unwanted,
                    wanted,
                    Repair(
                        kind=REMOVE_RULE,
                        explanation=f"{atom} comes from this rule; drop it",
                        remove=(rule,),
                    ),
                )
            )
            repairs.extend(
                self._specialisations(rule, atom, model, unwanted, wanted)
            )
        return repairs

    def _specialisations(
        self,
        rule: Rule,
        atom: Atom,
        model: Model,
        unwanted: Sequence[Atom],
        wanted: Sequence[Atom],
    ) -> list[Repair]:
        """Try adding one condition that excludes this case and keeps the rest.

        This is the repair worth having: the rule was nearly right, and one
        more condition makes it right. Dropping it outright usually costs more
        than it saves, which the measured ``breaks`` list makes obvious.
        """
        found: list[Repair] = []
        for extra in self._specialisation_candidates(rule):
            if _already_present(rule, extra):
                continue
            specialised = Rule(
                head=rule.head,
                body=rule.body + (extra,),
                source=rule.source,
                line=rule.line,
                label=rule.label,
            )
            try:
                specialised.plan()
            except SafetyError:
                continue
            candidate = self._clone()
            try:
                candidate.rules.rules.remove(rule)
            except ValueError:
                continue
            candidate.rules.add(specialised)
            repair = self._measure(
                candidate,
                model,
                unwanted,
                wanted,
                Repair(
                    kind=SPECIALISE,
                    explanation=f"narrow the rule so it no longer covers {atom}",
                    remove=(rule,),
                    add=(specialised,),
                ),
            )
            if repair.fixes:
                found.append(repair)
        found.sort(key=Repair.score)
        return found[:3]

    # -- things that should hold and do not ---------------------------------

    def _repair_missing(
        self, atom: Atom, model: Model, unwanted: Sequence[Atom]
    ) -> list[Repair]:
        if atom in model:
            return []
        repairs: list[Repair] = []

        candidate = self._clone()
        candidate.add_fact(atom, provenance="repair")
        repairs.append(
            self._measure(
                candidate,
                model,
                unwanted,
                [atom],
                Repair(
                    kind=ADD_FACT,
                    explanation=f"assert {atom} directly, if it is a fact rather "
                    "than something that should follow",
                    add_facts=(atom,),
                ),
            )
        )

        learned = self._induce(atom, model, unwanted)
        if learned is not None:
            candidate = self._clone()
            for rule in learned:
                candidate.rules.add(rule)
            repairs.append(
                self._measure(
                    candidate,
                    model,
                    unwanted,
                    [atom],
                    Repair(
                        kind=ADD_RULE,
                        explanation=f"a rule that would derive {atom} from what is "
                        "already known",
                        add=tuple(learned),
                    ),
                )
            )
        return repairs

    def _induce(
        self, atom: Atom, model: Model, unwanted: Sequence[Atom]
    ) -> Optional[list[Rule]]:
        """Learn a rule for the atom's predicate, treating the complaint as an
        example and everything currently derived for that predicate as the
        rest of the evidence."""
        signature = Signature(atom.predicate, atom.arity)
        positives = [atom] + [
            existing
            for existing in model.by_predicate(atom.predicate, atom.arity)
            if existing not in unwanted
        ]
        negatives = [a for a in unwanted if a.signature == atom.signature]
        bias = self.bias or LanguageBias.from_knowledge(
            self.knowledge,
            signature,
            max_variables=max(3, atom.arity + 1),
            max_body=3,
            allow_recursion=False,
            allow_comparison=True,
        )
        try:
            hypothesis = RuleLearner(
                self.knowledge,
                bias,
                Examples(positive=positives, negative=negatives),
                budget=20_000,
            ).learn()
        except Exception:
            return None
        return hypothesis.rules or None

    # -- measuring consequences ---------------------------------------------

    def _measure(
        self,
        candidate: KnowledgeBase,
        before: Model,
        unwanted: Sequence[Atom],
        wanted: Sequence[Atom],
        repair: Repair,
    ) -> Repair:
        """Fill in what a repair actually fixes, breaks and invents."""
        try:
            after = self._model(candidate)
        except Exception:
            repair.breaks = ("(the repair does not compile)",)
            return repair
        fixes = [a for a in unwanted if a in before and a not in after]
        fixes += [a for a in wanted if a not in before and a in after]
        # Anything that held, was not complained about, and no longer holds.
        breaks = sorted(
            (a for a in before.atoms if a not in after and a not in unwanted), key=str
        )
        introduces = sorted(
            (a for a in after.atoms if a not in before and a not in wanted), key=str
        )
        repair.fixes = tuple(fixes)
        repair.breaks = tuple(breaks)
        repair.introduces = tuple(introduces)
        return repair

    # -- helpers ------------------------------------------------------------

    def _model(self, knowledge: KnowledgeBase) -> Model:
        return ForwardChainer(knowledge.program(check=False)).run()

    def _clone(self) -> KnowledgeBase:
        return self.knowledge.copy()

    def _deriving_rules(self, atom: Atom, model: Model) -> list[Rule]:
        seen: list[Rule] = []
        for justification in model.justifications.get(atom, ()):
            rule = justification.rule
            if rule.is_fact or rule in seen:
                continue
            seen.append(rule)
        return seen

    def _specialisation_candidates(self, rule: Rule) -> list:
        """Conditions that could be added to *this* rule.

        Generated over the rule's own variables, plus one fresh one for joins.
        A generic candidate pool would be useless here: a literal over unrelated
        variables leaves them unbound, so every such rule is unsafe and gets
        discarded before it is ever tried.
        """
        import itertools

        from ..core.terms import Var

        assert rule.head is not None
        names = sorted(set(_rule_variable_names(rule)))
        fresh = next(
            letter for letter in "ZYWVUT" if letter not in names
        )
        pool = [Var(name) for name in names] + [Var(fresh)]

        signatures = [
            signature
            for signature in self._vocabulary()
            if signature != Signature(rule.head.predicate, rule.head.arity)
        ]
        candidates: list = []
        for signature in signatures:
            for arguments in itertools.product(pool, repeat=signature.arity):
                atom = Atom(signature.name, arguments)
                candidates.append(Literal(atom))
                candidates.append(Literal(atom, negated=True))
        for left, right in itertools.combinations(pool, 2):
            candidates.append(Compare("!=", left, right))
        return candidates

    def _vocabulary(self) -> list[Signature]:
        """Every predicate the knowledge base mentions."""
        found: dict[Signature, int] = {}
        for record in self.knowledge.facts:
            signature = Signature(record.atom.predicate, record.atom.arity)
            found[signature] = found.get(signature, 0) + 1
        for rule in self.knowledge.rules.rules:
            if rule.head is not None:
                found.setdefault(Signature(rule.head.predicate, rule.head.arity), 0)
            for part in rule.body:
                if isinstance(part, Literal):
                    found.setdefault(
                        Signature(part.atom.predicate, part.atom.arity), 0
                    )
        return sorted(found, key=lambda s: (-found[s], s))


def _atom(value: Union[Atom, str]) -> Atom:
    return parse_atom(value.rstrip(".")) if isinstance(value, str) else value


def _rule_variable_names(rule: Rule) -> list[str]:
    from ..core.terms import variables_in

    names: list[str] = []
    if rule.head is not None:
        names.extend(variables_in(rule.head))
    for part in rule.body:
        names.extend(variables_in(part))
    return names


def _already_present(rule: Rule, extra) -> bool:
    return any(str(part) == str(extra) for part in rule.body)
