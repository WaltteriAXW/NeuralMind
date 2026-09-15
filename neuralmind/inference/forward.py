"""Semi-naive forward chaining with justification tracking.

This is the default reasoning engine. It computes the least model of a
stratified Datalog program and, for every atom it derives, records the rule
instance that derived it. clingo is faster and far more expressive, but it
reports *which atoms are true*, not *why* -- and "why" is the entire point of
this pipeline. The two agree on every program both can run, which
:func:`neuralmind.inference.engine.cross_check` verifies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

from ..core.program import Program, Rule, Step
from ..core.terms import Atom, Const, evaluate
from .model import Justification, Model, Violation, match_atom

__all__ = ["ForwardChainer", "UnsupportedProgram", "ReasoningLimit", "solve", "match_body"]


class UnsupportedProgram(Exception):
    """The program uses ASP features this engine cannot evaluate."""


class ReasoningLimit(Exception):
    """Reasoning exceeded a configured bound (atoms or iterations)."""


@dataclass
class _CompiledRule:
    rule: Rule
    steps: tuple[Step, ...]
    body_signatures: frozenset[tuple[str, int]]


class ForwardChainer:
    """Computes the least model of a stratified program, with provenance.

    Parameters
    ----------
    program:
        The rules to run. Must be safe and stratified (both are checked).
    max_justifications:
        How many alternative derivations to keep per atom. Extra derivations
        are useful for "is there another reason this holds?" but cost memory,
        so the default keeps a handful.
    max_atoms / max_iterations:
        Guard rails. Arithmetic in rule heads can make a Datalog program
        diverge (``p(X+1) :- p(X).``); these turn an infinite loop into an
        error that names the rule responsible.
    strict:
        If True (default), hitting a limit raises. If False, reasoning stops
        and the model is flagged via :attr:`Model.truncated`.
    """

    def __init__(
        self,
        program: Program,
        max_justifications: int = 4,
        max_atoms: int = 500_000,
        max_iterations: int = 1_000,
        strict: bool = True,
    ) -> None:
        if program.raw_asp:
            features = ", ".join(program.requires_asp) or "unsupported constructs"
            raise UnsupportedProgram(
                f"this program uses {features}, which the Python engine cannot "
                "evaluate. Run it through the clingo backend "
                "(neuralmind.inference.clingo_backend) instead."
            )
        self.program = program
        self.max_justifications = max_justifications
        self.max_atoms = max_atoms
        self.max_iterations = max_iterations
        self.strict = strict
        self._strata = program.stratify()

    # -- public API ------------------------------------------------------

    def run(self) -> Model:
        """Compute the least model and return it with all justifications."""
        model = Model()
        total_iterations = 0
        for stratum in self._strata:
            compiled = [
                _CompiledRule(
                    rule=rule,
                    steps=tuple(rule.plan()),
                    body_signatures=frozenset(
                        lit.signature for lit in rule.positive_literals
                    ),
                )
                for rule in stratum
            ]
            total_iterations += self._saturate(model, compiled)
            if model.truncated:
                break
        model.iterations = total_iterations
        if not model.truncated:
            self._collect_violations(model)
        return model

    # -- fixpoint --------------------------------------------------------

    def _saturate(self, model: Model, compiled: list[_CompiledRule]) -> int:
        """Iterate this stratum's rules until nothing new can be derived."""
        facts = [c for c in compiled if c.rule.is_fact]
        derivations = [c for c in compiled if not c.rule.is_fact]

        for compiled_fact in facts:
            atom = compiled_fact.rule.head
            assert atom is not None
            if not atom.is_ground:
                raise UnsupportedProgram(
                    f"fact {atom} at {compiled_fact.rule.origin()} is not ground"
                )
            model.add(atom, Justification(compiled_fact.rule, (), (), (), 0), depth=0)

        # Everything is "fresh" on the first pass; after that a rule only needs
        # re-running if one of its body predicates gained an atom.
        changed_signatures = set(model._by_sig)
        iterations = 0
        while True:
            iterations += 1
            if iterations > self.max_iterations:
                return self._limit(
                    model, f"exceeded {self.max_iterations} iterations", iterations
                )
            newly_added: set[tuple[str, int]] = set()
            produced = False
            for compiled_rule in derivations:
                if iterations > 1 and not (compiled_rule.body_signatures & changed_signatures):
                    continue
                for bindings, support, negative in self._match(compiled_rule, model):
                    head = compiled_rule.rule.head
                    assert head is not None
                    try:
                        atom = head.ground(bindings)
                    except ValueError as exc:
                        raise UnsupportedProgram(
                            f"cannot instantiate head of {compiled_rule.rule.origin()}: {exc}"
                        ) from exc
                    depth = 1 + max((model.depth.get(s, 0) for s in support), default=0)
                    justification = Justification(
                        rule=compiled_rule.rule,
                        bindings=tuple(sorted(bindings.items())),
                        support=support,
                        negative_support=negative,
                        depth=depth,
                    )
                    if atom in model.atoms:
                        existing = model.justifications.setdefault(atom, [])
                        if (
                            len(existing) < self.max_justifications
                            and justification not in existing
                        ):
                            existing.append(justification)
                        continue
                    if len(model.atoms) >= self.max_atoms:
                        return self._limit(
                            model,
                            f"reached the {self.max_atoms}-atom limit while applying "
                            f"{compiled_rule.rule.origin()}",
                            iterations,
                        )
                    model.add(atom, justification, depth=depth)
                    newly_added.add(atom.signature)
                    produced = True
            if not produced:
                return iterations
            changed_signatures = newly_added

    def _limit(self, model: Model, message: str, iterations: int) -> int:
        if self.strict:
            raise ReasoningLimit(
                message + ". Rules with arithmetic in the head can derive "
                "infinitely many atoms -- add a bound, or raise the limit "
                "explicitly if the program really is this large."
            )
        model.truncated = message
        return iterations

    # -- body matching ---------------------------------------------------

    def _match(
        self, compiled_rule: _CompiledRule, model: Model
    ) -> Iterator[tuple[dict[str, Const], tuple[Atom, ...], tuple[Atom, ...]]]:
        """Enumerate every way this rule's body is satisfied by the model."""
        yield from match_body(compiled_rule.steps, model)

    # -- constraints -----------------------------------------------------

    def _collect_violations(self, model: Model) -> None:
        """Report every instantiation that breaks an integrity constraint."""
        for rule in self.program.constraints:
            compiled_rule = _CompiledRule(
                rule=rule,
                steps=tuple(rule.plan()),
                body_signatures=frozenset(lit.signature for lit in rule.positive_literals),
            )
            for bindings, support, negative in self._match(compiled_rule, model):
                model.violations.append(
                    Violation(
                        rule=rule,
                        bindings=tuple(sorted(bindings.items())),
                        support=support,
                        negative_support=negative,
                    )
                )


def match_body(
    steps: Sequence[Step], model: Model
) -> Iterator[tuple[dict[str, Const], tuple[Atom, ...], tuple[Atom, ...]]]:
    """Enumerate the groundings of a planned rule body against a model.

    Yields ``(bindings, positive support, negative support)`` for each way the
    body holds. Shared by the forward chainer and the consistency layer so both
    ground rules identically -- the Type 5 check must see the same
    instantiations the crisp engine does.
    """

    def walk(
        index: int,
        bindings: dict[str, Const],
        support: tuple[Atom, ...],
        negative: tuple[Atom, ...],
    ) -> Iterator[tuple[dict[str, Const], tuple[Atom, ...], tuple[Atom, ...]]]:
        if index == len(steps):
            yield bindings, support, negative
            return
        step = steps[index]
        if step.kind == "match":
            assert step.literal is not None
            pattern = step.literal.atom.ground(bindings)
            for fact in model.candidates(pattern):
                extended = match_atom(pattern, fact, bindings)
                if extended is not None:
                    yield from walk(index + 1, extended, support + (fact,), negative)
        elif step.kind == "absent":
            assert step.literal is not None
            absent = step.literal.atom.ground(bindings)
            if absent not in model.atoms:
                yield from walk(index + 1, bindings, support, negative + (absent,))
        elif step.kind == "filter":
            assert step.compare is not None
            if step.compare.holds(bindings):
                yield from walk(index + 1, bindings, support, negative)
        elif step.kind == "assign":
            assert step.variable is not None
            value = evaluate(step.expression, bindings)  # type: ignore[arg-type]
            extended = dict(bindings)
            extended[step.variable] = value
            yield from walk(index + 1, extended, support, negative)
        else:  # pragma: no cover - guarded by Rule.plan
            raise AssertionError(f"unknown plan step {step.kind!r}")

    yield from walk(0, {}, (), ())


def solve(program: Program, **kwargs) -> Model:
    """Convenience wrapper: build a chainer, run it, return the model."""
    return ForwardChainer(program, **kwargs).run()
