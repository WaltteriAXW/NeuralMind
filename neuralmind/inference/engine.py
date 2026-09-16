"""The reasoning facade: one object to load a program, ask it things, and get
back both an answer and the derivation behind it.

Answers are never bare. A positive answer carries a proof tree; a negative
answer carries a diagnosis naming the body literal that could not be
satisfied. Under the closed-world assumption "no" is a real conclusion, and it
deserves an explanation just as much as "yes" does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Union

from ..core.parser import parse_atom, parse_file, parse_program
from ..core.program import Program, Rule
from ..core.terms import Atom, Const, Var
from .clingo_backend import ClingoBackend, CrossCheckResult, clingo_available, cross_check
from .forward import ForwardChainer, UnsupportedProgram
from .model import Model, Violation, match_atom
from .proof import ProofNode, explain, explain_violation

__all__ = ["ReasoningEngine", "Answer", "FailureDiagnosis", "RuleAttempt"]


@dataclass
class RuleAttempt:
    """How far one candidate rule got before its body failed."""

    rule: Rule
    progress: int
    total_steps: int
    blocked_on: str
    bindings: dict[str, Const] = field(default_factory=dict)
    #: The body literals that did match, in rule order. These are the things
    #: the knowledge base *does* establish on the way to the goal.
    established: tuple[Atom, ...] = ()
    #: The ground literal the rule stalled on, when it stalled on a lookup.
    #: A brief line needs the atom, not a message about it.
    missing: Optional[Atom] = None

    def describe(self) -> str:
        return (
            f"{self.rule.origin()} got {self.progress}/{self.total_steps} body "
            f"literals in, then stalled: {self.blocked_on}"
        )


@dataclass
class FailureDiagnosis:
    """Why a query is false: the rules that could have proved it, and where
    each one ran out of support."""

    goal: Atom
    attempts: list[RuleAttempt] = field(default_factory=list)

    @property
    def has_candidates(self) -> bool:
        return bool(self.attempts)

    def describe(self) -> str:
        if not self.attempts:
            return (
                f"{self.goal} is false: no rule in the knowledge base has a head "
                f"matching it, and it was not given as a fact."
            )
        lines = [f"{self.goal} is false. The rules that could have derived it:"]
        for attempt in sorted(self.attempts, key=lambda a: -a.progress):
            lines.append(f"  - {attempt.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "goal": str(self.goal),
            "reason": "no matching rule or fact" if not self.attempts else "unsatisfied body",
            "attempts": [
                {
                    "rule": str(a.rule),
                    "source": a.rule.origin(),
                    "progress": a.progress,
                    "total_steps": a.total_steps,
                    "blocked_on": a.blocked_on,
                }
                for a in sorted(self.attempts, key=lambda a: -a.progress)
            ],
        }

    def __str__(self) -> str:
        return self.describe()


#: The three things a query can come back as. ``unknown`` is not ``no``.
YES, NO, UNKNOWN = "yes", "no", "unknown"


@dataclass
class Answer:
    """The result of a query: what holds, and why.

    Answers are three-valued. ``yes`` means the goal was derived, ``no`` means
    its falsity was established, and ``unknown`` means neither -- which is a
    real answer and not a soft ``no``. Which of the last two applies depends on
    how the predicate was declared:

    * a **closed** predicate (the default) is fully described by the program,
      so failing to derive it *is* the proof that it is false;
    * an **open** predicate (``#open flies/1.``) may hold for reasons the
      program never mentions, so only a derived ``-flies(x)`` makes it ``no``.

    ``bool(answer)`` and :attr:`holds` stay exactly as they were: true only for
    ``yes``. Code that predates the third value keeps working, and reads
    ``unknown`` as "not proven", which is what it always meant.
    """

    query: Atom
    holds: bool
    atoms: list[Atom] = field(default_factory=list)
    bindings: list[dict[str, Const]] = field(default_factory=list)
    proofs: list[ProofNode] = field(default_factory=list)
    diagnosis: Optional[FailureDiagnosis] = None
    #: ``yes`` | ``no`` | ``unknown``.
    status: str = YES
    #: The derived ``-p(x)`` behind a ``no``, when there is one.
    refutation: Optional[ProofNode] = None

    def __bool__(self) -> bool:
        return self.holds

    @property
    def unknown(self) -> bool:
        return self.status == UNKNOWN

    @property
    def evidence(self) -> Optional[ProofNode]:
        """The proof behind whichever answer was given, positive or negative."""
        if self.status == YES:
            return self.proof
        return self.refutation

    @property
    def proof(self) -> Optional[ProofNode]:
        """The proof of the first matching atom, if any."""
        return self.proofs[0] if self.proofs else None

    def to_dict(self, include_proofs: bool = True) -> dict:
        payload: dict = {
            "query": str(self.query),
            "status": self.status,
            "holds": self.holds,
            "answers": [
                {
                    "atom": str(atom),
                    "bindings": {k: (v.value if isinstance(v, Const) else str(v))
                                 for k, v in binding.items()},
                }
                for atom, binding in zip(self.atoms, self.bindings)
            ],
        }
        if include_proofs and self.proofs:
            payload["proofs"] = [p.to_dict() for p in self.proofs]
        if include_proofs and self.refutation is not None:
            payload["refutation"] = self.refutation.to_dict()
        if self.diagnosis is not None:
            payload["why_not"] = self.diagnosis.to_dict()
        return payload

    def __str__(self) -> str:
        if self.status == UNKNOWN:
            return (
                f"{self.query}: unknown -- {self.query.predicate}/"
                f"{self.query.arity} is open, so failing to derive it "
                f"proves nothing."
            )
        if not self.holds:
            if self.refutation is not None:
                return str(self.refutation)
            return self.diagnosis.describe() if self.diagnosis else f"{self.query}: no"
        if self.proof is not None:
            return str(self.proof)
        return "\n".join(str(a) for a in self.atoms)


class ReasoningEngine:
    """Load a program, compute its model once, then query it.

    Parameters
    ----------
    program:
        A :class:`Program`, ASP source, or a path to a ``.lp`` file.
    backend:
        ``"python"`` forces the proof-producing forward chainer, ``"clingo"``
        forces the ASP solver, and ``"auto"`` (default) uses the chainer unless
        the program needs features only clingo has.
    """

    def __init__(
        self,
        program: Union[Program, str, Path],
        backend: str = "auto",
        **chainer_options,
    ) -> None:
        self.program = _coerce_program(program)
        self.backend = backend
        self.chainer_options = chainer_options
        self._model: Optional[Model] = None
        self._resolved_backend: Optional[str] = None

    # -- construction helpers -------------------------------------------

    @classmethod
    def from_files(cls, *paths: Union[str, Path], **kwargs) -> "ReasoningEngine":
        """Build an engine from several ``.lp`` files merged in order."""
        program = Program()
        for path in paths:
            program = program.merge(parse_file(path, check=False))
        if not program.raw_asp:
            program.check()
        return cls(program, **kwargs)

    def add_facts(self, facts: Iterable[Union[Atom, str]]) -> "ReasoningEngine":
        """Add ground facts and invalidate the cached model."""
        for fact in facts:
            atom = parse_atom(fact) if isinstance(fact, str) else fact
            if not atom.is_ground:
                raise ValueError(f"fact {atom} must be ground")
            self.program.add(Rule(atom, (), source="runtime", label="given"))
        self._model = None
        return self

    # -- solving ---------------------------------------------------------

    @property
    def resolved_backend(self) -> str:
        """Which backend will actually run: ``python`` or ``clingo``."""
        if self._resolved_backend:
            return self._resolved_backend
        if self.backend != "auto":
            return self.backend
        return "clingo" if self.program.raw_asp else "python"

    def solve(self, force: bool = False) -> Model:
        """Compute (and cache) the model."""
        if self._model is not None and not force:
            return self._model
        backend = self.resolved_backend
        if backend == "python":
            self._model = ForwardChainer(self.program, **self.chainer_options).run()
        elif backend == "clingo":
            result = ClingoBackend().solve(self.program)
            if not result.satisfiable:
                raise UnsupportedProgram(
                    "the program is unsatisfiable: some integrity constraint cannot "
                    "be met. Run with backend='python' to see which one."
                )
            answer = result.first()
            assert answer is not None
            self._model = answer.to_model(self.program)
        else:
            raise ValueError(f"unknown backend {backend!r}; use 'auto', 'python' or 'clingo'")
        self._resolved_backend = backend
        return self._model

    @property
    def model(self) -> Model:
        return self.solve()

    def answer_sets(self, limit: int = 0) -> list:
        """Enumerate answer sets with clingo (``limit=0`` means all of them)."""
        return ClingoBackend(models=limit).solve(self.program).answer_sets

    # -- querying --------------------------------------------------------

    def ask(self, query: Union[Atom, str], explain_answer: bool = True, limit: int = 50) -> Answer:
        """Query the model. Variables in the query are solved for."""
        goal = parse_atom(query) if isinstance(query, str) else query
        model = self.solve()
        matches = model.query(goal)[:limit]
        if matches:
            proofs = []
            if explain_answer:
                for atom, _ in matches:
                    try:
                        proofs.append(explain(model, atom))
                    except Exception:  # pragma: no cover - proof is best-effort
                        pass
            return Answer(
                query=goal,
                holds=True,
                status=YES,
                atoms=[atom for atom, _ in matches],
                bindings=[binding for _, binding in matches],
                proofs=proofs,
            )

        # Nothing derived the goal. Whether that settles the question depends
        # on the predicate: a closed one is fully described by the program, so
        # silence is a refutation; an open one may hold for reasons the program
        # never mentions, and only a derived -p(x) rules it out.
        refutation = None
        opposite = model.query(goal.complement())
        if opposite and explain_answer:
            try:
                refutation = explain(model, opposite[0][0])
            except Exception:  # pragma: no cover - proof is best-effort
                pass
        if opposite:
            status = NO
        elif self.program.is_open(goal.signature):
            status = UNKNOWN
        else:
            status = NO
        return Answer(
            query=goal,
            holds=False,
            status=status,
            refutation=refutation,
            # The diagnosis is what makes "unknown" useful rather than blank:
            # it names how far the rules got and which literal stopped them.
            diagnosis=self.why_not(goal),
        )

    def explain(self, atom: Union[Atom, str]) -> ProofNode:
        """Proof tree for a single ground atom."""
        goal = parse_atom(atom) if isinstance(atom, str) else atom
        return explain(self.solve(), goal)

    def why_not(self, atom: Union[Atom, str], max_branches: int = 2000) -> FailureDiagnosis:
        """Diagnose a false query by finding where each candidate rule stalls."""
        goal = parse_atom(atom) if isinstance(atom, str) else atom
        model = self.solve()
        diagnosis = FailureDiagnosis(goal=goal)
        for rule in self.program.rules:
            if rule.head is None or rule.head.signature != goal.signature:
                continue
            attempt = _diagnose_rule(model, rule, goal, max_branches)
            if attempt is not None:
                diagnosis.attempts.append(attempt)
        return diagnosis

    # -- consistency ------------------------------------------------------

    @property
    def violations(self) -> list[Violation]:
        """Integrity-constraint violations in the current model."""
        return self.solve().violations

    def explain_violations(self) -> list[ProofNode]:
        """A proof tree per violation, showing the facts that broke each rule."""
        model = self.solve()
        return [explain_violation(model, violation) for violation in model.violations]

    @property
    def consistent(self) -> bool:
        return self.solve().consistent

    # -- verification -----------------------------------------------------

    def cross_check(self) -> CrossCheckResult:
        """Re-derive the model with clingo and compare. A no-op if clingo is absent."""
        return cross_check(self.program, self.solve() if self.resolved_backend == "python" else None)


def _coerce_program(program: Union[Program, str, Path]) -> Program:
    if isinstance(program, Program):
        return program
    if isinstance(program, Path):
        return parse_file(program)
    text = str(program)
    if "\n" not in text and text.endswith(".lp") and Path(text).exists():
        return parse_file(Path(text))
    return parse_program(text)


def _diagnose_rule(
    model: Model, rule: Rule, goal: Atom, max_branches: int
) -> Optional[RuleAttempt]:
    """Walk a rule's body against the model and report where it stalls."""
    assert rule.head is not None
    head_bindings = match_atom(rule.head, goal, {})
    if head_bindings is None:
        return None
    try:
        steps = rule.plan()
    except Exception:  # pragma: no cover - unsafe rules never reach here
        return None
    if not steps:
        return RuleAttempt(rule, 0, 0, "the fact is not present", head_bindings)

    best: dict = {
        "progress": -1,
        "blocked_on": "the body is unsatisfiable",
        "bindings": {},
        "missing": None,
        "established": (),
    }
    budget = [max_branches]

    def record(
        index: int, message: str, bindings: dict, missing: Optional[Atom] = None
    ) -> None:
        if index > best["progress"]:
            best.update(
                progress=index,
                blocked_on=message,
                bindings=dict(bindings),
                missing=missing,
                established=_matched_so_far(steps, index, bindings),
            )

    def walk(index: int, bindings: dict) -> None:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if index == len(steps):
            record(index, "the body is satisfied (the model may be stale)", bindings)
            return
        step = steps[index]
        if step.kind == "match":
            assert step.literal is not None
            pattern = step.literal.atom.ground(bindings)
            found = False
            for fact in model.candidates(pattern):
                extended = match_atom(pattern, fact, bindings)
                if extended is not None:
                    found = True
                    walk(index + 1, extended)
            if not found:
                record(index, f"nothing in the model matches {pattern}", bindings, pattern)
        elif step.kind == "absent":
            assert step.literal is not None
            absent = step.literal.atom.ground(bindings)
            if absent in model.atoms:
                record(index, f"{absent} is true, so 'not {absent}' fails", bindings)
            else:
                walk(index + 1, bindings)
        elif step.kind == "filter":
            assert step.compare is not None
            if step.compare.holds(bindings):
                walk(index + 1, bindings)
            else:
                grounded = step.compare
                record(index, f"the comparison {grounded} is false", bindings)
        elif step.kind == "assign":
            assert step.variable is not None
            from ..core.terms import evaluate

            try:
                value = evaluate(step.expression, bindings)  # type: ignore[arg-type]
            except ValueError as exc:
                record(index, f"could not evaluate {step.expression}: {exc}", bindings)
                return
            extended = dict(bindings)
            if step.variable in bindings and bindings[step.variable] != value:
                record(
                    index,
                    f"{step.variable} is already {bindings[step.variable]}, "
                    f"but the rule computes {value}",
                    bindings,
                )
                return
            extended[step.variable] = value
            walk(index + 1, extended)

    walk(0, head_bindings)
    return RuleAttempt(
        rule=rule,
        progress=max(best["progress"], 0),
        total_steps=len(steps),
        blocked_on=best["blocked_on"],
        bindings=best["bindings"],
        established=best["established"],
        missing=best["missing"],
    )


def _matched_so_far(steps, index: int, bindings: dict) -> tuple[Atom, ...]:
    """The positive body literals before ``index``, ground by what was bound.

    These are what the rule did establish before it stalled, which is the
    informative half of an "unknown": not "I have nothing", but "I have this
    much and then I ran out".
    """
    matched: list[Atom] = []
    for step in steps[:index]:
        if step.kind == "match" and step.literal is not None:
            grounded = step.literal.atom.ground(bindings)
            if grounded.is_ground:
                matched.append(grounded)
    return tuple(matched)
