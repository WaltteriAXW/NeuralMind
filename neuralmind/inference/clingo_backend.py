"""Optional clingo backend: the authoritative ASP solver.

Two jobs. First, it runs the programs the Python engine deliberately does not
support -- choice rules, aggregates, optimisation, unstratified negation --
which is everything combinatorial. Second, it acts as an oracle: any program
both engines can run should give both the same atoms, and
:func:`cross_check` asserts exactly that.

clingo is an optional dependency (``pip install neuralmind[asp]``). Everything
here degrades to a clear error message when it is missing, so the core pipeline
never hard-depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..core.program import Program
from ..core.terms import Atom, Const
from .model import Justification, Model

__all__ = [
    "ClingoBackend",
    "ClingoUnavailable",
    "AnswerSet",
    "SolveResult",
    "clingo_available",
    "cross_check",
    "CrossCheckResult",
]


class ClingoUnavailable(ImportError):
    """clingo is not installed."""


def clingo_available() -> bool:
    """True if the clingo Python module can be imported."""
    try:
        import clingo  # noqa: F401
    except ImportError:
        return False
    return True


def _require_clingo():
    try:
        import clingo
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ClingoUnavailable(
            "clingo is not installed. Install it with `pip install clingo` "
            "(MIT licensed) to use the ASP backend."
        ) from exc
    return clingo


@dataclass
class AnswerSet:
    """One stable model returned by clingo."""

    atoms: frozenset[Atom]
    cost: tuple[int, ...] = ()
    optimal: bool = False

    def by_predicate(self, predicate: str) -> list[Atom]:
        return sorted((a for a in self.atoms if a.predicate == predicate), key=str)

    def to_model(self, program: Optional[Program] = None) -> Model:
        """Convert to a :class:`Model`.

        The atoms are exact; the justifications are not -- clingo does not
        report them. Every atom therefore arrives marked as input. Use the
        forward chainer when you need proof trees.
        """
        model = Model()
        for atom in sorted(self.atoms, key=str):
            model.add(atom, depth=0)
        return model

    def __len__(self) -> int:
        return len(self.atoms)

    def __iter__(self):
        return iter(sorted(self.atoms, key=str))


@dataclass
class SolveResult:
    """The outcome of a clingo call."""

    satisfiable: bool
    answer_sets: list[AnswerSet] = field(default_factory=list)
    exhausted: bool = True

    @property
    def unsatisfiable(self) -> bool:
        return not self.satisfiable

    def first(self) -> Optional[AnswerSet]:
        return self.answer_sets[0] if self.answer_sets else None

    def best(self) -> Optional[AnswerSet]:
        """The optimal answer set when optimising, else the first one."""
        optimal = [a for a in self.answer_sets if a.optimal]
        if optimal:
            return optimal[-1]
        return self.first()


class ClingoBackend:
    """Thin wrapper over the clingo Python API."""

    def __init__(self, models: int = 1, options: Sequence[str] = ()) -> None:
        #: 0 means "enumerate every answer set".
        self.models = models
        self.options = list(options)

    def solve(self, program: Program | str, models: Optional[int] = None) -> SolveResult:
        """Ground and solve, returning up to ``models`` answer sets."""
        clingo = _require_clingo()
        source = program.to_asp() if isinstance(program, Program) else program
        limit = self.models if models is None else models
        control = clingo.Control([str(limit), *self.options])
        control.add("base", [], source)
        control.ground([("base", [])])

        answer_sets: list[AnswerSet] = []

        def on_model(model) -> None:
            answer_sets.append(
                AnswerSet(
                    atoms=frozenset(
                        symbol_to_atom(symbol) for symbol in model.symbols(shown=True)
                    ),
                    cost=tuple(model.cost),
                    optimal=model.optimality_proven,
                )
            )

        handle = control.solve(on_model=on_model)
        return SolveResult(
            satisfiable=bool(handle.satisfiable),
            answer_sets=answer_sets,
            exhausted=bool(handle.exhausted),
        )

    def satisfiable(self, program: Program | str) -> bool:
        """True if the program has at least one answer set."""
        return self.solve(program, models=1).satisfiable

    def optimise(self, program: Program | str) -> Optional[AnswerSet]:
        """Solve an optimisation program and return the optimal answer set."""
        return self.solve(program, models=0).best()


def symbol_to_atom(symbol) -> Atom:
    """Convert a clingo symbol into a :class:`~neuralmind.core.terms.Atom`."""
    clingo = _require_clingo()
    if symbol.type == clingo.SymbolType.Function:
        args = tuple(_symbol_to_term(a) for a in symbol.arguments)
        return Atom(symbol.name, args)
    # A bare number or string shown on its own; wrap it so it stays an atom.
    return Atom(str(symbol), ())


def _symbol_to_term(symbol):
    clingo = _require_clingo()
    if symbol.type == clingo.SymbolType.Number:
        return Const(symbol.number)
    if symbol.type == clingo.SymbolType.String:
        return Const(symbol.string, quoted=True)
    if symbol.type == clingo.SymbolType.Function and not symbol.arguments:
        return Const(symbol.name)
    # Nested function terms have no first-class representation in this data
    # model; keep them readable rather than dropping information.
    return Const(str(symbol), quoted=True)


@dataclass
class CrossCheckResult:
    """Whether the Python engine and clingo agree on a program."""

    agree: bool
    python_only: list[Atom] = field(default_factory=list)
    clingo_only: list[Atom] = field(default_factory=list)
    python_violations: int = 0
    clingo_satisfiable: bool = True
    skipped: Optional[str] = None

    def report(self) -> str:
        if self.skipped:
            return f"cross-check skipped: {self.skipped}"
        if self.agree:
            return "cross-check passed: the Python engine and clingo derive the same atoms"
        lines = ["cross-check FAILED"]
        if self.python_only:
            lines.append(
                "  only the Python engine derived: "
                + ", ".join(str(a) for a in self.python_only[:10])
            )
        if self.clingo_only:
            lines.append(
                "  only clingo derived: " + ", ".join(str(a) for a in self.clingo_only[:10])
            )
        if self.python_violations and self.clingo_satisfiable:
            lines.append(
                f"  the Python engine found {self.python_violations} constraint "
                "violation(s) but clingo reports the program satisfiable"
            )
        return "\n".join(lines)

    def __bool__(self) -> bool:
        return self.agree


def cross_check(program: Program, model: Optional[Model] = None) -> CrossCheckResult:
    """Verify the Python engine's model against clingo.

    Integrity constraints are stripped before handing the program to clingo:
    the Python engine reports violations *and* keeps the model, while clingo
    would simply answer UNSAT. The two views are compared separately -- the
    atoms must match, and a violation must correspond to an UNSAT answer on the
    program *with* its constraints.
    """
    if not clingo_available():
        return CrossCheckResult(agree=True, skipped="clingo is not installed")
    if program.raw_asp:
        return CrossCheckResult(
            agree=True, skipped="program uses ASP-only features the Python engine cannot run"
        )

    from .forward import ForwardChainer

    if model is None:
        model = ForwardChainer(program).run()

    backend = ClingoBackend()
    without_constraints = Program(
        rules=[r for r in program.rules if not r.is_constraint],
        shown=set(program.shown),
        constants=dict(program.constants),
    )
    # Show everything, not just #show-declared predicates, so the comparison is
    # over the whole model rather than a projection of it.
    source = without_constraints.to_asp(include_shown=False)
    result = backend.solve(source, models=1)
    answer = result.first()
    clingo_atoms = set(answer.atoms) if answer else set()
    python_atoms = set(model.atoms)

    with_constraints = backend.solve(program.to_asp(include_shown=False), models=1)

    python_only = sorted(python_atoms - clingo_atoms, key=str)
    clingo_only = sorted(clingo_atoms - python_atoms, key=str)
    violations_agree = bool(model.violations) == (not with_constraints.satisfiable)
    return CrossCheckResult(
        agree=not python_only and not clingo_only and violations_agree,
        python_only=python_only,
        clingo_only=clingo_only,
        python_violations=len(model.violations),
        clingo_satisfiable=with_constraints.satisfiable,
    )
