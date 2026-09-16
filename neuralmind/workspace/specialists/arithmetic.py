"""The arithmetic specialist: a constraint solver behind the same protocol.

Datalog can already compare and add. What it cannot do is run a relation
*backwards*. Given ``total = x + y``, a Datalog program needs one rule per
direction -- one to get the total, one to get ``x``, one to get ``y`` -- and
the author has to know in advance which will be missing. A constraint solver
needs the relation stated once and answers whichever question arrives.

So the vocabulary here is relational, not functional. A constraint is an
ordinary ground atom, which means it can be given as a fact, derived by a rule,
or posted by another specialist, and nothing new had to be added to the
language to express it:

===========================  ==========================================
``value(q, 12)``             the quantity ``q`` is 12
``plus(total, x, y)``        ``total = x + y``
``minus(d, a, b)``           ``d = a - b``
``times(area, w, h)``        ``area = w * h``
``leq(load, rating)``        ``load <= rating`` (also ``lt``/``geq``/``gt``)
===========================  ==========================================

The same atoms are goals. ``value(y, V)`` asks the solver to find ``y``;
``leq(load, rating)`` asks whether the constraints *entail* the inequality,
which is not the same as whether it was stated.

Entailment is checked the way a solver should: by asking whether the negation
is satisfiable. If the system plus ``load > rating`` has no model, the
inequality holds in every model of the system, and that is a proof rather than
a lucky assignment. Solving for a value uses the same idea -- a value is only
reported when no *other* value is consistent, so "unique" means unique.

The proof cites the **unsat core**, not the whole system. That matters for
honesty as much as for brevity: the core is a minimal set of constraints that
already forces the conclusion, so every constraint named in the explanation
genuinely contributed, and none that contributed is missing.

Optional. Without ``z3-solver`` installed the specialist reports that it is
missing and the goals it would have taken come back ``unknown`` with the
reason, which is design rule 2 working as intended.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ...core.terms import Atom, Const, Var
from ..specialist import Budget, Finding, Result, specialist_proof

__all__ = ["ArithmeticSpecialist", "z3_available", "RELATIONS", "COMPARISONS"]

#: Ternary relations, as ``head = left <op> right``.
RELATIONS = {"plus": "+", "minus": "-", "times": "*"}

#: Binary orderings, usable as constraints and as goals.
COMPARISONS = {"leq": "<=", "lt": "<", "geq": ">=", "gt": ">"}

#: The variable a "what is this quantity?" subgoal asks about.
_VALUE_VAR = Var("V")

_MISSING = (
    "the arithmetic specialist needs z3-solver: "
    "`pip install neuralmind[arithmetic]`"
)


def z3_available() -> bool:
    try:
        import z3  # noqa: F401
    except ImportError:
        return False
    return True


class ArithmeticSpecialist:
    """Solves and checks numeric constraints over named quantities."""

    name = "arithmetic"

    def __init__(self, timeout_ms: int = 1000) -> None:
        self.timeout_ms = timeout_ms

    def prepare(self) -> None:
        """Import z3 and build one throwaway solver, so the first query is warm."""
        if not z3_available():
            return
        import z3

        solver = z3.Solver()
        solver.add(z3.Real("_warm") == 1)
        solver.check()

    def accepts(self, goal: Atom, workspace) -> float:
        if goal.predicate == "value" and goal.arity == 2:
            return 0.95
        if goal.predicate in COMPARISONS and goal.arity == 2:
            return 0.9
        return 0.0

    def run(self, goal: Atom, workspace, budget: Budget) -> Result:
        if not z3_available():
            return Result.nothing(_MISSING)
        import z3

        constraints = list(_constraints(workspace))
        wanted = _needs_values(goal, workspace, constraints)
        if wanted:
            # The quantity has a unit but no plain number yet. The units
            # specialist is the one that knows what "5 kN" is in base units,
            # so ask for that rather than guessing at the magnitude.
            return Result(
                subgoals=wanted,
                reason="needs " + ", ".join(str(a) for a in wanted),
            )
        if not constraints:
            return Result.nothing("no numeric constraints on the blackboard")

        variables: dict[str, object] = {}

        def var(name: str):
            if name not in variables:
                variables[name] = z3.Real(name)
            return variables[name]

        solver = z3.Solver()
        solver.set("timeout", int(min(self.timeout_ms, max(budget.remaining_ms, 1))))
        solver.set(unsat_core=True)
        # Each constraint is tracked by a label so the unsat core can be read
        # back as the atoms that produced it, rather than as opaque terms.
        tracked: dict[str, Atom] = {}
        used: list[Atom] = []
        for index, atom in enumerate(constraints):
            encoded = _encode(atom, var, z3)
            if encoded is None:
                continue
            label = f"c{index}"
            tracked[label] = atom
            solver.assert_and_track(encoded, z3.Bool(label))
            used.append(atom)

        if goal.predicate in COMPARISONS:
            return self._prove_comparison(goal, solver, var, tracked, workspace, z3)
        return self._solve_value(goal, solver, var, tracked, workspace, z3)

    # -- the two kinds of question -----------------------------------------

    def _prove_comparison(self, goal, solver, var, tracked, workspace, z3) -> Result:
        """Entailed, not merely consistent: check that the negation has no model."""
        left, right = (_name(arg) for arg in goal.args)
        if left is None or right is None:
            return Result.nothing(f"{goal} does not name two quantities")
        negation = _COMPARISON_NEGATION[goal.predicate](var(left), var(right), z3)
        outcome, core = _refute(solver, negation, tracked, z3)
        if outcome == z3.unknown:
            return Result(
                reason=f"the solver could not decide {goal} in the time available",
                exhausted=True,
            )
        if outcome == z3.sat:
            return Result.nothing(
                f"the constraints allow {goal.predicate}({left}, {right}) to fail"
            )
        step = _explain_comparison(goal, left, right, core, var, solver, z3)
        return Result(
            findings=[
                Finding(
                    atom=goal,
                    proof=specialist_proof(
                        goal, self.name, step, _because(core, workspace)
                    ),
                )
            ]
        )

    def _solve_value(self, goal, solver, var, tracked, workspace, z3) -> Result:
        """A value is only reported when no other value is consistent."""
        name = _name(goal.args[0]) if goal.args else None
        if name is None:
            return Result.nothing(f"{goal} does not name a quantity")
        wanted = goal.args[1] if goal.arity == 2 else None
        if isinstance(wanted, Const) and wanted.is_numeric:
            return self._prove_comparison(
                Atom("leq", goal.args), solver, var, tracked, workspace, z3
            )

        if solver.check() != z3.sat:
            return Result.nothing("the numeric constraints are unsatisfiable")
        candidate = solver.model().eval(var(name), model_completion=True)
        alternative, core = _refute(solver, var(name) != candidate, tracked, z3)
        if alternative == z3.sat:
            return Result.nothing(
                f"the constraints do not pin down {name}; "
                f"{_render(candidate)} is one of several values"
            )
        if alternative == z3.unknown:
            return Result(
                reason=f"the solver could not show {name} is unique in time",
                exhausted=True,
            )
        value = _as_const(candidate)
        if value is None:
            return Result.nothing(f"{name} solves to a value that is not a number")
        solved = Atom("value", (goal.args[0], value))
        step = f"{name} = {value} from " + ", ".join(_readable(atom) for atom in core)
        return Result(
            findings=[
                Finding(
                    atom=solved,
                    proof=specialist_proof(
                        solved, self.name, step, _because(core, workspace)
                    ),
                )
            ]
        )


# -- reading the blackboard ------------------------------------------------


def _needs_values(goal: Atom, workspace, constraints) -> list:
    """Quantities the goal mentions that carry a unit but no number yet."""
    known = {
        _name(atom.args[0])
        for atom in constraints
        if atom.predicate == "value" and atom.args
    }
    wanted = []
    for argument in goal.args:
        name = _name(argument)
        if name is None or name.startswith("#") or name in known:
            continue
        declared = [
            atom
            for atom in workspace
            if atom.predicate == "quantity"
            and atom.arity == 3
            and _name(atom.args[0]) == name
        ]
        if declared:
            wanted.append(Atom("value", (argument, _VALUE_VAR)))
    return wanted


def _constraints(workspace) -> Iterable[Atom]:
    """Every atom on the blackboard this specialist understands."""
    for atom in workspace:
        if atom.predicate == "value" and atom.arity == 2:
            yield atom
        elif atom.predicate in RELATIONS and atom.arity == 3:
            yield atom
        elif atom.predicate in COMPARISONS and atom.arity == 2:
            yield atom


def _encode(atom: Atom, var, z3):
    """One atom as a z3 assertion, or None if it does not name quantities."""
    names = [_name(arg) for arg in atom.args]
    if atom.predicate == "value":
        target, number = atom.args
        if _name(target) is None or not _is_numeric(number):
            return None
        return var(_name(target)) == _number(number)
    if atom.predicate in RELATIONS:
        if any(n is None for n in names):
            return None
        head, left, right = (var(n) for n in names)
        operator = RELATIONS[atom.predicate]
        if operator == "+":
            return head == left + right
        if operator == "-":
            return head == left - right
        return head == left * right
    if atom.predicate in COMPARISONS:
        if any(n is None for n in names):
            return None
        left, right = (var(n) for n in names)
        return _COMPARISON_ASSERT[atom.predicate](left, right)
    return None


def _name(term) -> Optional[str]:
    """A quantity name: a symbol, or a literal number as its own constant."""
    if isinstance(term, Const):
        if term.is_numeric:
            return f"#{term.value}"
        return str(term.value)
    return None


def _is_numeric(term) -> bool:
    return isinstance(term, Const) and term.is_numeric


def _number(term) -> float:
    return float(term.value)


_COMPARISON_ASSERT = {
    "leq": lambda a, b: a <= b,
    "lt": lambda a, b: a < b,
    "geq": lambda a, b: a >= b,
    "gt": lambda a, b: a > b,
}

_COMPARISON_NEGATION = {
    "leq": lambda a, b, z3: a > b,
    "lt": lambda a, b, z3: a >= b,
    "geq": lambda a, b, z3: a < b,
    "gt": lambda a, b, z3: a <= b,
}


# -- explaining ------------------------------------------------------------


def _refute(solver, negation, tracked: dict, z3) -> tuple:
    """Assert the negation and read back the constraints that ruled it out.

    On ``unsat`` the core is a minimal subset of the tracked constraints that
    is already inconsistent with the negation -- exactly the ones that force
    the conclusion, which is what belongs in the explanation. The atoms are
    returned in blackboard order so the citation reads predictably.
    """
    solver.push()
    solver.add(negation)
    outcome = solver.check()
    core: list = []
    if outcome == z3.unsat:
        labels = {str(term) for term in solver.unsat_core()}
        core = [atom for label, atom in tracked.items() if label in labels]
    solver.pop()
    return outcome, core


def _because(atoms, workspace) -> list:
    """The blackboard proofs for the constraints used, so the tree connects."""
    nodes = []
    for atom in atoms:
        proof = workspace.proof(atom)
        if proof is not None:
            nodes.append(proof)
    return nodes


def _readable(atom: Atom) -> str:
    """A constraint in ordinary arithmetic notation."""
    names = [_display(arg) for arg in atom.args]
    if atom.predicate == "value":
        return f"{names[0]} = {names[1]}"
    if atom.predicate in RELATIONS:
        return f"{names[0]} = {names[1]} {RELATIONS[atom.predicate]} {names[2]}"
    if atom.predicate in COMPARISONS:
        return f"{names[0]} {COMPARISONS[atom.predicate]} {names[1]}"
    return str(atom)


def _display(term) -> str:
    name = _name(term)
    return name[1:] if name and name.startswith("#") else (name or str(term))


def _explain_comparison(goal, left, right, core, var, solver, z3) -> str:
    """"3200 <= 5000" when both sides are pinned; the core constraints otherwise."""
    operator = COMPARISONS[goal.predicate]
    if solver.check() == z3.sat:
        model = solver.model()
        if _pinned(solver, var(left), model, z3) and _pinned(solver, var(right), model, z3):
            left_value = _render(model.eval(var(left), model_completion=True))
            right_value = _render(model.eval(var(right), model_completion=True))
            return f"{left_value} {operator} {right_value}"
    return " and ".join(_readable(atom) for atom in core) or str(goal)


def _pinned(solver, variable, model, z3) -> bool:
    """True if the constraints admit exactly one value for this variable."""
    solver.push()
    solver.add(variable != model.eval(variable, model_completion=True))
    outcome = solver.check()
    solver.pop()
    return outcome == z3.unsat


def _render(value) -> str:
    text = str(value)
    if "/" in text:  # z3 rationals print as 3/2
        numerator, denominator = text.split("/")
        return f"{float(numerator) / float(denominator):g}"
    return text.rstrip(".0") if text.endswith(".0") else text


def _as_const(value) -> Optional[Const]:
    text = _render(value)
    try:
        number = float(text)
    except ValueError:
        return None
    return Const(int(number)) if number.is_integer() else Const(number)
