"""The units specialist: dimensions, conversions, and the errors they catch.

A number without its unit is not a fact, it is half of one. This specialist
keeps the other half, and it earns its place mostly by refusing things:
``plus(total, length, duration)`` is a sentence the arithmetic solver will
happily satisfy and that nothing in the world satisfies. Catching that is
cheaper than catching it later, and the error names the dimensions rather than
the numbers, which is what makes it fixable.

Vocabulary, all ordinary ground atoms:

==================================  ======================================
``quantity(beam1_load, 3200, n)``   3200 newtons
``in_unit(beam1_load, kn, V)``      ask: what is that in kilonewtons?
``same_dimension(a, b)``            ask: are these comparable at all?
``dimension(a, D)``                 ask: what kind of quantity is this?
==================================  ======================================

It also posts a plain ``value/2`` for every quantity, converted to **base
units**, which is what lets the arithmetic specialist compare a length in
millimetres against one in metres without either specialist knowing the other
exists. That conversion is the specialist's real contribution to the
blackboard: agreement about what the numbers mean.

Optional. Without ``pint`` installed the goals it would take come back
``unknown`` with the reason.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ...core.terms import Atom, Const
from ..specialist import Budget, Finding, Result, specialist_proof

__all__ = ["UnitsSpecialist", "pint_available"]

_MISSING = "the units specialist needs pint: `pip install neuralmind[units]`"

#: Unit symbols that ASP's lower-case constants cannot spell directly.
#: Logic constants are lower-case, so "kN" arrives as "kn" and would otherwise
#: read as kelvin-nanometres.
ALIASES = {
    "kn": "kilonewton",
    "n": "newton",
    "mn": "meganewton",
    "mpa": "megapascal",
    "kpa": "kilopascal",
    "pa": "pascal",
    "mm": "millimeter",
    "cm": "centimeter",
    "km": "kilometer",
    "m": "meter",
    "kg": "kilogram",
    "g": "gram",
    "mg": "milligram",
    "s": "second",
    "h": "hour",
    "c": "degC",
    "k": "kelvin",
    "l": "liter",
    "ml": "milliliter",
}


def pint_available() -> bool:
    try:
        import pint  # noqa: F401
    except ImportError:
        return False
    return True


_REGISTRY = None


def _registry():
    global _REGISTRY
    if _REGISTRY is None:
        import pint

        _REGISTRY = pint.UnitRegistry()
    return _REGISTRY


class UnitsSpecialist:
    """Dimensional analysis and conversion over ``quantity/3`` facts."""

    name = "units"

    def prepare(self) -> None:
        """Build the unit registry now. It is the most expensive warm-up here."""
        if not pint_available():
            return
        registry = _registry()
        (1 * registry("meter")).to_base_units()

    def accepts(self, goal: Atom, workspace) -> float:
        if goal.predicate in ("in_unit", "dimension") or (
            goal.predicate == "same_dimension" and goal.arity == 2
        ):
            return 0.95
        # A bare value question about something declared with a unit is this
        # specialist's too: it is the one that knows the base-unit number.
        if goal.predicate == "value" and goal.arity == 2:
            return 0.5 if _quantities(workspace) else 0.0
        return 0.0

    def run(self, goal: Atom, workspace, budget: Budget) -> Result:
        if not pint_available():
            return Result.nothing(_MISSING)
        registry = _registry()
        known = _quantities(workspace)
        if not known:
            return Result.nothing("no quantities with units on the blackboard")

        if goal.predicate == "same_dimension":
            return self._compare_dimensions(goal, known, workspace, registry)
        if goal.predicate == "dimension":
            return self._name_dimension(goal, known, workspace, registry)
        if goal.predicate == "in_unit":
            return self._convert(goal, known, workspace, registry)
        return self._base_value(goal, known, workspace, registry)

    # -- the questions ------------------------------------------------------

    def _base_value(self, goal, known, workspace, registry) -> Result:
        name = _symbol(goal.args[0])
        source = known.get(name)
        if source is None:
            return Result.nothing(f"{name} has no declared unit")
        measure = _measure(source, registry)
        if measure is None:
            return Result.nothing(f"{source[2]} is not a unit this specialist knows")
        base = measure.to_base_units()
        atom = Atom("value", (goal.args[0], _as_const(base.magnitude)))
        step = f"{_render(measure)} = {_render(base)}"
        return Result(
            findings=[
                Finding(atom, specialist_proof(atom, self.name, step,
                                               _because([source[3]], workspace)))
            ]
        )

    def _convert(self, goal, known, workspace, registry) -> Result:
        if goal.arity != 3:
            return Result.nothing("in_unit takes a quantity, a unit and a value")
        name, unit_term = _symbol(goal.args[0]), _symbol(goal.args[1])
        source = known.get(name)
        if source is None:
            return Result.nothing(f"{name} has no declared unit")
        measure = _measure(source, registry)
        if measure is None:
            return Result.nothing(f"{source[2]} is not a unit this specialist knows")
        try:
            converted = measure.to(_unit(unit_term))
        except Exception as exc:
            return Result.nothing(f"cannot convert {name} to {unit_term}: {_brief(exc)}")
        atom = Atom("in_unit", (goal.args[0], goal.args[1], _as_const(converted.magnitude)))
        step = f"{_render(measure)} = {_render(converted)}"
        return Result(
            findings=[
                Finding(atom, specialist_proof(atom, self.name, step,
                                               _because([source[3]], workspace)))
            ]
        )

    def _compare_dimensions(self, goal, known, workspace, registry) -> Result:
        left, right = (_symbol(arg) for arg in goal.args)
        pair = [known.get(left), known.get(right)]
        if any(item is None for item in pair):
            missing = left if pair[0] is None else right
            return Result.nothing(f"{missing} has no declared unit")
        measures = [_measure(item, registry) for item in pair]
        if any(m is None for m in measures):
            return Result.nothing("one of the units is not one this specialist knows")
        if measures[0].dimensionality != measures[1].dimensionality:
            return Result.nothing(
                f"{left} is {_dimension(measures[0])} and {right} is "
                f"{_dimension(measures[1])}; they are not comparable"
            )
        step = f"{left} and {right} are both {_dimension(measures[0])}"
        return Result(
            findings=[
                Finding(goal, specialist_proof(goal, self.name, step,
                                               _because([pair[0][3], pair[1][3]], workspace)))
            ]
        )

    def _name_dimension(self, goal, known, workspace, registry) -> Result:
        name = _symbol(goal.args[0])
        source = known.get(name)
        if source is None:
            return Result.nothing(f"{name} has no declared unit")
        measure = _measure(source, registry)
        if measure is None:
            return Result.nothing(f"{source[2]} is not a unit this specialist knows")
        label = _dimension(measure)
        atom = Atom("dimension", (goal.args[0], Const(label)))
        step = f"{source[2]} measures {label}"
        return Result(
            findings=[
                Finding(atom, specialist_proof(atom, self.name, step,
                                               _because([source[3]], workspace)))
            ]
        )

    # -- consistency --------------------------------------------------------

    def check(self, workspace) -> list[str]:
        """Dimension errors in the arithmetic constraints, in plain words.

        Run by the controller before the arithmetic specialist, because adding
        a length to a duration is not a hard problem the solver should work on
        -- it is a mistake, and the solver will happily satisfy it.
        """
        if not pint_available():
            return []
        registry = _registry()
        known = _quantities(workspace)
        problems: list[str] = []
        for atom in workspace:
            if atom.predicate not in ("plus", "minus") or atom.arity != 3:
                continue
            dimensions = []
            for argument in atom.args:
                source = known.get(_symbol(argument))
                measure = _measure(source, registry) if source else None
                dimensions.append(_dimension(measure) if measure else None)
            present = [d for d in dimensions if d]
            if len(present) > 1 and len(set(present)) > 1:
                problems.append(
                    f"{atom} mixes " + " and ".join(sorted(set(present)))
                )
        return problems


# -- reading the blackboard ------------------------------------------------


def _quantities(workspace) -> dict:
    """``name -> (name, number, unit, atom)`` for every quantity/3 on the board."""
    found = {}
    for atom in workspace:
        if atom.predicate != "quantity" or atom.arity != 3:
            continue
        name, number, unit = atom.args
        if not (isinstance(number, Const) and number.is_numeric):
            continue
        found[_symbol(name)] = (_symbol(name), float(number.value), _symbol(unit), atom)
    return found


def _measure(source, registry):
    if source is None:
        return None
    try:
        return source[1] * registry(_unit(source[2]))
    except Exception:
        return None


def _unit(symbol: str) -> str:
    return ALIASES.get(symbol.lower(), symbol)


def _symbol(term) -> str:
    return str(term.value) if isinstance(term, Const) else str(term)


#: Dimensionality as an exponent map, so the name does not depend on how pint
#: happens to order the factors in its string form.
_DIMENSION_NAMES = {
    (("[length]", 1),): "length",
    (("[mass]", 1),): "mass",
    (("[time]", 1),): "time",
    (("[temperature]", 1),): "temperature",
    (("[substance]", 1),): "amount",
    (("[current]", 1),): "current",
    (("[length]", 2),): "area",
    (("[length]", 3),): "volume",
    (("[length]", 1), ("[time]", -1)): "speed",
    (("[length]", 1), ("[time]", -2)): "acceleration",
    (("[length]", 1), ("[mass]", 1), ("[time]", -2)): "force",
    (("[length]", -1), ("[mass]", 1), ("[time]", -2)): "pressure",
    (("[length]", 2), ("[mass]", 1), ("[time]", -2)): "energy",
    (("[length]", 2), ("[mass]", 1), ("[time]", -3)): "power",
}


def _dimension(measure) -> str:
    """"length", "force", ... rather than "[mass] * [length] / [time] ** 2"."""
    key = tuple(sorted((name, int(power))
                       for name, power in measure.dimensionality.items()))
    if not key:
        return "dimensionless"
    named = _DIMENSION_NAMES.get(key)
    return named or str(measure.dimensionality)


def _because(atoms, workspace) -> list:
    nodes = []
    for atom in atoms:
        proof = workspace.proof(atom)
        if proof is not None:
            nodes.append(proof)
    return nodes


def _render(measure) -> str:
    return f"{measure.magnitude:g} {measure.units:~P}"


def _as_const(magnitude) -> Const:
    number = float(magnitude)
    return Const(int(number)) if number.is_integer() else Const(number)


def _brief(exc: Exception) -> str:
    return str(exc).split("\n")[0]
