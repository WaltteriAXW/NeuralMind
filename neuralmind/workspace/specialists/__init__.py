"""The specialists, and a default set built from whatever is installed.

Every one but :class:`~neuralmind.workspace.specialists.logic.LogicSpecialist`
is optional, and :func:`default_specialists` simply leaves out what is not
installed. It does *not* leave out a specialist whose dependency is missing:
that one is included so it can say so, which turns "unknown" into "unknown,
because the arithmetic specialist needs z3-solver". Silence would be the
unhelpful version of the same answer.
"""

from .arithmetic import ArithmeticSpecialist, z3_available
from .graph import GraphSpecialist, networkx_available
from .logic import LogicSpecialist
from .units import UnitsSpecialist, pint_available

__all__ = [
    "LogicSpecialist",
    "ArithmeticSpecialist",
    "UnitsSpecialist",
    "GraphSpecialist",
    "default_specialists",
    "installed",
]


def default_specialists(knowledge) -> list:
    """Logic, plus every optional specialist -- installed or not.

    They are all here on purpose. A specialist whose backend is missing still
    accepts the goals it would have taken and reports what is missing, so the
    answer names the gap instead of being a bare "unknown".
    """
    return [
        LogicSpecialist(knowledge),
        ArithmeticSpecialist(),
        UnitsSpecialist(),
        GraphSpecialist(),
    ]


def installed() -> dict[str, bool]:
    """Which optional backends are actually available, for the self-report."""
    return {
        "logic": True,
        "arithmetic": z3_available(),
        "units": pint_available(),
        "graph": networkx_available(),
    }
