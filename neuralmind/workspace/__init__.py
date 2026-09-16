"""The workspace: a blackboard, specialists, and a controller over them.

Phase One was a pipeline, which cannot hold two kinds of reasoning that are not
downstream of each other. This package is the place where they meet. See
:mod:`neuralmind.workspace.blackboard` for why a blackboard is the right shape
and :mod:`neuralmind.workspace.specialist` for the one rule every specialist
obeys -- every result carries its own proof, so the tree spans all of them.
"""

from .blackboard import Entry, Goal, OpenQuestion, Workspace
from .controller import BUDGET, NO_SPECIALIST, NOT_ESTABLISHED, Conclusion, Controller
from .router import Candidate, Router
from .specialist import Budget, Finding, Result, Specialist, specialist_proof

__all__ = [
    "Workspace",
    "Entry",
    "Goal",
    "OpenQuestion",
    "Controller",
    "Conclusion",
    "BUDGET",
    "NO_SPECIALIST",
    "NOT_ESTABLISHED",
    "Router",
    "Candidate",
    "Specialist",
    "Budget",
    "Finding",
    "Result",
    "specialist_proof",
]
