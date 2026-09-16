"""A small world that needs every specialist, and the questions to ask it.

P2.1's bar is a set of mixed questions — logic, numbers, units and paths —
each answered with **one** proof tree rather than four separate answers. That
is only a real test if the questions genuinely interlock, so this is one
scenario rather than four unrelated ones: a workshop assembling a frame, where
the load check needs a unit conversion before the arithmetic, the schedule
needs the assembly graph, and the sign-off rule needs all of it.

Everything here is ordinary ground atoms and ordinary rules. No specialist has
a private format, which is what lets a result from one be an input to another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

__all__ = ["WORKSHOP_FACTS", "WORKSHOP_RULES", "WORKSHOP_QUESTIONS", "MixedQuestion"]


@dataclass(frozen=True)
class MixedQuestion:
    """One question, its expected status, and which specialist should settle it."""

    goal: str
    status: str
    #: The specialist expected to contribute the decisive step.
    by: str
    note: str = ""


#: Quantities carry units; the numbers alone would not be facts.
WORKSHOP_FACTS = """
quantity(beam_a_load, 3200, n).
quantity(beam_a_rating, 5, kn).
quantity(beam_b_load, 7400, n).
quantity(beam_b_rating, 5, kn).
quantity(beam_c_load, 1, kn).
quantity(beam_c_rating, 900, n).
quantity(span, 4200, mm).
quantity(clearance, 5, m).
quantity(cure_time, 2, h).
quantity(bay_width, 3, m).
quantity(bay_height, 7, m).

beam(beam_a).
beam(beam_b).
beam(beam_c).
rated(beam_a, beam_a_load, beam_a_rating).
rated(beam_b, beam_b_load, beam_b_rating).
rated(beam_c, beam_c_load, beam_c_rating).

edge(cut, drill).
edge(drill, weld).
edge(weld, paint).
edge(cut, deburr).
edge(deburr, paint).
edge(paint, inspect).
edge(inspect, ship).

plus(frame_cost, steel_cost, labour_cost).
value(frame_cost, 1450).
value(steel_cost, 900).
times(bay_area, bay_width_m, bay_height_m).
value(bay_width_m, 3).
value(bay_height_m, 7).
"""

#: Rules that reach across specialists: each body mixes plain facts with
#: something only a specialist can establish.
WORKSHOP_RULES = """
%@ a beam is safe when its load is within its rating
safe(beam_a) :- beam(beam_a), leq(beam_a_load, beam_a_rating).
safe(beam_b) :- beam(beam_b), leq(beam_b_load, beam_b_rating).
safe(beam_c) :- beam(beam_c), leq(beam_c_load, beam_c_rating).

%@ the frame fits when the span clears the opening
fits :- leq(span, clearance).

%@ nothing ships until it has been inspected
shippable :- before(cut, ship).

%@ sign off needs every beam safe and a frame that fits
signed_off :- safe(beam_a), shippable, fits.
"""

#: The mixed set. Each entry names the specialist expected to do the decisive
#: work, so a regression that routes a question elsewhere is visible even when
#: the answer happens to stay the same.
WORKSHOP_QUESTIONS = [
    # -- units: conversion and dimension -----------------------------------
    MixedQuestion("in_unit(beam_a_load, kn, V)", "yes", "units", "3200 N is 3.2 kN"),
    MixedQuestion("in_unit(beam_a_rating, n, V)", "yes", "units"),
    MixedQuestion("in_unit(span, m, V)", "yes", "units"),
    MixedQuestion("in_unit(cure_time, s, V)", "yes", "units"),
    MixedQuestion("dimension(span, D)", "yes", "units", "length"),
    MixedQuestion("dimension(beam_a_load, D)", "yes", "units", "force"),
    MixedQuestion("dimension(cure_time, D)", "yes", "units", "time"),
    MixedQuestion("same_dimension(beam_a_load, beam_a_rating)", "yes", "units"),
    MixedQuestion("same_dimension(span, clearance)", "yes", "units"),
    MixedQuestion("same_dimension(span, cure_time)", "unknown", "units",
                  "a length and a duration are not comparable"),
    MixedQuestion("same_dimension(beam_a_load, span)", "unknown", "units"),
    MixedQuestion("value(beam_a_rating, V)", "yes", "units", "5 kN in base units"),
    MixedQuestion("value(span, V)", "yes", "units"),
    MixedQuestion("dimension(bay_width, D)", "yes", "units"),
    MixedQuestion("in_unit(bay_width, mm, V)", "yes", "units"),

    # -- arithmetic: comparison and multi-way solving ----------------------
    MixedQuestion("leq(beam_a_load, beam_a_rating)", "yes", "arithmetic",
                  "3200 <= 5000, after the unit conversion"),
    MixedQuestion("leq(beam_b_load, beam_b_rating)", "unknown", "arithmetic",
                  "7400 > 5000, so the inequality is not entailed"),
    MixedQuestion("gt(beam_b_load, beam_b_rating)", "yes", "arithmetic"),
    MixedQuestion("leq(beam_c_load, beam_c_rating)", "unknown", "arithmetic"),
    MixedQuestion("gt(beam_c_load, beam_c_rating)", "yes", "arithmetic"),
    MixedQuestion("leq(span, clearance)", "yes", "arithmetic",
                  "4200 mm against 5 m: the comparison only works in base units"),
    MixedQuestion("gt(span, clearance)", "unknown", "arithmetic"),
    MixedQuestion("value(labour_cost, V)", "yes", "arithmetic",
                  "solved backwards: 1450 - 900"),
    MixedQuestion("value(bay_area, V)", "yes", "arithmetic", "3 * 7"),
    MixedQuestion("lt(steel_cost, frame_cost)", "yes", "arithmetic"),
    MixedQuestion("geq(frame_cost, steel_cost)", "yes", "arithmetic"),
    MixedQuestion("gt(steel_cost, frame_cost)", "unknown", "arithmetic"),
    MixedQuestion("value(unknown_cost, V)", "unknown", "arithmetic",
                  "nothing constrains it, so no value is entailed"),

    # -- graph: reachability, distance, order ------------------------------
    MixedQuestion("reachable(cut, ship)", "yes", "graph"),
    MixedQuestion("reachable(ship, cut)", "unknown", "graph", "the process is one-way"),
    MixedQuestion("reachable(drill, paint)", "yes", "graph"),
    MixedQuestion("reachable(deburr, weld)", "unknown", "graph"),
    MixedQuestion("distance(cut, ship, N)", "yes", "graph"),
    MixedQuestion("distance(cut, paint, N)", "yes", "graph", "via deburr, not via weld"),
    MixedQuestion("path(cut, ship, R)", "yes", "graph"),
    MixedQuestion("path(drill, inspect, R)", "yes", "graph"),
    MixedQuestion("before(cut, weld)", "yes", "graph"),
    MixedQuestion("before(weld, cut)", "unknown", "graph"),
    MixedQuestion("before(drill, ship)", "yes", "graph"),
    MixedQuestion("cycle(cut)", "unknown", "graph", "an assembly order has no cycles"),

    # -- logic, and logic over the others ----------------------------------
    MixedQuestion("beam(beam_a)", "yes", "logic"),
    MixedQuestion("beam(gantry)", "unknown", "logic"),
    MixedQuestion("rated(beam_a, L, R)", "yes", "logic"),
    MixedQuestion("safe(beam_a)", "yes", "logic", "logic over arithmetic over units"),
    MixedQuestion("safe(beam_b)", "unknown", "logic", "the load exceeds the rating"),
    MixedQuestion("safe(beam_c)", "unknown", "logic", "1 kN against a 900 N rating"),
    MixedQuestion("shippable", "yes", "logic", "logic over graph"),
    MixedQuestion("fits", "yes", "logic", "logic over arithmetic over units"),
    MixedQuestion("signed_off", "yes", "logic", "three specialists in one proof"),
    MixedQuestion("scrapped", "unknown", "logic", "no rule defines it"),
]


def workshop(knowledge=None):
    """Build the scenario: a knowledge base and a workspace holding the facts."""
    from ..knowledge.base import KnowledgeBase
    from .blackboard import Workspace

    kb = knowledge if knowledge is not None else KnowledgeBase("workshop")
    kb.add_rules(WORKSHOP_RULES)
    kb.add_rules(WORKSHOP_FACTS)
    workspace = Workspace()
    _seed(workspace, kb)
    return kb, workspace


def _seed(workspace, knowledge) -> None:
    """Put the knowledge base's ground facts on the blackboard as inputs."""
    from ..inference.proof import FACT, ProofNode

    for record in knowledge.facts:
        workspace.post(
            record.atom,
            ProofNode(record.atom, FACT, confidence=None),
            source="given",
            confidence=record.confidence,
        )
