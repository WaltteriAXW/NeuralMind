"""MiniGrid and BabyAI, read as facts.

The adapter's whole job is translation, and the direction of the translation
is the design: the grid becomes *facts*, the instruction becomes a *rule*, and
what to do about them is the planner's business. Nothing in here decides
anything, which is why an adapter for another environment is a day's work
rather than a rewrite.

Two choices worth defending.

**Geometry arrives as facts, not as arithmetic.** ``ahead(2, 3, 0, 3, 3)``
says the cell east of (2,3) is (3,3). The action language has no arithmetic
and does not need any: where cells are is something about this grid, so the
adapter that can see the grid computes it once, and the planner reasons over
it like any other fact. A 6x6 room is about 140 of these, which clingo does
not notice.

**The partial view is the default.** BabyAI's agent sees a 7x7 cone, not the
room, and a planner given the whole grid is solving an easier problem than
the benchmark poses. Planning on what has actually been seen means the plan
is sometimes wrong -- the target turns out to be somewhere else, a wall
appears -- and that is the point: it is what the replanning in
:mod:`neuralmind.agency.execute` is for, and running it on the full grid
would never exercise it. ``view="full"`` is available for comparison, and the
difference between the two numbers is worth knowing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.parser import parse_atom
from ..core.terms import Atom
from .missions import GOAL_ATOM, Mission, MissionError, read_mission

__all__ = [
    "GridReader",
    "GridWorld",
    "ACTIONS",
    "STATE_RULES",
    "minigrid_available",
    "PLAN_TO_ACTION",
]


def minigrid_available() -> bool:
    try:
        import minigrid  # noqa: F401
        import gymnasium  # noqa: F401
    except ImportError:
        return False
    return True


#: MiniGrid's four headings, in its own order: 0 east, 1 south, 2 west, 3 north.
HEADINGS = ("east", "south", "west", "north")
DELTAS = {"east": (1, 0), "south": (0, 1), "west": (-1, 0), "north": (0, -1)}
LEFT = {"east": "north", "north": "west", "west": "south", "south": "east"}
RIGHT = {v: k for k, v in LEFT.items()}

#: Which environment action carries out each planned action.
PLAN_TO_ACTION = {
    "turn_left": "left",
    "turn_right": "right",
    "forward": "forward",
    "pick_up": "pickup",
    "put_down": "drop",
    "toggle": "toggle",
}

#: What the agent can do. Written out rather than generated, because this is
#: the part a person most needs to be able to read and disagree with.
ACTIONS = """
% Turning is free and always possible; it is the only thing that is.
action turn_left(D, E):
    needs  facing(D), turns_left(D, E)
    causes facing(E), -facing(D)

action turn_right(D, E):
    needs  facing(D), turns_right(D, E)
    causes facing(E), -facing(D)

% Moving needs somewhere to move to that is known, empty and not a wall.
action forward(X, Y, X2, Y2):
    needs  at(agent, X, Y), facing(D), ahead(X, Y, D, X2, Y2),
           seen(X2, Y2), not wall(X2, Y2), not blocked(X2, Y2)
    causes at(agent, X2, Y2), -at(agent, X, Y)

action pick_up(O, X2, Y2):
    needs  at(agent, X, Y), facing(D), ahead(X, Y, D, X2, Y2),
           object_at(O, X2, Y2), hand_empty
    causes carrying(O), -object_at(O, X2, Y2), -hand_empty, -blocked(X2, Y2)

action put_down(O, X2, Y2):
    needs  at(agent, X, Y), facing(D), ahead(X, Y, D, X2, Y2),
           carrying(O), seen(X2, Y2), not wall(X2, Y2), not blocked(X2, Y2)
    causes object_at(O, X2, Y2), hand_empty, blocked(X2, Y2), -carrying(O)

action toggle(O, X2, Y2):
    needs  at(agent, X, Y), facing(D), ahead(X, Y, D, X2, Y2),
           door_at(O, X2, Y2), door_shut(O), not door_locked(O)
    causes door_open(O), -door_shut(O), -blocked(X2, Y2)
"""

#: What follows from where things are, rather than from anything being done.
#: ``in_front`` is the one that matters: no action causes it, every move can
#: make it true, and it is what "go to the red key" actually asks for.
STATE_RULES = """
in_front(O) :- at(agent, X, Y), facing(D), ahead(X, Y, D, X2, Y2),
               object_at(O, X2, Y2).
in_front(O) :- at(agent, X, Y), facing(D), ahead(X, Y, D, X2, Y2),
               door_at(O, X2, Y2).
"""


@dataclass
class GridWorld:
    """One reading of the environment: the facts, the goal, and the geometry."""

    facts: tuple[Atom, ...] = ()
    mission: Optional[Mission] = None
    #: Cells seen at least once, which is what the agent may plan over.
    seen: frozenset = frozenset()
    agent: tuple[int, int] = (0, 0)
    facing: str = "east"
    carrying: Optional[str] = None
    #: Objects whose description matched the instruction, if any are in view.
    targets: tuple[str, ...] = ()

    @property
    def target_visible(self) -> bool:
        return bool(self.targets)

    def rules(self) -> str:
        rules = STATE_RULES
        if self.mission is not None:
            rules += "\n" + self.mission.rule() + "\n"
        return rules

    def describe(self) -> str:
        return (
            f"at {self.agent} facing {self.facing}, {len(self.seen)} cell(s) seen, "
            f"{'target in view' if self.target_visible else 'target not in view'}"
        )


class GridReader:
    """Turns a MiniGrid environment into facts, and back again.

    Keeps a memory of what has been seen, because an agent that forgets a cell
    the moment it leaves its view cannot plan a route through it. The memory
    is only ever *added* to from observation -- nothing is assumed about a
    cell that has never been looked at, which is what makes an unseen wall a
    surprise rather than a bug.
    """

    def __init__(self, view: str = "agent") -> None:
        if view not in ("agent", "full"):
            raise ValueError("view must be 'agent' or 'full'")
        self.view = view
        self.seen: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int]] = set()
        self.known: dict[tuple[int, int], tuple[str, str, str]] = {}
        self.names: dict[int, str] = {}
        self.mission: Optional[Mission] = None
        self.start: Optional[tuple[tuple[int, int], str]] = None
        #: Where each object was relative to the agent when it was first seen.
        #: Fixed once, because that is what the instruction meant: a ball that
        #: was on your right is still the ball that was on your right after
        #: you have turned around.
        self.places: dict[str, tuple[str, ...]] = {}

    # -- reading ------------------------------------------------------------

    def read(self, env, obs: Optional[dict] = None) -> GridWorld:
        """Take everything currently visible and fold it into what is known."""
        core = env.unwrapped
        agent = (int(core.agent_pos[0]), int(core.agent_pos[1]))
        facing = HEADINGS[int(core.agent_dir)]

        if obs is not None and self.mission is None:
            self.mission = read_mission(obs["mission"])
        if self.start is None:
            self.start = (agent, facing)

        for position in self._visible(core):
            self._look(core, position)

        facts: list[Atom] = []
        facts.append(parse_atom(f"at(agent, {agent[0]}, {agent[1]})"))
        facts.append(parse_atom(f"facing({facing})"))
        for heading, left in LEFT.items():
            facts.append(parse_atom(f"turns_left({heading}, {left})"))
        for heading, right in RIGHT.items():
            facts.append(parse_atom(f"turns_right({heading}, {right})"))

        for (x, y) in sorted(self.seen):
            facts.append(parse_atom(f"seen({x}, {y})"))
            for heading, (dx, dy) in DELTAS.items():
                nx, ny = x + dx, y + dy
                if 0 <= nx < core.grid.width and 0 <= ny < core.grid.height:
                    facts.append(
                        parse_atom(f"ahead({x}, {y}, {heading}, {nx}, {ny})")
                    )

        for (x, y) in sorted(self.walls):
            facts.append(parse_atom(f"wall({x}, {y})"))

        targets = []
        for (x, y), (name, kind, colour) in sorted(self.known.items()):
            facts.append(parse_atom(f"kind({name}, {kind})"))
            facts.append(parse_atom(f"colour({name}, {colour})"))
            if kind == "door":
                facts.append(parse_atom(f"door_at({name}, {x}, {y})"))
            else:
                facts.append(parse_atom(f"object_at({name}, {x}, {y})"))
                facts.append(parse_atom(f"blocked({x}, {y})"))
            for place in self.places.get(name, ()):
                facts.append(parse_atom(f"{place}({name})"))
            if self._matches(name, kind, colour):
                targets.append(name)

        facts += self._door_states(core)
        carrying = self._carrying(core)
        if carrying is None:
            facts.append(parse_atom("hand_empty"))
        else:
            facts.append(parse_atom(f"carrying({carrying})"))

        return GridWorld(
            facts=tuple(facts),
            mission=self.mission,
            seen=frozenset(self.seen),
            agent=agent,
            facing=facing,
            carrying=carrying,
            targets=tuple(targets),
        )

    # -- what is visible ----------------------------------------------------

    def _visible(self, core) -> list[tuple[int, int]]:
        if self.view == "full":
            return [
                (x, y)
                for x in range(core.grid.width)
                for y in range(core.grid.height)
            ]

        # The agent's cone, from MiniGrid's own visibility mask, mapped back
        # to absolute cells with MiniGrid's own transform. The view it hands
        # out is rotated to face the agent, so reading it as if it were
        # aligned with the grid works only when the agent happens to face
        # east -- which is exactly the sort of bug that looks like bad luck
        # on three seeds out of four.
        _, mask = core.gen_obs_grid()
        size = core.agent_view_size
        forward = core.dir_vec
        right = core.right_vec
        top_left = (
            core.agent_pos + forward * (size - 1) - right * (size // 2)
        )

        cells = []
        for j in range(size):
            for i in range(size):
                if not mask[i, j]:
                    continue
                x, y = top_left - (forward * j) + (right * i)
                if 0 <= x < core.grid.width and 0 <= y < core.grid.height:
                    cells.append((int(x), int(y)))
        return cells

    def _look(self, core, position: tuple[int, int]) -> None:
        x, y = position
        self.seen.add(position)
        cell = core.grid.get(x, y)
        if cell is None:
            self.walls.discard(position)
            self.known.pop(position, None)
            return
        if cell.type == "wall":
            self.walls.add(position)
            return
        name = self._name(cell)
        self.known[position] = (name, cell.type, cell.color)
        if name not in self.places:
            self.places[name] = self._place(position)

    def _name(self, cell) -> str:
        key = id(cell)
        if key not in self.names:
            self.names[key] = f"{cell.color}_{cell.type}_{len(self.names) + 1}"
        return self.names[key]

    def _place(self, position: tuple[int, int]) -> tuple[str, ...]:
        """Where something is relative to where the agent started.

        MiniGrid's own basis: the heading and its perpendicular, with the
        sign of each dot product naming the side. Anything on the axis is on
        neither side, which is why this returns a tuple and not one answer.
        """
        if self.start is None:
            return ()
        (ax, ay), facing = self.start
        fx, fy = DELTAS[facing]
        rx, ry = -fy, fx
        vx, vy = position[0] - ax, position[1] - ay
        along, across = vx * fx + vy * fy, vx * rx + vy * ry
        places = []
        if along > 0:
            places.append("ahead_of_start")
        if along < 0:
            places.append("behind_start")
        if across < 0:
            places.append("left_of_start")
        if across > 0:
            places.append("right_of_start")
        return tuple(places)

    def _door_states(self, core) -> list[Atom]:
        facts = []
        for (x, y), (name, kind, _) in self.known.items():
            if kind != "door":
                continue
            cell = core.grid.get(x, y)
            if cell is None:
                continue
            if cell.is_open:
                facts.append(parse_atom(f"door_open({name})"))
            else:
                facts.append(parse_atom(f"door_shut({name})"))
                facts.append(parse_atom(f"blocked({x}, {y})"))
            if cell.is_locked:
                facts.append(parse_atom(f"door_locked({name})"))
        return facts

    def _carrying(self, core) -> Optional[str]:
        cell = core.carrying
        return self._name(cell) if cell is not None else None

    def _matches(self, name: str, kind: str, colour: str) -> bool:
        if self.mission is None:
            return False
        if self.mission.kind and self.mission.kind != kind:
            return False
        if self.mission.colour and self.mission.colour != colour:
            return False
        if self.mission.place:
            from .missions import PLACES

            if PLACES[self.mission.place] not in self.places.get(name, ()):
                return False
        return True

    # -- going the other way ------------------------------------------------

    def to_env_action(self, env, step) -> int:
        """Turn a planned step into the environment's action number."""
        name = PLAN_TO_ACTION.get(step.name)
        if name is None:
            raise KeyError(
                f"{step.name} is not something this environment can do; "
                f"it offers {', '.join(sorted(PLAN_TO_ACTION))}"
            )
        return int(getattr(env.unwrapped.actions, name))
