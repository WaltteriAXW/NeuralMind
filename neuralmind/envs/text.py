"""TextWorld, read as facts.

A second environment, chosen because it is unlike the first. MiniGrid is a
grid with a heading; TextWorld is rooms, containers, supporters and keys.
The action language, the planner and the replanning are the same code in
both, which is the claim this adapter exists to test -- an action model that
only ever worked on grids would be a grid solver with extra steps.

**What this does not do, stated plainly: it does not read the prose.**
TextWorld describes its goal in generated English several sentences long
("Hey, thanks for coming over to the TextWorld today..."), and the quest also
exists as a structured condition. This takes the structured one. Reading the
prose is P2.5's problem and pretending otherwise here would make the language
work look further along than it is. BabyAI is where the English-to-goal path
is actually exercised, and it is exercised on every episode.

What the two share is everything after the goal: the same planner, the same
causal links, the same replanning, and the same learner working out the
preconditions nobody wrote down -- which in TextWorld is usually that a
container has to be open before you can take what is inside it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.parser import parse_atom
from ..core.terms import Atom

__all__ = [
    "TextReader",
    "TextWorldGame",
    "ACTIONS",
    "NAIVE_ACTIONS",
    "textworld_available",
    "make_simple_game",
    "to_command",
]

#: Where the player is, named the same way in facts and in goals.
PLAYER = "player"

DIRECTIONS = {
    "north_of": "north",
    "south_of": "south",
    "east_of": "east",
    "west_of": "west",
}

#: TextWorld's ``open`` would collide with nothing, but ``open`` reads as a
#: verb and this is a state, so it is spelled out.
ACTIONS = """
action go(From, Dir, To):
    needs  at(player, From), exit(From, Dir, To), passable(From, To)
    causes at(player, To), -at(player, From)

action open_it(C, Room):
    needs  at(player, Room), at(C, Room), shut(C), not locked(C)
    causes opened(C), -shut(C)

% Unlocking leaves it *shut*, not open: TextWorld holds "locked" and
% "closed" as alternatives rather than as layers, so a box stops being
% locked and starts being closed in the same move.
action unlock_it(C, K, Room):
    needs  at(player, Room), at(C, Room), locked(C), carried(K), match(K, C)
    causes shut(C), -locked(C)

action take_from(O, C, Room):
    needs  at(player, Room), at(C, Room), inside(O, C), opened(C)
    causes carried(O), -inside(O, C)

action take_here(O, Room):
    needs  at(player, Room), at(O, Room)
    causes carried(O), -at(O, Room)

action take_off(O, S, Room):
    needs  at(player, Room), at(S, Room), on(O, S)
    causes carried(O), -on(O, S)

action put_on(O, S, Room):
    needs  at(player, Room), at(S, Room), carried(O)
    causes on(O, S), -carried(O)

action insert_into(O, C, Room):
    needs  at(player, Room), at(C, Room), carried(O), opened(C)
    causes inside(O, C), -carried(O)

action drop_it(O, Room):
    needs  at(player, Room), carried(O)
    causes at(O, Room), -carried(O)

action close_it(C, Room):
    needs  at(player, Room), at(C, Room), opened(C)
    causes shut(C), -opened(C)

action lock_it(C, K, Room):
    needs  at(player, Room), at(C, Room), shut(C), carried(K), match(K, C)
    causes locked(C), -shut(C)

% No ``edible`` guard: TextWorld emits that fact for some games and not
% others, so requiring it makes "eat the cashew" unplannable in exactly the
% games that ask for it. Left off, the game refuses anything inedible -- and
% being refused is what the learner is for.
action eat_it(O):
    needs  carried(O)
    causes eaten(O), -carried(O)
"""

#: The same library with one condition missing: that a container has to be
#: open before you can take what is inside it. Not a straw man -- it is the
#: single most natural thing to leave out, because "take the teapot from the
#: refrigerator" is one action in English and the opening is implied. The
#: mind is given this one and has to find out.
NAIVE_ACTIONS = ACTIONS.replace(
    """action take_from(O, C, Room):
    needs  at(player, Room), at(C, Room), inside(O, C), opened(C)""",
    """action take_from(O, C, Room):
    needs  at(player, Room), at(C, Room), inside(O, C)""",
)

#: How a planned step becomes something to type.
COMMANDS = {
    "go": lambda a, n: f"go {a[1]}",
    "open_it": lambda a, n: f"open {n(a[0])}",
    "unlock_it": lambda a, n: f"unlock {n(a[0])} with {n(a[1])}",
    "take_from": lambda a, n: f"take {n(a[0])} from {n(a[1])}",
    "take_here": lambda a, n: f"take {n(a[0])}",
    "take_off": lambda a, n: f"take {n(a[0])} from {n(a[1])}",
    "put_on": lambda a, n: f"put {n(a[0])} on {n(a[1])}",
    "insert_into": lambda a, n: f"insert {n(a[0])} into {n(a[1])}",
    "drop_it": lambda a, n: f"drop {n(a[0])}",
    "eat_it": lambda a, n: f"eat {n(a[0])}",
    "close_it": lambda a, n: f"close {n(a[0])}",
    "lock_it": lambda a, n: f"lock {n(a[0])} with {n(a[1])}",
}


def textworld_available() -> bool:
    try:
        import textworld  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class TextWorldGame:
    """One reading of a game: the facts, and what would count as winning."""

    facts: tuple[Atom, ...] = ()
    goal: tuple[Atom, ...] = ()
    #: Symbol -> the words to type for it.
    names: dict = field(default_factory=dict)
    objective: str = ""

    def describe(self) -> str:
        return (
            f"{len(self.facts)} fact(s); winning means "
            + ", ".join(str(a) for a in self.goal)
        )


class TextReader:
    """Turns TextWorld's propositions into atoms, and steps back into text."""

    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    # -- reading ------------------------------------------------------------

    def read(self, state, game=None) -> TextWorldGame:
        facts = self._facts(state["facts"])
        goal = self._goal(game) if game is not None else ()
        return TextWorldGame(
            facts=tuple(facts),
            goal=tuple(goal),
            names=dict(self.names),
            objective=state.get("objective", "") or "",
        )

    def _facts(self, propositions: Iterable) -> list[Atom]:
        atoms: list[Atom] = []
        exits: list[tuple[str, str, str]] = []
        for proposition in propositions:
            name = proposition.name
            args = [self._symbol(a) for a in proposition.arguments]

            if name in DIRECTIONS and len(args) == 2:
                # ``north_of(a, b)``: a is north of b, so from b you go north.
                exits.append((args[1], DIRECTIONS[name], args[0]))
                continue
            if name == "free" and len(args) == 2:
                atoms.append(parse_atom(f"passable({args[1]}, {args[0]})"))
                continue
            if name == "at" and args and args[0] == "p":
                atoms.append(parse_atom(f"at({PLAYER}, {args[1]})"))
                continue
            if name == "in" and len(args) == 2 and args[1] == "i":
                atoms.append(parse_atom(f"carried({args[0]})"))
                continue

            renamed = {
                "closed": "shut",
                "open": "opened",
                "in": "inside",
            }.get(name, name)
            atoms.append(
                parse_atom(f"{renamed}({', '.join(args)})" if args else renamed)
            )

        for room, direction, target in exits:
            atoms.append(parse_atom(f"exit({room}, {direction}, {target})"))
        return atoms

    def _goal(self, game) -> list[Atom]:
        """The win condition, as a conjunction of ground atoms."""
        wanted: list[Atom] = []
        for quest in game.quests:
            for event in quest.win_events:
                for proposition in event.condition.preconditions:
                    atom = self._goal_atom(proposition, game)
                    if atom is not None:
                        wanted.append(atom)
        # Duplicates are common across events and mean nothing extra.
        seen, unique = set(), []
        for atom in wanted:
            if str(atom) not in seen:
                seen.add(str(atom))
                unique.append(atom)
        return unique

    def _goal_atom(self, proposition, game) -> Optional[Atom]:
        name = proposition.name
        args = [self._quest_symbol(a, game) for a in proposition.arguments]
        if any(a is None for a in args):
            return None
        if name == "at" and args and args[0] == "p":
            return parse_atom(f"at({PLAYER}, {args[1]})")
        if name == "in" and len(args) == 2 and args[1] == "i":
            return parse_atom(f"carried({args[0]})")
        renamed = {"closed": "shut", "open": "opened", "in": "inside"}.get(name, name)
        return parse_atom(f"{renamed}({', '.join(args)})" if args else renamed)

    def _quest_symbol(self, variable, game) -> Optional[str]:
        """Quests name things by generation id; facts name them by word."""
        if variable.type == "P":
            return "p"
        if variable.type == "I":
            return "i"
        info = game.infos.get(variable.name)
        words = getattr(info, "name", None) if info is not None else None
        if not words:
            return _sanitise(variable.name)
        symbol = _sanitise(words)
        self.names[symbol] = words
        return symbol

    def _symbol(self, variable) -> str:
        if variable.type == "P":
            return "p"
        if variable.type == "I":
            return "i"
        words = getattr(variable, "name", str(variable))
        symbol = _sanitise(words)
        self.names.setdefault(symbol, words)
        return symbol

    # -- going the other way ------------------------------------------------

    def to_command(self, step) -> str:
        """Turn a planned step into something to type."""
        return to_command(step, self.names)


def to_command(step, names: dict) -> str:
    builder = COMMANDS.get(step.name)
    if builder is None:
        raise KeyError(
            f"{step.name} is not something to type; this world offers "
            + ", ".join(sorted(COMMANDS))
        )
    spell = lambda symbol: names.get(str(symbol), str(symbol))
    return builder([str(a) for a in step.action.args], spell)


def make_simple_game(
    seed: int = 7, rooms: int = 3, objects: int = 4, quest_length: int = 3
):
    """Generate and compile one 'simple' game, returning (path, game).

    Generated rather than downloaded: TextWorld's whole point is that it
    makes its own games, so there is nothing to fetch and the seed is the
    entire description of what was tested.
    """
    import tempfile

    from textworld import GameOptions
    from textworld.generator import compile_game, make_game

    options = GameOptions()
    options.path = tempfile.mkdtemp()
    options.seeds = seed
    options.nb_rooms = rooms
    options.nb_objects = objects
    options.quest_length = quest_length
    game = make_game(options)
    return compile_game(game, options), game


def _sanitise(words: str) -> str:
    """``dish-pit: r`` -> ``dish_pit``: a symbol the parser will accept."""
    text = str(words).split(":")[0].strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text).strip("_")
    return text or "thing"
