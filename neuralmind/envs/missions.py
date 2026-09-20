"""Reading a BabyAI instruction into a goal.

BabyAI's instructions are *generated*, from a small closed grammar, which is
why reading them with a matching grammar is reading rather than guessing. The
whole language is a verb, an optional colour and type, and an optional place:

    go to a red key
    pick up the box behind you
    open a door on your left
    put a ball next to the blue key

An instruction becomes a **rule**, not a target. "A red key" means any red
key, and a rule says that directly::

    reached :- in_front(O), kind(O, key), colour(O, red).

A ground goal atom would have had to pick one of the red keys and commit to
it, which is both a worse plan and a worse description of what was asked.
The goal the planner is given is then just ``reached``, and which object
satisfies it falls out of the proof.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["Mission", "read_mission", "MissionError", "GOAL_ATOM"]

#: The atom the planner is asked for. Zero-arity, because "the instruction
#: was carried out" is not about any particular object.
GOAL_ATOM = "reached"

COLOURS = ("red", "green", "blue", "purple", "yellow", "grey")
KINDS = ("key", "ball", "box", "door")

#: Verb -> the fluent that means it has been done to an object.
VERBS = {
    "go to": "in_front",
    "pick up": "carrying",
    "open": "door_open",
}

#: Where the object is, relative to the agent when the instruction was given.
PLACES = {
    "in front of you": "ahead_of_start",
    "behind you": "behind_start",
    "on your left": "left_of_start",
    "on your right": "right_of_start",
}


class MissionError(ValueError):
    """An instruction outside the grammar. Never guessed at."""


@dataclass(frozen=True)
class Mission:
    """One instruction, read."""

    text: str
    verb: str
    kind: Optional[str] = None
    colour: Optional[str] = None
    place: Optional[str] = None
    #: A second object, for "put X next to Y".
    second: Optional["Mission"] = None

    @property
    def fluent(self) -> str:
        return VERBS[self.verb]

    def rule(self) -> str:
        """The instruction as a rule the planner can be given a goal from."""
        body = [f"{self.fluent}(O)"]
        if self.kind:
            body.append(f"kind(O, {self.kind})")
        if self.colour:
            body.append(f"colour(O, {self.colour})")
        if self.place:
            body.append(f"{PLACES[self.place]}(O)")
        return f"{GOAL_ATOM} :- {', '.join(body)}."

    def describe(self) -> str:
        what = " ".join(p for p in (self.colour, self.kind) if p) or "something"
        where = f" {self.place}" if self.place else ""
        return f"{self.verb} a {what}{where}"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "verb": self.verb,
            "kind": self.kind,
            "colour": self.colour,
            "place": self.place,
            "rule": self.rule(),
        }

    def __str__(self) -> str:
        return self.describe()


_ARTICLE = re.compile(r"^(a|an|the)\s+")


def read_mission(text: str) -> Mission:
    """Read an instruction, or say which part was not understood.

    Refusing is the point. An instruction half-read is a goal that looks
    reasonable and is not what was asked, and there is no way to notice that
    downstream -- the plan will be perfectly good at doing the wrong thing.
    """
    original = text
    remainder = text.strip().lower()

    verb = next((v for v in sorted(VERBS, key=len, reverse=True)
                 if remainder.startswith(v)), None)
    if verb is None:
        raise MissionError(
            f"{original!r}: no verb I know. I can do: "
            + ", ".join(sorted(VERBS))
        )
    remainder = remainder[len(verb):].strip()

    place = next((p for p in PLACES if remainder.endswith(p)), None)
    if place is not None:
        remainder = remainder[: -len(place)].strip()

    remainder = _ARTICLE.sub("", remainder).strip()

    colour = None
    words = remainder.split()
    if words and words[0] in COLOURS:
        colour, words = words[0], words[1:]

    kind = None
    if words and words[0] in KINDS:
        kind, words = words[0], words[1:]

    if words:
        raise MissionError(
            f"{original!r}: I got as far as "
            f"{Mission(original, verb, kind, colour, place).describe()!r} and "
            f"then {' '.join(words)!r} was left over"
        )
    if kind is None and colour is None and place is None:
        raise MissionError(f"{original!r}: nothing says which object")

    return Mission(original, verb, kind, colour, place)
