"""A domain no pack covers, as an observation log.

The roadmap's test case for the builder is "a simulated pump system": a host
whose vocabulary no facet profile knows, observed rather than described. This
is that simulator, and the shape of what it emits is the whole point --

    {"tick": 3, "pump_a": "running", "valve_b": "open",
     "flow": "12 l/min", "pressure": "2.4 bar", "alarm": false,
     "commands": ["start_pump_a", "stop_pump_a", "open_valve_b", ...],
     "action": "open_valve_b"}

-- because everything the builder concludes has to come out of exactly this
and nothing else. No schema, no documentation, no type declarations. Fields
with a small fixed set of values, fields with numbers and units, fields that
change between ticks and fields that do not, a list of commands the host says
it accepts, and which one was taken. That is what a real host can be expected
to provide, and it is enough.

The physics is deliberately simple but not trivial: flow depends on the pump
running *and* the valve being open, pressure rises when the pump pushes
against a closed valve, and the alarm is that over-pressure condition. So
there is a rule to find (``alarm`` is not any single field), an action whose
effect is conditional (``start_pump_a`` changes flow only when the valve is
open), and a quantity whose unit matters.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

__all__ = [
    "PumpStation",
    "pump_station_log",
    "HELD_OUT",
    "COMMANDS",
    "write_log",
    "read_log",
    "Vault",
    "vault_log",
    "VAULT_COMMANDS",
    "VAULT_GOAL",
]

#: What the host says it accepts. The builder gets these as strings, with no
#: hint of what any of them do.
COMMANDS = (
    "start_pump_a",
    "stop_pump_a",
    "open_valve_b",
    "close_valve_b",
    "reset_alarm",
)

#: Over this, the alarm trips. Nothing tells the builder the number.
ALARM_PRESSURE = 3.0


class PumpStation:
    """The world behind the log. Never shown to the builder."""

    def __init__(self, seed: int = 0) -> None:
        self.random = random.Random(seed)
        self.pump = "idle"
        self.valve = "closed"
        self.pressure = 2.0
        self.flow = 0
        self.alarm = False
        self.tick = 0

    # -- what a watcher would see ------------------------------------------

    def snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "pump_a": self.pump,
            "valve_b": self.valve,
            "flow": f"{self.flow} l/min",
            "pressure": f"{self.pressure:.1f} bar",
            "alarm": self.alarm,
            "commands": list(COMMANDS),
        }

    # -- the physics --------------------------------------------------------

    def apply(self, command: str) -> None:
        if command == "start_pump_a":
            self.pump = "running"
        elif command == "stop_pump_a":
            self.pump = "idle"
        elif command == "open_valve_b":
            self.valve = "open"
        elif command == "close_valve_b":
            self.valve = "closed"
        elif command == "reset_alarm":
            # Only clears if the cause has gone, which is a precondition the
            # builder has to discover from being refused.
            if self.pressure <= ALARM_PRESSURE:
                self.alarm = False
        self._settle()
        self.tick += 1

    def _settle(self) -> None:
        running, open_valve = self.pump == "running", self.valve == "open"
        if running and open_valve:
            self.flow = 12
            self.pressure = max(2.0, self.pressure - 0.4)
        elif running and not open_valve:
            self.flow = 0
            self.pressure = min(3.6, self.pressure + 0.4)
        else:
            self.flow = 0
            self.pressure = max(2.0, self.pressure - 0.2)
        self.pressure = round(self.pressure, 1)
        if self.pressure > ALARM_PRESSURE:
            self.alarm = True

    def possible(self) -> list[str]:
        return list(COMMANDS)


def pump_station_log(
    steps: int = 120, seed: int = 0, explore: float = 1.0
) -> list[dict]:
    """Watch a station being operated, and write down what was seen.

    Commands are chosen at random on purpose. A log of an *expert* operating
    the station would never show ``start_pump_a`` against a closed valve, and
    then the builder could not learn that it does nothing -- the cases that
    teach an action model are exactly the ones a competent operator avoids.
    """
    station = PumpStation(seed)
    chooser = random.Random(seed + 1)
    log = []
    for _ in range(steps):
        before = station.snapshot()
        command = chooser.choice(COMMANDS)
        station.apply(command)
        record = dict(before)
        record["action"] = command
        log.append(record)
    log.append(station.snapshot())
    return log


#: Questions the finished pack has to answer, held back from everything the
#: builder is shown. Structured rather than English on purpose: the pack is a
#: structure, and wrapping the check in a sentence parser would be measuring
#: the parser.
#:
#: Each is ``(query, expected)``. The negatives matter as much as the
#: positives -- a builder that calls everything a quantity scores well on a
#: set with no negatives in it.
HELD_OUT = (
    (("is_a", "flow", "quantity"), True),
    (("is_a", "pressure", "quantity"), True),
    (("is_a", "pump_a", "state"), True),
    (("is_a", "valve_b", "state"), True),
    (("is_a", "alarm", "flag"), True),
    (("is_a", "tick", "counter"), True),
    (("is_a", "tick", "quantity"), False),
    (("is_a", "alarm", "quantity"), False),
    (("is_a", "commands", "state"), False),
    (("unit", "pressure", "bar"), True),
    (("unit", "flow", "l/min"), True),
    (("unit", "pressure", "psi"), False),
    (("values", "pump_a", ("idle", "running")), True),
    (("values", "valve_b", ("closed", "open")), True),
    (("values", "pump_a", ("on", "off")), False),
    (("effect", "start_pump_a", "pump_a", "running"), True),
    (("effect", "stop_pump_a", "pump_a", "idle"), True),
    (("effect", "open_valve_b", "valve_b", "open"), True),
    (("effect", "close_valve_b", "valve_b", "closed"), True),
    (("effect", "start_pump_a", "valve_b", "open"), False),
    (("effect", "reset_alarm", "pump_a", "running"), False),
    (("conditional", "open_valve_b", "flow", "pump_a"), True),
    (("conditional", "start_pump_a", "flow", "valve_b"), True),
)


def write_log(log: Sequence[dict], path: Path) -> Path:
    """One JSON object per line, because that is what a host can append to."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in log:
            handle.write(json.dumps(record) + "\n")
    return path


def read_log(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# -- a small game, for the second acceptance test --------------------------


#: What the vault game accepts. Again, just strings.
VAULT_COMMANDS = (
    "take_key",
    "drop_key",
    "unlock_door",
    "open_door",
    "go_in",
    "light_lamp",
)


class Vault:
    """A room, a key, a locked door, and a lamp that matters.

    Deliberately not a grid: the point of the second acceptance test is that
    the builder is not a grid reader. Flat fields, a handful of commands, and
    a chain three deep -- take the key, unlock the door, open it, go in --
    with one condition that has to be discovered (you cannot see to unlock
    the door in the dark).
    """

    def __init__(self, seed: int = 0) -> None:
        self.random = random.Random(seed)
        self.holding = "nothing"
        self.door = "locked"
        self.lamp = "off"
        self.inside = False
        self.tick = 0

    def snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "holding": self.holding,
            "door": self.door,
            "lamp": self.lamp,
            "inside": self.inside,
            "commands": list(VAULT_COMMANDS),
        }

    def apply(self, command: str) -> None:
        if command == "take_key" and self.holding == "nothing":
            self.holding = "key"
        elif command == "drop_key" and self.holding == "key":
            self.holding = "nothing"
        elif command == "light_lamp":
            self.lamp = "on"
        elif command == "unlock_door":
            # Needs the key *and* the light: two conditions, one command.
            if self.holding == "key" and self.lamp == "on" and self.door == "locked":
                self.door = "shut"
        elif command == "open_door" and self.door == "shut":
            self.door = "open"
        elif command == "go_in" and self.door == "open":
            self.inside = True
        self.tick += 1

    def possible(self) -> list[str]:
        return list(VAULT_COMMANDS)


def vault_log(steps: int = 400, seed: int = 0) -> list[dict]:
    """Watch somebody press buttons in the vault room.

    Randomly again, and for the same reason: a log of somebody who knows the
    trick never shows ``unlock_door`` failing in the dark, and that failure
    is the only thing that teaches the condition.
    """
    chooser = random.Random(seed + 3)
    log: list[dict] = []
    game = Vault(seed)
    tick = 0
    for step in range(steps):
        before = game.snapshot()
        before["tick"] = tick
        command = chooser.choice(VAULT_COMMANDS)
        game.apply(command)
        record = dict(before)
        record["action"] = command
        log.append(record)
        tick += 1

        if game.inside:
            # Record the true state that go_in produced *before* starting a
            # fresh room. Resetting between a command and its result would
            # make the log say go_in put the key down and turned the lamp
            # off -- a fixture lying to the thing it is meant to test.
            done = game.snapshot()
            done["tick"] = tick
            log.append(done)
            tick += 1
            game = Vault(seed + step)
    final = game.snapshot()
    final["tick"] = tick
    log.append(final)
    return log


#: What a drafted vault pack has to get right to be playable.
VAULT_GOAL = "inside(true)"
