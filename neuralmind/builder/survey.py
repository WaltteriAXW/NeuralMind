"""What is in this log, worked out from its shape.

The builder is handed records and nothing else: no schema, no types, no
documentation. Everything below is inferred from how values behave across the
log, and every conclusion carries the evidence that produced it, because a
draft schema nobody can argue with is a draft schema nobody can correct.

The readings, and what each is read from:

``quantity``
    values that are a number and a unit (``"2.4 bar"``), where the unit is one
    a unit library recognises. The unit is part of the finding, not a guess.
``state``
    a small, closed set of string values (``idle``/``running``). Small and
    closed matters: two values seen once each is not an enumeration, it is two
    values.
``flag``
    booleans.
``counter``
    an integer that only ever goes up, by one, in step with the record index.
    Distinguished from a quantity on purpose: a tick is not a measurement, and
    a schema that calls it one invites questions about its units.
``identifier``
    a value unique to each record.
``commands``
    a list of strings that turn up as the log's ``action`` values. That is what
    makes them commands rather than a list of words.
``text``
    anything left that is a string.

Each reading is also either **changing** or **fixed**: a field that never
differs between one record and the next is an attribute of the thing being
watched, not part of its state, and the difference decides whether the
planner should carry it forward.

What the roadmap wants here and this does not do is clustering unfamiliar
vocabulary with sentence embeddings. Those models are not reachable from this
machine, and inventing a worse clusterer to fill the gap would be worse than
saying so: fields are grouped by *behaviour* -- which move together, which
share a value vocabulary -- which is weaker at naming concepts and much
easier to check.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

__all__ = [
    "Survey",
    "Finding",
    "survey",
    "QUANTITY",
    "STATE",
    "FLAG",
    "COUNTER",
    "IDENTIFIER",
    "COMMANDS",
    "TEXT",
    "KINDS",
]

QUANTITY = "quantity"
STATE = "state"
FLAG = "flag"
COUNTER = "counter"
IDENTIFIER = "identifier"
COMMANDS = "commands"
TEXT = "text"

KINDS = (QUANTITY, STATE, FLAG, COUNTER, IDENTIFIER, COMMANDS, TEXT)

#: Over this many distinct values, a string field is not an enumeration.
MAX_STATE_VALUES = 8
#: Under this many sightings each, a value set is too thin to call closed.
MIN_VALUE_SIGHTINGS = 2

_NUMBER_UNIT = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z°%][A-Za-z°%/·\^\-0-9]*)\s*$"
)


@dataclass
class Finding:
    """One field, what it appears to be, and what says so."""

    name: str
    kind: str
    #: Set for a quantity.
    unit: Optional[str] = None
    #: Set for a state: the values seen, in first-seen order.
    values: tuple[str, ...] = ()
    #: Whether it ever differs between consecutive records.
    changing: bool = False
    #: How many records carried this field.
    seen: int = 0
    #: How many times it changed.
    changes: int = 0
    #: Why this reading rather than another, in one line.
    because: str = ""
    #: True when the facets already explain this field and the builder should
    #: leave it alone.
    explained_by: Optional[str] = None

    @property
    def is_new(self) -> bool:
        return self.explained_by is None

    def describe(self) -> str:
        what = self.kind
        if self.unit:
            what = f"{self.kind} in {self.unit}"
        if self.values:
            what = f"{self.kind} ({', '.join(self.values)})"
        state = "changes" if self.changing else "fixed"
        covered = f" — already covered by {self.explained_by}" if self.explained_by else ""
        return f"{self.name}: {what}, {state}{covered}"

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "kind": self.kind,
            "changing": self.changing,
            "seen": self.seen,
            "changes": self.changes,
            "because": self.because,
        }
        if self.unit:
            payload["unit"] = self.unit
        if self.values:
            payload["values"] = list(self.values)
        if self.explained_by:
            payload["explained_by"] = self.explained_by
        return payload

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Survey:
    """Everything the log says about itself."""

    findings: list[Finding] = field(default_factory=list)
    records: int = 0
    #: The field naming what was done, if the log has one.
    action_field: Optional[str] = None
    #: Commands the host says it accepts.
    commands: tuple[str, ...] = ()
    #: Fields that move together, as sets. Weak evidence, offered as such.
    move_together: tuple[frozenset, ...] = ()

    def of(self, name: str) -> Optional[Finding]:
        return next((f for f in self.findings if f.name == name), None)

    def by_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    @property
    def new(self) -> list[Finding]:
        """What the facets did not already explain -- the builder's work."""
        return [f for f in self.findings if f.is_new]

    @property
    def fluents(self) -> list[Finding]:
        """What the planner should carry forward and actions should change.

        Not counters, which go up on their own, and not the command list,
        which describes the host rather than its state.
        """
        return [
            f for f in self.new
            if f.changing and f.kind not in (COUNTER, COMMANDS)
        ]

    @property
    def attributes(self) -> list[Finding]:
        return [f for f in self.new if not f.changing and f.kind != COMMANDS]

    def describe(self) -> str:
        lines = [
            f"{self.records} record(s), {len(self.findings)} field(s), "
            f"{len(self.new)} the facets do not explain"
        ]
        for finding in self.findings:
            lines.append(f"  {finding.describe()}")
        if self.commands:
            lines.append(f"  commands offered: {', '.join(self.commands)}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "records": self.records,
            "fields": [f.to_dict() for f in self.findings],
            "commands": list(self.commands),
            "action_field": self.action_field,
            "move_together": [sorted(group) for group in self.move_together],
        }

    def __str__(self) -> str:
        return self.describe()


def survey(
    log: Sequence[dict],
    explained: Optional[dict] = None,
    unit_check=None,
) -> Survey:
    """Read a log into findings.

    ``explained`` maps a field name to the facet that already covers it, so
    the builder works only on what is left -- the roadmap's first step, and
    the one that keeps a new pack small.
    """
    explained = explained or {}
    if not log:
        return Survey()

    values: dict[str, list] = defaultdict(list)
    for record in log:
        for name, value in record.items():
            values[name].append(value)

    action_field = "action" if "action" in values else None
    taken = {str(v) for v in values.get(action_field, [])} if action_field else set()

    findings = []
    for name, seen in values.items():
        if name == action_field:
            continue
        finding = _read_field(name, seen, taken, unit_check)
        finding.explained_by = explained.get(name)
        finding.changes = _count_changes(log, name)
        finding.changing = finding.changes > 0
        findings.append(finding)

    commands: tuple[str, ...] = ()
    for finding in findings:
        if finding.kind == COMMANDS:
            commands = finding.values
            break
    if not commands and taken:
        commands = tuple(sorted(taken))

    return Survey(
        findings=sorted(findings, key=lambda f: f.name),
        records=len(log),
        action_field=action_field,
        commands=commands,
        move_together=_move_together(log, [f.name for f in findings]),
    )


# -- reading one field ------------------------------------------------------


def _read_field(name: str, seen: Sequence, taken: set, unit_check) -> Finding:
    present = [v for v in seen if v is not None]
    total = len(present)

    if present and all(isinstance(v, bool) for v in present):
        return Finding(name, FLAG, seen=total, because="every value is true or false")

    if present and all(isinstance(v, list) for v in present):
        flat = tuple(dict.fromkeys(str(x) for group in present for x in group))
        if taken and set(flat) >= taken:
            return Finding(
                name, COMMANDS, values=flat, seen=total,
                because=f"a list whose entries appear as actions ({len(taken)} seen taken)",
            )
        return Finding(
            name, TEXT, values=flat[:MAX_STATE_VALUES], seen=total,
            because="a list of values, none of which was ever acted on",
        )

    if present and all(isinstance(v, int) and not isinstance(v, bool) for v in present):
        if _is_counter(present):
            return Finding(
                name, COUNTER, seen=total,
                because="whole numbers going up by one, in step with the records",
            )
        return Finding(
            name, QUANTITY, seen=total,
            because="whole numbers with no unit given",
        )

    if present and all(isinstance(v, (int, float)) for v in present):
        return Finding(name, QUANTITY, seen=total, because="numbers with no unit given")

    texts = [str(v) for v in present]
    units = [_NUMBER_UNIT.match(t) for t in texts]
    if texts and all(units):
        unit = Counter(m.group(2) for m in units if m).most_common(1)[0][0]
        known = _unit_known(unit, unit_check)
        return Finding(
            name, QUANTITY, unit=unit, seen=total,
            because=(
                f"every value is a number and a unit; {unit!r} is "
                + ("a unit the unit library knows" if known else "not a unit I know")
            ),
        )

    distinct = tuple(dict.fromkeys(texts))
    counts = Counter(texts)
    closed = (
        1 < len(distinct) <= MAX_STATE_VALUES
        and min(counts.values()) >= MIN_VALUE_SIGHTINGS
    )
    if closed:
        return Finding(
            name, STATE, values=distinct, seen=total,
            because=(
                f"{len(distinct)} values, each seen at least "
                f"{min(counts.values())} time(s)"
            ),
        )
    if len(distinct) == total and total > 1:
        return Finding(
            name, IDENTIFIER, seen=total, because="a different value in every record"
        )
    return Finding(
        name, TEXT, values=distinct[:MAX_STATE_VALUES], seen=total,
        because=f"{len(distinct)} distinct value(s), too many or too few to be a state",
    )


def _is_counter(values: Sequence[int]) -> bool:
    if len(values) < 3:
        return False
    return all(b - a == 1 for a, b in zip(values, values[1:]))


def _unit_known(unit: str, unit_check=None) -> bool:
    """Ask a unit library, when there is one. Never invent the answer."""
    if unit_check is not None:
        return bool(unit_check(unit))
    try:
        from ..workspace.specialists.units import pint_available
    except ImportError:
        return False
    if not pint_available():
        return False
    import pint

    try:
        pint.UnitRegistry().Unit(unit)
    except Exception:
        return False
    return True


#: A stand-in for "there was no previous record", distinct from any value a
#: log could contain -- including ``None``, which a host may well send.
_UNSET = object()


def _count_changes(log: Sequence[dict], name: str) -> int:
    changes = 0
    previous = _UNSET
    for record in log:
        value = record.get(name)
        if previous is not _UNSET and previous != value:
            changes += 1
        previous = value
    return changes


def _move_together(log: Sequence[dict], names: Sequence[str]) -> tuple[frozenset, ...]:
    """Fields that changed on the same records.

    Offered as a hint and nothing more. Two fields that always move together
    might be one concept or might be two things driven by a third, and the
    log cannot tell those apart -- so this groups them and lets a person say.
    """
    moved: dict[str, set[int]] = {name: set() for name in names}
    for index in range(1, len(log)):
        for name in names:
            if log[index].get(name) != log[index - 1].get(name):
                moved[name].add(index)

    groups: dict[frozenset, set[str]] = defaultdict(set)
    for name, where in moved.items():
        if where:
            groups[frozenset(where)].add(name)
    return tuple(
        frozenset(names_) for names_ in groups.values() if len(names_) > 1
    )
