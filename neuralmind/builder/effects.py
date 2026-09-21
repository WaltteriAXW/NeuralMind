"""What each command does, worked out from watching it done.

P2.7's learner narrows an action model that already exists: it finds the
precondition somebody left out. This is the other half -- there is no model
at all, only a log saying which command was sent and what the world looked
like either side of it.

The method is counting, and the counting is reported rather than hidden:

    open_valve_b set valve_b to open in 23 of 23 cases

Three outcomes, kept apart because they mean different things to whoever has
to confirm them:

**Always.** The field went to the same value every single time. Proposed as a
plain effect.

**Sometimes, and here is when.** The field went to that value in some cases
and not others, and a condition present in exactly the cases where it did
separates them. Proposed as a conditional effect with the condition named --
``start_pump_a`` sets ``flow`` to 12 l/min, but only when the valve is open,
and no log of an expert operator would ever show that.

**Sometimes, and I cannot say when.** The field moved, no condition in the
log separates the cases, and so nothing is proposed -- it becomes a question.
Pressure is the honest example: it drifts by an amount no single command
determines, and a builder that proposed ``sets pressure to 2.4 bar`` because
that happened most often would be making something up.

Nothing here decides anything. Every proposal carries its counts and the
transitions behind them, and a person says yes or no.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from .survey import COUNTER, FLAG, QUANTITY, STATE, Finding, Survey

__all__ = ["Transition", "Effect", "ActionDraft", "learn_effects", "MIN_CASES"]

#: Fewer cases than this and a pattern is a coincidence, not an effect.
MIN_CASES = 3

#: A quantity that takes more distinct values than this in the log is not
#: something a command *sets*. Flow is 0 or 12 and "sets flow to 12" is a
#: real finding; pressure drifts through a dozen values and "sets pressure to
#: 3.2 bar" is the log's regularity being mistaken for the world's. The
#: distinction cannot be made from the type -- both are quantities in the
#: same units notation -- so it is made from behaviour.
MAX_SETTABLE_VALUES = 4


@dataclass(frozen=True)
class Transition:
    """One command, and the world either side of it."""

    action: str
    before: dict
    after: dict

    def changed(self, name: str) -> bool:
        return self.before.get(name) != self.after.get(name)


@dataclass(frozen=True)
class Effect:
    """A proposed effect, with the arithmetic that produced it."""

    action: str
    field: str
    to_value: str
    happened: int
    cases: int
    #: What picks out the cases where it happened: ``(field, value)``, or a
    #: pair of those when one condition was not enough.
    condition: Optional[tuple] = None

    @property
    def always(self) -> bool:
        return self.happened == self.cases

    @property
    def conditions(self) -> tuple:
        """The condition as a tuple of ``(field, value)`` pairs, always."""
        if self.condition is None:
            return ()
        if isinstance(self.condition[0], tuple):
            return tuple(self.condition)
        return (tuple(self.condition),)

    def when(self) -> str:
        return " and ".join(f"{name} was {value}" for name, value in self.conditions)

    @property
    def share(self) -> float:
        return self.happened / self.cases if self.cases else 0.0

    def describe(self) -> str:
        if self.condition is not None:
            return (
                f"{self.action} set {self.field} to {self.to_value} in "
                f"{self.happened} of the {self.cases} case(s) where "
                f"{self.when()} and it was not already"
            )
        return (
            f"{self.action} set {self.field} to {self.to_value} in "
            f"{self.happened} of the {self.cases} case(s) where it was not "
            "already"
        )

    def question(self) -> str:
        """The roadmap's phrasing: state the counts, then ask."""
        return f"{self.describe()}. Is that its effect?"

    def to_dict(self) -> dict:
        payload = {
            "action": self.action,
            "field": self.field,
            "to": self.to_value,
            "happened": self.happened,
            "cases": self.cases,
            "always": self.always,
        }
        if self.condition is not None:
            payload["when"] = [
                {"field": name, "is": value} for name, value in self.conditions
            ]
        return payload

    def __str__(self) -> str:
        return self.describe()


@dataclass
class ActionDraft:
    """One command, everything proposed about it, and what stayed unclear."""

    name: str
    cases: int = 0
    effects: list[Effect] = field(default_factory=list)
    #: Fields that moved but that nothing explains. These become questions.
    unexplained: list[str] = field(default_factory=list)
    #: Fields whose movement a learned rule accounts for, so not this action's.
    explained_by_rule: list[str] = field(default_factory=list)
    #: Cases where nothing at all changed -- evidence of a precondition.
    refused: int = 0

    @property
    def modelled(self) -> bool:
        return bool(self.effects)

    def describe(self) -> str:
        lines = [f"{self.name} ({self.cases} case(s) seen)"]
        for effect in self.effects:
            lines.append(f"    {effect.describe()}")
        for name in self.unexplained:
            lines.append(f"    {name} moved, but nothing in the log says when")
        for name in self.explained_by_rule:
            lines.append(f"    {name} moved, but a learned rule accounts for it")
        if self.refused:
            lines.append(
                f"    {self.refused} case(s) changed nothing — something "
                "stops it that is not modelled yet"
            )
        if not self.effects and not self.unexplained:
            lines.append("    changed nothing, ever")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cases": self.cases,
            "effects": [e.to_dict() for e in self.effects],
            "unexplained": list(self.unexplained),
            "explained_by_rule": list(self.explained_by_rule),
            "refused": self.refused,
        }

    def __str__(self) -> str:
        return self.describe()


def transitions(log: Sequence[dict], action_field: str = "action") -> list[Transition]:
    """Pair each record carrying an action with the one that followed it."""
    found = []
    for before, after in zip(log, log[1:]):
        command = before.get(action_field)
        if command is not None:
            found.append(Transition(str(command), before, after))
    return found


def learn_effects(
    log: Sequence[dict],
    reading: Survey,
    min_cases: int = MIN_CASES,
    accounted_for: Sequence[str] = (),
    derived: Sequence[str] = (),
) -> list[ActionDraft]:
    """Work out what each command does. One draft per command seen.

    ``accounted_for`` names fields a learned rule already explains. They still
    move, and they are still not this action's doing: the alarm going off when
    a command pushed the pressure up is the *rule* firing, and listing it as
    an unexplained effect of five different commands would bury the one
    explanation under five false mysteries.
    """
    moves = transitions(log, reading.action_field or "action")
    # A derived concept is not something an action causes -- it follows from
    # what the action caused. Proposing "stop_pump_a sets high_pressure to
    # false" states the consequence and hides the cause.
    settable = {
        f.name for f in reading.fluents
        if f.kind != QUANTITY or len(_values_of(log, f.name)) <= MAX_SETTABLE_VALUES
    }
    watched = [
        f.name for f in reading.fluents
        if f.name not in set(derived) and f.name in settable
    ]
    drifting = [
        f.name for f in reading.fluents
        if f.name not in set(derived) and f.name not in settable
    ]
    # Conditions are searched over discrete fields only. A continuous
    # quantity has a different value in almost every record, so "only when
    # pressure is 3.2 bar" will always be available and will almost always
    # be an accident.
    discrete = [
        f.name for f in reading.fluents
        if f.kind in (STATE, FLAG) and f.name not in set(derived)
    ] + [name for name in derived]
    by_action: dict[str, list[Transition]] = defaultdict(list)
    for move in moves:
        by_action[move.action].append(move)

    drafts = []
    for command in sorted(by_action):
        cases = by_action[command]
        draft = ActionDraft(name=command, cases=len(cases))
        draft.refused = sum(
            1 for c in cases if not any(c.changed(n) for n in watched)
        )
        draft.unexplained = [
            name for name in drifting if any(c.changed(name) for c in cases)
        ]
        if len(cases) < min_cases:
            draft.unexplained += [
                n for n in watched if any(c.changed(n) for c in cases)
            ]
            drafts.append(draft)
            continue

        for name in watched:
            effect = _effect_for(command, name, cases, by_action, discrete)
            if effect is not None:
                draft.effects.append(effect)
            elif any(c.changed(name) for c in cases):
                if name in accounted_for:
                    draft.explained_by_rule.append(name)
                else:
                    draft.unexplained.append(name)
        drafts.append(draft)
    return drafts


def _values_of(log: Sequence[dict], name: str) -> set:
    return {str(record.get(name)) for record in log if record.get(name) is not None}


def _effect_for(
    command: str,
    name: str,
    cases: Sequence[Transition],
    by_action: dict,
    watched: Sequence[str],
) -> Optional[Effect]:
    """The value this command drove a field to, if there is one.

    Counted over **opportunities**, not over every case. A command that ran
    fifty times while the alarm was already off did not switch the alarm off
    fifty times; it had nothing to do. Including those cases in the
    denominator makes a reliable effect look unreliable, and -- much worse --
    puts them in the pool a condition is searched over, where their
    irrelevant variation hides the condition that was really there.
    """
    moved = Counter(
        str(case.after.get(name)) for case in cases if case.changed(name)
    )
    if not moved:
        return None
    value, _ = moved.most_common(1)[0]

    chances = [case for case in cases if str(case.before.get(name)) != value]
    if not chances:
        return None
    worked = [case for case in chances if str(case.after.get(name)) == value]

    if len(worked) == len(chances):
        return Effect(command, name, value, len(worked), len(chances))

    condition = _separating_condition(name, value, chances, watched)
    if condition is None:
        return None
    pairs = (
        condition if isinstance(condition[0], tuple) else (condition,)
    )
    inside = [
        c for c in chances
        if all(str(c.before.get(f)) == v for f, v in pairs)
    ]
    hit = [c for c in inside if str(c.after.get(name)) == value]
    return Effect(command, name, value, len(hit), len(inside), condition)


def _separating_condition(
    name: str, value: str, chances: Sequence[Transition], watched: Sequence[str]
) -> Optional[tuple]:
    """A field value true in every case it worked and in none where it did not.

    The same shape as P2.7's precondition learner, and for the same reason:
    a condition that holds in some of the failures explains nothing, and one
    that holds in all of them explains everything. ``chances`` is already
    narrowed to cases where the field was not already at the target value.

    The field being changed is searched too. "open_door works when the door
    was shut" is a condition on the door, and excluding a field from being
    its own condition rules out most of the preconditions there are.

    Pairs are searched after singles, and only then. Two conditions can
    always be found if you look hard enough -- with enough fields, some pair
    separates any split by luck -- so the rule is: take a single if one
    exists, take a pair only if none does, and never go past two.
    """
    worked = [c for c in chances if str(c.after.get(name)) == value]
    didnt = [c for c in chances if str(c.after.get(name)) != value]
    if not worked or not didnt:
        return None

    others = [n for n in dict.fromkeys(watched) if n != name]

    # The field's own previous value comes first and is not counted as one of
    # the two. "open_door works when the door was shut" is not a discovered
    # correlation, it is what changing the door *from* something means, and
    # spending one of two condition slots on it would leave a command that
    # needs a key and a light unlearnable.
    found: list[tuple] = []
    before_value = _single_value(worked, name)
    if before_value is not None:
        remaining = [c for c in didnt if str(c.before.get(name)) == before_value]
        if not remaining:
            return (name, before_value)
        found.append((name, before_value))
        didnt = remaining

    for other in others:
        only = _single_value(worked, other)
        if only is None:
            continue
        if all(str(c.before.get(other)) != only for c in didnt):
            return _condition(found + [(other, only)])

    for i, first in enumerate(others):
        left = _single_value(worked, first)
        if left is None:
            continue
        for second in others[i + 1:]:
            right = _single_value(worked, second)
            if right is None:
                continue
            if all(
                not (
                    str(c.before.get(first)) == left
                    and str(c.before.get(second)) == right
                )
                for c in didnt
            ):
                return _condition(found + [(first, left), (second, right)])
    return None


def _condition(pairs: Sequence[tuple]):
    """One pair stays a pair; several become a tuple of them."""
    return pairs[0] if len(pairs) == 1 else tuple(pairs)


def _single_value(cases: Sequence[Transition], name: str) -> Optional[str]:
    """The one value this field had across every case, if there is one."""
    seen = {str(c.before.get(name)) for c in cases}
    return next(iter(seen)) if len(seen) == 1 else None
