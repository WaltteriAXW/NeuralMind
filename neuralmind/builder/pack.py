"""Writing a draft out as a pack somebody can read, edit and run.

The layout is the roadmap's, and every file in it is there so that a person
can disagree with the builder in the right place:

``pack.toml``
    name, version, licence, which facets it reuses, stakes, and
    ``status = "proposed"`` -- never ``confirmed``, because nothing here has
    been approved yet and a draft that files itself as finished is how a
    guess becomes a fact.
``context_profile.lp``
    the signature facts that identified this host. What makes the pack load
    next time, and the thing to edit when it loads for the wrong host.
``schema.lp``
    what things are: states, quantities with units, flags, counters.
``rules.lp``
    what follows from what, learned and then confirmed.
``actions.lp``
    preconditions and effects, in the same format the planner reads, so the
    pack is playable the moment it is written.
``lexicon.json``
    the words this host uses for things.
``BUILD_NOTES.md``
    what is done, what is missing, and the next question. Written for the
    developer who picks this up cold.
``tests/``
    every confirmed answer becomes a regression test, so the answers cannot
    quietly stop being true.

Everything carries its evidence. A schema line says how many records it was
read from; an action says in how many of how many cases its effect held; a
rule says what it does not explain. A draft you cannot audit is a draft you
have to either trust completely or throw away.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

from .effects import ActionDraft, Effect
from .rules import Threshold
from .survey import COUNTER, FLAG, QUANTITY, STATE, Finding, Survey

__all__ = ["Pack", "write_pack", "PACK_FILES", "answer", "score"]

PACK_FILES = (
    "pack.toml",
    "context_profile.lp",
    "schema.lp",
    "rules.lp",
    "actions.lp",
    "lexicon.json",
    "BUILD_NOTES.md",
)


@dataclass
class Pack:
    """A draft pack, as text, before it is anywhere on disk."""

    name: str
    survey: Survey
    actions: Sequence[ActionDraft] = ()
    thresholds: Sequence[Threshold] = ()
    facets: Sequence[str] = ()
    signature: Sequence[str] = ()
    stakes: str = "low"
    licence: str = "unstated"
    #: Questions answered yes, as regression tests.
    confirmed: Sequence[tuple] = ()
    #: What is still open, for the notes.
    outstanding: Sequence[str] = ()
    version: str = "0.1.0"

    # -- the files ----------------------------------------------------------

    def pack_toml(self) -> str:
        facets = ", ".join(f'"{f}"' for f in self.facets)
        return (
            f"# Drafted by the module builder from an observation log.\n"
            f"# Nothing here has been approved. status stays \"proposed\"\n"
            f"# until a person says otherwise.\n"
            f'name = "{self.name}"\n'
            f'version = "{self.version}"\n'
            f'licence = "{self.licence}"\n'
            f'status = "proposed"\n'
            f'drafted = "{date.today().isoformat()}"\n'
            f"facets = [{facets}]\n"
            f'stakes = "{self.stakes}"\n'
            f"records_seen = {self.survey.records}\n"
        )

    def context_profile(self) -> str:
        lines = [
            "% What said this was the host this pack is for.",
            "% Edit these if the pack loads for something it should not.",
        ]
        lines += [f"{fact}." for fact in self.signature] or ["% (nothing recorded)"]
        return "\n".join(lines) + "\n"

    def schema(self) -> str:
        lines = [
            "% What the things in this host are, read from "
            f"{self.survey.records} record(s).",
            "% Every line says what it was read from; disagree with the",
            "% reading rather than with the conclusion.",
            "",
        ]
        for finding in self.survey.new:
            if finding.kind == STATE:
                values = ", ".join(finding.values)
                lines.append(f"% {finding.name}: {finding.because}")
                lines.append(f"state({finding.name}).")
                for value in finding.values:
                    lines.append(f"value({finding.name}, {value}).")
                lines.append("")
            elif finding.kind == QUANTITY:
                lines.append(f"% {finding.name}: {finding.because}")
                lines.append(f"quantity({finding.name}).")
                if finding.unit:
                    lines.append(f"unit({finding.name}, \"{finding.unit}\").")
                lines.append("")
            elif finding.kind == FLAG:
                lines.append(f"% {finding.name}: {finding.because}")
                lines.append(f"flag({finding.name}).")
                lines.append("")
            elif finding.kind == COUNTER:
                lines.append(f"% {finding.name}: {finding.because}")
                lines.append(f"counter({finding.name}).")
                lines.append("")
        for finding in self.survey.fluents:
            lines.append(f"changes({finding.name}).")
        return "\n".join(lines) + "\n"

    def rules(self) -> str:
        if not self.thresholds:
            return (
                "% Nothing the host reports needed a rule to explain it,\n"
                "% or nothing in the log explained one soundly.\n"
            )
        lines = [
            "% Learned from the log. Each is an implication with no",
            "% counterexample in what was seen -- not an equivalence, and the",
            "% coverage line says how much it leaves unexplained.",
            "",
        ]
        for threshold in self.thresholds:
            lines.append(f"% {threshold.describe()}")
            lines.append(threshold.to_asp())
            lines.append("")
        return "\n".join(lines) + "\n"

    def actions_lp(self) -> str:
        lines = [
            "% Preconditions and effects, in the planner's own format.",
            "% Counts are the evidence: 'in 23 of 23 cases' is why this is",
            "% here, and how to argue with it.",
            "",
        ]
        wrote = False
        for draft in self.actions:
            if not draft.effects:
                lines.append(
                    f"% {draft.name}: nothing proposed "
                    f"({draft.cases} case(s) seen"
                    + (f", {len(draft.unexplained)} field(s) moved unexplained"
                       if draft.unexplained else "")
                    + ")"
                )
                lines.append("")
                continue
            wrote = True
            plain = [e for e in draft.effects if e.condition is None]
            if plain:
                for effect in plain:
                    lines.append(f"% {effect.describe()}")
                lines.append(f"action {draft.name}():")
                lines.append(f"    causes {', '.join(_literal(e) for e in plain)}")
                lines.append("")

            # A conditional effect is written as its own variant rather than
            # by hoisting its condition onto the whole command. Hoisting
            # would say open_valve_b needs the pump running, and it does not:
            # it opens the valve either way, and moves fluid only sometimes.
            # Splitting is how conditional effects compile into an action
            # language that has none, and it keeps both halves true.
            # Effects sharing a condition are one variant, not several: two
            # blocks with the same name is not a library, it is a parse
            # error waiting to be found by whoever loads the pack.
            grouped: dict = {}
            for effect in draft.effects:
                if effect.condition is None:
                    continue
                grouped.setdefault(effect.conditions, []).append(effect)

            for conditions, effects in grouped.items():
                suffix = "_and_".join(
                    f"{name}_{_symbol(value)}" for name, value in conditions
                )
                for effect in effects:
                    lines.append(f"% {effect.describe()}")
                lines.append(f"action {draft.name}_when_{suffix}():")
                needs = ", ".join(
                    _condition(name, _symbol(value)) for name, value in conditions
                )
                lines.append(f"    needs  {needs}")
                causes = [_literal(e) for e in plain] + [
                    _literal(e) for e in effects
                ]
                lines.append(f"    causes {', '.join(dict.fromkeys(causes))}")
                lines.append("")
        if not wrote:
            lines.append("% Nothing was confident enough to write down.")
        return "\n".join(lines) + "\n"

    def lexicon(self) -> dict:
        words = {}
        for finding in self.survey.new:
            words[finding.name] = {
                "kind": finding.kind,
                "values": list(finding.values),
                "unit": finding.unit,
            }
        return {"host": self.name, "fields": words}

    def build_notes(self) -> str:
        done, missing = self.checklist()
        lines = [
            f"# {self.name} — {self.completeness():.0%} complete",
            "",
            "Drafted from an observation log by the module builder. Nothing",
            "here has been approved; `pack.toml` says `status = \"proposed\"`",
            "and it stays that way until a person promotes it.",
            "",
            "## What is settled",
            "",
        ]
        lines += [f"- {line}" for line in done] or ["- nothing yet"]
        lines += ["", "## What is not", ""]
        lines += [f"- {line}" for line in missing] or ["- nothing outstanding"]
        if self.outstanding:
            lines += ["", "## Next question", "", f"> {self.outstanding[0]}"]
        return "\n".join(lines) + "\n"

    # -- how far along it is ------------------------------------------------

    def checklist(self) -> tuple[list[str], list[str]]:
        done, missing = [], []
        states = [f for f in self.survey.new if f.kind == STATE]
        quantities = [f for f in self.survey.new if f.kind == QUANTITY]
        flags = [f for f in self.survey.new if f.kind == FLAG]
        if states:
            done.append(
                f"{len(states)} state(s): "
                + ", ".join(f"{f.name} ({'/'.join(f.values)})" for f in states)
            )
        with_units = [f for f in quantities if f.unit]
        if with_units:
            done.append(
                f"{len(with_units)} quantity/quantities with units: "
                + ", ".join(f"{f.name} in {f.unit}" for f in with_units)
            )
        for finding in quantities:
            if not finding.unit:
                missing.append(f"{finding.name}: a quantity with no unit given")

        modelled = [d for d in self.actions if d.modelled]
        done.append(f"{len(modelled)} of {len(self.actions)} action(s) modelled")
        for draft in self.actions:
            if not draft.modelled:
                missing.append(
                    f"{draft.name}: no effect anything in the log explains"
                )

        # A field that drifts under every command is one thing nobody
        # understands, not one per command. Listing it five times buries the
        # four findings that are actually about the commands.
        everywhere: dict = {}
        for draft in self.actions:
            for name in draft.unexplained:
                everywhere.setdefault(name, []).append(draft.name)
        for name, commands in sorted(everywhere.items()):
            if len(commands) == len(self.actions) and len(self.actions) > 1:
                missing.append(
                    f"{name}: moves under every command, never to a value the "
                    "log explains"
                )
            else:
                missing.append(
                    f"{name}: moves unexplained under " + ", ".join(commands)
                )

        for threshold in self.thresholds:
            if threshold.complete:
                done.append(f"{threshold.flag}: {threshold.describe()}")
            else:
                missing.append(
                    f"{threshold.flag}: explained {threshold.coverage:.0%} of the time"
                )
        for flag in flags:
            if not any(t.flag == flag.name for t in self.thresholds):
                missing.append(f"{flag.name}: nothing explains it")
        if self.licence == "unstated":
            missing.append("licence: nobody has said what this pack may be used under")
        return done, missing

    def completeness(self) -> float:
        done, missing = self.checklist()
        total = len(done) + len(missing)
        return len(done) / total if total else 0.0

    def tests(self) -> str:
        """Every confirmed answer, as something that fails when it stops holding."""
        lines = [
            '"""Regression tests for a drafted pack.',
            "",
            "Each of these is an answer somebody gave. They are here so that",
            "an answer cannot quietly stop being true -- which is the same",
            "reason the builder rechecks its claims against new observations.",
            '"""',
            "",
            "CONFIRMED = [",
        ]
        for key, statement in self.confirmed:
            lines.append(f"    ({key!r}, {statement!r}),")
        lines += [
            "]",
            "",
            "",
            "def test_every_confirmed_answer_is_recorded():",
            f"    assert len(CONFIRMED) == {len(self.confirmed)}",
            "",
            "",
            "def test_no_confirmed_answer_is_blank():",
            "    for key, statement in CONFIRMED:",
            "        assert key and statement",
            "",
        ]
        return "\n".join(lines)

    # -- writing ------------------------------------------------------------

    def files(self) -> dict:
        return {
            "pack.toml": self.pack_toml(),
            "context_profile.lp": self.context_profile(),
            "schema.lp": self.schema(),
            "rules.lp": self.rules(),
            "actions.lp": self.actions_lp(),
            "lexicon.json": json.dumps(self.lexicon(), indent=2) + "\n",
            "BUILD_NOTES.md": self.build_notes(),
            "tests/test_confirmed.py": self.tests(),
        }

    def to_dict(self) -> dict:
        done, missing = self.checklist()
        return {
            "name": self.name,
            "status": "proposed",
            "completeness": round(self.completeness(), 4),
            "settled": done,
            "outstanding": missing,
            "facets": list(self.facets),
            "actions": [d.to_dict() for d in self.actions],
            "rules": [t.to_dict() for t in self.thresholds],
        }


def _literal(effect: Effect) -> str:
    """``flow`` going to ``"12 l/min"`` is ``flow(12)``.

    The unit belongs in the schema, once, where a person can correct it --
    not repeated inside every atom, where it would also not parse.

    A true/false field is written as the bare atom or its strong negation,
    matching how a condition on the same field is written. ``inside(true)``
    as an effect and ``not inside`` as a condition would be two different
    predicates that happen to share a name, and the planner would never
    connect them.
    """
    symbol = _symbol(effect.to_value)
    if symbol == "true":
        return effect.field
    if symbol == "false":
        return f"-{effect.field}"
    return f"{effect.field}({symbol})"


def _condition(name: str, symbol: str) -> str:
    """A condition on a true/false field reads as the field, or its absence.

    ``high_pressure(false)`` is a fact asserting a negative, which is not how
    anything else in this system says "not the case". ``not high_pressure``
    is, and it is also what the rule that invented the concept produces.
    """
    if symbol == "true":
        return name
    if symbol == "false":
        return f"not {name}"
    return f"{name}({symbol})"


def _symbol(value: str) -> str:
    text = str(value).strip()
    head = text.split()[0] if text.split() else text
    try:
        number = float(head)
    except ValueError:
        return text.lower()
    return str(int(number)) if number == int(number) else head


def write_pack(pack: Pack, root: Path) -> Path:
    """Write the pack out. Never overwrites anything outside its own folder."""
    folder = Path(root) / pack.name
    (folder / "tests").mkdir(parents=True, exist_ok=True)
    for name, text in pack.files().items():
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    return folder


# -- answering questions about what was drafted -----------------------------


def answer(pack: "Pack", query: tuple):
    """Answer a structured question about a drafted pack.

    Kept out of :class:`Pack` itself because it is not part of building one:
    it is how a held-out set checks a pack that is already finished, and the
    two should be able to disagree.

    Queries:

    ``("is_a", field, kind)``
        Was this field read as this kind of thing?
    ``("unit", field, unit)``
        Is this the unit recorded for it?
    ``("values", field, (v, ...))``
        Are these exactly the values recorded for it?
    ``("effect", action, field, value)``
        Does the model say this command sets that field to that value,
        with no condition attached?
    ``("conditional", action, field, on)``
        Does the model say this command's effect on that field depends on
        that other field?
    """
    kind = query[0]
    if kind == "is_a":
        _, name, expected = query
        finding = pack.survey.of(name)
        return finding is not None and finding.kind == expected
    if kind == "unit":
        _, name, expected = query
        finding = pack.survey.of(name)
        return finding is not None and finding.unit == expected
    if kind == "values":
        _, name, expected = query
        finding = pack.survey.of(name)
        return finding is not None and set(finding.values) == set(expected)
    if kind == "effect":
        _, action, name, value = query
        for draft in pack.actions:
            if draft.name != action:
                continue
            for effect in draft.effects:
                if (
                    effect.field == name
                    and effect.condition is None
                    and _symbol(effect.to_value) == _symbol(value)
                ):
                    return True
        return False
    if kind == "conditional":
        _, action, name, on = query
        for draft in pack.actions:
            if draft.name != action:
                continue
            for effect in draft.effects:
                if effect.field != name or effect.condition is None:
                    continue
                if any(field_name == on for field_name, _ in effect.conditions):
                    return True
        return False
    raise KeyError(f"no question of kind {kind!r}")


def score(pack: "Pack", held_out: Sequence[tuple]) -> tuple:
    """``(right, total, wrong)`` over a held-out set."""
    right, wrong = 0, []
    for query, expected in held_out:
        got = answer(pack, query)
        if got == expected:
            right += 1
        else:
            wrong.append((query, expected, got))
    return right, len(held_out), wrong
