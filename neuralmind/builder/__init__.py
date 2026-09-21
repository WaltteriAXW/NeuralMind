"""No pack fits? Draft one.

The mind has been able to say "I do not recognise this host" since P2.4, and
to name the vocabulary it could not place. This is what happens next: it
drafts a pack from what it has watched, and asks a person the few questions
the log cannot answer.

The order is the roadmap's, and each step only works because the one before
it narrowed the problem:

1. **Reuse the facets.** Whatever money, quantities or dialogue already
   explain is taken as given. The builder works on what is left, which is why
   a new pack is small.
2. **Survey the rest** (:mod:`~neuralmind.builder.survey`) -- what each
   remaining field *is*, read from how its values behave.
3. **Learn the action model** (:mod:`~neuralmind.builder.effects`) -- what
   each command the host offers actually did, counted over the transitions.
4. **Learn rules** (:mod:`~neuralmind.builder.rules`) for what the host
   reports and nothing can derive, and give what they find a name.
5. **Use that name.** The invented concept goes back into step 3, which is
   how an action whose condition was inexpressible becomes expressible.
6. **Ask**, one question at a time, most-unlocking first
   (:mod:`~neuralmind.builder.questions`).
7. **Write the pack** (:mod:`~neuralmind.builder.pack`), status ``proposed``.
8. **Shadow-test** (:mod:`~neuralmind.builder.shadow`) -- the draft answers in
   the sandbox, beside the live mind, until its tests and every core canary
   pass and a person promotes it.

Two properties are worth stating plainly, because they are what separate this
from a schema guesser.

**Everything carries its evidence.** "In 23 of 23 cases" is not decoration;
it is the whole basis for the line above it, and it is what somebody argues
with. Where the log does not settle something, the builder says so instead of
picking the most common answer.

**An answer is a claim, and claims are rechecked.** A developer who answers
wrongly is not permanently believed. Every yes becomes a statement about
future observations, and an observation that contradicts one reopens the
question with the record attached. The log is not overruled by an opinion,
and the opinion is not overruled by the builder -- it is put back to the
person, which is the only move that respects both.

Usage::

    builder = Builder.from_log(log, name="pump_station")
    builder.step()                       # survey, effects, rules
    while builder.interview.next:
        question = builder.interview.next
        builder.answer(question.key, "yes")
    builder.write(Path("packs"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .effects import ActionDraft, Effect, Transition, learn_effects, transitions
from .pack import PACK_FILES, Pack, write_pack
from .questions import (
    NO,
    UNSURE,
    YES,
    Answer,
    Claim,
    Counterexample,
    Interview,
    Question,
)
from .rules import SCALE, Threshold, invented_name, learn_thresholds
from .survey import (
    COUNTER,
    FLAG,
    QUANTITY,
    STATE,
    Finding,
    Survey,
    survey,
)

__all__ = [
    "Builder",
    "triggered_by",
    # the steps
    "Survey", "survey", "Finding",
    "learn_effects", "ActionDraft", "Effect", "Transition", "transitions",
    "learn_thresholds", "Threshold", "invented_name",
    "Interview", "Question", "Answer", "Claim", "Counterexample",
    "Pack", "write_pack", "PACK_FILES",
    "YES", "NO", "UNSURE",
    # kinds
    "STATE", "QUANTITY", "FLAG", "COUNTER",
]

#: A context hypothesis at or below this is not a reading, it is a shrug.
UNRECOGNISED = 0.6
#: This many unplaced words in one stream is a domain, not a stray field.
ENOUGH_UNPLACED = 2


def triggered_by(reading) -> Optional[str]:
    """Whether a context reading is a reason to start building, and which.

    The roadmap lists three triggers and they are not alternatives so much as
    three views of the same thing: the mind is looking at something it has no
    words for. Returning *which* one fired matters, because it is the first
    line of the build notes -- somebody reading the pack next month wants to
    know what made it exist.
    """
    confidence = getattr(reading, "domain_confidence", 1.0) or 0.0
    unplaced = list(getattr(reading, "unexplained", ()) or ())
    covered = getattr(reading, "covered", 1.0)

    if confidence <= UNRECOGNISED and len(unplaced) >= ENOUGH_UNPLACED:
        return (
            f"the host was read at {confidence:.0%} confidence with "
            f"{len(unplaced)} word(s) unplaced: {', '.join(unplaced)}"
        )
    if len(unplaced) >= ENOUGH_UNPLACED and covered < 0.6:
        return (
            f"only {covered:.0%} of the vocabulary is explained; unplaced: "
            + ", ".join(unplaced)
        )
    return None


@dataclass
class Builder:
    """Drafts a pack from a log, and asks about what the log cannot settle."""

    log: list = field(default_factory=list)
    name: str = "new_module"
    #: Field -> the facet that already explains it.
    explained: dict = field(default_factory=dict)
    #: Why this build started, for the notes.
    because: str = ""
    #: Signature facts that identified this host.
    signature: tuple = ()
    facets: tuple = ()
    stakes: str = "low"

    reading: Optional[Survey] = None
    actions: list = field(default_factory=list)
    thresholds: list = field(default_factory=list)
    interview: Interview = field(default_factory=Interview)
    #: Every question ever offered, by key, so a reopened one keeps its words.
    catalogue: dict = field(default_factory=dict)

    # -- starting -----------------------------------------------------------

    @classmethod
    def from_log(
        cls,
        log: Sequence[dict],
        name: str = "new_module",
        explained: Optional[dict] = None,
        because: str = "",
        **kwargs,
    ) -> "Builder":
        return cls(
            log=list(log),
            name=name,
            explained=dict(explained or {}),
            because=because,
            **kwargs,
        )

    @classmethod
    def from_context(
        cls, log: Sequence[dict], reading, name: str = "new_module", **kwargs
    ) -> "Builder":
        """Start from what context discovery already worked out.

        The facets it recognised are subtracted rather than re-derived: they
        are the part of this host that is *not* new, and a builder that
        redrafted them would produce a pack that duplicates the facet library
        and drifts from it.
        """
        explained = {}
        for name_ in getattr(reading, "names", ()) or ():
            pass
        covered = set(getattr(reading, "unexplained", ()) or ())
        for record in log[:1]:
            for key in record:
                if key not in covered and key not in ("action", "commands", "tick"):
                    facet = _facet_for(key, reading)
                    if facet:
                        explained[key] = facet
        return cls.from_log(
            log,
            name=name,
            explained=explained,
            because=triggered_by(reading) or "",
            facets=tuple(getattr(reading, "names", ()) or ()),
            stakes=getattr(reading, "stakes", "low"),
            **kwargs,
        )

    # -- the build ----------------------------------------------------------

    def step(self) -> "Builder":
        """Survey, learn the action model, learn rules, then use what was learned.

        Run again after more observations and it redoes all of it -- which is
        the point: the second pass has the invented concepts from the first,
        and can explain things the first could not.
        """
        self.reading = survey(self.log, self.explained)
        self.thresholds = learn_thresholds(self.log, self.reading)

        # Fields a sound rule accounts for are not any action's mystery.
        accounted = tuple(t.flag for t in self.thresholds if t.sound)

        # First pass: what the raw fields explain.
        self.actions = learn_effects(
            self.log, self.reading, accounted_for=accounted
        )

        # Second pass: with the learned concepts available as fields, an
        # action whose condition could not be said before may become sayable.
        enriched = self._with_concepts()
        if enriched is not None:
            invented = tuple(t.name for t in self.thresholds if t.sound)
            reading = survey(enriched, self.explained)
            better = learn_effects(
                enriched, reading,
                accounted_for=accounted + invented,
                derived=invented,
            )
            self.actions = _prefer_explained(self.actions, better)

        self.interview.offer(self._questions())
        return self

    def _with_concepts(self) -> Optional[list]:
        """The log again, with each learned concept as a boolean field."""
        sound = [t for t in self.thresholds if t.sound]
        if not sound:
            return None
        enriched = []
        for record in self.log:
            copy = dict(record)
            for threshold in sound:
                value = _number(record.get(threshold.quantity))
                copy[threshold.name] = (
                    value is not None and value > threshold.above
                )
            enriched.append(copy)
        return enriched

    # -- asking -------------------------------------------------------------

    def _questions(self) -> list:
        questions: list[Question] = []
        if self.reading is None:
            return questions

        for finding in self.reading.new:
            if finding.kind == QUANTITY and finding.unit:
                questions.append(self._unit_question(finding))
        for draft in self.actions:
            for effect in draft.effects:
                questions.append(self._effect_question(draft, effect))
        for threshold in self.thresholds:
            questions.append(self._rule_question(threshold))

        for question in questions:
            self.catalogue[question.key] = question
        return questions

    def _unit_question(self, finding: Finding) -> Question:
        key = f"unit:{finding.name}"
        unit = finding.unit
        return Question(
            key=key,
            kind="unit",
            text=f"{finding.name} looks like a quantity. Unit: {unit}?",
            options=(YES, NO, "other"),
            detail=finding.because,
            claim=Claim(
                key=key,
                statement=f"{finding.name} is measured in {unit}",
                check=_unit_checker(finding.name, unit),
            ),
        )

    def _effect_question(self, draft: ActionDraft, effect: Effect) -> Question:
        key = f"effect:{draft.name}:{effect.field}"
        return Question(
            key=key,
            kind="effect",
            text=effect.question(),
            detail=effect.describe(),
            claim=Claim(
                key=key,
                statement=effect.describe(),
                check=_effect_checker(effect),
            ),
        )

    def _rule_question(self, threshold: Threshold) -> Question:
        key = f"rule:{threshold.flag}:{threshold.quantity}"
        unit = f" {threshold.unit}" if threshold.unit else ""
        tail = (
            ""
            if threshold.complete
            else f" That leaves {1 - threshold.coverage:.0%} of them unexplained."
        )
        return Question(
            key=key,
            kind="rule",
            text=(
                f"{threshold.flag} was on in every record where "
                f"{threshold.quantity} was above {threshold.above:g}{unit}. "
                f"Is that the rule?{tail}"
            ),
            detail=threshold.describe(),
            claim=Claim(
                key=key,
                statement=(
                    f"{threshold.quantity} above {threshold.above:g}{unit} "
                    f"means {threshold.flag}"
                ),
                check=_threshold_checker(threshold),
            ),
        )

    def answer(self, key: str, reply: str, note: str = "") -> Answer:
        return self.interview.answer(key, reply, note)

    def observe(self, records: Iterable[dict]) -> list:
        """Take more observations, and recheck everything already agreed.

        Returns the questions that came back. This is where a wrong answer is
        caught: not by arguing with the person at the time, but by the world
        turning up later with a record that cannot be true if they were right.

        Rechecking lives here rather than in :meth:`step` because this is the
        only place new evidence arrives. Putting it in ``step`` meant a
        rebuild quietly consumed the contradiction, and the caller who asked
        for the reopened questions got an empty list.
        """
        self.log.extend(records)
        self.step()
        return self.interview.recheck(self.log, self.catalogue)

    # -- the result ---------------------------------------------------------

    def pack(self) -> Pack:
        if self.reading is None:
            self.step()
        confirmed = [
            (key, self.catalogue[key].claim.statement)
            for key in self.interview.answers
            if self.interview.agreed(key) and key in self.catalogue
            and self.catalogue[key].claim is not None
        ]
        outstanding = [q.describe() for q in self.interview.pending]
        return Pack(
            name=self.name,
            survey=self.reading,
            actions=self.actions,
            thresholds=self.thresholds,
            facets=self.facets,
            signature=self.signature,
            stakes=self.stakes,
            licence=self._licence(),
            confirmed=tuple(confirmed),
            outstanding=tuple(outstanding),
        )

    def _licence(self) -> str:
        given = self.interview.answers.get("licence")
        return given.note if given is not None and given.note else "unstated"

    def write(self, root: Path) -> Path:
        return write_pack(self.pack(), root)

    # -- the developer's view ----------------------------------------------

    def progress(self) -> str:
        """A short checklist, not an essay."""
        pack = self.pack()
        done, missing = pack.checklist()
        lines = [f'New module "{self.name}" — {pack.completeness():.0%} complete']
        if self.because:
            lines.append(f"  started because {self.because}")
        lines += [f"  ✔ {line}" for line in done]
        lines += [f"  ✘ {line}" for line in missing]
        if self.interview.reopenings:
            lines.append(
                f"  ! {len(self.interview.reopenings)} earlier answer(s) "
                "contradicted by later observations"
            )
        if self.interview.next is not None:
            lines.append(f"  next: {self.interview.next.describe()}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "because": self.because,
            "records": len(self.log),
            "pack": self.pack().to_dict(),
            "interview": self.interview.to_dict(),
        }

    def __repr__(self) -> str:
        return (
            f"Builder({self.name!r}, {len(self.log)} record(s), "
            f"{len(self.interview)} question(s) outstanding)"
        )


# -- checkers: what a yes commits to ---------------------------------------


def _unit_checker(name: str, unit: str):
    def check(log: Sequence[dict]) -> list:
        failures = []
        for index, record in enumerate(log):
            value = record.get(name)
            if not isinstance(value, str):
                continue
            parts = value.split()
            if len(parts) == 2 and parts[1] != unit:
                failures.append(
                    Counterexample(index, record, f"{name} is {value!r}, not in {unit}")
                )
        return failures

    return check


def _effect_checker(effect: Effect):
    def check(log: Sequence[dict]) -> list:
        failures = []
        for index, (before, after) in enumerate(zip(log, log[1:])):
            if str(before.get("action")) != effect.action:
                continue
            if effect.condition is not None:
                field_name, value = effect.condition
                if str(before.get(field_name)) != value:
                    continue
            landed = str(after.get(effect.field))
            if landed != effect.to_value:
                failures.append(
                    Counterexample(
                        index,
                        before,
                        f"{effect.action} left {effect.field} at {landed}, "
                        f"not {effect.to_value}",
                    )
                )
        return failures

    return check


def _threshold_checker(threshold: Threshold):
    def check(log: Sequence[dict]) -> list:
        failures = []
        for index, record in enumerate(log):
            value = _number(record.get(threshold.quantity))
            if value is None or value <= threshold.above:
                continue
            if not record.get(threshold.flag):
                failures.append(
                    Counterexample(
                        index,
                        record,
                        f"{threshold.quantity} is {value:g} but "
                        f"{threshold.flag} is off",
                    )
                )
        return failures

    return check


# -- odds and ends ----------------------------------------------------------


def _facet_for(key: str, reading) -> Optional[str]:
    """Which recognised facet, if any, already accounts for this field."""
    for facet in getattr(reading, "facets", ()) or ():
        evidence = " ".join(getattr(facet, "because", ()) or ())
        if key in evidence:
            return getattr(facet, "name", None)
    return None


def _prefer_explained(first: Sequence[ActionDraft], second: Sequence[ActionDraft]):
    """Keep whichever pass explained more of each action.

    The second pass has the invented concepts and usually wins, but not
    always -- adding a field can only ever help a conditional effect, and a
    draft that got worse means the concept was noise.
    """
    by_name = {draft.name: draft for draft in first}
    out = []
    for draft in second:
        original = by_name.get(draft.name)
        if original is None:
            out.append(draft)
            continue
        out.append(draft if len(draft.effects) >= len(original.effects) else original)
    for name, draft in by_name.items():
        if not any(d.name == name for d in out):
            out.append(draft)
    return sorted(out, key=lambda d: d.name)


def _number(raw) -> Optional[float]:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if not isinstance(raw, str):
        return None
    try:
        return float(raw.strip().split()[0])
    except (ValueError, IndexError):
        return None
