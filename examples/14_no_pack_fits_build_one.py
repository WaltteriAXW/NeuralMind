#!/usr/bin/env python3
"""No pack fits this host, so draft one from watching it.

The mind has been able to say "I do not recognise this" since P2.4, and to
name the words it could not place. This is what happens next. It is handed a
log of a pump station being operated — nothing else, no schema, no
documentation — and works out what the things are, what each command does,
and what explains the alarm.

Four things worth watching:

* every conclusion carries the counting that produced it, because a draft
  nobody can argue with is a draft nobody can correct;
* where the log does not settle something, it says so instead of picking the
  most common answer;
* a concept it invented to explain the alarm turns out to be what makes a
  command's condition sayable — which is the strongest evidence a log can
  give that the concept is real;
* and an answer somebody gives is a claim, rechecked against everything seen
  afterwards.
"""

from neuralmind.builder import YES, Builder
from neuralmind.builder.pack import score
from neuralmind.builder.stations import (
    HELD_OUT,
    PumpStation,
    pump_station_log,
)

print("what it was given (one record of 301)")
print("-" * 74)
log = pump_station_log(steps=300)
for key, value in log[3].items():
    print(f"  {key:10} {value}")

print()
print("what it made of it")
print("-" * 74)
builder = Builder.from_log(log, name="pump_station").step()
print(builder.reading.describe())

print()
print("what each command does, and how it knows")
print("-" * 74)
for draft in builder.actions:
    print(draft.describe())

print()
print("what explains the alarm")
print("-" * 74)
for threshold in builder.thresholds:
    print(f"  {threshold.describe()}")
print()
print("  ...and the concept that finding invented is what let it finally say")
print("  when reset_alarm works. Nothing could express 'the pressure is low")
print("  enough' before there was a name for the pressure being high.")

print()
print("the checklist a developer sees")
print("-" * 74)
print(builder.progress())

print()
print("questions it never saw, answered from the draft")
print("-" * 74)
right, total, wrong = score(builder.pack(), HELD_OUT)
print(f"  {right} of {total}")
for query, expected, got in wrong:
    print(f"  wrong: {query} expected {expected}, got {got}")

print()
print("an answer is a claim, and claims are rechecked")
print("-" * 74)


def drive(station, commands, into):
    for command in commands:
        before = station.snapshot()
        station.apply(command)
        record = dict(before)
        record["action"] = command
        into.append(record)


CALM = [
    "close_valve_b", "start_pump_a", "start_pump_a", "start_pump_a",
    "start_pump_a", "open_valve_b", "open_valve_b", "open_valve_b",
    "open_valve_b", "reset_alarm",
]

station = PumpStation(0)
short = []
for _ in range(4):
    drive(station, CALM, short)
short.append(station.snapshot())

second = Builder.from_log(short, name="pump_station").step()
question = next(
    q for q in second.interview.pending if q.key == "effect:reset_alarm:alarm"
)
print(f"  asked:    {question.text}")
print("  answered: yes   (and it is wrong — the log was just calm)")
second.answer(question.key, YES)

later = []
drive(station, ["close_valve_b"] + ["start_pump_a"] * 4 + ["reset_alarm"] * 2, later)
later.append(station.snapshot())
reopened = second.observe(later)

print()
print("  then it watched the command fail:")
for back in reopened:
    print(f"    {back.describe()}")
print()
print("  the builder does not overrule the person — it puts the question back")
print("  with the record that disagrees, because they may know something the")
print("  log does not show.")
