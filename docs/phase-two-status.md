# Phase Two status

Phase One proved the architectural claim: the reasoning is exact, so every
failure is a perception or a knowledge failure. Phase Two turns that reasoner
into something a host can embed — one that works out its own situation rather
than being told where it is.

This file tracks the milestones in the same style as `docs/roadmap-status.md`.
Every "done when" is a named test in `tests/test_phase_two.py`; a milestone
with no test is not done, whatever the code says.

| Milestone | Goal | State |
|---|---|---|
| P2.0 — `Mind` facade and honest answers | One entry point, three-valued answers, short lines | **done** |
| P2.1 — Workspace, specialists, controller | A place where kinds of reasoning meet | **done** |
| P2.2 — Safety kernel | Explore without breaking; always a way back | not started |
| P2.3 — Growth loop | Notice gaps and close them | not started |
| P2.4 — Self-model and context discovery | Work out where it is, and say so | not started |
| P2.5 — Language that grows | Widen perception by correction | not started |
| P2.6 — Background knowledge as defaults | The obvious facts text never states | not started |
| P2.7 — Action, time and goals | Decide, not just answer | not started |
| P2.8 — Guided module building | No pack fits? Draft one | not started |
| P2.9–P2.12 — Facets, packs, vision | Reusable building blocks | not started |
| P2.13 — The school | One command reproduces every number | not started |
| P2.14 — Packs and release | Drop it into new software | not started |

## P2.1 — the workspace

### Why a blackboard rather than another pipeline stage

Phase One was a pipeline, and a pipeline cannot hold two kinds of reasoning
that are not downstream of each other. An arithmetic solver and a graph search
have nothing to say to one another through a pipe, because neither comes after
the other.

A blackboard is the classical answer and the right one here for a specific
reason: everything on it is a ground atom with a justification, so a result
posted by the constraint solver is, to the logic engine, indistinguishable from
a fact that was given. They reason over each other's conclusions without either
knowing the other exists.

### Four specialists, one protocol

| Specialist | Backend | What it adds over Datalog |
|---|---|---|
| `logic` | the Phase One engine | deduction; never optional |
| `arithmetic` | Z3 | runs a relation **backwards** — one statement of `total = x + y` answers for any of the three |
| `units` | pint | dimensions, conversion, and refusing to add a length to a duration |
| `graph` | NetworkX | *shortest* path — a least model cannot express a minimum over derivations |

Each implements `accepts(goal) -> float` and `run(goal, workspace, budget)`,
and obeys one rule: **every result carries its own proof**. That is what lets
a tree span all of them:

```
signed_off  (by sign off needs every beam safe and a frame that fits)
├── safe(beam_a)  (by a beam is safe when its load is within its rating)
│   ├── beam(beam_a)  [given]
│   └── leq(beam_a_load, beam_a_rating)  [by arithmetic: 3200 <= 5000]
│       ├── value(beam_a_load, 3200)  [by units: 3200 N = 3200 kg·m/s²]
│       └── value(beam_a_rating, 5000)  [by units: 5 kN = 5000 kg·m/s²]
├── shippable  (by nothing ships until it has been inspected)
│   └── before(cut, ship)  [by graph: cut -> deburr -> paint -> inspect -> ship]
└── fits  (by the frame fits when the span clears the opening)
    └── leq(span, clearance)  [by arithmetic: 4.2 <= 5]
```

One derivation, four specialists, no seams. The arithmetic proof cites Z3's
**unsat core** rather than the whole constraint system, so every constraint
named genuinely contributed and none that contributed is missing.

### Measured

Fifty hand-written mixed questions (`neuralmind/workspace/scenarios.py`) over
one workshop scenario — unit conversions, load checks, multi-way solving,
assembly ordering, and rules that span all of it:

| Budget | Worst overshoot | Correct |
|---|---|---|
| 200 ms | −94% | 50/50 |
| 50 ms | −74% | 50/50 |
| 20 ms | −40% | 50/50 |
| 10 ms | **+8.9%** | 50/50 |
| 5 ms | +49% | 50/50 |
| 1 ms | +318% | 41/50 |

The 10% bar holds from 10ms up. Below that it cannot: nothing in Python can
safely interrupt a call, so the overshoot is bounded by **one specialist call**
rather than by a percentage, and the controller does not pretend otherwise.

The property that makes a short budget safe is the direction of failure. Every
miss at 1ms is `yes → unknown`; a query cut short never answers `no` and never
answers a different `yes`. Warming is paid up front and outside the budget —
importing z3 and building pint's registry cost ~300ms together, more than ten
times a realistic query, and charging that to whichever question came first
would make the budget meaningless.

Degrading is per-specialist. Removing one costs exactly the questions that
needed it, and a missing *backend* is reported rather than hidden:

```
leq(load, rating) → unknown
  arithmetic: the arithmetic specialist needs z3-solver: pip install neuralmind[arithmetic]
```

### Bugs this cost

- **Goals leaked between queries.** Facts persist on a blackboard; agendas must
  not. A leftover subgoal was being taken instead of the current question and
  answered with a reason belonging to another query.
- **A livelock on an unreachable subgoal.** The controller re-posts a goal
  under its subgoals so it is retried once they land — and then re-derived the
  same failing subgoal until the budget ran out, reporting "out of time" for
  something it had settled on the first attempt.
- **The logic specialist re-pushed the whole blackboard on every call**,
  invalidating its cached model each time. Handing over only the delta took the
  mixed set from ~400ms to 142ms.
- **`max_rounds` was a performance knob pretending to be a safety net.** A
  chain across three specialists needs about twenty rounds; the limit was 16.
- **Floats had nowhere to live.** 4200 mm is 4.2 m and rounding it is a wrong
  answer, not a simplification. `Const.is_number` stays integer-only — it is
  what the engine's ASP arithmetic checks — and `is_numeric` includes floats
  for the specialists. The engine now refuses float arithmetic rather than
  truncating it.

### What P2.1 does not do

- **The router is rule-based.** The roadmap has it becoming a classifier over
  logged routing decisions; until those logs exist, a learned router would be a
  guess wearing a confidence score.
- **No parallelism.** Specialists run one at a time.
- **Budgets are cooperative**, not enforced. See above for why.

## P2.0 — the `Mind` facade and honest answers

### Unknown is not no

Phase One answered yes or no, and under the closed-world assumption that was
right: a program that fully describes its world makes failure-to-derive a proof
of falsity. Outside that setting it is a lie. A mind that does not know whether
mammals are warm-blooded should not say they are not.

So answers are now three-valued, and the program says which reading applies per
predicate:

```prolog
#open flies/1.        % silence about flies means unknown
#closed roster/1.     % silence about roster means no
#open.                % silence about anything means unknown
```

Closed stays the default, because that is what Phase One's benchmarks are
graded against and none of those numbers moved.

`no` also became something that can be *proven*, through strong negation.
`-flies(pingu)` is a claim that Pingu does not fly, as opposed to `not
flies(pingu)`, which only says nothing derived it. The two are different
predicates to the engine and different answers to a user, and deriving both
`p` and `-p` is now reported as a contradiction rather than absorbed
(`Model.contradictions`).

### The gap behind an unknown

"I don't know" is useless. "I don't know whether Bob is furry" is a question
someone can answer. Every unknown carries its failure diagnosis, which now
reports structurally rather than as a message: `established` is what the rule
did prove, `missing` is the literal it stalled on.

```
Unknown — Bob is a mammal, but I don't know whether Bob is furry.
```

### Brief lines that are still true

`neuralmind/output/brief.py` compresses a proof into one sentence under a word
budget. It has one hard rule: **every clause comes from a node of the proof**.
Nothing is added for fluency and nothing is inferred to fill a gap, so a test
can read each line back and find the node behind it. Measured over 1,440
answers from four OWA splits:

| | |
|---|---|
| brief lines within the 25-word budget | **100.0%** (target 95%) |
| mean length | 11.2 words |
| clauses traceable to a proof node | **2050/2050** |

### The mind is never told where it is

`Mind.__init__` takes no domain, pack or mode argument, and a test asserts it
never will. A host that has to declare "you are a bank assistant" has already
made the mind's hardest decision for it, and whatever it declares will be wrong
in the cases that matter — the training simulator that is half game and half
engineering tool, the bank app with a shop inside it.

Until context discovery lands (P2.4), `self_report()` says the context is
unresolved rather than implying one has been worked out. `observe()` keeps
every structured payload whole and unread: interpreting shape is P2.4's job,
and guessing at it here would be the same assumption by another route.

### Measured: the open-world splits

The corpus ships both readings. The OWA splits answer True, False *or*
Unknown, and about 46% of their questions are ones the theory does not settle —
so a system that answers "no" to whatever it cannot derive scores around 54%
on them while scoring 100% on CWA. Every question in every OWA test split:

| Split | Accuracy | Questions | Theories exact | Unknown in gold |
|---|---|---|---|---|
| depth-0 | **100.0%** | 20,024 | 5389/5389 | 46% |
| depth-1 | **100.0%** | 20,210 | 2607/2607 | 48% |
| depth-2 | **100.0%** | 19,840 | 1794/1794 | 46% |
| depth-3 | **100.0%** | 20,346 | 1405/1405 | 45% |
| depth-5 | **100.0%** | 20,030 | 948/948 | 43% |
| birds-electricity | **99.9%** | 5,270 | 139/140 | 75% |
| NatLang *(crowdsourced)* | **70.5%** | 8,008 | 1/482 | 50% |

```bash
python scripts/download_proofwriter.py
neuralmind eval --corpus depth-5 --world owa -n 200
```

Zero failures attributable to the engine, on either reading. The generated
splits match their CWA numbers exactly, which is what the milestone asked for,
and NatLang lands in the same place under both readings — the perception gap
dominates and the third answer neither helps nor hurts it.

### What P2.0 does not do

- **No context discovery.** `self_report()` says "unresolved" because it is.
- **No specialists, no workspace.** One engine, as in Phase One.
- **`questions()` returns nothing.** It is the growth loop's entry point
  (P2.3), present so the interface does not change when that lands.
- **`budget_ms` is recorded, not enforced.** The controller that enforces it
  is P2.1.
