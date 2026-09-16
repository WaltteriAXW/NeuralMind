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
| P2.1 — Workspace, specialists, controller | A place where kinds of reasoning meet | not started |
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
