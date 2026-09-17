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
| P2.2 — Safety kernel | Explore without breaking; always a way back | **done** |
| P2.3 — Growth loop | Notice gaps and close them | **done** |
| P2.4 — Self-model and context discovery | Work out where it is, and say so | **done** |
| P2.5 — Language that grows | Widen perception by correction | not started |
| P2.6 — Background knowledge as defaults | The obvious facts text never states | not started |
| P2.7 — Action, time and goals | Decide, not just answer | not started |
| P2.8 — Guided module building | No pack fits? Draft one | not started |
| P2.9–P2.12 — Facets, packs, vision | Reusable building blocks | not started |
| P2.13 — The school | One command reproduces every number | **done** |
| P2.14 — Packs and release | Drop it into new software | not started |

## P2.13 — the school

By P2.4 the numbers this project claims lived in five places: a fuzz harness, a
growth benchmark, two `eval` invocations and an intent training script. A set of
figures you have to reassemble by hand is a set of figures that quietly stops
being true. So: one command.

```
neuralmind school            # the quick pass, about 36 seconds
neuralmind school --full     # every question and 10,000 fuzz inputs
```

### The quick pass

```
curriculum: 10 passed, 0 failed, 5 unavailable, 0 blocked (quick, 36s)

  proofwriter-cwa            ok      100.0% (need 99.9%)  1,408/1,408 questions, 0 engine failure(s)
  proofwriter-owa            ok      100.0% (need 99.9%)  1,456 questions, 45% of gold answers are unknown
  proofwriter-natlang        ok       72.9% (need 68.0%)  1,004 questions; the ceiling is perception, not reasoning
  mixed-specialists          ok      100.0% (need 100.0%)  50 questions inside a 10ms budget, one proof each
  kernel-fuzz                ok   100.0% intact (need 100.0% intact)  300 hostile inputs, 180 quarantined
  growth-by-asking           ok   68.3% saved (need 50.0% saved)  2/2 learned correctly; active 19 vs random 60 question(s)
  service-intents            ok   50.5% rejected (need 40.0% rejected)  200 out-of-scope utterances; banking family 90%
  where-am-i                 ok      100.0% (need 100.0%)  9 unlabelled hosts
  context-drift              ok      100.0% (need 100.0%)  noticed the host change after 4 observation(s)
  mnist-weak-supervision     ok       97.5% (need 95.0%)  2,000 digits, from a model trained with no digit labels at all
  babi                       —        unavailable: no reachable mirror of the bAbI tasks
  entailment-bank            —        unavailable: not downloaded; no importer written yet
  babyai                     —        unavailable: minigrid is not installed (pip install neuralmind[games])
  textworld                  —        unavailable: textworld is not installed (pip install neuralmind[games])
  build-a-module             —        unavailable: the module builder is P2.8 and is not built

failures by cause, most upstream first:
  perception              272   the text layer extracted the wrong symbols
```

Every stage carries the target it has to beat, so "ok" is a comparison and not
an opinion. `--full` widens the same stages: all CWA test splits (113,632
questions), the full NatLang set, and the 10,000-input fuzz run — about an hour,
mostly the fuzz.

### Three things a plain test suite does not do

**It gates.** There is no information in a NatLang score from a mind that
cannot do generated ProofWriter, so that stage does not run until the earlier
one passes, and a blocked stage names the stage that blocked it. `unavailable`
deliberately does *not* block: a missing dataset should not hide a stage that
would otherwise run.

**It says what it cannot do.** Five stages are listed precisely because they
cannot run. A curriculum that omits what it cannot reach reports a smaller,
better-looking mind than the one that exists — and the five reasons are four
different kinds of problem (no mirror, no importer, no dependency, not built
yet), which is information a silence would throw away.

**Every failure has exactly one attribution.** Seven categories, and the
*order* is the design:

| # | Category | What it means | Remedy |
|---|---|---|---|
| 1 | `engine` | the reasoner disagreed with a gold answer it should have got | fix the engine — nothing downstream matters |
| 2 | `budget` | the answer ran out of time | raise the budget or speed the specialist |
| 3 | `specialist` | no specialist accepted the subgoal | write or enable one |
| 4 | `wrong_context` | the theory or question did not match the host | fix context discovery |
| 5 | `perception` | the text layer extracted the wrong symbols | widen the readers |
| 6 | `missing_knowledge` | a fact nobody supplied | supply it, or ask |
| 7 | `missing_rule` | the rules stalled short of the goal | learn or write the rule |

A failure usually looks like several of these at once. A question that ran out
of budget *also* has an unproved goal, so it *also* looks like a missing rule —
and counting it in both places is how a report ends up recommending you write
rules when the real problem is a slow specialist. Priority order makes "exactly
one attribution" a claim about which cause is furthest upstream, not a claim
that causes never overlap. Engine failures come first because an engine that is
wrong makes every number below it meaningless.

### The dashboard

Each run appends to `reports/school.json`, which keeps the last twenty runs and
a `regressions` list naming any stage that scored worse than its previous run.
A regression fails **by name**, rather than by someone noticing a table looks
different. The file survives being corrupt or truncated — a broken dashboard
starts a new history instead of taking the curriculum down with it.

### It found a real bug on its first run

This is the strongest argument for having built it. The first end-to-end run
crashed:

```
ValueError: arithmetic on non-numeric terms: bob + 1
```

Nothing in the growth stage does arithmetic on people. What happened: an earlier
fuzz stage had a rule `n(Y) :- n(X), Y = X + 1.` accepted into a layer. A later
unstratified *candidate* forced `local_strata()`, which grounds the whole merged
program — and there, `n(Y) :- n(X), Y = X + 1.` grounds against `bob`, a
constant from the family fixture, and the grounder raised.

Fixed in two places, because the bug is two bugs:

- `Program.ground()` now skips an instance whose arithmetic cannot be evaluated.
  An instance that can never fire is not part of the grounding, and raising over
  it is the grounder having an opinion about a rule it was only enumerating.
- `Firewall.check` now catches bare `Exception`. A firewall that can be made to
  raise is not a firewall — a rule that crashes the checker has to come back
  `rejected`, not as a traceback in the caller.

Neither stage alone reaches this. Only running them in one process, in order,
against one accumulated state does — which is what a curriculum is for and what
15 separate test files were never going to find.

### And then a bare environment found two more

Running the same curriculum in a virtualenv with nothing installed but the
standard library was the second most useful thing the school did:

```
curriculum: 6 passed, 0 failed, 9 unavailable, 0 blocked (quick, 31s)
```

Before the fixes it was `6 passed, 1 failed, 7 unavailable, 1 blocked` — and
both of those numbers were lies.

- **An availability check raised instead of reporting.** `_mnist_available`
  imported the loader before asking whether NumPy was installed, and the loader
  imports NumPy at module level. So in the one environment the check exists for,
  it threw — and took down the listing of what *else* could have run, which is
  the whole point of the school. `Stage.why_not()` now wraps every check and
  turns an exception into the reason, because a check is the likeliest thing in
  the curriculum to blow up: finding out whether a dependency is there means
  importing something.
- **Two stages reported a number that meant something different.**
  `mixed-specialists` scored 34% with three of its four backends absent, and
  `proofwriter-natlang` scored 51.7% instead of 72.9% with no spaCy. Both read
  as "the mind cannot do this" when the truth was "pint is not installed". A
  measurement whose meaning depends on the environment is worse than no
  measurement, so both stages now declare what they need and step aside.

That second one is a rule worth stating plainly: **a stage either measures the
thing it claims to measure, or it is unavailable.** There is no third state
where it measures something else and reports the number anyway.

### What P2.13 does not do

- **Five stages are declared, not run.** bAbI has no reachable mirror from this
  machine; EntailmentBank's repo is reachable but has no importer; BabyAI and
  TextWorld need packages that are not installed; the module builder is P2.8.
- **`--full` is not run on every commit.** An hour is too long for that, so the
  quick pass is the gate and the full numbers are reproduced deliberately.
- **Targets are floors, not contracts.** A stage that beats its target by a
  large margin is not flagged, so a target that has gone stale stays stale until
  someone raises it.
- **Nine of fifteen stages are unavailable in a bare environment.** That is
  honest rather than good. The quick pass is only a real gate where the optional
  dependencies are installed.

## P2.4 — working out where it is

`Mind.__init__` has taken no domain argument since P2.0, with a test asserting
it never will. This is the milestone that makes that rule survivable rather
than merely principled.

### Shape first, and nothing else

An observation becomes facts about its *shape*: "records carry a field named
stock", "a value is a number with a currency attached", "this field changed
between ticks". Never "this is a shop". Whether those add up to a shop is a
question for rules, which a person can read and disagree with, rather than for
a classifier, which they cannot.

The facet rules live in `neuralmind/self/profiles/facets.lp` and run on the
ordinary engine, so a context hypothesis is a derivation:

```
facet(inventory, 7)  (by facet(F, W) :- suggests(F, W), not beaten(F, W).)
├── suggests(inventory, 7)  (by suggests(inventory, 7) :- record_with(stock), record_with(sku).)
│   ├── record_with(stock)  [given]
│   └── record_with(sku)  [given]
└── not beaten(inventory, 7)  [not derivable]
```

### Nine unlabelled hosts

| Host | Stakes | Domain | Covered | Could not place |
|---|---|---|---|---|
| grid game | low | game | 0.44 | position, direction, actions |
| text game | low | — | 1.00 | |
| CAD parameters | low | engineering | 0.67 | part, material |
| banking chat *(real: BANKING77)* | **high** | bank_chat | 1.00 | |
| service dialogue | low | — | 1.00 | |
| financial tables | medium | investing | 0.83 | period |
| grocery feed | medium | grocery | 0.50 | product, category |
| pump system *(no pack covers it)* | low | engineering | 0.50 | pump_a, valve_b, alarm |
| money transfer | **high** | banking_records | 0.50 | account_id, counterparty |

Being straight about the data: the banking chat and the out-of-scope stream
are **real utterances** (BANKING77 and CLINC150). The rest are synthesised *in
the shape of* the roadmap's hosts — they mirror the structure, because
structure is what the mind reads, but none of them is the real corpus and
calling them "eight hosts" without saying so would be overselling.

The pump system is the interesting one. It is outside every facet pack, and the
right answer is what it gives: the facets it does recognise (`quantities`),
and the vocabulary it could not place named rather than ignored. That list is
what the module builder (P2.8) is handed.

### Caution before recognition

Stakes are raised **before** any domain is matched. After a *single*
observation of a money-moving host:

```
stakes high, domain banking_records at 0.40
L2: act after confirmation; granted L3, bound by stakes
transfer(500): confirm — this needs a person to confirm it
```

The mind has not worked out where it is and is already careful, which is the
only order that is safe. The context can lower autonomy and can never raise
it — design rule 11, now reachable through a second path.

### Reading an utterance

A sentence has almost no shape, so shape alone sees "free text, a question,
first person" and stops. `neuralmind/self/intent.py` is a TF-IDF
nearest-centroid classifier trained on BANKING77 + CLINC150 — it classifies
and never generates, and the out-of-scope threshold is *calibrated* on held-out
data rather than chosen:

| target in-scope recall | family | intent | out-of-scope kept out |
|---|---|---|---|
| 0.85 | 82.0% | 68.6% | 67.3% |
| **0.90** | **86.9%** | **71.5%** | **56.1%** |
| 0.95 | 92.4% | 74.3% | 31.9% |

A bag-of-words baseline, and the numbers are a baseline's. The roadmap wants
multilingual-e5 here and is right to; what this buys is a legible out-of-scope
decision with no download and no new dependency — a cosine against a centroid
of counted words can be taken apart by hand.

### What it cost to get right

- **Drift cannot be seen from an accumulated signature.** Once the mind has
  seen a shop it has seen one for ever, so comparing against the running total
  never notices anything. It needs a window over recent observations.
- **And it is *loss*, not disjointness.** While the window straddles two hosts
  it shows both, which looks like agreement with each. Drift is what was
  settled disappearing.
- **The baseline slid during the transition.** Updating it every steady step
  meant it absorbed the new host before the switch was noticed, so it has to
  hold at the last settled point.
- **`stakes(medium) :- not stakes(high)` is recursion through negation.** It
  reads perfectly and has no least model; two flag predicates express the same
  thing and stratify.
- **"A is the smallest" is not `A <= B, A <= C`.** That guard only fires when
  the facets happen to be listed in the right order, so `domain(grocery)` was
  never derived. The minimum has to be computed.
- **Coverage over-credited itself.** `record_with` is emitted for every field,
  so any facet citing it marked the whole vocabulary explained. Specific
  evidence credits what it names; only a short list of genuinely generic facts
  credits everything of their kind.

### What P2.4 does not do

- **No embeddings.** Vocabulary matching is exact, so `amount` and `menge` are
  different symbols. The roadmap closes that with multilingual-e5.
- **Seven of nine hosts are synthesised.** Real BabyAI, TextWorld, MultiWOZ and
  FinQA streams would be a better test and are not reachable from here.
- **Calibration is recorded, not used.** `SelfModel` tracks whether confidence
  held up; nothing yet consults it to soften a brief line.

## P2.3 — the growth loop

Phase One could learn a rule when you handed it examples and named the target.
That is a tool, not growth. What was missing is the part that notices a gap
without being pointed at one, works out which question would settle it, and
then *does not use the answer* until someone confirms it.

### Gaps are collected, not inferred

Every part of the system already reports its own failures — `why_not()` names
the literal a rule stalled on, perception reports refused sentences, a
three-valued answer carries its diagnosis. None of that was built for learning
and all of it is what a learner needs. So a gap always arrives with its
evidence and never has to be justified after the fact.

### The question that settles the most

When several definitions fit equally, the examples have not determined the
answer. A probe *splits* the candidates, so the best one splits them most
evenly — the halving argument. Measured on the family domain, one seed example
each:

| Target | Active | Random | Learned correctly |
|---|---|---|---|
| `grandparent/2` | **6** | 80+ | yes |
| `sibling/2` | **13** | 80+ | yes |
| `aunt/2` | **8** | 80+ | yes |
| `ancestor/2` *(recursive)* | **9** | 80+ | yes |
| **total** | **36** | 320+ | active needs **11%** |

The bar was ≤50%. Random hit its 80-question cap on every target, so 320 is a
floor on what it costs, not the cost.

### Refutation probes, and why they were needed

Pure disagreement-based selection cannot ask about a hypothesis it has not
formed. `ancestor` is the case: with examples that are all parent-child pairs,
every candidate is the base clause, they agree on everything, and the loop
settles confidently on a definition missing its recursion.

So when no probe splits the candidates, the loop asks one they all say *no* to
— a "yes" refutes the whole space at once. Ordering matters more here than for
splitters, because nothing predicts the answer: **compositions first**. If
`p(a,b)` and `p(b,c)` both hold, `p(a,c)` is exactly what a transitive
definition would add and a non-recursive one would not.

These are budgeted (three consecutive noes and it settles). Without a budget
the loop asks them until it runs out — 74 questions for `grandparent` instead
of 6, since a "no" confirms what every candidate already said. Three noes are
weak evidence; what the sample buys is catching the case where *nothing* fits,
not proving the definition right.

### Nothing is believed until it is confirmed

A settled definition is a **proposal**. It is stored with belief state
`proposed`, and `Memory.to_asp()` returns only `confirmed` beliefs, so a
proposal cannot answer anything. Confirming runs it through the safety kernel,
which refuses it anyway if it breaks something — and then the proposal is
`retired`, not silently dropped.

Re-learning something a person retired does **not** resurrect it. "Latest
write wins" would quietly undo their decision.

### Predicate invention

A knowledge base grown by induction grows sideways: the learner cannot say
"this combination is a thing", so it spells the combination out again. The
consolidator finds conjunctions that recur, proposes a predicate, and rewrites
— then *verifies* that the model is unchanged, because "this should be
equivalent" is how equivalence-preserving transformations stop preserving
equivalence. The name is a placeholder and is labelled as one; what the
conjunction means is the part a person knows.

Finding nothing in the hand-written access policy is the correct answer there:
`cleared`, `permitted_by_role` and `blocked` *are* the invented predicates
already. That has its own test, so a future change that starts "finding"
things there gets looked at.

### What it cost to get right

- **Closed-world negatives killed recursion.** Falling back to the learner
  passed `negative=None`, which completes every unasked pair as false —
  including the true ones nobody had been asked about yet, which is exactly
  the evidence a recursive clause needs and was being rejected for using.
- **Comparisons are off by default in `LanguageBias`.** `sibling` is
  `parent(P,X), parent(P,Y), X != Y`; without the disequality the true rule is
  not in the space at all and the loop settles confidently on something else.
- **"I don't know" looped forever.** The atom was removed from the random
  walk's universe but not from the picker's, and the active strategy re-ranks
  from scratch each round.
- **A canary on the thing being learned refuses the lesson.** Correct
  behaviour, and a trap: learning is *supposed* to change that answer, so
  canaries belong on what you are not trying to learn.
- **The family was too small.** With five people, genuinely different
  definitions derive identical atoms and "settled" meant "the data cannot
  tell". That is a fact about the data, not the loop, and the fix was more
  data rather than a cleverer tie-break.

### What P2.3 does not do

- **`:grow` asks one question per invocation.** The shell cannot block for an
  answer, so a cycle is several turns: it asks, you state the fact (or its
  strong negation), you run it again.
- **The version space is single-clause.** Multi-clause and recursive
  definitions come from the learner's sequential covering, which is bolted on
  as an extra candidate rather than enumerated.
- **Consolidation does not follow `Compare` literals.** `R1 < R2` is skipped,
  so `toxic_combination` cannot be factored.

## P2.2 — the safety kernel

This lands before the growth loop for the reason the roadmap gives: anything
that changes itself needs a guaranteed way back, and building the changing part
first would mean a period where there was none.

### Eight mechanisms, one idea

A mind that learns will be wrong sometimes. It will induce a rule from two
examples, accept a pack that contradicts something it knew, read a sentence
badly. None of that is avoidable. What is avoidable is letting any of it reach
what the mind came with.

| Module | What it stops |
|---|---|
| `layers` | core → confirmed → tenant → session → sandbox. The core is read-only at runtime; the sandbox is excluded from queries by default |
| `snapshot` | a risky step commits or leaves no trace; an exception rolls back and re-raises |
| `canaries` | questions with known answers, checked after every change — including `unknown → yes`, which is what an over-general rule does |
| `firewall` | safety, stratification, consistency with the core, and a budgeted dry run, all against a **copy** |
| `quarantine` | what failed is disabled with the reason and a repeat count, never quietly deleted |
| `modes` | full → core only → observe only, stepping down on trouble and up only on evidence |
| `autonomy` | L0–L3: `min(grant, stakes ceiling)`. Caution rises on a guess; autonomy rises only on a grant |
| `privacy` | personal data routed into a confined layer, held for review when generalised, deleted on a retention sweep |

`Kernel.learn()` wires them so the safe path is the short one — firewall,
snapshot, canaries, rollback or quarantine — and there is no briefer way to add
a learned rule that skips any of it.

### The asymmetry that matters

Stakes are inferred; grants come from the host. So stakes can only ever
*lower* the level in force, and there is deliberately no API to lower stakes
once raised. Guesswork that raises autonomy is a way for a misread observation
to authorise a transfer; guesswork that lowers it is a way for a misread
observation to be annoying. Those are not comparable risks, so one of them is
impossible rather than unlikely.

### Measured

**Fuzz.** 10,000 random, malformed and contradictory inputs — junk bytes,
unbalanced syntax, unsafe rules, recursive negation, contradictions with the
core, over-general rules, unbounded recursion. After every single one the core
was byte-identical and every canary still gave its answer.

| | |
|---|---|
| inputs | 10,000 in 3,858s |
| accepted | 1,605 |
| rejected | 8,395 |
| quarantined | 5,839 |
| core modified | **never** |
| canary failures surviving a rollback | **none** |
| final mode | `full` |

Where the firewall caught things: parse 2,370, safety 2,059, stratification
1,273, budget 107, consistency 3. The rest were caught by the canaries and
rolled back, or refused outright because that exact rule was already
quarantined.

Two of those numbers repay a careful reading. **Consistency is only 3** because
a rule that contradicts the core is quarantined on its first attempt and
refused by name afterwards — the check runs once per distinct rule, not once
per attempt. And the mode stayed **full** throughout: rolled-back changes do
not degrade (see below), and one accepted rule clears the strike count, so a
mind that is mostly succeeding does not step down for occasional failures.

The run takes just over an hour, and almost all of it is the tail. The fuzz
accepts 1,605 junk-but-valid rules, so by the end every check runs against a
1,590-rule theory — the fuzz being adversarial about volume as well as
content, not a cost a real session pays.

```bash
python benchmarks/fuzz_kernel.py    # exits non-zero on any breach
```

**A bad pack.** Injected with all three failure modes at once (an unsafe rule,
a contradiction with the core, runaway recursion): each is quarantined with the
check that caught it, and the mind keeps answering from the core.

**Tenants.** Marker records planted in one tenant appear in no part of
another's instance — not its answers, not its rules, not a snapshot of it.

**Autonomy.** No action above the granted level reaches the host, and every
decision is logged with which of the two bound it.

### One redundancy worth removing

`learn()` ran the firewall's dry run and then solved the knowledge base again
to check the canaries. Those are the same program — the dry run solves exactly
what will exist if the rule is admitted — so the firewall now hands its model
over. **1.87×** on the accepted path at 500 rules, and the saving grows with
the knowledge base.

It makes no difference to the fuzz figure, which is dominated by *rejected*
inputs: those never reach the canary check and only ever paid one solve. Worth
saying, because the first A/B looked like a 1.3× win and was measuring a
smaller program rather than a faster path.

Reading answers off a model needed one thing the model did not carry: whether
an underivable atom is false or merely unknown is a property of the *program*,
not of the model. `ForwardChainer` now records the program it solved, which is
the honest fix — a model that cannot say which of the three answers an absence
means is a model you can misread.

### What it cost to get right

- **`recover()` checked the wrong mode.** A degraded mode answers differently
  on purpose: core-only cannot see the facts a canary was captured with, so
  checking there said "still broken" forever and the mind never came back up.
  The question that matters is whether the mode being stepped *into* is healthy.
- **A rolled-back change was degrading the mode.** A canary failure the
  rollback already fixed is the system working, not the system failing.
  Stepping down is for a failure still present after the way back was taken —
  otherwise one bad induced rule disabled learning entirely.
- **The first fuzz harness tested the wrong claim.** It checked core-only
  answers against canaries captured with facts from another layer, which fail
  for a reason that has nothing to do with safety. "The core is read-only" is a
  claim about the core's own text, and is now checked as that.

### What P2.2 does not do

- **No GLiNER entity typing.** `PrivacyPolicy` takes a detector and works
  without one; host-declared predicates and a short list of patterns are the
  default. Patterns find what they match — a host that knows its schema should
  declare it rather than hope.
- **Nothing stops a caller reaching past the kernel.** Python does not work
  that way, and a kernel that claimed otherwise would be lying. What it does is
  make the checked route the convenient one and everything it refuses visible.
- **No packs yet.** `quarantine` handles rules; packs arrive with P2.14.

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
