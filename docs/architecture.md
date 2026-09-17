# Architecture

The system is five layers with narrow interfaces. Each is replaceable; the
contract between them is ground atoms with confidences.

```
raw input ─→ perception ─→ knowledge base ─→ inference ─→ Type 5 check ─→ output
```

## 0. The entry point (`neuralmind/mind.py`)

`Mind` is what a host holds. It wraps the knowledge base, engine and perceptor,
and takes **no domain, pack or mode argument** — a test asserts it never will.
A host that has to declare what it is has already made the mind's hardest
decision for it, and later phases work the situation out from observations
instead.

Answers are three-valued. `unknown` is a real answer, not a soft `no`, and
which one applies is declared per predicate (`#open`, `#closed`, or a bare
`#open.` for the whole program). A `no` can be proven rather than assumed,
through strong negation: `-flies(pingu)` is a claim, `not flies(pingu)` is an
absence, and deriving both `p` and `-p` is reported as a contradiction.

`Mind.solve()` goes through the workspace (below) rather than straight to the
engine, so a question that needs numbers, units or paths as well as rules gets
one answer with one proof.

See `docs/phase-two-status.md` for what is built and what is not.

## 0b. The workspace (`neuralmind/workspace/`)

A blackboard, four specialists and a controller over them. The blackboard is
the right shape because a pipeline cannot hold two kinds of reasoning that are
not downstream of each other — a constraint solver and a graph search have
nothing to say to one another through a pipe.

Everything posted is a ground atom with a justification, so a result from the
arithmetic specialist is, to the logic engine, indistinguishable from a given
fact. That one property is what makes a proof tree span all four specialists
instead of stopping at the first seam.

The controller works an agenda under a cooperative time budget and always
answers: running out of time yields `unknown` with the reason, never a hang.
Every specialist but `logic` is optional, and a missing backend costs exactly
its own questions.

## 0c. The safety kernel (`neuralmind/kernel/`)

Knowledge in five layers, core first and read-only at runtime, sandbox last and
excluded from queries by default. Everything the mind *learns* goes through
`Kernel.learn()`: firewall (safety, stratification, consistency with the core,
a budgeted dry run — against a copy), inside a transaction, with canaries
checked afterwards and a rollback plus quarantine when one fails.

Two asymmetries are load-bearing. Personal data is routed into a confined layer
whatever the caller asked for, because the caller asking for the wrong layer is
the mistake the mechanism exists to stop. And autonomy is the *minimum* of the
host's grant and what the inferred stakes allow, with no way to lower stakes —
caution rises on a guess, freedom only on a grant.

## 0d. The growth loop (`neuralmind/growth/`)

Gaps are *collected* from diagnoses the system already produces, never
inferred, so each arrives with its evidence. Induction over a gap keeps the
definitions that **compete** rather than only the winner, because the
competitors are what a question is for: the best probe splits them most evenly.

When they all agree, the loop asks one they all say no to — the only way out of
a space that does not contain the answer, and what makes recursive definitions
reachable at all.

Nothing here writes to the knowledge base. A settled definition is a proposal;
the safety kernel decides whether it may be believed. That split is why the
kernel came first.

## 1. Perception (`neuralmind/perception/`)

The only place anything is learned. Its output is always
`FactRecord(atom, confidence, provenance, evidence)` — never a conclusion.

**Text.** Three extractors, used for different jobs rather than as fallbacks.
`ControlledEnglishParser` handles sentences that state *rules*, because a
dependency parse does not tell you a sentence is an implication — the grammar
does. `SpacyTripleExtractor` handles sentences that state *facts*, where the
parse copes with modifiers, prepositions, conjunctions and passives.
`NarrativeExtractor` handles free-form prose, where the distinction between a
rule and a fact is itself a question about the sentence's structure:

> Charlie is green, but often kind, even when he is blue and cold.

Four facts in one sentence, a pronoun standing in for the subject, and an "even
when" that looks like a conditional and is not. The controlled grammar refuses
this outright. The narrative reader splits the sentence into spans and collects
every predicative adjective in each, resolves pronouns to the previous
sentence's subject, and decides *per sentence* whether a marker conditions
(opening the sentence, or before a "then") or merely modifies.

`TextPerceptor` routes between them and works without spaCy installed. With
`prefer="auto"` the routing is decided per input by the grammar's own refusal:
if it reads every sentence it is exact and wins; if it refuses any, the text is
outside its register and the narrative reader takes the whole passage. That
rule exists because the two readers are near-opposites on the ProofWriter
corpus — neither is better everywhere — so choosing once, up front, would give
up one or the other.

All three run verbs through `base_verb()` so the grammar's `purrs` and spaCy's
lemmatised `purr` land on the same predicate. Without that the paths produce
facts that never unify — a bug this project shipped briefly and now has a
regression test for.

**Confidences mean different things by source, and the code says which.** A
spaCy-derived fact carries a *structural reliability weight*: an SVO triple read
straight off the parse (0.90) is more trustworthy than one recovered through a
preposition (0.75), and anything the narrative reader guesses at sits below
both (0.70). These are an ordering, not calibrated probabilities. A CNN digit
fact carries an actual softmax probability. Only the second is a probability,
and only the second should be treated as one.

**Vision.** `perception/nn.py` is a ~13k-parameter CNN in NumPy — conv/pool/
dense with im2col, Adam, and a finite-difference gradient check in the tests.
It reaches 98.5% on MNIST. NumPy rather than PyTorch because the network is
small enough that it does not matter and the pipeline stays installable
anywhere; a torch model drops in by implementing `predict_proba`.

The perceptor keeps the **full distribution**, not just the argmax. That is what
makes Type 5 repair possible: when the rules reject the top prediction, the
answer is usually the runner-up.

## 2. Knowledge (`neuralmind/knowledge/`)

Rules the domain expert writes, plus facts the world supplies. Facts carry
provenance and confidence; the symbolic engine ignores confidence entirely — it
reasons over what is asserted, and the Type 5 layer decides what may be
asserted.

## 3. Inference (`neuralmind/inference/`)

Semi-naive forward chaining over stratified Datalog, computing the least model.
For every derived atom it records the rule instances that derived it.

**Semi-naive evaluation.** Each round derives only what uses at least one atom
found in the previous round. A rule fires once per body position that could
match something new, with that position drawing from the delta and the rest
from the whole model; the union of those variants is exactly the set of new
derivations. Naive iteration re-derives the whole relation every round, costing
an extra factor of the number of rounds -- on a transitive closure, the chain
length. The benchmark reports atoms examined per atom derived, which is flat at
about 3 under semi-naive evaluation and grows without bound under naive
iteration.

**Join ordering.** `Rule.plan()` reorders body literals so each step shares a
variable with something already bound. Written as
`cousin(A,B) :- parent(P,A), parent(Q,B), sibling(P,Q).` the first two literals
share nothing, so source order enumerates every pair of parents before the
sibling check discards almost all of them. Reordered, each step is an indexed
lookup: on 800 families that is 4,000 atoms examined instead of 1,282,400.
Conjunction is commutative, so this changes only the cost -- and proofs are
rendered back in source order, so a reader still sees the rule they wrote.

**Proof trees fall out of that record.** `explain()` reads the tree off the
justifications; nothing is reconstructed afterwards, so a proof cannot disagree
with the answer it explains. Recursion terminates because a justification is
usable in a proof only when every support is strictly shallower than the atom
it supports — `ancestor` can support `ancestor`, but never the instance that
produced it.

**Negative answers are explained too.** `why_not()` walks each rule whose head
matches the goal and reports the furthest point its body got to, which turns
"no" into something actionable.

**Stratification, locally.** Negation must not be recursive, or there is no
unique model. Checking that at the predicate level is the textbook version and
is too coarse for real rule bases: `p(a) :- not p(b).` is perfectly well
defined, and 6-12% of the ProofWriter corpus looks like it. When the coarse
check fails, the program is ground over its own constants and stratified atom
by atom -- *local* stratification, the same perfect-model semantics on a
strictly larger class of programs. Grounding is bounded and raises rather than
exhausting memory, and genuine recursion through negation is still refused.

**clingo is the oracle.** It is faster and far more expressive, but it reports
which atoms are true, not why. `cross_check()` runs both engines on the same
program and asserts they derive the same atoms, with constraints handled
separately (the Python engine reports violations and keeps the model; clingo
answers UNSAT). This caught a real bug: clingo reads `_anon1` as a constant, so
generated anonymous variables were silently matching nothing.

## 4. Consistency (`neuralmind/consistency/`)

The Type 5 layer, in the sense of Logic Tensor Networks: the same first-order
rules, evaluated over truth values in [0, 1].

**Propagation.** `propagate()` pushes input confidences through the
derivations. A derived atom takes the fuzzy conjunction of its supports,
combined across alternative derivations with the t-conorm — two independent
reasons to believe something are better than one. Without this the check is
vacuous, because a crisply-derived atom always reads as 1.0.

**Aggregation.** Rules are aggregated with `p_mean_error`, the aggregator LTN
uses for `forall`. A plain mean lets a thousand satisfied groundings drown out
one badly violated one; for a consistency check the violation is the point.

**Repair.** `resolve()` searches the classifier's candidates for the most
probable assignment the hard rules allow — exact weighted model counting over a
small candidate set, which is what Scallop and DeepProbLog do with gradients.
Exact by enumeration is both feasible and easier to trust at this size, and
`O(candidates ** slots)` is honestly documented as the limit.

The connectives are configurable because the choice matters: product is the
probabilistic reading and the default, Łukasiewicz behaves better when many
weak facts combine, Gödel (min) is the most pessimistic and suits hard safety
checks.

## 5. Output (`neuralmind/output/`)

JSON is the primary format, and for machine-to-machine use it is also the
clearest for people — a proof tree is already a tree. `render.py` draws it as
ASCII for a terminal. `nlg.py` is a deterministic template realiser in the
spirit of SimpleNLG: articles, agreement, capitalisation, list punctuation, and
nothing more. It generates no text it was not given a template for.

## 6. Learning (`neuralmind/learning/`)

Every other layer runs the logic forward. This one runs it backward, and it is
what turns a reasoning system into a learning one.

For a program with neural predicates over a finite domain, the probability that
a query follows is a weighted model count: sum, over the assignments that
entail the query, of the product of their per-slot probabilities. The *set* of
entailing assignments depends only on the rules, not on the probabilities, so
it is computed once by the symbolic engine and stored as a boolean **truth
tensor** with one axis per neural predicate.

After that the whole thing is multilinear. The probability is a tensor
contraction, and the derivative with respect to any slot is the same
contraction with that slot's axis left out. Both are a few lines of `tensordot`,
and both are exact -- the tests check them against finite differences at 1e-10.

This is what DeepProbLog and Scallop compile to arithmetic circuits to compute.
Here the circuit is a dense tensor, which is the right representation when the
domain is small and the wrong one when it is large. The cost is
`|domain| ** slots`; `SemanticLoss` refuses to build a table past `max_table`
rather than quietly exhausting memory, and says so in the error.

Integrity constraints shape the gradient too: an assignment that violates a
hard rule contributes nothing to the count, so the network is never pushed
toward a reading the rules forbid.

**What this buys.** `scripts/train_weak_supervision.py` trains the digit
classifier on image pairs labelled only with their sum. No digit label is used
at any point. The gradient of P(sum = s) through
`sum(S) :- digit(d0, A), digit(d1, B), S = A + B.` is the entire training
signal, and it is enough.

## 7. Induction (`neuralmind/induction/`)

Layer 6 learns perception from the rules. This one learns the rules.

The loop is generate-test-constrain. **Generate**: enumerate clauses inside a
declared :class:`LanguageBias`, shortest first, so the simplest working rule is
the one found. **Test**: run the actual inference engine on background knowledge
plus the candidate -- the same engine that will run the rule in production, so a
hypothesis cannot pass here and behave differently later. **Constrain**: a body
covering no positive example cannot start covering one when literals are added,
because a conjunction only narrows; every superset of it is pruned unseen.

Three filters cut the space before anything is tested: variable safety (the
engine's own check), connectivity (a body literal sharing no variable with the
rest constrains nothing), and canonical renaming (`q(A,C),r(C,B)` and
`q(A,D),r(D,B)` are one clause, not two).

Multi-clause definitions come from sequential covering: learn a clause, set
aside what it explains, repeat -- with the already-learned clauses left in the
program while later candidates are tested. That is what lets recursion
bootstrap. `ancestor`'s base case is found first because it covers the most,
and the recursive clause can then fire on top of it.

**Repair** (`induction/repair.py`) runs the same machinery from the other end.
Given "this should follow" or "this should not", it proposes changes: narrow the
offending rule with one more condition, drop it, retract a fact, or learn a new
rule. Each proposal is measured against the model before and after, so it
carries what it fixes, what it breaks, and what it newly derives. Nothing is
applied automatically — a change that repairs one case and silently breaks four
is worse than none, and only showing both makes that decidable.

Specialisation candidates are generated over *the rule's own variables*, not a
fresh pool. That sounds like a detail and is not: a literal over unrelated
variables leaves them unbound, so every such rule is unsafe and is discarded
before it is tried. The first version did exactly that and proposed nothing but
"delete the rule".

**Three honesty mechanisms**, all added after the implementation misled its own
author:

* If the background already entails a positive example, the hypothesis is
  credited with coverage it did not earn. `strip_target=True` removes the
  target's existing definition by default, and `Hypothesis.leaked` reports it
  loudly when that is switched off.
* `Examples.closed_world` turns every unlisted atom into a negative. With an
  incomplete positive list the correct rule derives an unlisted one and is
  rejected for it. The CLI says so when a closed-world run finds nothing.
* When several clauses cover exactly the same examples, choosing one is
  arbitrary. `Hypothesis.underdetermined` reports the competing rules instead of
  presenting the arbitrary pick as a conclusion.

## The session shell (`neuralmind/shell.py`)

`Shell.handle(line) -> str` is a pure function from an input line to the text
to print; `run()` is a thin loop around it. That split is why the shell has 34
tests and none of them need a terminal, and why a session can be scripted from
Python as easily as typed.

Input is dispatched four ways -- a leading `:` is a command, a trailing `?` is
a question, logic syntax is read as ASP, everything else goes to the perception
layer -- and both readings are tried before anything is refused.

Two behaviours matter more than the convenience:

* **Nothing is absorbed silently.** Every assertion echoes the symbols it
  produced. A sentence the grammar cannot read is reported; one it can only
  guess at is marked as a guess, with the confidence the perception layer
  assigned it.
* **An assertion that would break the knowledge base is rolled back.** Without
  this, a single unsafe rule makes every later command fail and `:clear` --
  which discards the whole session -- is the only way out.

## Design decisions worth knowing

**Why a hand-written engine when clingo exists.** Proof trees. clingo gives
answers; this system's product is the derivation. The Python engine exists to
produce that, and clingo exists to keep it honest.

**Why static checks run eagerly.** An unsafe or unstratified program does not
error at runtime — it computes a *different* fixpoint. Catching it at load time
is the difference between a clear message and a wrong answer nobody notices.

**Why perception never concludes anything.** Every conclusion comes from a rule,
so every conclusion has a proof. The moment perception is allowed to infer, the
proof tree stops being complete.

**Why the inference engine tests ILP candidates, rather than a fast
approximation.** A specialised coverage checker would be several times quicker.
It would also be a second implementation of entailment, and the moment the two
disagreed the learner would be confidently producing rules that do not work.
The search is fast enough as it is -- `ancestor` takes 43 candidate tests.

**Why the truth tensor rather than circuit compilation.** Compiling to an SDD is
the scalable answer and the reason DeepProbLog can handle larger programs. A
dense tensor is exact, twenty lines, and obviously correct by inspection --
which matters more here, where the engine that fills it is the thing being
trusted. The limit is stated in the error message rather than discovered in
production.
