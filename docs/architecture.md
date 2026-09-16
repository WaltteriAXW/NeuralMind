# Architecture

The system is five layers with narrow interfaces. Each is replaceable; the
contract between them is ground atoms with confidences.

```
raw input ─→ perception ─→ knowledge base ─→ inference ─→ Type 5 check ─→ output
```

## 1. Perception (`neuralmind/perception/`)

The only place anything is learned. Its output is always
`FactRecord(atom, confidence, provenance, evidence)` — never a conclusion.

**Text.** Two extractors, used for different jobs rather than as fallbacks.
`ControlledEnglishParser` handles sentences that state *rules*, because a
dependency parse does not tell you a sentence is an implication — the grammar
does. `SpacyTripleExtractor` handles sentences that state *facts*, where the
parse copes with modifiers, prepositions, conjunctions and passives.
`TextPerceptor` routes between them and works without spaCy installed.

Both run verbs through `base_verb()` so the grammar's `purrs` and spaCy's
lemmatised `purr` land on the same predicate. Without that the two paths
produce facts that never unify — a bug this project shipped briefly and now has
a regression test for.

**Confidences mean different things by source, and the code says which.** A
spaCy-derived fact carries a *structural reliability weight*: an SVO triple read
straight off the parse (0.90) is more trustworthy than one recovered through a
preposition (0.75). These are an ordering, not calibrated probabilities. A CNN
digit fact carries an actual softmax probability. Only the second is a
probability, and only the second should be treated as one.

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

**Two honesty mechanisms**, both added after the implementation misled its own
author:

* If the background already entails a positive example, the hypothesis is
  credited with coverage it did not earn. `strip_target=True` removes the
  target's existing definition by default, and `Hypothesis.leaked` reports it
  loudly when that is switched off.
* `Examples.closed_world` turns every unlisted atom into a negative. With an
  incomplete positive list the correct rule derives an unlisted one and is
  rejected for it. The CLI says so when a closed-world run finds nothing.

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
