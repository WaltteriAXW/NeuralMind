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

**Proof trees fall out of that record.** `explain()` reads the tree off the
justifications; nothing is reconstructed afterwards, so a proof cannot disagree
with the answer it explains. Recursion terminates because a justification is
usable in a proof only when every support is strictly shallower than the atom
it supports — `ancestor` can support `ancestor`, but never the instance that
produced it.

**Negative answers are explained too.** `why_not()` walks each rule whose head
matches the goal and reports the furthest point its body got to, which turns
"no" into something actionable.

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
