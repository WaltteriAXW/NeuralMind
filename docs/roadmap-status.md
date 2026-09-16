# Roadmap status

The blueprint's phase table, with what was built and what was measured. Every
milestone is asserted in `tests/test_roadmap_phases.py`.

| Phase | Done-when | Status |
|---|---|---|
| 0 — Setup | Call the solver from Python and get a result back | Done. `clingo` 5.8.2 via the optional ASP backend; the pure-Python engine needs nothing at all. |
| 1 — Toy pipeline | MNIST digit-pair addition, >90% end to end on 100 images | **98/100.** A 12,810-parameter CNN (98.5% on the MNIST test set) plus `mnist_sum.lp`. |
| 2 — Knowledge base | 10–30 rules for one domain; correct on hand-written cases | Done, four times over: `family.lp` (24 rules), `access_policy.lp` (30+ rules), `mnist_sum.lp`, `triples.lp`. Each cross-checks against clingo. |
| 3 — Perception integration | 80%+ of a held-out set produces correctly extracted facts | **100%** of generated theories extracted exactly (0 theory mismatches over 150 problems). See the caveat below. |
| 4 — Consistency layer | Correctly flags rule violations it previously missed | Exceeded: it **repairs** them. 10/10 misread digit pairs recovered from the sum constraint. |
| 5 — Explanation layer | A non-technical person can read the output and understand *why* | JSON proof trees, ASCII trees, and template-generated prose. |
| 6 — Domain pilot | Ship something narrow but real | Access-policy compliance checker: RBAC, role inheritance, a classification lattice, training gates, separation of duties, four-eyes. |
| 7 — Evaluate & iterate | Say in one sentence what fraction of failures are perception vs rules | Built into `neuralmind eval`. **1200/1200** on generated problems to proof depth 4. |

## Reading the Phase 3 and 7 numbers honestly

They are measured on **generated** controlled-English problems in the
ProofWriter register, not on the real ProofWriter dataset. The perception layer
covers that register by construction, so a perfect score means the grammar and
the generator agree — which is worth knowing, but is not the same as handling
ProofWriter's messier phrasing.

What the numbers do establish: the reasoning is exact (0 engine failures across
every run), so end-to-end accuracy is bounded entirely by perception. That is
the architectural claim, and it holds.

The gold labels come from `datasets/proofwriter.closure`, a plain-Python
fixpoint that shares no code with the inference engine. Grading the engine with
the engine would be circular.

## Deviations from the blueprint, and why

**Stanford CoreNLP OpenIE → spaCy dependency rules.** The blueprint's own
licensing section recommends this: CoreNLP is GPL, and a subject/verb/object
extraction over spaCy's parse keeps the whole stack MIT/Apache. Done as
recommended.

**Scallop / LTNtorch → implemented directly.** The blueprint flags both as
research-grade with slowing maintenance, and says to confirm Scallop's licence
before depending on it. The Type 5 semantics needed here are the Real Logic
connectives and the `forall` aggregator, about 150 lines; inference-time repair
is exact weighted model counting, another 80; and training-time gradients are a
truth tensor plus two contractions, in `neuralmind/learning/`. Implementing them
directly removed a dependency the blueprint was itself cautious about, and kept
the connectives configurable. What a real Scallop integration would still buy is
scale — circuit compilation instead of a dense tensor.

**PyTorch → NumPy.** For a 13k-parameter CNN the difference does not matter, and
the pipeline stays installable with no heavy dependency. Swapping in a torch
model means implementing `predict_proba`; nothing above the perception layer
changes.

**SimpleNLG → a template realiser in Python.** SimpleNLG is Java. The English
needed here is articles, agreement, capitalisation and list punctuation, which
is small enough to do directly and keeps the stack to one language.

## Beyond the roadmap: gradients through the logic

The blueprint stops at Phase 7, and the Type 5 layer as specified checks and
repairs at *inference* time. `neuralmind/learning/` adds the training-time
counterpart -- the DeepProbLog/Scallop capability proper -- by exact weighted
model counting over a truth tensor the symbolic engine fills.

The demonstration is MNIST addition with **sum-only supervision**: pairs of
images labelled with their sum and never with a digit.

| Supervision | Digit labels | Test accuracy |
|---|---|---|
| Per-image digit labels | 30,000 | 98.48% |
| Pair sums only | 0 | 98.20% |

Same architecture, same 30,000 images, same optimiser; 12 epochs against 10.
Both checkpoints are committed under `neuralmind/perception/weights/`, and
`tests/test_learning.py` asserts the weak one still hits its number, so the
claim cannot rot.

It matters because it answers the obvious objection to a Type 3 pipeline -- that
the neural and symbolic halves must be trained separately, and that the
perception layer therefore needs its own labelled data. It does not.

## Beyond the roadmap: learning the rules

The source report names the knowledge acquisition bottleneck as the central
practical cost of this architecture, and points at ILP (Popper) as the
mitigation. `neuralmind/induction/` implements generate-test-constrain search
over a bounded hypothesis space, with the inference engine judging candidates.

Measured on the classic targets: `grandparent` in 12 candidate tests, recursive
`ancestor` (two clauses, found in the right order) in 43, and the negated
exception `flies(A) :- bird(A), not penguin(A).` in 5.

Together with the weak-supervision layer this closes both halves of the
bottleneck as far as they can be closed: perception no longer needs labels, and
rules no longer need to be written by hand *when examples of the relation exist
and the bias can be stated*. Neither condition is free, and neither replaces
knowing the domain.

## What is genuinely not built
- **Vision beyond digits.** No Detectron2 or SAM integration. The perceptor
  interface takes any model with `predict_proba`, so the work is wiring, not
  architecture.
- **Ontology learning.** `induction/` learns a definition of one predicate from
  examples of it. It does not invent predicates, propose a type hierarchy, or
  decide what the domain's concepts should be. That is still the expert's job,
  and it is the larger part of Phase 2.
- **Large-scale knowledge.** No Wikidata or ConceptNet import. `knowledge/rdf.py`
  is the bridge, but nothing is loaded through it at scale, and the forward
  chainer is not a Datalog engine for millions of facts — that is what Soufflé
  is for.
- **Open-domain language.** Stated plainly in the README. The controlled grammar
  covers a register, not English; the narrative reader covers a second register
  approximately, and `prefer="auto"` picks between them per input. Together
  they take ProofWriter's crowdsourced split from 52% to 70%, which is
  progress and is not a solution. The remaining errors are genuine paraphrase
  problems — idioms read as attributes, world knowledge the text assumes — and
  no amount of further dependency-pattern work reaches them.
