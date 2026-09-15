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

**Scallop / LTNtorch → a fuzzy layer implemented here.** The blueprint flags both
as research-grade with slowing maintenance, and says to confirm Scallop's
licence before depending on it. The Type 5 semantics needed here are the Real
Logic connectives and the `forall` aggregator, which are about 150 lines; the
repair step is exact weighted model counting, which is another 80. Implementing
them directly removed a dependency the blueprint itself was cautious about, and
the connectives stay configurable. A gradient-based version — training the
perception model through the logic — is where a real Scallop or LTNtorch
integration would earn its place, and that is not built here.

**PyTorch → NumPy.** For a 13k-parameter CNN the difference does not matter, and
the pipeline stays installable with no heavy dependency. Swapping in a torch
model means implementing `predict_proba`; nothing above the perception layer
changes.

**SimpleNLG → a template realiser in Python.** SimpleNLG is Java. The English
needed here is articles, agreement, capitalisation and list punctuation, which
is small enough to do directly and keeps the stack to one language.

## What is genuinely not built

- **Gradient flow through the logic.** The Type 5 layer checks and repairs at
  inference time; it does not train the CNN through the rules. That is the
  DeepProbLog/Scallop capability proper.
- **Vision beyond digits.** No Detectron2 or SAM integration. The perceptor
  interface takes any model with `predict_proba`, so the work is wiring, not
  architecture.
- **Large-scale knowledge.** No Wikidata or ConceptNet import. `knowledge/rdf.py`
  is the bridge, but nothing is loaded through it at scale, and the forward
  chainer is not a Datalog engine for millions of facts — that is what Soufflé
  is for.
- **Open-domain language.** Stated plainly in the README. The controlled grammar
  covers a register, not English.
