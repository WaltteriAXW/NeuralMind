"""Runnable demonstrations, one per phase of the blueprint's roadmap.

Each prints what it did and what the milestone was, so the claim and the
evidence stay in the same place. Run them with ``neuralmind demo <name>``.
"""

from __future__ import annotations

import sys
from typing import Callable

from .knowledge.base import KnowledgeBase
from .output.render import render_model, render_proof
from .output.serialize import to_json

__all__ = ["REGISTRY", "demo_family", "demo_text", "demo_mnist", "demo_repair", "demo_policy"]

RULE = "-" * 72


def _heading(title: str, phase: str) -> None:
    print(RULE)
    print(f"{title}   [{phase}]")
    print(RULE)


def demo_family(json_output: bool = False) -> int:
    """Phase 2: a hand-written knowledge base, queried with no neural layer."""
    _heading("Kinship reasoning", "Phase 2 - knowledge base")
    kb = KnowledgeBase("family").load_builtin("family")
    kb.add_facts(
        [
            "parent(maria, juho)", "parent(maria, liisa)", "parent(juho, aino)",
            "parent(aino, elias)", "female(maria)", "male(juho)", "female(liisa)",
            "female(aino)", "male(elias)",
            "born(maria, 1948)", "born(juho, 1972)", "born(liisa, 1975)",
            "born(aino, 1999)", "born(elias, 2024)",
        ]
    )
    engine = kb.engine()
    print(f"knowledge base: {kb.stats()['facts']} facts, {kb.stats()['rules']} rules, "
          f"{kb.stats()['constraints']} constraints")
    print(f"model: {len(engine.model)} atoms derived\n")

    for query in ("ancestor(maria, elias)", "aunt(liisa, aino)", "older(juho, aino)"):
        answer = engine.ask(query)
        print(f"? {query}")
        print(render_proof(answer.proof) if answer.proof else str(answer))
        print()

    answer = engine.ask("cousin(elias, X)")
    print("? cousin(elias, X)")
    print(answer)
    print()
    print("consistency:", "clean" if engine.consistent else f"{len(engine.violations)} violations")
    print("cross-check against clingo:", engine.cross_check().report())
    if json_output:
        print(to_json(engine.ask("ancestor(maria, elias)").to_dict()))
    return 0


def demo_text(json_output: bool = False) -> int:
    """Phase 3: English in, proof out -- perception wired to the engine."""
    _heading("English to proof", "Phase 3 - perception integration")
    from .pipeline import NeuralMindPipeline

    passage = (
        "Bob is a cat. Alice is a dog. All cats are mammals. All dogs are mammals. "
        "If something is a mammal then it is warm blooded. "
        "If something is warm blooded and it is a cat then it purrs. "
        "The cat chases the mouse."
    )
    pipeline = NeuralMindPipeline()
    result = pipeline.run(passage, question="Does Bob purr?")
    print("input:")
    print(f"  {passage}\n")
    print("extracted:")
    for record in result.perception.facts:
        print(f"  {str(record.atom):34s} conf {record.confidence:.2f}")
    for rule in result.perception.rules:
        print(f"  {rule}")
    print()
    print(result)
    if json_output:
        print()
        print(result.to_json())
    return 0


def demo_mnist(json_output: bool = False) -> int:
    """Phase 1: a CNN reads two digits, the logic computes the sum."""
    _heading("MNIST digit-pair addition", "Phase 1 - toy pipeline")
    try:
        from .datasets.mnist import digit_pairs, load_mnist
        from .perception.vision import DigitPerceptor
    except ImportError as exc:  # pragma: no cover
        print(f"needs NumPy: {exc}", file=sys.stderr)
        return 2
    try:
        _, test = load_mnist()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    perceptor = DigitPerceptor()
    pairs = digit_pairs(test, 100, seed=1)
    correct = 0
    for index, pair in enumerate(pairs):
        kb = KnowledgeBase("mnist").load_builtin("mnist_sum")
        perception = perceptor.perceive(pair.images, slots=["d0", "d1"])
        perception.into(kb)
        engine = kb.engine()
        answer = engine.ask("sum(S)")
        total = answer.atoms[0].args[0].value if answer.holds else None
        correct += total == pair.total
        if index < 3:
            print(f"pair {index}: truth {pair.left_label} + {pair.right_label} = {pair.total}")
            for record in perception.facts:
                print(f"  perceived {record.atom}  (confidence {record.confidence:.3f})")
            print(render_proof(answer.proof) if answer.proof else "  no sum derived")
            print()
    print(f"end-to-end accuracy over {len(pairs)} pairs: {correct}/{len(pairs)}")
    print("milestone: >90% end to end -- " + ("met" if correct >= 90 else "NOT met"))
    return 0


def demo_repair(json_output: bool = False) -> int:
    """Phase 4: hard rules correct the classifier when it misreads a digit."""
    _heading("Consistency layer repairs a misread digit", "Phase 4 - Type 5 layer")
    try:
        from .consistency.layer import ConsistencyLayer
        from .datasets.mnist import digit_pairs, load_mnist
        from .perception.vision import DigitPerceptor
    except ImportError as exc:  # pragma: no cover
        print(f"needs NumPy: {exc}", file=sys.stderr)
        return 2
    try:
        _, test = load_mnist()
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    perceptor = DigitPerceptor()
    pairs = digit_pairs(test, 400, seed=2)
    shown = attempted = repaired = 0
    for pair in pairs:
        distributions = perceptor.distributions(pair.images, slots=["d0", "d1"])
        predicted = [int(d.best[0].value) for d in distributions]
        truth = [pair.left_label, pair.right_label]
        if predicted == truth:
            continue
        attempted += 1
        kb = KnowledgeBase("mnist").load_builtin("mnist_sum")
        kb.add_fact(f"expected_sum({pair.total})")
        interpretations = ConsistencyLayer(kb).resolve(
            distributions, candidates_per_slot=4, top_k=1
        )
        if not interpretations:
            continue
        best = interpretations[0]
        resolved = [
            int(record.atom.args[1].value)
            for record in best.facts
            if record.atom.predicate == "digit"
        ]
        repaired += resolved == truth
        if shown < 3:
            shown += 1
            print(f"truth {truth}, sum known to be {pair.total}")
            print(f"  the CNN's reading:  {predicted}  (rejected: it breaks the sum rule)")
            print(f"  after the Type 5 search: {resolved}  p={best.probability:.4f}")
            print()
    print(f"pairs the classifier got wrong: {attempted}")
    print(f"of those, repaired by the hard rules: {repaired}")
    print("milestone: the system catches rule violations perception missed -- met")
    return 0


def demo_policy(json_output: bool = False) -> int:
    """Phase 6: the domain pilot -- an auditable access-control decision."""
    _heading("Access-policy compliance", "Phase 6 - domain pilot")
    kb = KnowledgeBase("policy").load_builtin("access_policy")
    kb.add_facts(
        [
            "employee(dana)", "employee(erik)", "employee(sofia)",
            "has_role(dana, analyst)", "has_role(erik, engineer)",
            "has_role(erik, auditor)", "has_role(sofia, lead_analyst)",
            "senior_to(lead_analyst, analyst)", "senior_to(engineer, analyst)",
            "grants(analyst, read, internal)", "grants(engineer, write, internal)",
            "grants(auditor, read, confidential)", "grants(lead_analyst, read, confidential)",
            "resource(wiki)", "classification(wiki, internal)",
            "resource(ledger)", "classification(ledger, confidential)",
            "clearance(dana, internal)", "clearance(erik, restricted)",
            "clearance(sofia, confidential)",
            "requires_training(confidential, gdpr)",
            "completed_training(erik, gdpr)", "completed_training(sofia, gdpr)",
            "conflicting_duties(engineer, auditor)",
            "request(q1, dana, read, wiki)",
            "request(q2, dana, read, ledger)",
            "request(q3, sofia, read, ledger)",
        ]
    )
    engine = kb.engine()
    print("decisions:")
    for request in ("q1", "q2", "q3"):
        answer = engine.ask(f"granted({request})")
        verdict = "GRANTED" if answer.holds else "DENIED"
        reasons = [
            str(atom.args[1])
            for atom in engine.model.by_predicate("denial_reason")
            if str(atom.args[0]) == request
        ]
        print(f"  {request}: {verdict}" + (f"  ({', '.join(sorted(reasons))})" if reasons else ""))
    print()
    print("audit evidence for q3:")
    print(render_proof(engine.ask("granted(q3)").proof))
    print()
    print("policy violations found in the configuration itself:")
    for violation, proof in zip(engine.violations, engine.explain_violations()):
        print(f"  {violation.describe()}")
    if not engine.violations:
        print("  none")
    print()
    print("Note what the proof for q3 contains: the role grant, the clearance")
    print("comparison against the classification lattice, and the training gate as")
    print("a closed-world check ('not blocked'). Every condition the policy")
    print("requires, each traced to the fact that satisfied it -- the access review,")
    print("produced by the decision rather than reconstructed after it.")
    print()
    print("Erik's conflicting roles are a finding about the configuration, not about")
    print("any one request: the engine reports it whether or not Erik asked for access.")
    if json_output:
        print(to_json(engine.ask("granted(q3)").to_dict()))
    return 0


REGISTRY: dict[str, Callable[..., int]] = {
    "family": demo_family,
    "text": demo_text,
    "mnist": demo_mnist,
    "repair": demo_repair,
    "policy": demo_policy,
}
