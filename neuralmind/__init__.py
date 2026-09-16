"""NeuralMind -- a non-LLM neurosymbolic reasoning system.

A Type 3 pipeline in Kautz's taxonomy: small task-specific neural models do
perception, a symbolic engine does all the reasoning, and the proof tree the
engine produces *is* the explanation. No generative language model anywhere.

Quick start::

    from neuralmind import NeuralMindPipeline

    pipeline = NeuralMindPipeline()
    result = pipeline.run(
        "Bob is a cat. All cats are mammals. "
        "If something is a mammal then it is warm blooded.",
        question="Is Bob warm blooded?",
    )
    print(result.proof)

Learn the rules instead of writing them::

    from neuralmind import KnowledgeBase, learn_rules

    kb = KnowledgeBase().add_facts(["parent(a, b)", "parent(b, c)"])
    hypothesis = learn_rules(
        kb, "grandparent/2",
        positive=["grandparent(a, c)"], negative=["grandparent(c, a)"],
        body_predicates=["parent/2"], allow_recursion=False,
    )
    print(hypothesis)   # grandparent(A, B) :- parent(A, C), parent(C, B).

Or purely symbolically::

    from neuralmind import KnowledgeBase

    kb = KnowledgeBase().load_builtin("family")
    kb.add_facts(["parent(alice, bob)", "parent(bob, carol)"])
    print(kb.engine().ask("ancestor(alice, carol)").proof)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .consistency.fuzzy import FuzzySemantics
from .consistency.layer import ConsistencyLayer, ConsistencyReport
from .core.parser import parse_atom, parse_file, parse_program
from .core.program import Program, Rule
from .core.terms import Atom, Const, Literal, Var
from .induction import Examples, Hypothesis, LanguageBias, RuleLearner, learn_rules
from .inference.engine import Answer, ReasoningEngine
from .inference.model import Model
from .inference.proof import ProofNode, explain
from .knowledge.base import FactRecord, KnowledgeBase
from .mind import Mind, Observation
from .output.brief import Brief, brief
from .output.nlg import Realiser
from .output.render import render_model, render_proof
from .output.serialize import to_json
from .perception.base import Perception
from .pipeline import NeuralMindPipeline, PipelineResult


def __getattr__(name: str):
    """Expose the learning layer without making NumPy a hard dependency."""
    if name in ("SemanticLoss", "NeuralPredicate"):
        from . import learning

        return getattr(learning, name)
    if name in ("WeaklySupervisedTrainer", "WeakExample"):
        from .learning import weak

        return getattr(weak, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    # pipeline
    "Mind",
    "Observation",
    "NeuralMindPipeline",
    "PipelineResult",
    # knowledge
    "KnowledgeBase",
    "FactRecord",
    "Program",
    "Rule",
    # reasoning
    "ReasoningEngine",
    "Answer",
    "Model",
    "ProofNode",
    "explain",
    # symbols
    "Atom",
    "Const",
    "Var",
    "Literal",
    "parse_atom",
    "parse_program",
    "parse_file",
    # induction (pure Python, no NumPy needed)
    "RuleLearner",
    "LanguageBias",
    "Examples",
    "Hypothesis",
    "learn_rules",
    # consistency
    "ConsistencyLayer",
    "ConsistencyReport",
    "FuzzySemantics",
    # learning (needs NumPy; imported on demand)
    "SemanticLoss",
    "NeuralPredicate",
    "WeaklySupervisedTrainer",
    "WeakExample",
    # perception and output
    "Perception",
    "Realiser",
    "Brief",
    "brief",
    "render_proof",
    "render_model",
    "to_json",
]
