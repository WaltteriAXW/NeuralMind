"""A ProofWriter/RuleTaker-style benchmark generator.

The real ProofWriter (Allen AI) is the right benchmark for this pipeline: it
gives rules and facts in natural language and asks whether a conclusion
follows. This module generates problems in the same shape so the pipeline can
be evaluated offline, with two properties the evaluation depends on.

First, every problem carries its **gold symbolic theory** alongside the English.
That is what makes failure attribution possible: if the extracted theory
differs from the gold theory, the perception layer was wrong; if it matches and
the answer is still wrong, the reasoning was.

Second, gold labels are computed by :func:`closure`, a plain-Python fixpoint
that shares no code with the inference engine. Grading the engine with the
engine would be circular.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

__all__ = ["Problem", "Question", "generate", "generate_problem", "closure"]

NAMES = [
    "Anne", "Bob", "Charlie", "Dave", "Erin", "Fiona", "Gary", "Harry",
    "Ivan", "Julia", "Karl", "Laura",
]
ATTRIBUTES = [
    "red", "blue", "green", "round", "big", "small", "kind", "nice",
    "rough", "smooth", "cold", "young", "quiet", "furry",
]
CLASSES = ["cat", "dog", "bear", "rabbit", "squirrel", "mouse", "lion"]


@dataclass(frozen=True)
class Question:
    """One yes/no question with its gold answer."""

    text: str
    #: Gold symbolic form, e.g. ``("attr", "bob", "green")``.
    goal: tuple
    answer: bool
    #: Number of rule applications needed; 0 means it was stated outright.
    depth: int
    #: True when the question asserts the *absence* of the goal, as in
    #: "The mouse is not blue." Under the closed-world reading such a question
    #: is true exactly when the goal is not derivable; under the open-world
    #: reading it asks about the strongly negated atom ``-blue(mouse)``.
    negated: bool = False
    #: Open-world only: the corpus says the theory settles neither polarity.
    #: ``answer`` is meaningless when this is set, so read :attr:`status`.
    unknown: bool = False

    @property
    def status(self) -> str:
        """The gold answer as one of ``yes`` / ``no`` / ``unknown``."""
        if self.unknown:
            return "unknown"
        return "yes" if self.answer else "no"

    def to_dict(self) -> dict:
        return {
            "question": self.text,
            "goal": "%s(%s)" % (self.goal[0], ", ".join(self.goal[1:])),
            "answer": self.answer,
            "status": self.status,
            "depth": self.depth,
            "negated": self.negated,
        }


@dataclass
class Problem:
    """A theory in English, its gold symbolic form, and its questions."""

    theory: str
    questions: list[Question] = field(default_factory=list)
    #: ``cwa`` or ``owa``. Under ``owa`` every predicate is open, the theory
    #: may assert and derive strongly negated atoms, and a question the theory
    #: does not settle is answered ``unknown`` rather than ``no``.
    world: str = "cwa"
    #: Gold facts as ``(predicate, *arguments)`` tuples.
    gold_facts: set[tuple] = field(default_factory=set)
    #: Gold rules as ``(head_template, [body_templates])`` over the variable X.
    gold_rules: list[tuple] = field(default_factory=list)
    seed: Optional[int] = None

    @property
    def gold_closure(self) -> set[tuple]:
        return closure(self.gold_facts, self.gold_rules)

    def to_dict(self) -> dict:
        return {
            "theory": self.theory,
            "questions": [q.to_dict() for q in self.questions],
            "gold_facts": sorted("%s(%s)" % (f[0], ", ".join(f[1:])) for f in self.gold_facts),
            "gold_rules": [_render_rule(rule) for rule in self.gold_rules],
        }


def closure(facts: Iterable[tuple], rules: Sequence[tuple]) -> set[tuple]:
    """Forward-chain to a fixpoint, independently of the inference engine.

    Rules are ``(head, body)`` where each literal is a tuple whose entity slot
    holds the sentinel ``"?"`` for the universally quantified individual. The
    restricted shape makes this short enough to be obviously correct, which is
    the point -- it is the reference the engine is graded against.
    """
    derived = set(facts)
    changed = True
    while changed:
        changed = False
        entities = {fact[1] for fact in derived}
        for head, body in rules:
            for entity in entities:
                grounded_body = {_bind(literal, entity) for literal in body}
                if grounded_body <= derived:
                    conclusion = _bind(head, entity)
                    if conclusion not in derived:
                        derived.add(conclusion)
                        changed = True
    return derived


def _bind(literal: tuple, entity: str) -> tuple:
    return tuple(entity if part == "?" else part for part in literal)


def _render_rule(rule: tuple) -> str:
    head, body = rule
    render = lambda lit: "%s(%s)" % (lit[0], ", ".join("X" if p == "?" else p for p in lit[1:]))
    return f"{render(head)} :- " + ", ".join(render(literal) for literal in body) + "."


def generate_problem(
    seed: int = 0,
    entities: int = 3,
    base_facts: int = 4,
    rules: int = 4,
    max_body: int = 2,
    questions: int = 6,
) -> Problem:
    """Generate one problem with a mix of true and false questions."""
    rng = random.Random(seed)
    names = rng.sample(NAMES, entities)
    attributes = rng.sample(ATTRIBUTES, min(len(ATTRIBUTES), 3 + rules))
    class_name = rng.choice(CLASSES)

    facts: set[tuple] = set()
    sentences: list[str] = []

    # Ground facts.
    for _ in range(base_facts):
        name = rng.choice(names)
        attribute = rng.choice(attributes[: max(2, len(attributes) // 2)])
        fact = ("attr", name.lower(), attribute)
        if fact in facts:
            continue
        facts.add(fact)
        sentences.append(f"{name} is {attribute}.")

    # One class membership, so the theory exercises isa/2 as well.
    member = rng.choice(names)
    facts.add(("isa", member.lower(), class_name))
    sentences.append(f"{member} is a {class_name}.")

    # Rules, built as a chain so that later rules can fire on earlier
    # conclusions -- that is what creates multi-step proofs.
    gold_rules: list[tuple] = []
    available = list(attributes[: max(2, len(attributes) // 2)])
    for index in range(rules):
        conclusion = attributes[min(len(attributes) - 1, len(available))]
        body_size = 1 if index == 0 else rng.randint(1, max_body)
        body_attributes = rng.sample(available, min(body_size, len(available)))
        body = [("attr", "?", attribute) for attribute in body_attributes]
        if index == 1:
            body.append(("isa", "?", class_name))
        head = ("attr", "?", conclusion)
        if head in body or any(head == literal for literal in body):
            continue
        gold_rules.append((head, body))
        sentences.append(_rule_sentence(head, body))
        available.append(conclusion)

    derived = closure(facts, gold_rules)
    depths = _depths(facts, gold_rules)

    # Questions: half drawn from what follows, half from what does not.
    question_list: list[Question] = []
    # Prefer conclusions that needed rule steps: a benchmark made of restated
    # input facts would not test the inference layer at all.
    positives = sorted(derived, key=lambda goal: (-depths.get(goal, 0), str(goal)))
    deep = [goal for goal in positives if depths.get(goal, 0) > 0]
    shallow = [goal for goal in positives if depths.get(goal, 0) == 0]
    rng.shuffle(shallow)
    wanted = max(1, questions // 2)
    chosen = deep[: max(1, wanted - 1)] + shallow
    for goal in chosen[:wanted]:
        question_list.append(
            Question(
                text=_question_sentence(goal),
                goal=goal,
                answer=True,
                depth=depths.get(goal, 0),
            )
        )
    tried = 0
    while len(question_list) < questions and tried < 200:
        tried += 1
        goal = ("attr", rng.choice(names).lower(), rng.choice(attributes))
        if goal in derived or any(q.goal == goal for q in question_list):
            continue
        question_list.append(
            Question(text=_question_sentence(goal), goal=goal, answer=False, depth=0)
        )

    rng.shuffle(question_list)
    return Problem(
        theory=" ".join(sentences),
        questions=question_list,
        gold_facts=facts,
        gold_rules=gold_rules,
        seed=seed,
    )


def generate(count: int = 20, seed: int = 0, **kwargs) -> list[Problem]:
    """Generate a dataset of problems."""
    return [generate_problem(seed=seed + index, **kwargs) for index in range(count)]


def _rule_sentence(head: tuple, body: Sequence[tuple]) -> str:
    conditions = []
    for literal in body:
        if literal[0] == "isa":
            conditions.append(f"a {literal[2]}")
        else:
            conditions.append(literal[2])
    if len(conditions) == 1:
        condition = conditions[0]
    else:
        condition = " and ".join(conditions)
    return f"If someone is {condition} then they are {head[2]}."


def _question_sentence(goal: tuple) -> str:
    name = goal[1].capitalize()
    if goal[0] == "isa":
        return f"Is {name} a {goal[2]}?"
    return f"Is {name} {goal[2]}?"


def _depths(facts: set[tuple], rules: Sequence[tuple]) -> dict[tuple, int]:
    """Proof depth of each derived fact: 0 for given, n for n rule steps."""
    depth = {fact: 0 for fact in facts}
    derived = set(facts)
    changed = True
    while changed:
        changed = False
        entities = {fact[1] for fact in derived}
        for head, body in rules:
            for entity in entities:
                grounded = [_bind(literal, entity) for literal in body]
                if not all(literal in derived for literal in grounded):
                    continue
                conclusion = _bind(head, entity)
                candidate = 1 + max(depth[literal] for literal in grounded)
                if conclusion not in derived or candidate < depth.get(conclusion, 10**9):
                    derived.add(conclusion)
                    depth[conclusion] = candidate
                    changed = True
    return depth
