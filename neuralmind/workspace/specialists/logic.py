"""The logic specialist: Phase One's engine, behind the specialist protocol.

This one is not optional and never will be. Every other specialist produces
atoms; this is what reasons over them, and what turns a pile of results into a
conclusion. It also owns the proof format the others borrow, so a tree that
spans four specialists still reads as one derivation.

It accepts any goal, at a middling score. That is deliberate: a specialist that
claims everything strongly would starve the others, and one that claims nothing
would never run on the plain deductive questions that are most of the work. The
score rises for a goal whose predicate the rules actually define, because then
there is something to try.
"""

from __future__ import annotations

from typing import Optional

from ...core.terms import Atom
from ...inference.engine import ReasoningEngine
from ...inference.proof import ProofError, explain
from ..specialist import Budget, Finding, Result

__all__ = ["LogicSpecialist"]


class LogicSpecialist:
    """Deduction over the knowledge base."""

    name = "logic"

    def __init__(self, knowledge) -> None:
        self.knowledge = knowledge
        self._engine: Optional[ReasoningEngine] = None
        #: The blackboard revision whose entries have already been handed to
        #: the engine. Everything at or below it is in the program already.
        self._absorbed = 0

    def engine(self) -> ReasoningEngine:
        """The engine over the knowledge base as it currently stands."""
        if self._engine is None:
            self._engine = self.knowledge.engine()
        return self._engine

    def invalidate(self) -> None:
        self._engine = None
        self._absorbed = 0

    def prepare(self) -> None:
        """Compute the model once, so the first query does not pay for it."""
        try:
            self.engine().solve()
        except Exception:  # a broken program is a query-time problem, not this
            pass

    def accepts(self, goal: Atom, workspace) -> float:
        defined = {
            rule.head.signature
            for rule in self.knowledge.rules.rules
            if rule.head is not None
        }
        given = {record.atom.signature for record in self.knowledge.facts}
        if goal.signature in defined:
            return 0.9
        if goal.signature in given:
            return 0.8
        # Still worth a try: the goal may be derivable through a rule whose
        # head unifies at a different arity, and saying "not mine" to every
        # unfamiliar predicate would make the mind unable to answer "no".
        return 0.4

    def run(self, goal: Atom, workspace, budget: Budget) -> Result:
        engine = self.engine()
        # Anything another specialist has posted is input to this one. That is
        # the whole point of the blackboard: an arithmetic result is, to the
        # reasoner, indistinguishable from a fact that was given.
        #
        # Only what is *new* is handed over. Re-adding the whole board on every
        # call would invalidate the cached model each time and make the engine
        # re-derive a fixpoint it had already computed -- which turned out to
        # be most of the cost of a query that consults several specialists.
        borrowed = [
            entry
            for entry in workspace.since(self._absorbed)
            if entry.source != self.name and entry.atom.is_ground
        ]
        self._absorbed = workspace.revision
        if borrowed:
            engine.add_facts([entry.atom for entry in borrowed])

        answer = engine.ask(goal, explain_answer=True)
        if not answer.holds:
            # A stalled rule names the literal it needed. Handing that back as
            # a subgoal is what lets another specialist supply it: the rule
            # "safe(B) :- beam(B), leq(load, rating)" fails here and succeeds
            # once the arithmetic specialist has posted the inequality.
            subgoals = _wanted(answer)
            if subgoals:
                return Result(
                    subgoals=subgoals,
                    reason=f"{goal} needs " + ", ".join(str(a) for a in subgoals),
                )
            if answer.status == "no":
                return Result.nothing(f"{goal} is false under the rules given")
            return Result.nothing(f"nothing in the rules settles {goal}")

        findings = []
        for atom in answer.atoms:
            try:
                proof = explain(engine.model, atom)
            except ProofError:  # pragma: no cover - the model just derived it
                continue
            findings.append(Finding(atom=atom, proof=_reattach(proof, workspace)))
        return Result(findings=findings)


def _wanted(answer) -> list:
    """The literals the closest-matching rules stalled on, best attempt first.

    Only ground ones: an unbound literal is not a goal another specialist can
    take, and offering it would send the controller round a loop that cannot
    close.
    """
    if answer.diagnosis is None or not answer.diagnosis.attempts:
        return []
    wanted = []
    for attempt in sorted(answer.diagnosis.attempts, key=lambda a: -a.progress):
        if attempt.missing is not None and attempt.missing.is_ground:
            if attempt.missing not in wanted:
                wanted.append(attempt.missing)
    return wanted[:4]


def _reattach(proof, workspace):
    """Replace 'given' leaves with the proof whoever actually posted them.

    Without this the tree stops at "load_ok(beam1) [given]" and loses the
    arithmetic that established it -- the specialists would each be correct and
    the explanation would still be a lie by omission.
    """
    for index, child in enumerate(proof.children):
        proof.children[index] = _reattach(child, workspace)
    if not proof.children:
        posted = workspace.proof(proof.conclusion)
        if posted is not None and posted.kind != proof.kind:
            return posted
    return proof
