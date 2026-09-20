"""Finding a plan, one step of horizon at a time.

The encoding is the standard incremental one: a ``base`` program for what is
true at time 0, a ``step(t)`` program instantiated once per time point, and a
``check(t)`` program holding the goal constraint behind an external, so the
same solver can be asked "in t steps?" repeatedly without rebuilding.

Growing the horizon one step at a time is not just an implementation detail.
It is what makes the planner *anytime* in the only sense that matters here:
at any moment it either has the shortest plan or knows that none of the
lengths it has tried can work, and the second is a fact worth reporting. A
planner that picks a horizon up front and solves once cannot tell "no plan in
5 steps" from "no plan"; this one can, and :meth:`Planner.plan` says which.

clingo does the search because this is combinatorial and the Python engine
deliberately is not. The plan that comes back is then replayed through the
ordinary state representation, so what gets returned is checked against the
action definitions rather than trusted from the solver.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from ..core.terms import Atom, Const, Term, Var
from ..inference.clingo_backend import ClingoUnavailable, clingo_available
from .actions import Action, ActionLibrary
from .domain import Domain, Goal, State
from .plan import Plan, Step, build_links

__all__ = [
    "Planner", "PlanningError", "to_asp", "possible_actions",
    "SAT", "UNSAT", "TIMEOUT",
]

#: What one horizon came back with.
SAT = "sat"
UNSAT = "unsat"
TIMEOUT = "timeout"

#: A horizon nobody asked for is a horizon that runs for ever.
DEFAULT_HORIZON = 20
#: Wall-clock budget. Anytime means this is a real answer, not a failure.
DEFAULT_BUDGET_MS = 5_000.0


class PlanningError(RuntimeError):
    """The domain could not be planned over at all."""


class Planner:
    """Grows the horizon until a plan appears, the budget runs out, or the
    problem is shown to have no plan at any length tried."""

    def __init__(
        self,
        max_horizon: int = DEFAULT_HORIZON,
        budget_ms: float = DEFAULT_BUDGET_MS,
        optimise: bool = False,
    ) -> None:
        self.max_horizon = max_horizon
        self.budget_ms = budget_ms
        #: Minimise total action cost at the first horizon that has a plan.
        self.optimise = optimise
        self.last_program: str = ""

    # -- the entry point ----------------------------------------------------

    def plan(self, domain: Domain, goal: Optional[Goal] = None) -> Plan:
        """Find the shortest plan for ``goal``, or say why there is none."""
        target = goal if goal is not None else domain.goal
        started = time.perf_counter()

        if not len(target):
            return Plan(goal=target, already=True, reason="no goal was given")
        if target.satisfied_by(domain.state):
            return Plan(goal=target, already=True)

        unreachable = Domain(
            domain.library, domain.state, target, domain.rules
        ).unreachable()
        if unreachable:
            # Worth its own answer: no horizon will help, and saying "not
            # within 20 steps" would send someone looking for 21.
            names = ", ".join(str(a) for a in unreachable)
            return Plan(
                goal=target,
                reason=f"no action can bring about {names}",
                seconds=time.perf_counter() - started,
            )

        if not clingo_available():
            raise ClingoUnavailable(
                "planning needs clingo, which searches; the Python engine "
                "derives. Install it with `pip install neuralmind[asp]`."
            )

        return self._search(domain, target, started)

    # -- the incremental loop -----------------------------------------------

    def _search(self, domain: Domain, goal: Goal, started: float) -> Plan:
        import clingo

        base, step, check = to_asp(domain, goal)
        self.last_program = "\n".join(
            [f"#program base.\n{base}", f"#program step(t).\n{step}",
             f"#program check(t).\n{check}"]
        )

        control = clingo.Control(["--models=1", "--warn=none"])
        control.add("base", [], base)
        control.add("step", ["t"], step)
        control.add("check", ["t"], check)

        query = lambda t: clingo.Function("query", [clingo.Number(t)])
        control.ground([("base", []), ("check", [clingo.Number(0)])])
        control.assign_external(query(0), True)

        horizon = 0
        while True:
            elapsed = (time.perf_counter() - started) * 1000.0
            if elapsed > self.budget_ms:
                return Plan(
                    goal=goal,
                    horizon=horizon,
                    seconds=time.perf_counter() - started,
                    reason=(
                        f"stopped after {elapsed:.0f}ms with no plan up to "
                        f"{horizon} step(s); the budget is {self.budget_ms:.0f}ms"
                    ),
                )

            status, occurrences = self._solve(control, self.budget_ms - elapsed)
            if status == SAT:
                plan = self._replay(domain, goal, occurrences or [])
                plan.horizon = horizon
                plan.seconds = time.perf_counter() - started
                return plan
            if status == TIMEOUT:
                # Distinct from UNSAT on purpose. "I ran out of time at 6
                # steps" and "6 steps is provably not enough" are different
                # facts, and only the second one means stop looking.
                return Plan(
                    goal=goal,
                    horizon=horizon,
                    seconds=time.perf_counter() - started,
                    reason=(
                        f"the search at {horizon} step(s) did not finish inside "
                        f"the {self.budget_ms:.0f}ms budget, so a plan of that "
                        "length may still exist"
                    ),
                )

            if horizon >= self.max_horizon:
                return Plan(
                    goal=goal,
                    horizon=horizon,
                    seconds=time.perf_counter() - started,
                    reason=(
                        f"no plan exists within {self.max_horizon} step(s); "
                        "every shorter length was ruled out, not skipped"
                    ),
                )

            control.assign_external(query(horizon), False)
            horizon += 1
            control.ground(
                [("step", [clingo.Number(horizon)]),
                 ("check", [clingo.Number(horizon)])]
            )
            control.assign_external(query(horizon), True)

    def _solve(
        self, control, remaining_ms: float
    ) -> tuple[str, Optional[list[tuple[int, Atom]]]]:
        """One horizon, under what is left of the budget.

        Three outcomes, kept apart: a plan, a proof that there is none of this
        length, and running out of time. Collapsing the last two into "no
        plan" is how a planner ends up claiming a search space is empty when
        it only ran out of patience.
        """
        from ..inference.clingo_backend import symbol_to_atom

        with control.solve(yield_=True, async_=True) as handle:
            handle.resume()
            if not handle.wait(max(remaining_ms, 1.0) / 1000.0):
                handle.cancel()
                return TIMEOUT, None
            model = handle.model()
            if model is None:
                return UNSAT, None
            occurrences = []
            for symbol in model.symbols(shown=True):
                if symbol.name != "occurs" or len(symbol.arguments) != 2:
                    continue
                action, when = symbol.arguments
                occurrences.append((when.number, symbol_to_atom(action)))
            return SAT, sorted(occurrences, key=lambda pair: pair[0])

    # -- checking what came back -------------------------------------------

    def _replay(
        self, domain: Domain, goal: Goal, occurrences: Sequence[tuple[int, Atom]]
    ) -> Plan:
        """Run the plan through the action definitions and record each state.

        The solver is trusted to search, not to be right. Replaying catches an
        encoding mistake here rather than in whatever acts on the plan, and it
        is where the ground preconditions come from -- clingo knows them but
        does not report them, and they are most of what makes a step
        explainable.
        """
        state = _derive(domain, domain.state)
        steps: list[Step] = []
        trace: list[State] = [state]

        for position, (_, action_atom) in enumerate(occurrences):
            schema = domain.library.get(action_atom.predicate, len(action_atom.args))
            if schema is None:
                raise PlanningError(
                    f"the solver returned {action_atom}, which is not an action "
                    "in this library"
                )
            binding = dict(zip((v.name for v in schema.params), action_atom.args))
            positive = [
                literal.atom
                for literal in schema.needs
                if not literal.absent and not literal.atom.is_negated
            ]
            # The action's own term does not have to carry every variable its
            # preconditions mention -- ``forward(X, Y, X2, Y2)`` says nothing
            # about which way the agent was facing. Those are recovered by
            # matching the preconditions against the state, which is also
            # what turns them into the ground facts the proof is built from.
            binding = _bind(positive, binding, state)
            if binding is None:
                raise PlanningError(
                    f"step {position + 1} ({action_atom}) has preconditions "
                    "that nothing in the state satisfies"
                )

            needs = tuple(_ground(atom, binding) for atom in positive)
            absent = tuple(
                _ground(literal.atom.positive, binding)
                for literal in schema.needs
                if literal.absent or literal.atom.is_negated
            )
            adds = tuple(_ground(a, binding) for a in schema.adds)
            cancels = tuple(_ground(a, binding) for a in schema.cancels)

            unmet = [n for n in needs if n not in state.atoms]
            if unmet:
                raise PlanningError(
                    f"step {position + 1} ({action_atom}) needs "
                    f"{', '.join(str(n) for n in unmet)}, which does not hold"
                )
            still_there = [a for a in absent if a in state.fluents]
            if still_there:
                raise PlanningError(
                    f"step {position + 1} ({action_atom}) needs "
                    f"{', '.join(str(a) for a in still_there)} out of the way, "
                    "and it is not"
                )

            steps.append(
                Step(
                    index=position,
                    action=action_atom,
                    schema=schema,
                    needs=tuple(n for n in needs if n in state.fluents),
                    needs_absent=tuple(
                        a for a in absent
                        if a.positive.predicate in domain.library.fluents
                    ),
                    adds=adds,
                    cancels=cancels,
                )
            )
            fluents = (set(state.fluents) - set(cancels)) | set(adds)
            state = _derive(domain, state.with_fluents(fluents))
            trace.append(state)

        plan = Plan(
            steps=steps,
            trace=trace,
            goal=goal,
            links=build_links(steps, goal, domain.state, trace),
        )
        problems = plan.check()
        if problems and not goal.satisfied_by(trace[-1]):
            raise PlanningError("; ".join(problems))
        return plan


# -- the encoding ----------------------------------------------------------


def to_asp(domain: Domain, goal: Optional[Goal] = None) -> tuple[str, str, str]:
    """Compile a domain into the three incremental subprograms.

    Returned as text rather than pushed straight into a solver so that it can
    be read, diffed and pasted into clingo by hand. A planner whose encoding
    you cannot look at is a planner you cannot debug.
    """
    target = goal if goal is not None else domain.goal
    library = domain.library

    base = ["% -- what is fixed, and what holds at time 0"]
    for atom in sorted(domain.state.statics, key=str):
        base.append(f"{atom}.")
    for atom in sorted(domain.state.fluents, key=str):
        base.append(f"holds({atom}, 0).")
    # Only the changing half of the goal goes to the solver. A goal literal
    # over a fixed fact is either already true -- in which case asking the
    # planner to achieve it is asking for a step that cannot exist -- or it
    # is unreachable, which Domain.unreachable() has already said before we
    # got here. Sending it anyway makes every horizon unsatisfiable, and the
    # planner then reports, at length and with total confidence, that no plan
    # exists.
    fluents = library.fluents
    for atom in target.wanted:
        if atom.positive.predicate in fluents:
            base.append(f"goal({atom}).")
    for atom in target.unwanted:
        if atom.positive.predicate in fluents:
            base.append(f"unwanted({atom}).")

    step = [
        "% -- which actions could happen now",
    ]
    for action in library.actions:
        step.append(f"% {action.describe()}")
        step.append(f"poss({action.head}, t) :- {_conditions(action, library)}.")
    step += [
        "",
        "% -- exactly one action per time point",
        "1 { occurs(A, t) : poss(A, t) } 1.",
        "",
        "% -- effects",
    ]
    for action in library.actions:
        for effect in action.adds:
            step.append(f"holds({effect}, t) :- occurs({action.head}, t).")
        for effect in action.cancels:
            step.append(f"cancelled({effect}, t) :- occurs({action.head}, t).")

    if domain.rules:
        step += ["", "% -- what follows from the state, re-derived each step"]
        for rule in domain.rules:
            step.append(_timed_rule(rule, library, domain.derived))

    step += [
        "",
        "% -- inertia: one axiom per changing predicate, so that the derived",
        "%    ones above are simply absent from this list rather than needing",
        "%    an exception to it",
    ]
    for predicate, arity in sorted(library.fluent_signatures()):
        if predicate in domain.derived:
            continue
        args = ", ".join(_letters(arity))
        term = f"{predicate}({args})" if arity else predicate
        step.append(
            f"holds({term}, t) :- holds({term}, t-1), not cancelled({term}, t)."
        )
    step += ["", "#show occurs/2."]

    check = [
        "#external query(t).",
        ":- query(t), goal(F), not holds(F, t).",
        ":- query(t), unwanted(F), holds(F, t).",
    ]

    return "\n".join(base), "\n".join(step), "\n".join(check)


def possible_actions(domain: Domain) -> list[Atom]:
    """Every ground action whose preconditions hold right now.

    "What can I do from here" is a different question from "how do I get
    there", and a mind that can only answer the second one has no way to
    experiment. It is also the question an environment with a command list
    is already answering, so having it here means the two can be compared --
    a mismatch is the action model being wrong about the world, which is
    exactly what is worth knowing.
    """
    state = _derive(domain, domain.state)
    found: list[Atom] = []
    for schema in domain.library.actions:
        positive = [
            literal.atom
            for literal in schema.needs
            if not literal.absent and not literal.atom.is_negated
        ]
        absent = [
            literal.atom.positive
            for literal in schema.needs
            if literal.absent or literal.atom.is_negated
        ]
        for binding in _bindings(positive, {}, state):
            if any(_ground(a, binding) in state.fluents for a in absent):
                continue
            action = _ground(schema.head, binding)
            if action.is_ground and action not in found:
                found.append(action)
    return sorted(found, key=str)


def _bindings(needs: Sequence[Atom], binding: dict, state: State):
    """Every way of satisfying these preconditions, not just the first."""
    if not needs:
        yield dict(binding)
        return
    head, rest = needs[0], needs[1:]
    candidate = _ground(head, binding)
    if candidate.is_ground:
        if candidate in state.atoms:
            yield from _bindings(rest, binding, state)
        return
    for atom in state.atoms:
        if atom.predicate != candidate.predicate or len(atom.args) != len(candidate.args):
            continue
        extended = dict(binding)
        for slot, value in zip(candidate.args, atom.args):
            if isinstance(slot, Var):
                if extended.setdefault(slot.name, value) != value:
                    break
            elif slot != value:
                break
        else:
            yield from _bindings(rest, extended, state)


def _derive(domain: Domain, state: State) -> State:
    """Close a state under the domain's state constraints.

    Run on the *Python* engine, not clingo. The plan came back from a search
    and is about to be checked against the action definitions; deriving the
    consequences with the other engine means the two have to agree about what
    the plan achieves, which is the same cross-check the rest of the system
    gets for free and planning would otherwise miss.
    """
    if not domain.rules:
        return state

    from ..core.program import Program, Rule
    from ..inference.forward import ForwardChainer

    derived_predicates = domain.derived
    given = [a for a in state.atoms if a.positive.predicate not in derived_predicates]
    program = Program(
        rules=[Rule(atom, (), source="state", label="given") for atom in given]
        + list(domain.rules)
    )
    model = ForwardChainer(program).run()
    fluents = {a for a in state.fluents if a.positive.predicate not in derived_predicates}
    fluents |= {a for a in model.atoms if a.positive.predicate in derived_predicates}
    return State(state.statics, frozenset(fluents))


def _conditions(action: Action, library: ActionLibrary) -> str:
    """A precondition reads from the timeless facts or from the last state."""
    fluents = library.fluents
    parts = []
    for literal in action.needs:
        atom = literal.atom
        positive = atom.positive
        timed = positive.predicate in fluents
        if literal.absent:
            inner = f"holds({positive}, t-1)" if timed else str(positive)
            parts.append(f"not {inner}")
        elif atom.is_negated:
            # A strong-negated precondition: known false, which for a fluent
            # under the closed world of one time point is "does not hold".
            inner = f"holds({positive}, t-1)" if timed else str(positive)
            parts.append(f"not {inner}")
        else:
            parts.append(f"holds({positive}, t-1)" if timed else str(positive))
    for param in action.params:
        if not any(param.name in _names(l.atom) for l in action.needs):
            parts.append(f"% unbound {param.name}")
    return ", ".join(p for p in parts if not p.startswith("%")) or "#true"


def _timed_rule(rule, library: ActionLibrary, derived: set) -> str:
    """Put a state constraint at time ``t``.

    Fluents in the body read from this time point, not the last one: a
    derived fact describes the state the actions have just produced, so it
    has to see them.
    """
    fluents = library.fluents
    head = rule.head
    body = []
    for part in rule.body:
        atom = getattr(part, "atom", part)
        negated = getattr(part, "negated", False) or getattr(part, "is_negative", False)
        if not isinstance(atom, Atom):
            body.append(str(part))
            continue
        positive = atom.positive
        inner = (
            f"holds({positive}, t)"
            if positive.predicate in fluents
            else str(positive)
        )
        body.append(f"not {inner}" if (negated or atom.is_negated) else inner)
    conclusion = f"holds({head}, t)" if head.predicate in fluents else str(head)
    return f"{conclusion} :- {', '.join(body)}." if body else f"{conclusion}."


def _letters(count: int) -> list:
    return [chr(ord("A") + i) for i in range(count)]


def _names(atom: Atom) -> set[str]:
    return {t.name for t in atom.args if isinstance(t, Var)}


def _bind(needs: Sequence[Atom], binding: dict, state: State) -> Optional[dict]:
    """Extend ``binding`` until every precondition matches something real.

    Straightforward backtracking over the state's atoms. The search is small
    because the preconditions are few and mostly ground once the action's
    parameters are filled in, and being exact matters more than being clever:
    this is the check that the plan clingo returned is a plan the action
    definitions actually licence.
    """
    if not needs:
        return binding
    head, rest = needs[0], needs[1:]
    candidate = _ground(head, binding)
    if candidate.is_ground:
        if candidate not in state.atoms:
            return None
        return _bind(rest, binding, state)

    for atom in state.atoms:
        if atom.predicate != candidate.predicate or len(atom.args) != len(candidate.args):
            continue
        extended = dict(binding)
        for slot, value in zip(candidate.args, atom.args):
            if isinstance(slot, Var):
                if extended.setdefault(slot.name, value) != value:
                    break
            elif slot != value:
                break
        else:
            found = _bind(rest, extended, state)
            if found is not None:
                return found
    return None


def _ground(atom: Atom, binding: dict) -> Atom:
    args = tuple(binding.get(a.name, a) if isinstance(a, Var) else a for a in atom.args)
    return Atom(atom.predicate, args)
