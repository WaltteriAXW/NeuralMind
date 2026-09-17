"""Asking the one question that settles the most.

When several rules fit the examples equally well, the examples have not
determined the answer -- the learner is choosing, not concluding, and Phase
One's :attr:`Hypothesis.underdetermined` already says so. The way out is to ask
about one more case. Which case is not arbitrary.

A probe *splits* the candidates: some of them derive it, the rest do not.
Whichever way the answer comes back, every candidate on the losing side is
eliminated. So the best probe is the one that splits them most evenly, because
that is the one whose worst case is smallest -- the standard halving argument,
and the reason active selection beats random selection by more the more
candidates there are.

Concretely, with ``n`` candidates a probe that splits them ``k`` / ``n-k``
leaves at most ``max(k, n-k)``. Minimising that maximum is maximising
``min(k, n-k)``, which is what :func:`split_score` computes. Entropy would
rank the same way here; the count is used because it is exact and reads as
what it is.

A probe nobody disagrees about is worthless, however interesting: if every
candidate derives it, the answer eliminates nothing. Those score zero and are
never asked, which is most of the saving.

With one exception, and it matters. When every candidate says *no*, a "yes"
refutes all of them at once -- and that is the only way out of a space that
does not contain the answer. ``ancestor`` is the case: with examples that are
all parent-child pairs, every candidate is the base clause, they agree on
everything, and pure disagreement-based selection settles confidently on a
definition that is missing its recursion. Asking about a pair two generations
apart, which no candidate predicts, is what reveals it. So these are asked
after the splitters and before giving up.

When *no* rule fits, there is nothing to split and the honest question is
different: ask for an example, because the space the learner was given does not
contain the answer and more negative evidence will not help.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

from ..core.program import Program, Rule
from ..core.terms import Atom, Const
from ..inference.forward import ForwardChainer

__all__ = [
    "Question",
    "QuestionPicker",
    "split_score",
    "ASK_MEMBERSHIP",
    "ASK_REFUTATION",
    "ASK_EXAMPLE",
]

#: "Is this true?" -- settles which candidate rules survive.
ASK_MEMBERSHIP = "membership"
#: "Is this true?" where every candidate says no -- a yes refutes them all.
ASK_REFUTATION = "refutation"
#: "Show me one" -- asked when no candidate fits at all.
ASK_EXAMPLE = "example"


@dataclass
class Question:
    """One thing worth asking, and what it would settle."""

    kind: str
    #: The atom to ask about, for a membership question.
    atom: Optional[Atom] = None
    text: str = ""
    #: How many candidates each answer would eliminate.
    eliminates_if_yes: int = 0
    eliminates_if_no: int = 0

    @property
    def worst_case(self) -> int:
        """Candidates still standing after the least helpful answer."""
        return max(
            self.eliminates_if_no,  # survivors when the answer is yes
            self.eliminates_if_yes,
        )

    def describe(self) -> str:
        if self.kind == ASK_EXAMPLE:
            return self.text
        return self.text or f"Is {self.atom} true?"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "atom": str(self.atom) if self.atom else None,
            "text": self.describe(),
            "eliminates_if_yes": self.eliminates_if_yes,
            "eliminates_if_no": self.eliminates_if_no,
        }

    def __str__(self) -> str:
        return self.describe()


def split_score(yes: int, no: int) -> int:
    """How much the worse answer still eliminates. Higher is a better probe.

    ``min(yes, no)``: a probe that splits 5/5 guarantees five eliminations, one
    that splits 9/1 guarantees one, and one nobody disagrees about guarantees
    nothing.
    """
    return min(yes, no)


class QuestionPicker:
    """Chooses the probe that settles the most, given competing candidates."""

    def __init__(
        self,
        background,
        candidates: Sequence[Sequence[Rule]],
        target: tuple[str, int],
        constants: Sequence[Const],
        known: Iterable[Atom] = (),
        positive: Iterable[Atom] = (),
    ) -> None:
        self.background = (
            background.program(check=False)
            if hasattr(background, "program")
            else background
        )
        #: Each candidate is a set of clauses -- one possible definition.
        self.candidates = [list(c) for c in candidates]
        self.target = target
        self.constants = list(constants)
        self.known = {a for a in known}
        #: The subset of ``known`` that is true. Needed to compose probes:
        #: "p(a,b) and p(b,c) hold, so does p(a,c)?" only makes sense over
        #: answers that came back yes.
        self.positive = {a for a in positive}
        self._derived: Optional[list[set]] = None

    # -- what each candidate says -------------------------------------------

    def derived(self) -> list[set]:
        """What each candidate definition derives over the constant universe."""
        if self._derived is None:
            self._derived = [self._closure(clauses) for clauses in self.candidates]
        return self._derived

    def _closure(self, clauses: Sequence[Rule]) -> set:
        program = Program(
            rules=list(self.background.rules) + list(clauses),
            shown=set(self.background.shown),
            constants=dict(self.background.constants),
        )
        try:
            model = ForwardChainer(program).run()
        except Exception:
            return set()
        name, arity = self.target
        return {a for a in model.atoms if a.signature == (name, arity)}

    def universe(self) -> list[Atom]:
        """Every ground target atom over the constants, minus what is known."""
        name, arity = self.target
        return [
            atom
            for arguments in itertools.product(self.constants, repeat=arity)
            if (atom := Atom(name, arguments)) not in self.known
        ]

    # -- choosing -----------------------------------------------------------

    def rank(self) -> list[Question]:
        """Every probe worth asking, best first."""
        derived = self.derived()
        total = len(derived)
        ranked: list[Question] = []
        for atom in self.universe():
            yes = sum(1 for found in derived if atom in found)
            no = total - yes
            if split_score(yes, no) == 0:
                # Nobody disagrees, so the answer eliminates nothing. Skipping
                # these is most of what makes active selection cheap.
                continue
            ranked.append(
                Question(
                    kind=ASK_MEMBERSHIP,
                    atom=atom,
                    text=f"Is {atom} true?",
                    eliminates_if_yes=no,
                    eliminates_if_no=yes,
                )
            )
        ranked.sort(key=lambda q: (-split_score(q.eliminates_if_yes, q.eliminates_if_no), str(q.atom)))
        return ranked

    def refutations(self, limit: int = 8) -> list[Question]:
        """Probes every candidate says no to. A yes refutes the whole space.

        Order matters more here than for splitters, because nothing predicts
        the answer and most of these are simply false. Two heuristics, in
        order:

        **Compositions first.** If ``p(a, b)`` and ``p(b, c)`` are both known
        to hold, then ``p(a, c)`` is what a *transitive* definition would add
        and a non-recursive one would not. That is precisely the question that
        reveals recursion, and without it the loop settles on ``ancestor`` 's
        base case and never learns the rest. Cheap to compute and aimed
        squarely at the structure recursion has.

        **Then contact.** Failing that, prefer an atom whose arguments already
        appear in something true -- a guess about known individuals beats a
        guess about arbitrary ones.
        """
        derived = self.derived()
        if not derived:
            return []
        seen = {c for atom in self.known for c in atom.args}
        compositions = self._compositions()
        found = []
        for atom in self.universe():
            if any(atom in candidate for candidate in derived):
                continue
            touching = sum(1 for argument in atom.args if argument in seen)
            found.append((0 if atom in compositions else 1, -touching, str(atom), atom))
        found.sort()
        return [
            Question(
                kind=ASK_REFUTATION,
                atom=atom,
                text=f"Is {atom} true?",
                eliminates_if_yes=len(derived),
                eliminates_if_no=0,
            )
            for _, _, _, atom in found[:limit]
        ]

    def _compositions(self) -> set:
        """``p(a, c)`` where ``p(a, b)`` and ``p(b, c)`` are both known true.

        Only meaningful for binary relations, which is where recursion of this
        shape lives. For anything else this is empty and the contact heuristic
        decides on its own.
        """
        name, arity = self.target
        if arity != 2:
            return set()
        pairs = [
            (atom.args[0], atom.args[1])
            for atom in self.positive
            if atom.signature == (name, 2)
        ]
        by_left: dict = {}
        for left, right in pairs:
            by_left.setdefault(left, []).append(right)
        found = set()
        for left, right in pairs:
            for far in by_left.get(right, ()):
                found.add(Atom(name, (left, far)))
        return found

    def best(self) -> Optional[Question]:
        """The single probe that settles the most, or None if none would."""
        ranked = self.rank()
        if ranked:
            return ranked[0]
        refuting = self.refutations()
        if refuting:
            return refuting[0]
        if not self.candidates:
            name, arity = self.target
            return Question(
                kind=ASK_EXAMPLE,
                text=(
                    f"No rule in the space I was given explains {name}/{arity}. "
                    f"Show me one more example?"
                ),
            )
        return None

    def answer(self, atom: Atom, holds: bool) -> int:
        """Eliminate the candidates that disagree. Returns how many went.

        This is the halving step: after it, every surviving candidate agrees
        with everything that has been answered so far.
        """
        derived = self.derived()
        keep = [
            index
            for index, found in enumerate(derived)
            if (atom in found) == holds
        ]
        removed = len(self.candidates) - len(keep)
        self.candidates = [self.candidates[i] for i in keep]
        self._derived = [derived[i] for i in keep]
        self.known.add(atom)
        if holds:
            self.positive.add(atom)
        return removed

    def settled(self) -> bool:
        """True when nothing left to ask would change anything.

        Both kinds of probe have to be exhausted: candidates that agree with
        each other can still all be wrong, and a refutation probe is how that
        is found out.
        """
        return not self.rank() and not self.refutations()

    def summary(self) -> dict:
        return {
            "candidates": len(self.candidates),
            "settled": self.settled(),
            "probes_worth_asking": len(self.rank()),
        }
