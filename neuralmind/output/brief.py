"""Answer-first one-liners, every clause traceable to the proof.

A proof tree is the honest output, and it is the wrong thing to put in a
status line, an NPC's speech bubble or a reorder alert. This module compresses
one into a single sentence under a word budget:

    Yes -- Bob is warm blooded, because he is a cat.

The compression has one hard rule: **every clause comes from a node of the
proof**. Nothing is added for fluency, nothing is inferred to fill a gap, and
:class:`Brief` keeps the atom behind each clause so a test can read the line
back and find the node it came from. That is what makes a short answer safe to
show: it is shorter than the proof, never different from it.

Three lengths, since different hosts have different room:

``brief``
    One sentence, at most :data:`BRIEF_WORDS` words. Clauses are dropped from
    the least important end until it fits, and the line says how many were
    dropped rather than pretending they were not there.
``short``
    At most three sentences -- the verdict and the reasoning behind it.
``full``
    The proof tree itself; see :mod:`neuralmind.output.render`.

An ``unknown`` answer is where this earns its keep. "I don't know" is useless;
"I don't know whether mammals are furry" is a question the user can answer, and
it comes straight out of the failure diagnosis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..core.terms import Atom
from ..inference.proof import CLOSED_WORLD, FACT, ProofNode
from .nlg import Realiser, join_clauses, sentence_case

__all__ = ["Brief", "Clause", "brief", "BRIEF_WORDS", "SHORT_SENTENCES", "LENGTHS"]

#: The word budget for a ``brief`` line.
BRIEF_WORDS = 25

#: The sentence budget for a ``short`` answer.
SHORT_SENTENCES = 3

#: Length policies, longest last.
LENGTHS = ("brief", "short", "full")

_VERDICT = {"yes": "Yes", "no": "No", "unknown": "Unknown"}


@dataclass(frozen=True)
class Clause:
    """One piece of a brief line, and the thing it states.

    ``atom`` is what the clause asserts. ``role`` says why it is in the line:
    ``conclusion`` for the answer itself, ``premise`` for a fact it rests on,
    ``established`` for something proven on the way to a goal that failed, and
    ``gap`` for the literal that stopped it.
    """

    text: str
    atom: Atom
    role: str
    negated: bool = False


@dataclass
class Brief:
    """A short answer plus the clauses it was built from."""

    status: str
    text: str
    clauses: tuple[Clause, ...] = ()
    #: Clauses that did not fit the budget, kept so nothing is silently lost.
    dropped: tuple[Clause, ...] = ()

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def atoms(self) -> tuple[Atom, ...]:
        """Every atom this line mentions -- what a traceability test checks."""
        return tuple(clause.atom for clause in self.clauses)

    def __str__(self) -> str:
        return self.text


def brief(
    answer,
    realiser: Optional[Realiser] = None,
    length: str = "brief",
    max_words: int = BRIEF_WORDS,
) -> Brief:
    """Render an :class:`~neuralmind.inference.engine.Answer` in one line.

    ``answer`` is duck-typed rather than imported, so the output layer does not
    depend on the inference layer.
    """
    if length not in LENGTHS:
        raise ValueError(f"length must be one of {', '.join(LENGTHS)}, not {length!r}")
    realiser = realiser or Realiser()
    status = getattr(answer, "status", "yes" if answer.holds else "no")

    if status == "yes":
        clauses = _because(answer, realiser)
    elif status == "no":
        clauses = _refutation(answer, realiser)
    else:
        clauses = _gap(answer, realiser)

    if length == "short":
        return _assemble(status, clauses, (), sentences=SHORT_SENTENCES)
    kept, dropped = _fit(status, clauses, max_words)
    return _assemble(status, kept, dropped)


# -- building the clause list ---------------------------------------------


def _because(answer, realiser: Realiser) -> list[Clause]:
    """The conclusion, then the facts it rests on."""
    conclusion = _conclusion_clause(answer, realiser)
    proof = answer.proof
    if proof is None:
        return [conclusion]
    premises = [
        Clause(realiser.realise(atom), atom, "premise")
        for atom in proof.premises()
        if atom != conclusion.atom
    ]
    return [conclusion] + premises


def _refutation(answer, realiser: Realiser) -> list[Clause]:
    """Why the answer is no: a derived ``-p(x)``, or nothing that shows ``p(x)``."""
    if answer.refutation is not None:
        node = answer.refutation
        stated = node.conclusion.positive
        clauses = [Clause(realiser.realise(stated, negated=True), node.conclusion, "conclusion")]
        clauses += [
            Clause(realiser.realise(atom), atom, "premise")
            for atom in node.premises()
            if atom != node.conclusion
        ]
        return clauses
    # Closed-world: the absence *is* the reason, so say that and then say what
    # the nearest rule would have needed.
    query = answer.query
    clauses = [
        Clause(
            f"nothing shows that {realiser.realise(query)}", query, "conclusion"
        )
    ]
    return clauses + _from_diagnosis(answer, realiser)


def _gap(answer, realiser: Realiser) -> list[Clause]:
    """What is known, and the one thing that would settle the question."""
    query = answer.query
    clauses = _from_diagnosis(answer, realiser)
    if not clauses:
        clauses = [
            Clause(
                f"nothing here says whether {realiser.realise(query)}", query, "gap"
            )
        ]
    return clauses


def _from_diagnosis(answer, realiser: Realiser) -> list[Clause]:
    """Read the furthest-progressing rule attempt as "this much, then this gap"."""
    diagnosis = getattr(answer, "diagnosis", None)
    if diagnosis is None or not diagnosis.attempts:
        return []
    attempt = max(diagnosis.attempts, key=lambda a: a.progress)
    clauses = [
        Clause(realiser.realise(atom), atom, "established")
        for atom in attempt.established
    ]
    if attempt.missing is not None:
        clauses.append(
            Clause(
                f"I don't know whether {realiser.realise(attempt.missing)}",
                attempt.missing,
                "gap",
            )
        )
    return clauses


def _conclusion_clause(answer, realiser: Realiser) -> Clause:
    atom = answer.atoms[0] if answer.atoms else answer.query
    return Clause(realiser.realise(atom), atom, "conclusion")


# -- fitting and joining ---------------------------------------------------


def _fit(status: str, clauses: Sequence[Clause], max_words: int):
    """Drop clauses from the end until the line fits, keeping the first.

    The first clause is the answer itself and is never dropped: a line that
    fits but says nothing is worse than a long one.
    """
    kept = list(clauses)
    while len(kept) > 1 and len(_render(status, kept, len(clauses) - len(kept)).split()) > max_words:
        kept.pop()
    return tuple(kept), tuple(clauses[len(kept) :])


def _assemble(
    status: str,
    kept: Sequence[Clause],
    dropped: Sequence[Clause],
    sentences: Optional[int] = None,
) -> Brief:
    if sentences is not None:
        text = _render_sentences(status, kept, sentences)
    else:
        text = _render(status, kept, len(dropped))
    return Brief(status=status, text=text, clauses=tuple(kept), dropped=tuple(dropped))


def _render(status: str, clauses: Sequence[Clause], dropped: int) -> str:
    """One sentence: verdict, the answer, then its reasons."""
    verdict = _VERDICT.get(status, status.capitalize())
    if not clauses:
        return f"{verdict}."
    head, *rest = clauses
    line = f"{verdict} — {head.text}"
    reasons = [c.text for c in rest]
    if dropped:
        reasons.append(f"and {dropped} more" if reasons else f"{dropped} more reasons")
    if reasons:
        line += f", {_connective(rest)} {join_clauses(reasons)}"
    return sentence_case(line.rstrip(",")) + "."


def _connective(rest: Sequence[Clause]) -> str:
    """"because" introduces support; "but" introduces what is missing."""
    if rest and rest[0].role == "gap":
        return "but"
    return "because"


def _render_sentences(status: str, clauses: Sequence[Clause], limit: int) -> str:
    """Up to ``limit`` sentences: the verdict, then one clause each."""
    verdict = _VERDICT.get(status, status.capitalize())
    if not clauses:
        return f"{verdict}."
    head, *rest = clauses
    out = [f"{verdict} — {head.text}."]
    for clause in rest[: max(limit - 1, 0)]:
        lead = "But " if clause.role == "gap" else ""
        out.append(sentence_case(f"{lead}{clause.text}.") if lead else sentence_case(f"{clause.text}."))
    return " ".join(sentence_case(s) for s in out)
