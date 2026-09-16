"""Template-based natural language generation -- deterministic, no model.

The blueprint is explicit that readable prose should come from a grammar, not
a generator: SimpleNLG or Grammatical Framework rather than an LLM. This is a
small pure-Python realiser in that spirit. It handles the morphology that
makes template output read like English rather than like filled-in slots --
articles, agreement, capitalisation, list punctuation -- and nothing more.

It is genuinely optional. For machine-to-machine use the JSON in
:mod:`neuralmind.output.serialize` is clearer and shorter, and the blueprint
says so. Use this when a person has to read the answer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from ..core.terms import Atom, Const, Term
from ..inference.proof import CLOSED_WORLD, FACT, ProofNode

__all__ = ["Realiser", "realise_atom", "realise_proof", "humanise"]

_VOWEL_SOUND = re.compile(r"^[aeiou]", re.IGNORECASE)
#: Words that take "an" despite starting with a consonant letter, and vice versa.
_ARTICLE_EXCEPTIONS = {"hour": "an", "honest": "an", "university": "a", "unit": "a", "user": "a"}


def indefinite_article(word: str) -> str:
    """``a`` or ``an``, by sound rather than by spelling where it matters."""
    head = word.strip().split()[0].lower() if word.strip() else ""
    if head in _ARTICLE_EXCEPTIONS:
        return _ARTICLE_EXCEPTIONS[head]
    return "an" if _VOWEL_SOUND.match(head) else "a"


def humanise(term: Term) -> str:
    """Turn a logic constant back into a readable phrase."""
    if isinstance(term, Const):
        if term.is_number:
            return str(term.value)
        return str(term.value).replace("_", " ")
    return str(term)


def third_person(verb: str) -> str:
    """Agreement for a singular subject: ``like`` -> ``likes``."""
    word = verb.strip()
    if not word or word.endswith("s"):
        return word
    if word.endswith(("sh", "ch", "x", "z", "o")):
        return word + "es"
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def sentence_case(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


def join_clauses(clauses: Sequence[str], conjunction: str = "and") -> str:
    """Oxford-comma list joining."""
    items = [c for c in clauses if c]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {conjunction} {items[1]}"
    return ", ".join(items[:-1]) + f", {conjunction} {items[-1]}"


@dataclass
class Realiser:
    """Renders atoms and proof trees as English.

    Templates are looked up by ``(predicate, arity)``. A template is either a
    format string over the humanised arguments, or a callable taking the atom.
    Predicates with no template fall back to a readable generic form, so an
    unregistered predicate degrades to plain wording rather than an error.
    """

    templates: dict[tuple[str, int], object] = field(default_factory=dict)
    negation_prefix: str = "not_"
    #: Constants to capitalise as proper names. Logic constants are all
    #: lower-case, so nothing in the symbols themselves says that ``bob`` is a
    #: person and ``cat`` is not -- the caller supplies that.
    proper_names: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # The default triple schema the text perception layer emits.
        self.register("attr", 2, lambda a: f"{a[0]} is {a[1]}")
        self.register("isa", 2, lambda a: f"{a[0]} is {indefinite_article(a[1])} {a[1]}")
        self.register("rel", 3, lambda a: f"{a[0]} {third_person(a[1])} {a[2]}")
        self.register("act", 2, lambda a: f"{a[0]} {third_person(a[1])}")

    def term(self, value: Term) -> str:
        """Humanise one argument, capitalising registered proper names."""
        text = humanise(value)
        if isinstance(value, Const) and str(value.value) in self.proper_names:
            return " ".join(word.capitalize() for word in text.split())
        return text

    def learn_names(self, names) -> "Realiser":
        """Register constants that should be capitalised as proper names."""
        for name in names:
            self.proper_names.add(str(name.value) if isinstance(name, Const) else str(name))
        return self

    def register(self, predicate: str, arity: int, template) -> "Realiser":
        """Add or override a template."""
        self.templates[(predicate, arity)] = template
        return self

    def realise(self, atom: Atom, negated: bool = False) -> str:
        """One atom as a clause, without terminal punctuation."""
        predicate = atom.predicate
        if predicate.startswith(self.negation_prefix):
            predicate = predicate[len(self.negation_prefix) :]
            negated = not negated
        key = (predicate, atom.arity)
        template = self.templates.get(key)
        stripped = Atom(predicate, atom.args)
        arguments = [self.term(a) for a in stripped.args]
        if template is None:
            clause = self._generic(stripped, arguments)
        elif callable(template):
            clause = template(arguments)
        else:
            clause = template.format(*arguments)
        return self._negate(clause) if negated else clause

    def _negate(self, clause: str) -> str:
        """Insert 'not' after the first auxiliary, or fall back to a prefix."""
        for auxiliary in (" is ", " are ", " was ", " were ", " has ", " have "):
            if auxiliary in clause:
                return clause.replace(auxiliary, f"{auxiliary.rstrip()} not ", 1)
        words = clause.split()
        if len(words) >= 2:
            return f"{words[0]} does not {_base_form(words[1])} " + " ".join(words[2:])
        return f"it is not the case that {clause}"

    def _generic(self, atom: Atom, arguments: list[str]) -> str:
        name = atom.predicate.replace("_", " ")
        if not arguments:
            return name
        if len(arguments) == 1:
            return f"{arguments[0]} is {name}"
        if len(arguments) == 2:
            return f"{arguments[0]} is the {name} of {arguments[1]}"
        return f"{name} holds of " + join_clauses(arguments)

    # -- proof trees -----------------------------------------------------

    def realise_proof(self, node: ProofNode, max_sentences: int = 40) -> str:
        """Render a proof tree as a short explanatory paragraph.

        The tree is walked deepest-first, so the reader gets the premises
        before the conclusions that rest on them -- which is the order an
        explanation has to be in to be followed.
        """
        sentences: list[str] = []
        seen: set[str] = set()

        def visit(current: ProofNode) -> None:
            if len(sentences) >= max_sentences:
                return
            for child in current.children:
                visit(child)
            key = str(current.conclusion) + str(current.negated)
            if key in seen:
                return
            seen.add(key)
            clause = self.realise(current.conclusion, current.negated)
            if current.kind == FACT:
                sentences.append(sentence_case(f"{clause}."))
            elif current.kind == CLOSED_WORLD:
                sentences.append(
                    sentence_case(f"nothing in the knowledge base shows that {clause}.")
                )
            else:
                reasons = [
                    self.realise(child.conclusion, child.negated) for child in current.children
                ]
                if reasons:
                    sentences.append(
                        sentence_case(f"{clause}, because {join_clauses(reasons)}.")
                    )
                else:
                    sentences.append(sentence_case(f"{clause}."))

        visit(node)
        if len(sentences) > 1:
            sentences[-1] = "Therefore " + sentences[-1]
        return " ".join(sentences)

    def realise_answer(self, answer) -> str:
        """A one-paragraph answer, with the reasoning behind it."""
        if not answer.holds:
            clause = self.realise(answer.query)
            return sentence_case(
                f"no: nothing in the knowledge base shows that {clause}."
            )
        if answer.proof is not None:
            return "Yes. " + self.realise_proof(answer.proof)
        return "Yes. " + join_clauses([self.realise(a) for a in answer.atoms]) + "."


def _base_form(verb: str) -> str:
    """Undo third-person agreement, for negation ('likes' -> 'like')."""
    from ..perception.lexicon import base_verb

    return base_verb(verb)


_DEFAULT = Realiser()


def realise_atom(atom: Atom, negated: bool = False) -> str:
    """Realise a single atom with the default templates."""
    return _DEFAULT.realise(atom, negated)


def realise_proof(node: ProofNode) -> str:
    """Realise a proof tree with the default templates."""
    return _DEFAULT.realise_proof(node)
