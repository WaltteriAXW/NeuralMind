"""What an action is: what it needs, and what it changes.

The representation is deliberately small. An action has a name, some
parameters, a set of literals that must hold before it can happen, and a set
of literals that hold afterwards. Everything else the planner needs -- which
predicates change over time, which are fixed, whether a variable is safe --
is *derived* from those, because a declaration you have to keep in sync with
the rules is a declaration that will eventually disagree with them.

The surface syntax is a block per action::

    % Opening a valve needs it closed, and makes it open.
    action open_valve(V):
        needs  valve(V), closed(V)
        causes open(V), -closed(V)

``needs`` takes positive literals and ``not`` literals; ``causes`` takes
positive literals and strong-negated ones, where ``-closed(V)`` means the
fluent stops holding. That is the same strong negation the engine already
uses, and it reads the way the domain expert says it out loud: opening a
valve causes it to be open and causes it *not* to be closed.

Fluents are inferred: a predicate any action causes is a fluent, and
everything else is static. So ``valve(v1)`` is a fact about the world that
never changes, and ``closed(v1)`` is a fact about the world right now, and
nobody had to say which is which. A ``fluent`` declaration exists for the
case where something changes but no modelled action changes it -- a door the
environment closes on its own -- because there the inference genuinely cannot
know.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional, Sequence

from ..core.parser import ParseError, parse_atom
from ..core.terms import Atom, Term, Var

__all__ = [
    "Action",
    "ActionLibrary",
    "ActionError",
    "parse_actions",
    "Literal",
]


class ActionError(ValueError):
    """An action that could not be read, or that could never be applied."""


@dataclass(frozen=True)
class Literal:
    """A precondition: an atom, and whether it is required to be absent.

    Two ways to be absent, kept apart because they mean different things. A
    ``not`` precondition is negation as failure -- "as far as I know, not
    locked". A strong-negated atom is an explicit negative fact -- "known to
    be not locked". A plan built on the first is a plan built on ignorance,
    and the brief line says so.
    """

    atom: Atom
    #: True for ``not p(X)``: satisfied when ``p(X)`` is not known to hold.
    absent: bool = False

    @property
    def predicate(self) -> str:
        return self.atom.positive.predicate

    def __str__(self) -> str:
        return f"not {self.atom}" if self.absent else str(self.atom)


@dataclass(frozen=True)
class Action:
    """One thing the mind can do, and what follows from doing it."""

    name: str
    params: tuple[Var, ...] = ()
    needs: tuple[Literal, ...] = ()
    causes: tuple[Atom, ...] = ()
    #: What it costs to do -- the planner minimises the total.
    cost: int = 1
    #: Free text carried through to the brief line.
    note: str = ""
    #: The autonomy level this action needs, when reversibility is not the
    #: whole story. ``None`` means "work it out", which is the usual case.
    authority: Optional[int] = None
    source: str = ""
    line: int = 0

    # -- the two halves of an effect ---------------------------------------

    @property
    def adds(self) -> tuple[Atom, ...]:
        """Fluents that hold afterwards."""
        return tuple(a for a in self.causes if not a.is_negated)

    @property
    def cancels(self) -> tuple[Atom, ...]:
        """Fluents that stop holding afterwards, as their positive form."""
        return tuple(a.positive for a in self.causes if a.is_negated)

    @property
    def signature(self) -> str:
        return f"{self.name}/{len(self.params)}"

    @property
    def head(self) -> Atom:
        """The action itself as a term, e.g. ``open_valve(V)``."""
        return Atom(self.name, tuple(self.params))

    # -- what it talks about ------------------------------------------------

    def changed_predicates(self) -> set[str]:
        return {atom.positive.predicate for atom in self.causes}

    def needed_predicates(self) -> set[str]:
        return {literal.predicate for literal in self.needs}

    def variables(self) -> set[str]:
        seen: set[str] = {v.name for v in self.params}
        for literal in self.needs:
            seen |= _vars(literal.atom)
        for atom in self.causes:
            seen |= _vars(atom)
        return seen

    # -- safety -------------------------------------------------------------

    def unsafe_variables(self) -> set[str]:
        """Variables a negative condition mentions that nothing positive binds.

        The same safety rule the rule engine applies, for the same reason: a
        condition on ``not open(X)`` where nothing says which ``X`` is not a
        condition, it is a wish. Catching it here turns a silently empty
        grounding into a message naming the variable.
        """
        bound = {v.name for v in self.params}
        for literal in self.needs:
            if not literal.absent and not literal.atom.is_negated:
                bound |= _vars(literal.atom)
        used: set[str] = set()
        for literal in self.needs:
            if literal.absent or literal.atom.is_negated:
                used |= _vars(literal.atom)
        return used - bound

    def unparameterised_effects(self) -> set[str]:
        """Effect variables that are not parameters of the action.

        Stricter than ordinary rule safety, and it has to be. An effect is
        attributed to *the action happening*, so whatever the effect names
        has to be recoverable from the action's own term -- ``forward`` that
        moves the agent to ``Y`` where ``Y`` came from a precondition leaves
        nothing in ``forward`` to say which ``Y``, and the honest reading of
        that action is ``forward(X, Y)``: a move from somewhere to somewhere,
        not a move in the abstract.

        Preconditions may still bind extra variables freely; only effects are
        held to this.
        """
        parameters = {v.name for v in self.params}
        used: set[str] = set()
        for atom in self.causes:
            used |= _vars(atom)
        return used - parameters

    def check(self) -> "Action":
        loose = self.unparameterised_effects()
        if loose:
            names = ", ".join(sorted(loose))
            suggestion = ", ".join(
                [v.name for v in self.params] + sorted(loose)
            )
            raise ActionError(
                f"action {self.signature} has effects on {names}, which are "
                f"not among its parameters. An effect is attributed to the "
                f"action, so the action has to name what it affects: write "
                f"{self.name}({suggestion})."
            )
        unsafe = self.unsafe_variables()
        if unsafe:
            raise ActionError(
                f"action {self.signature} is unsafe: nothing binds "
                f"{', '.join(sorted(unsafe))}. Every variable in an effect or "
                "a negative condition must appear in a positive condition or "
                "in the action's own parameters."
            )
        if not self.causes:
            raise ActionError(
                f"action {self.signature} causes nothing, so no plan could "
                "ever have a reason to include it"
            )
        contradictory = {a.positive for a in self.causes if a.is_negated} & set(
            self.adds
        )
        if contradictory:
            names = ", ".join(sorted(str(a) for a in contradictory))
            raise ActionError(
                f"action {self.signature} both causes and cancels {names}"
            )
        return self

    def describe(self) -> str:
        parts = [f"{self.head}"]
        if self.needs:
            parts.append("needs " + ", ".join(str(n) for n in self.needs))
        if self.causes:
            parts.append("causes " + ", ".join(str(c) for c in self.causes))
        return "; ".join(parts)

    def to_dict(self) -> dict:
        payload = {
            "name": self.name,
            "params": [v.name for v in self.params],
            "needs": [str(n) for n in self.needs],
            "causes": [str(c) for c in self.causes],
            "cost": self.cost,
        }
        if self.authority is not None:
            payload["authority"] = self.authority
        if self.note:
            payload["note"] = self.note
        return payload

    def __str__(self) -> str:
        return self.describe()


class ActionLibrary:
    """The actions available, plus what they imply about the world.

    A library knows which predicates are fluents, which are static, and which
    actions could ever establish a given fluent -- the last being what turns
    "no plan" from a shrug into a diagnosis.
    """

    def __init__(
        self, actions: Iterable[Action] = (), fluents: Iterable[str] = ()
    ) -> None:
        self._actions: dict[str, Action] = {}
        self._declared_fluents: set[str] = set(fluents)
        for action in actions:
            self.add(action)

    def add(self, action: Action) -> "ActionLibrary":
        action.check()
        if action.signature in self._actions:
            raise ActionError(
                f"two actions named {action.signature}; "
                "give one of them a different name or arity"
            )
        self._actions[action.signature] = action
        return self

    def replace(self, action: Action) -> "ActionLibrary":
        """Revise an action in place, keeping its name and arity.

        The revision path, as distinct from :meth:`add`: learning narrows an
        action rather than introducing a new one, and an action that changed
        its own signature would leave every plan that named it dangling.
        """
        action.check()
        if action.signature not in self._actions:
            raise ActionError(
                f"no action {action.signature} to replace; use add() for a new one"
            )
        self._actions[action.signature] = action
        return self

    def declare_fluent(self, predicate: str) -> "ActionLibrary":
        """Mark a predicate as changing even though no action changes it.

        For the things the environment does on its own. Without this they look
        static, and the planner would carry a stale value forward for ever.
        """
        self._declared_fluents.add(predicate)
        return self

    # -- what it knows ------------------------------------------------------

    @property
    def actions(self) -> list[Action]:
        return sorted(self._actions.values(), key=lambda a: a.name)

    @property
    def fluents(self) -> set[str]:
        """Predicates that change: anything an action causes, plus declared."""
        changing = set(self._declared_fluents)
        for action in self._actions.values():
            changing |= action.changed_predicates()
        return changing

    def fluent_signatures(self) -> set[tuple[str, int]]:
        """Each changing predicate with its arity.

        The planner writes one frame axiom per signature rather than one
        generic one, which keeps the generated program readable and -- more
        usefully -- lets derived fluents be left out of inertia by simply not
        appearing in this set.
        """
        signatures = set()
        for action in self._actions.values():
            for atom in action.causes:
                positive = atom.positive
                signatures.add((positive.predicate, len(positive.args)))
        return signatures

    def statics(self) -> set[str]:
        """Predicates the actions read but never change."""
        read: set[str] = set()
        for action in self._actions.values():
            read |= action.needed_predicates()
        return read - self.fluents

    def establishers(self, atom: Atom) -> list[Action]:
        """Actions that could make ``atom`` hold. Empty is a real answer."""
        target = atom.positive.predicate
        return [a for a in self.actions if target in {x.predicate for x in a.adds}]

    def undoes(self, action: Action) -> list[Action]:
        """Actions that could put back what ``action`` takes away.

        An action is reversible when everything it cancels, something else
        can restore, and everything it adds, something else can cancel. This
        is a *syntactic* check over the library -- it asks whether a way back
        is even expressible, not whether it would be available in the state
        the plan reaches. That is the right conservatism: a mind that treats
        an action as reversible because a recovery exists in principle is a
        mind that will one day discover the recovery needs the thing it just
        destroyed.
        """
        restorers = []
        for other in self.actions:
            if other.name == action.name:
                continue
            restores = {a.predicate for a in other.adds} & {
                a.predicate for a in action.cancels
            }
            removes = {a.predicate for a in other.cancels} & {
                a.predicate for a in action.adds
            }
            if restores or removes:
                restorers.append(other)
        return restorers

    def reversible(self, action: Action) -> bool:
        """True when every change this action makes could be put back."""
        undone_adds: set[str] = set()
        restored_cancels: set[str] = set()
        for other in self.undoes(action):
            undone_adds |= {a.predicate for a in other.cancels}
            restored_cancels |= {a.predicate for a in other.adds}
        adds = {a.predicate for a in action.adds}
        cancels = {a.predicate for a in action.cancels}
        return adds <= undone_adds and cancels <= restored_cancels

    def irreversible(self) -> list[Action]:
        """Actions with no way back. Worth listing on its own: these are the
        ones a plan should never take without someone agreeing to them."""
        return [a for a in self.actions if not self.reversible(a)]

    def get(self, name: str, arity: Optional[int] = None) -> Optional[Action]:
        if arity is not None:
            return self._actions.get(f"{name}/{arity}")
        matches = [a for a in self._actions.values() if a.name == name]
        return matches[0] if len(matches) == 1 else None

    def __len__(self) -> int:
        return len(self._actions)

    def __iter__(self) -> Iterator[Action]:
        return iter(self.actions)

    def __contains__(self, name: str) -> bool:
        return any(a.name == name for a in self._actions.values())

    def summary(self) -> dict:
        return {
            "actions": len(self._actions),
            "fluents": sorted(self.fluents),
            "statics": sorted(self.statics()),
        }

    def __repr__(self) -> str:
        return (
            f"ActionLibrary({len(self._actions)} action(s), "
            f"{len(self.fluents)} fluent(s))"
        )


# -- reading the block syntax ---------------------------------------------

_HEADER = re.compile(r"^action\s+(?P<name>[a-z_][A-Za-z0-9_]*)\s*(?P<args>\([^)]*\))?\s*:\s*$")
_CLAUSE = re.compile(
    r"^(?P<key>needs|causes|costs|note|when|authority)\b\s*(?P<rest>.*)$"
)
_FLUENT = re.compile(r"^fluent\s+(?P<names>[^.]+)\.?\s*$")


def parse_actions(source: str, name: str = "actions") -> ActionLibrary:
    """Read a block of action definitions into a library."""
    library = ActionLibrary()
    current: Optional[dict] = None
    header_line = 0
    #: Which clause a bare line continues, when the one above ended in a comma.
    continuing: Optional[str] = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        library.add(_build(current, name, header_line))
        current = None

    for number, raw in enumerate(source.splitlines(), start=1):
        line = raw.split("%", 1)[0].rstrip()
        if not line.strip():
            continue

        fluent = _FLUENT.match(line.strip())
        if fluent and current is None:
            for predicate in fluent.group("names").split(","):
                library.declare_fluent(predicate.strip())
            continue

        header = _HEADER.match(line.strip())
        if header:
            flush()
            continuing = None
            header_line = number
            current = {
                "name": header.group("name"),
                "args": header.group("args") or "()",
                "needs": [],
                "causes": [],
                "cost": 1,
                "note": "",
                "authority": None,
            }
            continue

        if current is None:
            raise ActionError(
                f"{name}:{number}: expected an 'action name(...):' header "
                f"before {line.strip()!r}"
            )

        clause = _CLAUSE.match(line.strip())
        if not clause:
            # A continuation, but only of a line that ended with a comma. A
            # trailing comma is an unambiguous "there is more"; without that
            # rule a misspelled keyword would be silently swallowed into the
            # clause above it.
            if continuing is not None:
                items = current[continuing]
                text, first_line = items[-1]
                items[-1] = (text.rstrip().rstrip(",") + ", " + line.strip(), first_line)
                continuing = continuing if line.strip().endswith(",") else None
                continue
            raise ActionError(
                f"{name}:{number}: expected needs/causes/costs/note, "
                f"got {line.strip()!r}"
            )
        key, rest = clause.group("key"), clause.group("rest").strip()
        if key == "when":
            key = "needs"  # a synonym, because both readings are natural
        if key == "costs":
            try:
                current["cost"] = int(rest)
            except ValueError as exc:
                raise ActionError(f"{name}:{number}: costs wants a whole number") from exc
        elif key == "authority":
            try:
                current["authority"] = int(rest)
            except ValueError as exc:
                raise ActionError(
                    f"{name}:{number}: authority wants a level 0-3"
                ) from exc
        elif key == "note":
            current["note"] = rest
        else:
            current[key].append((rest, number))
            continuing = key if rest.rstrip().endswith(",") else None

    flush()
    if not library:
        raise ActionError(f"{name} defines no actions")
    return library


def _build(spec: dict, source: str, line: int) -> Action:
    params = tuple(_params(spec["args"], source, line))
    needs = tuple(
        literal
        for text, number in spec["needs"]
        for literal in _literals(text, source, number)
    )
    causes = tuple(
        _atom(part, source, number)
        for text, number in spec["causes"]
        for part in _split(text)
    )
    return Action(
        name=spec["name"],
        params=params,
        needs=needs,
        causes=causes,
        cost=spec["cost"],
        note=spec["note"],
        authority=spec["authority"],
        source=source,
        line=line,
    )


def _params(args: str, source: str, line: int) -> Iterator[Var]:
    inner = args.strip()[1:-1].strip()
    if not inner:
        return
    for part in inner.split(","):
        token = part.strip()
        if not token or not token[0].isupper():
            raise ActionError(
                f"{source}:{line}: action parameters must be variables "
                f"(capitalised); got {token!r}"
            )
        yield Var(token)


def _literals(text: str, source: str, line: int) -> Iterator[Literal]:
    for part in _split(text):
        token = part.strip()
        if token.startswith("not "):
            yield Literal(_atom(token[4:], source, line), absent=True)
        else:
            yield Literal(_atom(token, source, line))


def _atom(text: str, source: str, line: int) -> Atom:
    try:
        return parse_atom(text.strip())
    except ParseError as exc:
        raise ActionError(f"{source}:{line}: {exc}") from exc


def _split(text: str) -> list[str]:
    """Split on commas that are not inside parentheses."""
    parts, depth, current = [], 0, []
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def _vars(atom: Atom) -> set[str]:
    return {t.name for t in atom.args if isinstance(t, Var)}
