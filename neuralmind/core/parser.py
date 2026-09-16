"""Parser for the ASP/Datalog subset the pure-Python engine executes.

The subset is deliberately the fragment with a unique least model: definite
rules, stratified default negation, integrity constraints, comparisons and
integer arithmetic. Anything outside it (choice rules, aggregates, disjunction,
optimisation) is *not* an error -- the statement is kept verbatim and flagged
via :attr:`Program.requires_asp`, so the clingo backend can run the program
even though the Python engine cannot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

from .program import Program, Rule
from .terms import Arith, Atom, Compare, Const, Literal, Term, Var

__all__ = ["parse_program", "parse_file", "parse_rule", "parse_atom", "ParseError"]


class ParseError(SyntaxError):
    """The source is not valid in the supported syntax."""


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    line: int
    col: int

    def __str__(self) -> str:
        return f"{self.value!r} (line {self.line})"


_TOKEN_SPEC = [
    ("WS", r"[ \t\r\n]+"),
    ("COMMENT", r"%\*.*?\*%|%[^\n]*", re.DOTALL),
    ("STRING", r'"(?:[^"\\]|\\.)*"'),
    ("DIRECTIVE", r"#[A-Za-z_]+"),
    ("NUM", r"\d+"),
    ("ID", r"[a-z][A-Za-z0-9_]*"),
    ("VAR", r"[A-Z_][A-Za-z0-9_]*"),
    ("RANGE", r"\.\."),
    ("IMPL", r":-"),
    ("WEAK", r":~"),
    ("OP", r"!=|<=|>=|\*\*|[=<>+\-*/\\%|(),;{}\[\]:@]"),
    ("DOT", r"\."),
]

_MASTER_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern, *_ in _TOKEN_SPEC),
    re.DOTALL,
)

#: Statement-level constructs the Python engine cannot evaluate.
_ASP_ONLY_TOKENS = {
    "{": "choice rules",
    "}": "choice rules",
    "|": "disjunctive heads",
    ";": "condition/pooling syntax",
    "[": "weight annotations",
    "]": "weight annotations",
    "@": "external functions",
}
_ASP_ONLY_DIRECTIVES = {
    "#minimize": "optimisation",
    "#maximize": "optimisation",
    "#sum": "aggregates",
    "#count": "aggregates",
    "#min": "aggregates",
    "#max": "aggregates",
    "#external": "external atoms",
}


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    line = 1
    line_start = 0
    pos = 0
    while pos < len(source):
        match = _MASTER_RE.match(source, pos)
        if match is None:
            col = pos - line_start + 1
            raise ParseError(f"unexpected character {source[pos]!r} at line {line}, column {col}")
        kind = match.lastgroup or ""
        text = match.group()
        if kind in ("WS", "COMMENT"):
            newlines = text.count("\n")
            if newlines:
                line += newlines
                line_start = match.start() + text.rfind("\n") + 1
        else:
            tokens.append(_Token(kind, text, line, match.start() - line_start + 1))
        pos = match.end()
    tokens.append(_Token("EOF", "", line, 0))
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[_Token], source_name: Optional[str]) -> None:
        self.tokens = list(tokens)
        self.pos = 0
        self.source_name = source_name
        self._anonymous_count = 0

    # -- token helpers ---------------------------------------------------

    @property
    def current(self) -> _Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> _Token:
        index = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[index]

    def advance(self) -> _Token:
        token = self.tokens[self.pos]
        if token.kind != "EOF":
            self.pos += 1
        return token

    def accept(self, kind: str, value: Optional[str] = None) -> Optional[_Token]:
        token = self.current
        if token.kind == kind and (value is None or token.value == value):
            return self.advance()
        return None

    def expect(self, kind: str, value: Optional[str] = None) -> _Token:
        token = self.accept(kind, value)
        if token is None:
            wanted = value if value is not None else kind
            raise ParseError(
                f"expected {wanted!r} but found {self.current} "
                f"in {self.source_name or '<string>'}"
            )
        return token

    # -- statements ------------------------------------------------------

    def parse(self) -> Program:
        program = Program()
        asp_features: list[str] = []
        while self.current.kind != "EOF":
            start = self.pos
            feature = self._scan_statement_for_asp_features()
            if feature is not None:
                raw = self._consume_raw_statement(start)
                program.raw_asp.append(raw)
                asp_features.append(feature)
                continue
            self.pos = start
            self._parse_statement(program)
        program.requires_asp = tuple(dict.fromkeys(asp_features))
        return program

    def _scan_statement_for_asp_features(self) -> Optional[str]:
        """Look ahead to the statement terminator for unsupported constructs."""
        index = self.pos
        while index < len(self.tokens):
            token = self.tokens[index]
            if token.kind in ("DOT", "EOF"):
                return None
            if token.kind == "RANGE":
                return "interval terms"
            if token.kind == "WEAK":
                return "weak constraints"
            if token.kind == "DIRECTIVE" and token.value in _ASP_ONLY_DIRECTIVES:
                return _ASP_ONLY_DIRECTIVES[token.value]
            if token.kind == "OP" and token.value in _ASP_ONLY_TOKENS:
                return _ASP_ONLY_TOKENS[token.value]
            index += 1
        return None

    def _consume_raw_statement(self, start: int) -> str:
        while self.current.kind not in ("DOT", "EOF"):
            self.advance()
        self.accept("DOT")
        pieces = [t.value for t in self.tokens[start : self.pos]]
        return " ".join(pieces[:-1]) + "."

    def _parse_statement(self, program: Program) -> None:
        token = self.current
        if token.kind == "DIRECTIVE":
            self._parse_directive(program)
            return
        line = token.line
        if self.accept("IMPL"):  # integrity constraint ":- body."
            body = self._parse_body()
            self.expect("DOT")
            program.add(Rule(None, body, source=self.source_name, line=line))
            return
        head = self._parse_atom()
        body: tuple = ()
        if self.accept("IMPL"):
            body = self._parse_body()
        self.expect("DOT")
        program.add(Rule(head, body, source=self.source_name, line=line))

    def _parse_directive(self, program: Program) -> None:
        directive = self.advance()
        if directive.value == "#show":
            name = self.expect("ID").value
            self.expect("OP", "/")
            arity = int(self.expect("NUM").value)
            self.expect("DOT")
            program.shown.add((name, arity))
            return
        if directive.value == "#open" and self.current.kind == "DOT":
            # A bare "#open." makes the whole program open-world.
            self.advance()
            program.open_world = True
            return
        if directive.value in ("#open", "#closed"):
            # Which predicates answer "unknown" when nothing derives them, and
            # which answer "no". Silence means something different in each case
            # and the program is the only place that can say which.
            while True:
                name = self.expect("ID").value
                self.expect("OP", "/")
                arity = int(self.expect("NUM").value)
                if directive.value == "#open":
                    program.open_predicates.add((name, arity))
                else:
                    program.open_predicates.discard((name, arity))
                    program.closed_predicates.add((name, arity))
                if not self.accept("OP", ","):
                    break
            self.expect("DOT")
            return
        if directive.value == "#const":
            name = self.expect("ID").value
            self.expect("OP", "=")
            value = self._parse_expression()
            self.expect("DOT")
            program.constants[name] = value
            return
        # Unknown but harmless directives are preserved for clingo.
        raw = self._consume_raw_statement(self.pos - 1)
        program.raw_asp.append(raw)

    # -- rule parts ------------------------------------------------------

    def _parse_body(self) -> tuple:
        parts = [self._parse_body_part()]
        while self.accept("OP", ","):
            parts.append(self._parse_body_part())
        return tuple(parts)

    def _parse_body_part(self) -> Union[Literal, Compare]:
        if self.current.kind == "ID" and self.current.value == "not":
            self.advance()
            return Literal(self._parse_atom(), negated=True)
        # An atom and the left side of a comparison start the same way, so
        # parse optimistically and reinterpret if a comparison operator turns up.
        if self.current.kind == "OP" and self.current.value == "-" and self.peek().kind == "ID":
            return Literal(self._parse_atom())
        if self.current.kind == "ID" and self.peek().value == "(":
            atom = self._parse_atom()
            if self._at_comparison():
                raise ParseError(
                    f"function terms are not supported; got '{atom}' on the left "
                    f"of a comparison at line {self.current.line}"
                )
            return Literal(atom)
        left = self._parse_expression()
        if self._at_comparison():
            op = self.advance().value
            right = self._parse_expression()
            return Compare(op, left, right)
        if isinstance(left, Const) and not left.quoted and isinstance(left.value, str):
            return Literal(Atom(left.value, ()))  # propositional atom
        raise ParseError(f"expected an atom or comparison near line {self.current.line}")

    def _at_comparison(self) -> bool:
        token = self.current
        return token.kind == "OP" and token.value in ("=", "!=", "<", "<=", ">", ">=")

    def _parse_atom(self) -> Atom:
        # "-p(x)" is strong negation: a claim that p(x) is false, as opposed to
        # "not p(x)", which only says p(x) could not be derived.
        strong = "-" if self.accept("OP", "-") else ""
        name = strong + self.expect("ID").value
        if not self.accept("OP", "("):
            return Atom(name, ())
        args = [self._parse_expression()]
        while self.accept("OP", ","):
            args.append(self._parse_expression())
        self.expect("OP", ")")
        return Atom(name, tuple(args))

    # -- expressions (precedence climbing) -------------------------------

    def _parse_expression(self) -> Term:
        return self._parse_additive()

    def _parse_additive(self) -> Term:
        node = self._parse_multiplicative()
        while self.current.kind == "OP" and self.current.value in ("+", "-"):
            op = self.advance().value
            node = Arith(op, node, self._parse_multiplicative())
        return node

    def _parse_multiplicative(self) -> Term:
        node = self._parse_power()
        while self.current.kind == "OP" and self.current.value in ("*", "/", "\\", "%"):
            op = self.advance().value
            node = Arith(op, node, self._parse_power())
        return node

    def _parse_power(self) -> Term:
        node = self._parse_primary()
        if self.current.kind == "OP" and self.current.value == "**":
            self.advance()
            return Arith("**", node, self._parse_power())  # right-associative
        return node

    def _parse_primary(self) -> Term:
        token = self.current
        if token.kind == "OP" and token.value == "-":
            self.advance()
            operand = self._parse_primary()
            if isinstance(operand, Const) and operand.is_number:
                return Const(-int(operand.value))
            return Arith("-", Const(0), operand)
        if token.kind == "OP" and token.value == "(":
            self.advance()
            inner = self._parse_expression()
            self.expect("OP", ")")
            return inner
        if token.kind == "NUM":
            return Const(int(self.advance().value))
        if token.kind == "VAR":
            name = self.advance().value
            if name == "_":
                # Each anonymous variable is distinct, as in ASP.
                self._anonymous_count += 1
                name = f"_Anon{self._anonymous_count}"
            return Var(name)
        if token.kind == "STRING":
            raw = self.advance().value[1:-1]
            return Const(raw.replace('\\"', '"').replace("\\\\", "\\"), quoted=True)
        if token.kind == "ID":
            return Const(self.advance().value)
        raise ParseError(f"unexpected token {token} while reading a term")


#: ``%@ Some label`` on the line(s) before a rule names it for humans.
_LABEL_RE = re.compile(r"^\s*%@\s*(.+?)\s*$")


def _collect_labels(source: str) -> dict[int, str]:
    """Map each labelled rule's line number to its label.

    A label applies to the next statement that starts at or after it, which is
    how a constraint gets to report itself as "Separation of duties" instead of
    as a line number.
    """
    labels: dict[int, str] = {}
    pending: Optional[str] = None
    for number, line in enumerate(source.splitlines(), start=1):
        match = _LABEL_RE.match(line)
        if match:
            pending = match.group(1)
            continue
        if pending and line.strip() and not line.lstrip().startswith("%"):
            labels[number] = pending
            pending = None
    return labels


def _apply_labels(program: Program, labels: dict[int, str]) -> None:
    import dataclasses

    for index, rule in enumerate(program.rules):
        if rule.line in labels:
            program.rules[index] = dataclasses.replace(rule, label=labels[rule.line])


def parse_program(source: str, source_name: Optional[str] = None, check: bool = True) -> Program:
    """Parse ASP/Datalog source into a :class:`Program`.

    With ``check=True`` the program is also validated for variable safety and
    stratified negation, which is almost always what you want -- a knowledge
    base that fails these checks would otherwise produce a wrong model.
    """
    program = _Parser(_tokenize(source), source_name).parse()
    _apply_labels(program, _collect_labels(source))
    if check and not program.raw_asp:
        program.check()
    return program


def parse_file(path: Union[str, Path], check: bool = True) -> Program:
    """Parse a ``.lp`` rule file from disk."""
    path = Path(path)
    return parse_program(path.read_text(encoding="utf-8"), source_name=path.name, check=check)


def parse_rule(source: str) -> Rule:
    """Parse exactly one rule, e.g. ``parse_rule('p(X) :- q(X).')``."""
    program = parse_program(source, check=False)
    if len(program.rules) != 1:
        raise ParseError(f"expected exactly one rule, found {len(program.rules)}")
    return program.rules[0]


def parse_atom(source: str) -> Atom:
    """Parse a single atom, e.g. ``parse_atom('parent(alice, bob)')``."""
    parser = _Parser(_tokenize(source), None)
    atom = parser._parse_atom()
    if parser.current.kind not in ("EOF", "DOT"):
        raise ParseError(f"trailing input after atom: {parser.current}")
    return atom
