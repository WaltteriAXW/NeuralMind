"""An interactive session for building and questioning a knowledge base.

This is a REPL, not a chat interface, and the difference is the whole point.
Nothing here generates language. You state facts and rules, it tells you
exactly what symbols it took from them; you ask a question, it answers and
shows the derivation. A sentence it cannot read is reported, and one it can
only guess at is marked as a guess -- neither is quietly absorbed. That is the
behaviour a language model cannot offer, and the reason this project exists.

    $ neuralmind shell
    > Bob is a cat.
      + isa(bob, cat)
    > All cats are mammals.
      + isa(X, mammal) :- isa(X, cat).
    > Is Bob a mammal?
    yes
      isa(bob, mammal)  (by All cats are mammals)
      └── isa(bob, cat)  [given]

Input is dispatched four ways: a line starting with ``:`` is a command,
a line ending with ``?`` is a question, anything with logic syntax is read as
ASP, and everything else goes to the perception layer as English. Both
readings are tried before anything is rejected.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, TextIO

from .core.parser import ParseError, parse_atom, parse_program
from .core.program import ProgramError
from .core.terms import Atom, Const
from .inference.proof import ProofError, explain
from .knowledge.base import KnowledgeBase, builtin_rulesets, rule_path
from .output.nlg import Realiser
from .output.render import render_model, render_proof

__all__ = ["Shell", "run_shell"]

BANNER = """NeuralMind interactive session.
State facts and rules in English or in logic; ask questions with '?'.
No language model is involved: every answer comes with its derivation.
Type :help for commands, :quit to leave."""

HELP = """
  Statements        Bob is a cat.            English, read into symbols
                    parent(alice, bob).      logic, asserted directly
                    All cats are mammals.    a rule
  Questions         Is Bob a mammal?         English
                    ancestor(alice, X)?      logic, solves for X

  :help                  this text
  :facts [predicate]     what has been asserted
  :rules                 the rules in force
  :model [predicate]     everything derivable
  :why <atom>            the derivation of one atom
  :check                 integrity-constraint violations
  :load <name|path>      add a bundled rule set (or a .lp file)
  :sets                  list the bundled rule sets
  :retract <atom>        remove an asserted fact
  :learn <target/arity> [from <pred/arity> ...]
                         induce a rule for the target from what is known
  :gaps                  what it could not answer, most-asked first
  :questions             what it would need to be told
  :grow [target/arity]   one growth cycle: propose a rule by asking
  :beliefs               what is remembered, and whether it is confirmed
  :bootstrap <log>       draft a module from an observation log
  :ask [yes|no]          answer the draft's next question
  :pack status|write|promote   where the draft has got to
  :expect <atom>         this should follow but does not -- propose fixes
  :wrong <atom>          this follows but should not -- propose fixes
  :accept <n>            apply one of the proposed fixes
  :undo                  undo the last change
  :history               everything asserted this session
  :open <path>           replace the session with a saved knowledge base
  :schema [direct|triple]  how English maps onto predicates
  :proof on|off          show derivations with answers
  :prose on|off          add an English rendering of the answer
  :save <path>           write the session's knowledge base as ASP
  :clear                 forget everything
  :quit
""".rstrip()

#: A leading "-" is strong negation -- "-flies(pingu)." is a claim that it is
#: false, and a line the session has to read as logic rather than as English.
_LOGIC_CALL = re.compile(r"^-?[a-z_][A-Za-z0-9_]*\s*\(.*\)\s*\.?\??$", re.DOTALL)


@dataclass
class Shell:
    """The session state and the one function that advances it.

    :meth:`handle` takes a line and returns what should be printed. It has no
    side effects beyond the session itself, which is what makes the shell
    testable without a terminal.
    """

    knowledge: KnowledgeBase = field(default_factory=lambda: KnowledgeBase("session"))
    schema_style: str = "triple"
    show_proof: bool = True
    show_prose: bool = False
    running: bool = True
    transcript: list[str] = field(default_factory=list)
    _perceptor: object = None
    _engine: object = None
    _gaps: object = None
    _memory: object = None
    _realiser: Realiser = field(default_factory=Realiser)
    _reported_violations: set = field(default_factory=set)
    _undo: list = field(default_factory=list)
    _repairs: list = field(default_factory=list)
    _undo_limit: int = 50

    # -- lazily built parts -------------------------------------------------

    @property
    def perceptor(self):
        if self._perceptor is None:
            from .perception.controlled import TripleSchema
            from .perception.text import TextPerceptor

            self._perceptor = TextPerceptor(schema=TripleSchema(self.schema_style))
        return self._perceptor

    @property
    def engine(self):
        if self._engine is None:
            self._engine = self.knowledge.engine()
        return self._engine

    def _invalidate(self) -> None:
        self._engine = None

    def _snapshot(self, label: str) -> None:
        """Remember the knowledge base so :undo can put it back."""
        self._undo.append(
            (label, list(self.knowledge.rules.rules), list(self.knowledge.facts))
        )
        del self._undo[: -self._undo_limit]

    def _restore(self, rules, facts) -> None:
        self.knowledge.rules.rules = list(rules)
        self.knowledge._facts = {record.atom: record for record in facts}
        self._reported_violations.clear()
        self._invalidate()

    def _reversible(self):
        """Context manager that can undo an assertion that breaks the session.

        A single unsafe or unstratifiable rule would otherwise wedge the whole
        knowledge base: every later command fails and the only way out is
        ``:clear``, which throws away the work. Anything that does not compile
        is rolled back and reported instead.
        """
        return _Reversible(self)

    # -- dispatch -----------------------------------------------------------

    def handle(self, line: str) -> str:
        """Process one line of input and return the response."""
        text = line.strip()
        if not text or text.startswith("%") or text.startswith("#"):
            return ""
        if text.startswith(":"):
            return self._command(text[1:].strip())
        try:
            if text.endswith("?"):
                return self._question(text)
            return self._assert(text)
        except ProgramError as exc:
            return f"! {exc}"
        except Exception as exc:  # pragma: no cover - last-resort guard
            return f"! {type(exc).__name__}: {exc}"

    # -- assertions ---------------------------------------------------------

    def _assert(self, text: str) -> str:
        logic_first = _looks_like_logic(text)
        attempts = (
            (self._assert_logic, self._assert_english)
            if logic_first
            else (self._assert_english, self._assert_logic)
        )
        problems = []
        for attempt in attempts:
            result = attempt(text)
            if result is not None:
                return result
            problems.append(attempt)
        return (
            "? I could not read that as English or as logic.\n"
            "  English I understand looks like 'Bob is a cat.', "
            "'The cat chases the mouse.',\n"
            "  'All cats are mammals.' or 'If something is a cat then it purrs.'"
        )

    def _assert_logic(self, text: str) -> Optional[str]:
        if not text.endswith("."):
            text += "."
        try:
            program = parse_program(text, "session", check=False)
        except (ParseError, ValueError):
            return None
        if not program.rules:
            return None
        self._snapshot(text)
        with self._reversible() as undo:
            added = []
            for rule in program.rules:
                if rule.is_fact and rule.head is not None and rule.head.is_ground:
                    self.knowledge.add_fact(rule.head, provenance="session")
                    added.append(str(rule.head))
                else:
                    self.knowledge.rules.add(rule)
                    added.append(str(rule))
            self._invalidate()
            rejected = undo.validate()
            if rejected:
                return rejected
        self.transcript.append(text)
        return self._added(added)

    def _assert_english(self, text: str) -> Optional[str]:
        perception = self.perceptor.perceive(text)
        if not perception.facts and not perception.rules:
            return None
        self._snapshot(text)
        with self._reversible() as undo:
            perception.into(self.knowledge)
            self._invalidate()
            rejected = undo.validate()
            if rejected:
                return rejected
        self._realiser.learn_names(perception.diagnostics.get("proper_names", ()))
        self.transcript.append(f"% {text}")
        added = []
        for record in perception.facts:
            label = str(record.atom)
            if record.confidence < 1.0:
                label += f"   [confidence {record.confidence:.2f} -- {_why_a_guess(record)}]"
            added.append(label)
            self.transcript.append(f"{record.atom}.")
        for rule in perception.rules:
            added.append(str(rule))
            self.transcript.append(str(rule))
        note = ""
        if perception.unparsed:
            note = "\n? not understood: " + "; ".join(
                fragment.split("  (")[0] for fragment in perception.unparsed
            )
        return self._added(added) + note

    def _added(self, items: Iterable[str]) -> str:
        lines = [f"  + {item}" for item in items]
        warning = self._new_violations()
        if warning:
            lines.append(warning)
        return "\n".join(lines) if lines else "  (nothing new)"

    def _new_violations(self) -> str:
        """Report any constraint broken since the last assertion."""
        try:
            violations = self.engine.violations
        except ProgramError as exc:
            return f"  ! the knowledge base no longer compiles: {exc}"
        fresh = [v for v in violations if v.describe() not in self._reported_violations]
        for violation in fresh:
            self._reported_violations.add(violation.describe())
        if not fresh:
            return ""
        return "\n".join(f"  ! {v.describe()}" for v in fresh)

    # -- questions ----------------------------------------------------------

    def _question(self, text: str) -> str:
        body = text.rstrip("?").strip()
        goal: Optional[Atom] = None
        if _looks_like_logic(body):
            try:
                goal = parse_atom(body)
            except (ParseError, ValueError):
                goal = None
        if goal is None:
            try:
                goal = self.perceptor.parse_question(text)
            except Exception:
                goal = None
        if goal is None:
            return (
                "? I could not read that question.\n"
                "  Try 'Is Bob a mammal?', 'Does the cat chase the mouse?' "
                "or a logic goal like 'ancestor(alice, X)?'"
            )

        try:
            answer = self.engine.ask(goal)
        except ProgramError as exc:
            return f"! {exc}"

        if not answer.holds:
            # A question the session could not answer is exactly what the
            # growth loop wants to hear about, and it is free to record here:
            # the diagnosis was produced anyway.
            self.gaps.from_answer(answer)
            lines = ["no" if answer.status == "no" else "unknown"]
            if answer.diagnosis is not None:
                lines.append(_indent(answer.diagnosis.describe()))
            return "\n".join(lines)

        lines = ["yes"]
        bound = [b for b in answer.bindings if b]
        if bound:
            for atom, binding in zip(answer.atoms, answer.bindings):
                if binding:
                    values = ", ".join(
                        f"{name} = {value.value if isinstance(value, Const) else value}"
                        for name, value in sorted(binding.items())
                    )
                    lines.append(f"  {values}    ({atom})")
        if self.show_proof and answer.proof is not None:
            lines.append(_indent(render_proof(answer.proof)))
        if self.show_prose:
            lines.append(_indent(self._realiser.realise_answer(answer)))
        return "\n".join(lines)

    # -- commands -----------------------------------------------------------

    def _command(self, text: str) -> str:
        name, _, argument = text.partition(" ")
        argument = argument.strip()
        handler = _COMMANDS.get(name.lower())
        if handler is None:
            return f"? unknown command :{name} -- try :help"
        return handler(self, argument)

    def cmd_help(self, argument: str) -> str:
        return HELP

    def cmd_quit(self, argument: str) -> str:
        self.running = False
        return "bye"

    def cmd_facts(self, argument: str) -> str:
        records = self.knowledge.facts
        if argument:
            records = [r for r in records if r.atom.predicate == argument]
        if not records:
            return "  (no facts)"
        lines = []
        for record in records:
            line = f"  {record.atom}"
            if record.confidence < 1.0:
                line += f"   [confidence {record.confidence:.2f}]"
            lines.append(line)
        return "\n".join(lines)

    def cmd_rules(self, argument: str) -> str:
        rules = self.knowledge.rules
        parts = [str(r) for r in rules.derivation_rules] + [
            str(r) for r in rules.constraints
        ]
        return "\n".join(f"  {p}" for p in parts) if parts else "  (no rules)"

    def cmd_sets(self, argument: str) -> str:
        return "  " + ", ".join(builtin_rulesets())

    def cmd_model(self, argument: str) -> str:
        try:
            model = self.engine.solve()
        except ProgramError as exc:
            return f"! {exc}"
        return _indent(render_model(model, [argument] if argument else None))

    def cmd_why(self, argument: str) -> str:
        if not argument:
            return "? :why needs an atom, e.g. ':why isa(bob, mammal)'"
        try:
            atom = parse_atom(argument.rstrip("."))
        except (ParseError, ValueError) as exc:
            return f"? {exc}"
        try:
            return _indent(render_proof(explain(self.engine.solve(), atom)))
        except ProofError as exc:
            return f"  {exc}"
        except ProgramError as exc:
            return f"! {exc}"

    def cmd_check(self, argument: str) -> str:
        try:
            violations = self.engine.violations
        except ProgramError as exc:
            return f"! {exc}"
        if not violations:
            return "  no violations: every hard rule holds"
        lines = []
        for violation, proof in zip(violations, self.engine.explain_violations()):
            lines.append(f"  ! {violation.describe()}")
            lines.append(_indent(render_proof(proof), "    "))
        return "\n".join(lines)

    def cmd_load(self, argument: str) -> str:
        if not argument:
            return f"? :load needs a name or path. Available: {', '.join(builtin_rulesets())}"
        path = Path(argument)
        try:
            source = path if path.exists() else rule_path(argument)
            before = len(self.knowledge.rules.rules)
            self.knowledge.load_rules(source)
        except (FileNotFoundError, ProgramError, ParseError) as exc:
            return f"? {exc}"
        self._invalidate()
        added = len(self.knowledge.rules.rules) - before
        return f"  loaded {source.name}: {added} rule(s), {len(self.knowledge)} fact(s) total"

    def cmd_retract(self, argument: str) -> str:
        if not argument:
            return "? :retract needs an atom"
        try:
            atom = parse_atom(argument.rstrip("."))
        except (ParseError, ValueError) as exc:
            return f"? {exc}"
        if self.knowledge.remove_fact(atom):
            self._invalidate()
            self._reported_violations.clear()
            return f"  - {atom}"
        return f"  {atom} was not asserted"

    def cmd_clear(self, argument: str) -> str:
        self.knowledge = KnowledgeBase("session")
        self.transcript.clear()
        self._reported_violations.clear()
        self._invalidate()
        return "  forgotten"

    def cmd_save(self, argument: str) -> str:
        if not argument:
            return "? :save needs a path"
        target = Path(argument)
        try:
            target.write_text(self.knowledge.to_asp() + "\n", encoding="utf-8")
        except OSError as exc:
            return f"? {exc}"
        return f"  wrote {target}"

    def cmd_schema(self, argument: str) -> str:
        if not argument:
            return f"  schema is '{self.schema_style}'"
        if argument not in ("direct", "triple"):
            return "? schema must be 'direct' or 'triple'"
        if argument != self.schema_style:
            self.schema_style = argument
            self._perceptor = None
            return (
                f"  schema is now '{argument}'. Facts already asserted keep their "
                "old shape -- :clear first for a clean slate."
            )
        return f"  schema is already '{argument}'"

    def cmd_proof(self, argument: str) -> str:
        return self._toggle("show_proof", argument, "proofs")

    def cmd_prose(self, argument: str) -> str:
        return self._toggle("show_prose", argument, "prose")

    def _toggle(self, attribute: str, argument: str, label: str) -> str:
        if argument in ("on", "off"):
            setattr(self, attribute, argument == "on")
        elif argument:
            return f"? {label} takes 'on' or 'off'"
        return f"  {label} {'on' if getattr(self, attribute) else 'off'}"

    def cmd_learn(self, argument: str) -> str:
        """Induce a rule for a predicate from what the session already knows."""
        match = re.match(r"^(\S+)(?:\s+from\s+(.+))?$", argument.strip())
        if not match:
            return (
                "? usage: :learn <target/arity> [from <pred/arity> ...]\n"
                "  e.g. ':learn grandparent/2' or ':learn grandparent/2 from parent/2'"
            )
        target = match.group(1)
        sources = match.group(2).split() if match.group(2) else None
        from .induction import Examples, LanguageBias, RuleLearner, Signature

        try:
            signature = Signature.parse(target)
        except ValueError as exc:
            return f"? {exc}"
        positives = [
            record.atom
            for record in self.knowledge.facts
            if record.atom.signature == (signature.name, signature.arity)
        ]
        if not positives:
            return (
                f"? no facts of {signature} are asserted, so there is nothing to "
                "generalise from. Assert some examples first."
            )
        universe = sorted(
            {
                str(argument_.value)
                for record in self.knowledge.facts
                for argument_ in record.atom.args
                if isinstance(argument_, Const) and not argument_.is_number
            }
        )
        try:
            examples = Examples.closed_world(positives, universe, signature)
            options = dict(
                max_variables=max(3, signature.arity + 2),
                max_body=3,
                allow_comparison=True,
                allow_recursion=False,
            )
            # With no "from" clause, take the vocabulary from the session
            # itself -- the caller should not have to know which predicates
            # are relevant before asking.
            bias = (
                LanguageBias.for_target(target, sources, **options)
                if sources is not None
                else LanguageBias.from_knowledge(self.knowledge, signature, **options)
            )
            hypothesis = RuleLearner(self.knowledge, bias, examples).learn()
        except (ValueError, ProgramError) as exc:
            return f"? {exc}"
        if not hypothesis.rules:
            return _indent(hypothesis.describe())
        if hypothesis.underdetermined:
            pass  # reported below, after the rules
        for rule in hypothesis.rules:
            self.knowledge.rules.add(rule)
            self.transcript.append(str(rule))
        self._invalidate()
        lines = [f"  + {rule}" for rule in hypothesis.rules]
        lines.append(
            f"  ({examples.summary()}, {hypothesis.candidates_evaluated} candidates tested)"
        )
        if hypothesis.underdetermined:
            lines.append(
                "  ! the examples do not single out one rule; these fit as well:"
            )
            for alternatives in hypothesis.ties.values():
                lines.extend(f"      {rule}" for rule in alternatives)
        if not hypothesis.correct:
            lines.append(f"  ! {hypothesis.incomplete_reason or 'not fully correct'}")
        return "\n".join(lines)

    # -- growth --------------------------------------------------------

    @property
    def gaps(self):
        if self._gaps is None:
            from .growth import GapCollector

            self._gaps = GapCollector()
        return self._gaps

    @property
    def memory(self):
        if self._memory is None:
            from .growth import Memory

            self._memory = Memory()
        return self._memory

    # -- drafting a module (P2.8) -------------------------------------------

    def cmd_bootstrap(self, argument: str) -> str:
        """Draft a module from an observation log: ``:bootstrap log.jsonl``."""
        from pathlib import Path as _Path

        from .builder import Builder
        from .builder.stations import read_log

        parts = argument.split()
        if not parts:
            return (
                "  :bootstrap <log.jsonl> [name] — draft a module from a log "
                "of what a host was seen doing"
            )
        path = _Path(parts[0])
        if not path.exists():
            return f"  no log at {path}"
        name = parts[1] if len(parts) > 1 else path.stem

        try:
            log = read_log(path)
        except Exception as exc:  # a log is somebody else's file
            return f"  could not read {path}: {exc}"
        if not log:
            return f"  {path} has no records in it"

        self.builder = Builder.from_log(log, name=name)
        self.builder.step()
        return self.builder.progress()

    def cmd_pack(self, argument: str) -> str:
        """``:pack status`` | ``:pack write <dir>`` | ``:pack promote <name>``."""
        builder = getattr(self, "builder", None)
        if builder is None:
            return "  nothing drafted yet — :bootstrap a log first"

        parts = argument.split()
        what = parts[0] if parts else "status"

        if what == "status":
            return builder.progress()
        if what == "write":
            from pathlib import Path as _Path

            root = _Path(parts[1]) if len(parts) > 1 else _Path("packs")
            folder = builder.write(root)
            return f"  written to {folder}"
        if what == "promote":
            if len(parts) < 2:
                return (
                    "  :pack promote <your name> — a drafted pack is a "
                    "proposal about somebody else's domain, so promoting it "
                    "is signed"
                )
            from .builder.shadow import Shadow

            shadow = Shadow(builder.pack(), self.kernel)
            shadow.run()
            return "  " + shadow.promote(approved_by=" ".join(parts[1:]))
        return f"  :pack status | write <dir> | promote <name>; not {what!r}"

    def cmd_ask(self, argument: str) -> str:
        """Answer the builder's next question: ``:ask yes`` / ``:ask no``."""
        builder = getattr(self, "builder", None)
        if builder is None:
            return "  nothing drafted yet — :bootstrap a log first"
        question = builder.interview.next
        if question is None:
            return "  nothing outstanding"
        reply = argument.strip().lower()
        if not reply:
            return "  " + question.describe()
        if reply not in question.options:
            return f"  answer one of: {', '.join(question.options)}"
        builder.answer(question.key, reply)
        following = builder.interview.next
        settled = f"  noted: {question.text} -> {reply}"
        if following is None:
            return settled + "\n  nothing else outstanding"
        return settled + "\n  next: " + following.describe()

    def cmd_gaps(self, argument: str) -> str:
        """What the session could not answer, most-asked first."""
        found = self.gaps.ranked()
        if not found:
            return "  no gaps yet — ask something it cannot answer"
        return "\n".join(f"  {gap.describe()}" for gap in found)

    def cmd_questions(self, argument: str) -> str:
        """What the session would need to be told to close its gaps."""
        found = self.gaps.ranked()
        if not found:
            return "  nothing to ask about"
        lines = []
        for gap in found:
            if gap.missing is not None:
                lines.append(f"  Is {gap.missing} true?")
            elif gap.goal is not None:
                lines.append(f"  Show me something that is {gap.goal.predicate}?")
            else:
                lines.append(f"  How should I read: {gap.subject}")
        return "\n".join(lines)

    def cmd_grow(self, argument: str) -> str:
        """Run one growth cycle, asking about the gap it names.

        Interactive rather than automatic: each question is put to you, and
        what comes back is a *proposal*. Nothing changes an answer until
        ``:accept`` confirms it.
        """
        target = argument.strip()
        if not target:
            found = [g for g in self.gaps.ranked() if g.goal is not None]
            if not found:
                return "? usage: :grow <target/arity>   (or ask something first)"
            goal = found[0].goal
            target = f"{goal.predicate}/{goal.arity}"
        from .growth import GrowthLoop, PROPOSED

        pending: list = []

        def ask(atom):
            pending.append(atom)
            return None  # the shell cannot block for an answer here

        try:
            session = GrowthLoop(self.knowledge, max_questions=1).learn(target, ask)
        except Exception as exc:
            return f"? {type(exc).__name__}: {exc}"
        if pending:
            return (
                f"  to learn {target} I need to know:\n"
                f"    Is {pending[0]} true?\n"
                "  answer it as a fact, then :grow again"
            )
        if not session.rules:
            return f"  nothing found for {target}: {session.stopped}"
        lines = [f"  proposed for {target} after {session.questions} question(s):"]
        for rule in session.rules:
            self.memory.remember(
                str(rule), state=PROPOSED, provenance=f"growth:{target}"
            )
            lines.append(f"    ? {rule}")
        lines.append("  nothing changes until you confirm it with :accept")
        return "\n".join(lines)

    def cmd_beliefs(self, argument: str) -> str:
        """Everything remembered, and what the session makes of it."""
        beliefs = self.memory.beliefs()
        if not beliefs:
            return "  nothing remembered yet"
        return "\n".join(f"  {belief.describe()}" for belief in beliefs)

    def cmd_undo(self, argument: str) -> str:
        if not self._undo:
            return "  nothing to undo"
        label, rules, facts = self._undo.pop()
        self._restore(rules, facts)
        if self.transcript:
            self.transcript.pop()
        return f"  undid: {label}"

    def cmd_history(self, argument: str) -> str:
        if not self.transcript:
            return "  (nothing yet)"
        return "\n".join(f"  {line}" for line in self.transcript)

    def cmd_open(self, argument: str) -> str:
        """Replace the session with a saved knowledge base."""
        if not argument:
            return "? :open needs a path"
        path = Path(argument)
        if not path.exists():
            return f"? no such file: {path}"
        self._snapshot(f":open {path}")
        fresh = KnowledgeBase("session")
        try:
            fresh.load_rules(path)
            fresh.program()
        except (ProgramError, ParseError, OSError) as exc:
            self._undo.pop()
            return f"? {path} would not load: {exc}"
        self.knowledge = fresh
        self._reported_violations.clear()
        self._invalidate()
        return (
            f"  opened {path}: {len(self.knowledge)} fact(s), "
            f"{len(self.knowledge.rules.derivation_rules)} rule(s)"
        )

    def cmd_expect(self, argument: str) -> str:
        """Complain that something should follow and does not."""
        return self._correct(expected=argument, label="should hold")

    def cmd_wrong(self, argument: str) -> str:
        """Complain that something follows and should not."""
        return self._correct(rejected=argument, label="should not hold")

    def _correct(self, expected: str = "", rejected: str = "", label: str = "") -> str:
        argument = expected or rejected
        if not argument:
            return "? give an atom, e.g. ':wrong flies(pingu)'"
        try:
            atom = parse_atom(argument.rstrip("."))
        except (ParseError, ValueError) as exc:
            return f"? {exc}"
        from .induction.repair import Corrector

        try:
            repairs = Corrector(self.knowledge).diagnose(
                expected=[atom] if expected else (),
                rejected=[atom] if rejected else (),
            )
        except ProgramError as exc:
            return f"! {exc}"
        if not repairs:
            holds = atom in self.engine.solve()
            if expected and holds:
                return f"  {atom} already follows -- nothing to fix"
            if rejected and not holds:
                return f"  {atom} does not follow -- nothing to fix"
            return "  I could not find any change that would help"
        self._repairs = repairs
        lines = [f"  {atom} {label}. Possible changes:"]
        for index, repair in enumerate(repairs, 1):
            lines.append(f"  [{index}] " + repair.describe().replace("\n", "\n  "))
        lines.append("  apply one with ':accept <n>'")
        return "\n".join(lines)

    def cmd_accept(self, argument: str) -> str:
        if not self._repairs:
            return "? nothing proposed -- use :expect or :wrong first"
        try:
            index = int(argument) - 1
        except ValueError:
            return "? :accept needs a number from the list"
        if not 0 <= index < len(self._repairs):
            return f"? choose between 1 and {len(self._repairs)}"
        repair = self._repairs[index]
        self._snapshot(f":accept {index + 1}")
        repair.apply(self.knowledge)
        self._reported_violations.clear()
        self._invalidate()
        try:
            self.knowledge.program()
        except ProgramError as exc:
            label, rules, facts = self._undo.pop()
            self._restore(rules, facts)
            return f"! rejected, that change would not compile:\n  {exc}"
        for rule in repair.add:
            self.transcript.append(str(rule))
        for atom in repair.add_facts:
            self.transcript.append(f"{atom}.")
        self._repairs = []
        return "  applied:\n" + "\n".join(
            f"    {line}" for line in repair.describe().splitlines()[1:]
        )

    # -- the loop -----------------------------------------------------------

    def run(
        self,
        stream: Optional[TextIO] = None,
        out: Optional[TextIO] = None,
        prompt: str = "> ",
        banner: bool = True,
    ) -> int:
        """Read lines until EOF or ``:quit``."""
        stream = stream or sys.stdin
        out = out or sys.stdout
        interactive = stream.isatty() if hasattr(stream, "isatty") else False
        if banner:
            print(BANNER, file=out)
        while self.running:
            if interactive:
                print(prompt, end="", file=out, flush=True)
            line = stream.readline()
            if not line:
                break
            if not interactive and line.strip():
                print(f"{prompt}{line.rstrip()}", file=out)
            response = self.handle(line)
            if response:
                print(response, file=out, flush=True)
        return 0


class _Reversible:
    """Snapshot the knowledge base, and restore it if the result will not run."""

    def __init__(self, shell: "Shell") -> None:
        self.shell = shell
        self.rules: list = []
        self.facts: list = []

    def __enter__(self) -> "_Reversible":
        knowledge = self.shell.knowledge
        self.rules = list(knowledge.rules.rules)
        self.facts = list(knowledge.facts)
        return self

    def __exit__(self, *exc_info) -> bool:
        return False

    def restore(self) -> None:
        knowledge = self.shell.knowledge
        knowledge.rules.rules = list(self.rules)
        knowledge._facts = {record.atom: record for record in self.facts}
        self.shell._invalidate()

    def validate(self) -> Optional[str]:
        """Return an error message and undo, or None if all is well."""
        try:
            self.shell.knowledge.program()
        except ProgramError as exc:
            self.restore()
            return f"! rejected, the knowledge base would not compile:\n  {exc}"
        return None


def _looks_like_logic(text: str) -> bool:
    """True if the line is more plausibly ASP than English."""
    stripped = text.strip()
    if ":-" in stripped:
        return True
    return bool(_LOGIC_CALL.match(stripped))


def _why_a_guess(record) -> str:
    """Why a fact came in below full confidence, in the reader's own terms.

    The two readers are uncertain for different reasons, and saying so is the
    point: a grammar reading with nothing to anchor it is a different kind of
    doubt from a dependency pattern that usually but not always means what it
    looks like.
    """
    if record.provenance.endswith("narrative"):
        return "read from a dependency parse of free text, so this is a guess"
    return (
        "no determiner or copula to anchor the reading, so this is a guess"
    )


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(prefix + line for line in text.splitlines())


_COMMANDS: dict[str, Callable[[Shell, str], str]] = {
    "help": Shell.cmd_help,
    "h": Shell.cmd_help,
    "?": Shell.cmd_help,
    "quit": Shell.cmd_quit,
    "q": Shell.cmd_quit,
    "exit": Shell.cmd_quit,
    "facts": Shell.cmd_facts,
    "rules": Shell.cmd_rules,
    "sets": Shell.cmd_sets,
    "model": Shell.cmd_model,
    "why": Shell.cmd_why,
    "check": Shell.cmd_check,
    "load": Shell.cmd_load,
    "retract": Shell.cmd_retract,
    "clear": Shell.cmd_clear,
    "save": Shell.cmd_save,
    "schema": Shell.cmd_schema,
    "proof": Shell.cmd_proof,
    "prose": Shell.cmd_prose,
    "learn": Shell.cmd_learn,
    "gaps": Shell.cmd_gaps,
    "questions": Shell.cmd_questions,
    "grow": Shell.cmd_grow,
    "beliefs": Shell.cmd_beliefs,
    "undo": Shell.cmd_undo,
    "history": Shell.cmd_history,
    "open": Shell.cmd_open,
    "expect": Shell.cmd_expect,
    "wrong": Shell.cmd_wrong,
    "accept": Shell.cmd_accept,
    "bootstrap": Shell.cmd_bootstrap,
    "pack": Shell.cmd_pack,
    "ask": Shell.cmd_ask,
}


def run_shell(**kwargs) -> int:
    """Entry point used by ``neuralmind shell``."""
    return Shell(**kwargs).run()
