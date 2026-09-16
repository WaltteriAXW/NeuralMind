"""Command-line interface.

    neuralmind shell    interactive session: state facts, ask questions
    neuralmind ask      query a knowledge base, with a proof
    neuralmind read     turn English into facts and rules
    neuralmind check    report integrity-constraint violations
    neuralmind solve    print the whole model
    neuralmind induce   learn rules from examples
    neuralmind demo     run one of the roadmap phases end to end
    neuralmind eval     benchmark the pipeline and attribute its failures
                        (--corpus for the real ProofWriter data)
    neuralmind verify   re-derive the model with clingo and compare
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

from .core.parser import ParseError
from .core.program import ProgramError
from .inference.forward import ReasoningLimit, UnsupportedProgram
from .knowledge.base import KnowledgeBase, builtin_rulesets, rule_path
from .perception.base import PerceptionError

#: Errors that mean "your input was wrong", as opposed to "this is a bug".
_USER_ERRORS = (
    FileNotFoundError,
    ValueError,
    ParseError,
    ProgramError,
    PerceptionError,
    ReasoningLimit,
    UnsupportedProgram,
)

__all__ = ["main"]


def _build_kb(rules: Sequence[str], facts: Sequence[str], fact_files: Sequence[str]) -> KnowledgeBase:
    kb = KnowledgeBase("cli")
    for source in rules:
        path = Path(source)
        if path.exists():
            kb.load_rules(path)
        elif source in builtin_rulesets():
            kb.load_rules(rule_path(source))
        else:
            kb.add_rules(source, name="--rules")
    for source in fact_files:
        kb.load_rules(Path(source))
    for fact in facts:
        kb.add_fact(fact)
    return kb


def _add_kb_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-r",
        "--rules",
        action="append",
        default=[],
        metavar="SOURCE",
        help="a .lp file, a bundled rule set name, or inline ASP source (repeatable)",
    )
    parser.add_argument(
        "-f", "--fact", action="append", default=[], metavar="ATOM",
        help="a ground fact such as 'parent(alice, bob)' (repeatable)",
    )
    parser.add_argument(
        "-F", "--facts-file", action="append", default=[], metavar="PATH",
        help="a .lp file of ground facts (repeatable)",
    )
    parser.add_argument("--text", metavar="TEXT", help="English to read facts and rules from")
    parser.add_argument("--text-file", type=Path, help="a file of English to read")


def _read_text(args, kb: KnowledgeBase) -> Optional[object]:
    text = args.text
    if getattr(args, "text_file", None):
        text = (text or "") + "\n" + Path(args.text_file).read_text(encoding="utf-8")
    if not text:
        return None
    from .perception.text import TextPerceptor

    perception = TextPerceptor().perceive(text)
    perception.into(kb)
    return perception


def cmd_ask(args) -> int:
    kb = _build_kb(args.rules, args.fact, args.facts_file)
    perception = _read_text(args, kb)
    from .pipeline import NeuralMindPipeline

    pipeline = NeuralMindPipeline(kb, prose=not args.no_prose)
    if perception is not None:
        pipeline.realiser.learn_names(perception.diagnostics.get("proper_names", ()))
    result = pipeline.ask(args.query)
    if args.json:
        print(result.to_json(include_facts=False))
    else:
        print(result)
    return 0 if result.holds else 1


def cmd_read(args) -> int:
    from .perception.text import TextPerceptor

    text = args.text
    if args.text_file:
        text = (text or "") + Path(args.text_file).read_text(encoding="utf-8")
    if not text:
        text = sys.stdin.read()
    perception = TextPerceptor().perceive(text)
    if args.json:
        from .output.serialize import to_json

        print(to_json(perception.to_dict()))
        return 0
    for record in perception.facts:
        print(f"{str(record.atom):50s} {record.confidence:.2f}  {record.provenance}")
    for rule in perception.rules:
        print(rule)
    for fragment in perception.unparsed:
        print(f"[unparsed] {fragment}", file=sys.stderr)
    return 0


def cmd_check(args) -> int:
    kb = _build_kb(args.rules, args.fact, args.facts_file)
    _read_text(args, kb)
    engine = kb.engine()
    violations = engine.violations
    if args.json:
        from .output.serialize import to_json, violations_document

        print(to_json(violations_document(violations)))
    else:
        from .output.render import render_proof

        if not violations:
            print("no violations: every hard rule holds")
        for violation, proof in zip(violations, engine.explain_violations()):
            print(f"VIOLATION {violation.describe()}")
            if args.explain:
                print(render_proof(proof))
                print()
    return 1 if violations else 0


def cmd_solve(args) -> int:
    kb = _build_kb(args.rules, args.fact, args.facts_file)
    _read_text(args, kb)
    engine = kb.engine()
    model = engine.solve()
    if args.json:
        from .output.serialize import model_document, to_json

        print(to_json(model_document(model, args.predicate or None)))
    else:
        from .output.render import render_model, render_violations

        print(render_model(model, args.predicate or None))
        if model.violations:
            print()
            print(render_violations(model.violations))
    return 0


def cmd_verify(args) -> int:
    kb = _build_kb(args.rules, args.fact, args.facts_file)
    _read_text(args, kb)
    result = kb.engine().cross_check()
    print(result.report())
    return 0 if result.agree else 1


def cmd_eval(args) -> int:
    from .evaluation import evaluate

    perceptor = None
    if args.corpus:
        from .datasets import proofwriter_corpus
        from .perception.controlled import TripleSchema
        from .perception.text import TextPerceptor

        if args.corpus not in proofwriter_corpus.SPLITS:
            print(
                f"unknown split {args.corpus!r}; choose from "
                + ", ".join(proofwriter_corpus.SPLITS),
                file=sys.stderr,
            )
            return 2
        if not proofwriter_corpus.available(args.corpus):
            print(
                f"the '{args.corpus}' split is not downloaded. Run "
                "`python scripts/download_proofwriter.py`.",
                file=sys.stderr,
            )
            return 2
        problems = proofwriter_corpus.load(args.corpus, limit=args.problems)
        # These theories need one predicate per attribute: under the triple
        # schema every atom is attr/2, and a rule like "if smart and not white
        # then round" makes attr depend negatively on itself.
        perceptor = TextPerceptor(schema=TripleSchema("direct"))
        if not args.json:
            print(f"ProofWriter corpus, {args.corpus}: {len(problems)} problems")
    else:
        from .datasets.proofwriter import generate

        problems = generate(args.problems, seed=args.seed, questions=args.questions)

    report = evaluate(problems, perceptor=perceptor)
    if args.json:
        from .output.serialize import to_json

        print(to_json(report.to_dict()))
    else:
        print(report.describe())
    return 0


def cmd_induce(args) -> int:
    from .core.terms import Const
    from .induction import Examples, LanguageBias, RuleLearner

    kb = _build_kb(args.rules, args.fact, args.facts_file)
    _read_text(args, kb)

    positives = list(args.positive)
    if args.positives_file:
        positives += _atoms_from_file(args.positives_file)
    if not positives:
        print("error: at least one --positive example is required", file=sys.stderr)
        return 2

    negatives = list(args.negative)
    if args.negatives_file:
        negatives += _atoms_from_file(args.negatives_file)

    bias = LanguageBias.for_target(
        args.target,
        args.body,
        max_variables=args.max_vars,
        max_body=args.max_body,
        max_clauses=args.max_clauses,
        allow_recursion=not args.no_recursion,
        allow_negation=args.negation,
    )

    if args.closed_world and not negatives:
        # Everything over the constants seen in the background is a negative,
        # unless it was given as a positive.
        constants = sorted(
            {
                str(argument.value)
                for record in kb.facts
                for argument in record.atom.args
                if isinstance(argument, Const) and not argument.is_number
            }
        )
        examples = Examples.closed_world(positives, constants, bias.target)
    else:
        examples = Examples.from_strings(positives, negatives)

    if not args.json:
        print(bias.describe())
        print(f"examples: {examples.summary()}\n")

    hypothesis = RuleLearner(kb, bias, examples, budget=args.budget).learn(
        verbose=not args.json
    )
    if args.json:
        from .output.serialize import to_json

        print(to_json(hypothesis.to_dict()))
    else:
        print()
        print(hypothesis.describe())
        if args.closed_world and not hypothesis.complete:
            print()
            print(
                "hint: --closed-world made every unlisted atom a negative example. "
                "If the positives you gave are not the complete relation, the "
                "correct rule will derive an unlisted one and be rejected for it. "
                "List every positive, or supply negatives explicitly."
            )
    return 0 if hypothesis.correct else 1


def _atoms_from_file(path: Path) -> list[str]:
    """One atom per line; blank lines and % comments ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [
        line.strip().rstrip(".")
        for line in lines
        if line.strip() and not line.lstrip().startswith("%")
    ]


def cmd_shell(args) -> int:
    from .shell import Shell

    kb = _build_kb(args.rules, args.fact, args.facts_file)
    shell = Shell(
        knowledge=kb,
        schema_style=args.schema,
        show_proof=not args.no_proof,
        show_prose=args.prose,
    )
    return shell.run(banner=not args.quiet)


def cmd_demo(args) -> int:
    from . import demos

    runner = demos.REGISTRY.get(args.name)
    if runner is None:
        print(f"unknown demo {args.name!r}; choose from {', '.join(demos.REGISTRY)}", file=sys.stderr)
        return 2
    return runner(json_output=args.json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralmind",
        description="A non-LLM neurosymbolic reasoning system.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--version", action="version", version=_version())
    sub = parser.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="query a knowledge base and print the proof")
    _add_kb_arguments(ask)
    ask.add_argument("query", help="an atom such as 'ancestor(alice, X)' or an English question")
    ask.add_argument("--json", action="store_true", help="emit JSON instead of a tree")
    ask.add_argument("--no-prose", action="store_true", help="skip the English explanation")
    ask.set_defaults(func=cmd_ask)

    read = sub.add_parser("read", help="extract facts and rules from English")
    read.add_argument("text", nargs="?", help="text to read (default: stdin)")
    read.add_argument("--text-file", type=Path)
    read.add_argument("--json", action="store_true")
    read.set_defaults(func=cmd_read)

    check = sub.add_parser("check", help="report integrity-constraint violations")
    _add_kb_arguments(check)
    check.add_argument("--explain", action="store_true", help="show a proof tree per violation")
    check.add_argument("--json", action="store_true")
    check.set_defaults(func=cmd_check)

    solve = sub.add_parser("solve", help="print the whole model")
    _add_kb_arguments(solve)
    solve.add_argument("-p", "--predicate", action="append", default=[], help="restrict output")
    solve.add_argument("--json", action="store_true")
    solve.set_defaults(func=cmd_solve)

    verify = sub.add_parser("verify", help="re-derive the model with clingo and compare")
    _add_kb_arguments(verify)
    verify.set_defaults(func=cmd_verify)

    evaluate_parser = sub.add_parser("eval", help="benchmark the pipeline")
    evaluate_parser.add_argument("-n", "--problems", type=int, default=50)
    evaluate_parser.add_argument(
        "--corpus",
        metavar="SPLIT",
        help="evaluate on the real ProofWriter corpus instead of generated "
        "problems: depth-0..depth-5, birds-electricity, NatLang",
    )
    evaluate_parser.add_argument("-q", "--questions", type=int, default=6)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--json", action="store_true")
    evaluate_parser.set_defaults(func=cmd_eval)

    induce = sub.add_parser("induce", help="learn rules from examples")
    _add_kb_arguments(induce)
    induce.add_argument("--target", required=True, metavar="PRED/ARITY",
                        help="the predicate to learn, e.g. 'grandparent/2'")
    induce.add_argument("--body", action="append", default=[], metavar="PRED/ARITY",
                        help="a predicate usable in rule bodies (repeatable)")
    induce.add_argument("--positive", action="append", default=[], metavar="ATOM",
                        help="a positive example (repeatable)")
    induce.add_argument("--negative", action="append", default=[], metavar="ATOM",
                        help="a negative example (repeatable)")
    induce.add_argument("--positives-file", type=Path, help="one positive atom per line")
    induce.add_argument("--negatives-file", type=Path, help="one negative atom per line")
    induce.add_argument("--closed-world", action="store_true",
                        help="treat everything not given as positive as negative")
    induce.add_argument("--max-body", type=int, default=3)
    induce.add_argument("--max-vars", type=int, default=4)
    induce.add_argument("--max-clauses", type=int, default=4)
    induce.add_argument("--no-recursion", action="store_true",
                        help="forbid the target in rule bodies, shrinking the search")
    induce.add_argument("--negation", action="store_true",
                        help="allow negated body literals, for learning exceptions")
    induce.add_argument("--budget", type=int, default=200_000,
                        help="maximum candidate clauses to test")
    induce.add_argument("--json", action="store_true")
    induce.set_defaults(func=cmd_induce)

    shell = sub.add_parser("shell", help="interactive session: state facts, ask questions")
    _add_kb_arguments(shell)
    shell.add_argument("--schema", choices=("triple", "direct"), default="triple",
                       help="how English maps onto predicates")
    shell.add_argument("--no-proof", action="store_true", help="answers without derivations")
    shell.add_argument("--prose", action="store_true", help="add an English rendering")
    shell.add_argument("--quiet", action="store_true", help="skip the banner")
    shell.set_defaults(func=cmd_shell)

    demo = sub.add_parser("demo", help="run a roadmap phase end to end")
    demo.add_argument(
        "name", help="one of: family, text, mnist, repair, policy, learn, induce"
    )
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=cmd_demo)
    return parser


def _version() -> str:
    from . import __version__

    return f"neuralmind {__version__}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    except _USER_ERRORS as exc:
        # A malformed rule or an unreadable sentence is the user's problem to
        # fix, not a crash to debug: report it as a message, not a traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
