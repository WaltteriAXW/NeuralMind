"""The curriculum: what the mind is measured on, in the order it builds up.

The roadmap lists thirteen stages and gates each on the one before it. This
file is those stages, with one rule applied throughout: **a stage that cannot
run says why, rather than being absent**. A curriculum that silently omits what
it cannot reach reports a smaller, better-looking mind than the one that exists.

So each stage has a ``check`` that decides whether it can run at all, and the
four states are:

``passed`` / ``failed``
    It ran and met, or missed, its target.
``unavailable``
    It cannot run here, with the reason -- a dataset not downloaded, a
    dependency not installed, an environment not reachable from this machine.
``blocked``
    An earlier stage it depends on did not pass. Gating, which is the point of
    an ordered curriculum: there is no information in a NatLang score from a
    mind that cannot do generated ProofWriter.

Every target below is a number from `docs/phase-two-status.md`, so a
regression fails by name rather than by someone noticing a table has changed.
"""

from __future__ import annotations

import string
import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

from .attribution import Attribution, Evidence, attribute, breakdown

__all__ = ["Stage", "Result", "CURRICULUM", "PASSED", "FAILED", "UNAVAILABLE", "BLOCKED"]

#: The checkpoint trained with no digit labels -- Phase One's headline result.
_WEAK_WEIGHTS = (
    __import__("pathlib").Path(__file__).resolve().parents[1]
    / "perception" / "weights" / "mnist_weak.npz"
)

PASSED = "passed"
FAILED = "failed"
UNAVAILABLE = "unavailable"
BLOCKED = "blocked"


@dataclass
class Result:
    """How one stage went."""

    stage: str
    status: str
    #: The number the stage is about, and what it had to beat.
    metric: Optional[float] = None
    target: Optional[float] = None
    unit: str = ""
    detail: str = ""
    attributions: list[Attribution] = field(default_factory=list)
    seconds: float = 0.0
    #: Numbers worth keeping that are not the headline one.
    extra: dict = field(default_factory=dict)

    @property
    def ran(self) -> bool:
        return self.status in (PASSED, FAILED)

    def describe(self) -> str:
        if self.status == UNAVAILABLE:
            return f"  {self.stage:26} —        unavailable: {self.detail}"
        if self.status == BLOCKED:
            return f"  {self.stage:26} —        blocked: {self.detail}"
        mark = "ok  " if self.status == PASSED else "FAIL"
        value = f"{self.metric:.1f}{self.unit}" if self.metric is not None else "—"
        against = f" (need {self.target:.1f}{self.unit})" if self.target is not None else ""
        return f"  {self.stage:26} {mark} {value:>9}{against}  {self.detail}"

    def to_dict(self) -> dict:
        payload = {
            "stage": self.stage,
            "status": self.status,
            "seconds": round(self.seconds, 2),
        }
        if self.metric is not None:
            payload["metric"] = round(self.metric, 4)
            payload["unit"] = self.unit
        if self.target is not None:
            payload["target"] = self.target
        if self.detail:
            payload["detail"] = self.detail
        if self.attributions:
            payload["failures"] = breakdown(self.attributions)
        if self.extra:
            payload["extra"] = self.extra
        return payload

    def __str__(self) -> str:
        return self.describe()


@dataclass
class Stage:
    """One thing the mind is measured on."""

    name: str
    #: What it is for, in one line, shown when the curriculum is listed.
    about: str
    #: ``() -> str`` returning why it cannot run, or "" if it can.
    check: Callable[[], str]
    #: ``(quick: bool) -> Result`` -- the measurement itself.
    run: Callable[[bool], Result]
    #: Stages that must have passed first.
    after: tuple[str, ...] = ()
    #: Which phase's claim this stage defends.
    milestone: str = ""
    #: Skipped unless the caller asks for the long run.
    slow: bool = False

    def why_not(self) -> str:
        """Why this stage cannot run, or ``""`` if it can.

        Always call this rather than ``check`` directly. A check reports a
        missing dependency, and the usual way to find out whether a dependency
        is there is to import something -- so the check itself is the most
        likely place in the whole curriculum to raise. A check that raises
        takes down the listing of what *else* could run, which is exactly the
        information the school exists to give, so the exception becomes the
        reason instead.
        """
        try:
            return self.check()
        except Exception as exc:  # noqa: BLE001 -- see the docstring
            return f"its availability check raised {type(exc).__name__}: {exc}"


# -- helpers shared by the stages -----------------------------------------


def _corpus_available(world: str = "cwa", split: str = "depth-2") -> str:
    from ..datasets import proofwriter_corpus as corpus

    if not corpus.available(split, world=world):
        return (
            f"the {split} {world.upper()} split is not downloaded "
            "(python scripts/download_proofwriter.py)"
        )
    return ""


def _natlang_available() -> str:
    """The free-text corpus needs the free-text reader.

    NatLang is unrestricted English, and the grammar reader is not built for
    it: without spaCy the same stage scores 51.7% instead of 72.9%, so the
    target would be measuring which packages are installed rather than how
    well the mind reads. Same rule as the specialists -- declare the
    dependency, do not quietly report a different number.
    """
    from ..perception.text import spacy_available

    if not spacy_available():
        return (
            "spaCy and en_core_web_sm are needed for free-text reading "
            "(pip install neuralmind[text] && python -m spacy download en_core_web_sm)"
        )
    return _corpus_available("cwa", "NatLang")


def _needs(module: str, extra: str) -> Callable[[], str]:
    def check() -> str:
        try:
            __import__(module)
        except ImportError:
            return f"{module} is not installed (pip install neuralmind[{extra}])"
        return ""
    return check


def _specialists_available() -> str:
    """All four backends, because the stage measures them cooperating.

    Run with three of the four missing, this stage scores about a third and
    calls it a failure -- which reads as "the mind cannot reason about units"
    when the truth is "pint is not installed". A number that means something
    different depending on the environment is worse than no number, so the
    stage declares what it needs and steps aside when it is not there.
    """
    from ..workspace.specialists import installed

    absent = [name for name, ready in installed().items() if not ready]
    if absent:
        return (
            f"{', '.join(absent)} specialist(s) unavailable "
            "(pip install neuralmind[specialists])"
        )
    return ""


def _always() -> str:
    return ""


def _never(reason: str) -> Callable[[], str]:
    return lambda: reason


def _timed(name: str, target: Optional[float], unit: str):
    """Wrap a measurement so every stage reports the same shape."""

    def decorate(function):
        def wrapped(quick: bool) -> Result:
            started = time.perf_counter()
            try:
                metric, detail, attributions, extra = function(quick)
            except Exception as exc:  # a stage that crashes is a failed stage
                return Result(
                    stage=name,
                    status=FAILED,
                    detail=f"{type(exc).__name__}: {exc}",
                    seconds=time.perf_counter() - started,
                )
            status = PASSED if target is None or metric >= target else FAILED
            return Result(
                stage=name,
                status=status,
                metric=metric,
                target=target,
                unit=unit,
                detail=detail,
                attributions=list(attributions),
                seconds=time.perf_counter() - started,
                extra=extra,
            )
        return wrapped
    return decorate


# -- 1. reasoning on generated problems ------------------------------------


@_timed("proofwriter-cwa", 99.9, "%")
def _run_proofwriter_cwa(quick: bool):
    from ..datasets import proofwriter_corpus as corpus
    from ..evaluation import evaluate
    from ..perception.controlled import TripleSchema
    from ..perception.text import TextPerceptor

    limit = 40 if quick else None
    perceptor = TextPerceptor(TripleSchema("direct"), prefer="auto")
    totals = [0, 0]
    engine_failures = 0
    for split in ("depth-0", "depth-2", "depth-5"):
        if not corpus.available(split):
            continue
        report = evaluate(corpus.load(split, limit=limit, schema="direct"), perceptor)
        totals[0] += report.correct
        totals[1] += report.total
        engine_failures += report.failure_breakdown().get("engine", 0)
    accuracy = 100.0 * totals[0] / totals[1] if totals[1] else 0.0
    attributions = [
        attribute(Evidence(gold_disagrees=True)) for _ in range(engine_failures)
    ]
    return (
        accuracy,
        f"{totals[0]:,}/{totals[1]:,} questions, {engine_failures} engine failure(s)",
        attributions,
        {"questions": totals[1], "engine_failures": engine_failures},
    )


# -- 2. the same theories, open-world -------------------------------------


@_timed("proofwriter-owa", 99.9, "%")
def _run_proofwriter_owa(quick: bool):
    from ..datasets import proofwriter_corpus as corpus
    from ..evaluation import evaluate
    from ..perception.controlled import TripleSchema
    from ..perception.text import TextPerceptor

    limit = 40 if quick else None
    perceptor = TextPerceptor(TripleSchema("direct", negation="-"), prefer="auto")
    totals, unknowns = [0, 0], 0
    for split in ("depth-0", "depth-2", "depth-5"):
        if not corpus.available(split, world="owa"):
            continue
        problems = corpus.load(split, limit=limit, world="owa", schema="direct")
        report = evaluate(problems, perceptor)
        totals[0] += report.correct
        totals[1] += report.total
        unknowns += sum(
            1 for p in problems for q in p.questions if q.status == "unknown"
        )
    accuracy = 100.0 * totals[0] / totals[1] if totals[1] else 0.0
    share = 100.0 * unknowns / totals[1] if totals[1] else 0.0
    return (
        accuracy,
        f"{totals[1]:,} questions, {share:.0f}% of gold answers are unknown",
        [],
        {"questions": totals[1], "unknown_share": round(share, 1)},
    )


# -- 3. the crowdsourced register -----------------------------------------


@_timed("proofwriter-natlang", 68.0, "%")
def _run_natlang(quick: bool):
    from ..datasets import proofwriter_corpus as corpus
    from ..evaluation import evaluate
    from ..perception.controlled import TripleSchema
    from ..perception.text import TextPerceptor

    limit = 60 if quick else None
    report = evaluate(
        corpus.load("NatLang", limit=limit, schema="direct"),
        TextPerceptor(TripleSchema("direct"), prefer="auto"),
    )
    counts = report.failure_breakdown()
    attributions = [
        attribute(Evidence(theory_mismatch=True))
        for _ in range(counts.get("perception", 0))
    ] + [
        attribute(Evidence(rules_stalled=True))
        for _ in range(counts.get("reasoning", 0))
    ] + [
        attribute(Evidence(gold_disagrees=True))
        for _ in range(counts.get("engine", 0))
    ]
    return (
        100.0 * report.accuracy,
        f"{report.total:,} questions; the ceiling is perception, not reasoning",
        attributions,
        {"questions": report.total},
    )


# -- 4. several kinds of reasoning at once --------------------------------


@_timed("mixed-specialists", 100.0, "%")
def _run_mixed(quick: bool):
    from ..core.parser import parse_atom
    from ..workspace import Controller
    from ..workspace.scenarios import WORKSHOP_QUESTIONS, workshop
    from ..workspace.specialists import default_specialists, installed

    kb, space = workshop()
    controller = Controller(space, default_specialists(kb), budget_ms=10).warm()
    right, attributions = 0, []
    missing = [name for name, ready in installed().items() if not ready]
    for question in WORKSHOP_QUESTIONS:
        conclusion = controller.solve(parse_atom(question.goal))
        if conclusion.status == question.status:
            right += 1
            continue
        attributions.append(
            attribute(
                Evidence(
                    out_of_budget=conclusion.cause == "budget",
                    specialist_missing=(question.by if question.by in missing else None),
                    rules_stalled=conclusion.cause == "not established",
                )
            )
        )
    return (
        100.0 * right / len(WORKSHOP_QUESTIONS),
        f"{len(WORKSHOP_QUESTIONS)} questions inside a 10ms budget, one proof each",
        attributions,
        {"questions": len(WORKSHOP_QUESTIONS)},
    )


# -- 5. learning a definition by asking -----------------------------------


@_timed("growth-by-asking", 50.0, "% saved")
def _run_growth(quick: bool):
    from ..growth.loop import GrowthLoop
    from ..knowledge.base import KnowledgeBase
    from ..inference.engine import ReasoningEngine

    from ..growth.family import FACTS, TARGETS

    targets = list(TARGETS)[:2] if quick else list(TARGETS)
    totals = {"active": 0, "random": 0}
    learned_right = 0
    for target in targets:
        support, truth = TARGETS[target]
        base = FACTS + support
        oracle = ReasoningEngine(base + truth).model
        name = target.split("/")[0]
        gold = {a for a in oracle.atoms if a.predicate == name}
        seed = sorted(gold, key=str)[:1]
        for strategy in totals:
            kb = KnowledgeBase().add_rules(base)
            session = GrowthLoop(kb, max_questions=30 if quick else 80).learn(
                target, lambda a: oracle.holds(a), positive=seed, strategy=strategy
            )
            totals[strategy] += session.questions
            if strategy == "active" and session.rules:
                model = ReasoningEngine(
                    base + "\n".join(str(r) for r in session.rules)
                ).model
                if {a for a in model.atoms if a.predicate == name} == gold:
                    learned_right += 1
    share = 100.0 * totals["active"] / totals["random"] if totals["random"] else 100.0
    # Reported as questions *saved*, so that -- as everywhere else in the
    # curriculum -- higher is better and the target reads as a floor. The
    # roadmap's bar is "at most half of random's questions", which is the same
    # statement as "saves at least half".
    return (
        100.0 - share,
        f"{learned_right}/{len(targets)} learned correctly; "
        f"active {totals['active']} vs random {totals['random']} question(s)",
        [],
        {"active": totals["active"], "random": totals["random"],
         "correct": learned_right, "targets": len(targets)},
    )


# -- 6. reading an utterance ----------------------------------------------


@_timed("service-intents", 40.0, "% rejected")
def _run_intents(quick: bool):
    from ..self.context import _default_intents
    from ..self.hosts import stream

    model = _default_intents()
    if model is None:
        raise RuntimeError("the intent classifier is not trained")
    out_of_scope = stream("out_of_scope", 200 if quick else 1000)
    rejected = sum(1 for text in out_of_scope if not model.predict(text).in_scope)
    banking = stream("banking_chat", 20 if quick else 200)
    families = [model.predict(text).family for text in banking]
    banking_right = 100.0 * families.count("banking") / len(banking)
    return (
        100.0 * rejected / len(out_of_scope),
        f"{len(out_of_scope):,} out-of-scope utterances; "
        f"banking family {banking_right:.0f}%",
        [],
        {"out_of_scope": len(out_of_scope), "banking_family": round(banking_right, 1)},
    )


# -- 7. where am I? -------------------------------------------------------


@_timed("where-am-i", 100.0, "%")
def _run_where_am_i(quick: bool):
    from ..self.context import ContextDiscovery
    from ..self.hosts import HOSTS, real_data_available, stream
    from ..self.intent import numpy_available

    full = real_data_available() and numpy_available()
    right, attributions = 0, []
    for host, (_builder, expected) in HOSTS.items():
        wanted = set(expected)
        if not full and host == "banking_chat":
            wanted -= {"money", "parties", "policy"}
        discovery = ContextDiscovery()
        discovery.observe_all(stream(host))
        found = {facet.name for facet in discovery.reading.facets}
        if wanted <= found:
            right += 1
        else:
            attributions.append(
                attribute(Evidence(context_wrong=discovery.reading.domain))
            )
    return (
        100.0 * right / len(HOSTS),
        f"{len(HOSTS)} unlabelled hosts"
        + ("" if full else " (intent classifier unavailable)"),
        attributions,
        {"hosts": len(HOSTS)},
    )


# -- 8. a switch under its feet -------------------------------------------


@_timed("context-drift", 100.0, "%")
def _run_drift(quick: bool):
    from ..self.context import ContextDiscovery
    from ..self.hosts import stream

    discovery = ContextDiscovery(drift_after=2)
    discovery.observe_all(stream("grocery_feed", 4))
    noticed = 0
    for index, payload in enumerate(stream("grid_game", 8), start=1):
        discovery.observe(payload)
        if discovery.drifting:
            noticed = index
            break
    passed = 100.0 if noticed and noticed <= 5 else 0.0
    return (
        passed,
        f"noticed the host change after {noticed or '>8'} observation(s)",
        [] if passed else [attribute(Evidence(context_wrong="the previous host"))],
        {"observations_to_notice": noticed},
    )


# -- 9. nothing learned may break the core --------------------------------


@_timed("kernel-fuzz", 100.0, "% intact")
def _run_fuzz(quick: bool):
    import random

    from ..kernel import CONFIRMED, Kernel

    core = """
    mammal(X) :- cat(X).
    mammal(X) :- dog(X).
    warm_blooded(X) :- mammal(X).
    :- cat(X), dog(X).
    #open flies/1.
    """
    kernel = Kernel("school")
    kernel.load_core(core)
    kernel.layers.add_fact("cat(bob)", CONFIRMED)
    kernel.layers.add_fact("dog(rex)", CONFIRMED)
    kernel.watch(
        ["mammal(bob)", "warm_blooded(bob)", "mammal(rex)", "flies(bob)", "dog(bob)"]
    )
    core_text = kernel.layers.core.to_asp()

    rng = random.Random(7)
    total = 300 if quick else 10_000
    breaches = 0
    for index in range(total):
        kernel.learn(_junk(rng), CONFIRMED, f"fuzz{index}")
        if kernel.layers.core.to_asp() != core_text:
            breaches += 1
            break
        if kernel.canaries.check(kernel.layers, kernel.mode_layers):
            breaches += 1
            break
    return (
        0.0 if breaches else 100.0,
        f"{total:,} hostile inputs, {len(kernel.quarantine)} quarantined",
        [],
        {"inputs": total, "quarantined": len(kernel.quarantine)},
    )


def _junk(rng) -> str:
    kind = rng.randrange(8)
    word = lambda: "".join(rng.choices(string.ascii_lowercase, k=rng.randint(1, 6)))
    var = lambda: rng.choice("XYZW")
    if kind == 0:
        return "".join(rng.choices(string.printable, k=rng.randint(1, 40)))
    if kind == 1:
        return f"{word()}({var()} :- {word()}({var()})."
    if kind == 2:
        return f"{word()}({var()}, {var()}) :- {word()}({var()})."
    if kind == 3:
        a, b = word(), word()
        return f"{a}(X) :- cat(X), not {b}(X). {b}(X) :- cat(X), not {a}(X)."
    if kind == 4:
        return rng.choice(["dog(X) :- cat(X).", "-mammal(X) :- cat(X)."])
    if kind == 5:
        return f"{rng.choice(['flies', 'mammal', 'warm_blooded'])}(X) :- {word()}(X)."
    if kind == 6:
        return "n(Y) :- n(X), Y = X + 1."
    return f"{word()}({var()}) :- cat({var()})."





# -- 10. perception learned through the rules ------------------------------


@_timed("mnist-weak-supervision", 95.0, "%")
def _run_mnist(quick: bool):
    from ..datasets.mnist import load_mnist
    from ..perception.nn import ConvNet

    _train, test = load_mnist()
    limit = 2000 if quick else len(test.labels)
    network = ConvNet.from_checkpoint(_WEAK_WEIGHTS)
    accuracy = network.accuracy(test.images[:limit], test.labels[:limit])
    return (
        100.0 * accuracy,
        f"{limit:,} digits, from a model trained with no digit labels at all",
        [],
        {"digits": limit},
    )


# -- 11. carrying out an instruction --------------------------------------


@_timed("babyai", 90.0, "% solved")
def _run_babyai(quick: bool):
    """The lowest of the three required levels, not the average.

    An average lets a strong level carry a weak one, and the roadmap's bar is
    per level: GoTo *and* PickUp *and* Open. Reporting the worst is the only
    figure that means what the bar says.
    """
    from benchmarks.bench_babyai import measure

    episodes = 40 if quick else 150
    results = [
        measure(level, episodes=episodes, view="agent")
        for level in ("goto", "pickup", "open")
    ]
    worst = min(results, key=lambda r: r.rate)
    attributions = []
    for result in results:
        for reason, count in result.reasons.items():
            evidence = (
                Evidence(out_of_budget=True)
                if "budget" in reason or "out of steps" in reason
                else Evidence(rules_stalled=True)
            )
            attributions += [attribute(evidence) for _ in range(count)]

    detail = ", ".join(f"{r.level} {r.rate:.0f}%" for r in results)
    return (
        worst.rate,
        f"{episodes} episodes per level, agent's own view; {detail}",
        attributions,
        {
            "episodes": episodes,
            "levels": {r.level: r.to_dict() for r in results},
            "replans_per_episode": round(
                sum(r.replans for r in results) / len(results), 3
            ),
        },
    )


@_timed("textworld", 90.0, "% won")
def _run_textworld(quick: bool):
    """Played twice: with a correct action model, and with a broken one.

    The reported number is the *broken* one, because that is the harder
    claim. A planner that wins when it is told the rules is a planner; one
    that wins when it is told the rules wrongly, and comes out knowing which
    rule was wrong, is the milestone.
    """
    from benchmarks.bench_textworld import measure

    games = 8 if quick else 30
    given = measure(games=games, naive=False)
    naive = measure(games=games, naive=True)

    attributions = [
        attribute(Evidence(rules_stalled=True))
        for _ in range(naive.games - naive.won)
    ]
    return (
        naive.rate,
        (
            f"{games} generated games; with the model given, {given.rate:.0f}%; "
            f"with a condition missing, {naive.rate:.0f}% and "
            f"{naive.correct_lessons} game(s) ended knowing what was missing"
        ),
        attributions,
        {"games": games, "given": given.to_dict(), "naive": naive.to_dict()},
    )


# -- the curriculum --------------------------------------------------------


def _mnist_available() -> str:
    # NumPy first: the loader imports it at module level, so asking whether
    # MNIST is on disk before asking whether NumPy is installed raises rather
    # than reporting.
    missing = _needs("numpy", "numeric")()
    if missing:
        return missing

    from ..datasets.mnist import mnist_available

    if not mnist_available():
        return "MNIST is not downloaded (python scripts/download_mnist.py)"
    if not _WEAK_WEIGHTS.exists():
        return (
            "the weakly supervised checkpoint is missing "
            "(python scripts/train_weak_supervision.py)"
        )
    return ""


def _intents_available() -> str:
    from ..self.hosts import real_data_available
    from ..self.intent import numpy_available

    if not numpy_available():
        return "NumPy is not installed (pip install neuralmind[numeric])"
    if not real_data_available():
        return (
            "the intent datasets are not downloaded "
            "(python scripts/download_contexts.py)"
        )
    from pathlib import Path

    if not (Path(__file__).resolve().parents[1] / "self" / "weights" / "intents.npz").exists():
        return "the classifier is not trained (python scripts/train_intents.py)"
    return ""


#: In order. Each stage is gated on the ones named in ``after``.
CURRICULUM: tuple[Stage, ...] = (
    Stage(
        name="proofwriter-cwa",
        about="deduction on generated problems, closed-world",
        check=lambda: _corpus_available("cwa"),
        run=_run_proofwriter_cwa,
        milestone="Phase One",
    ),
    Stage(
        name="proofwriter-owa",
        about="the same theories where 'unknown' is a real answer",
        check=lambda: _corpus_available("owa"),
        run=_run_proofwriter_owa,
        after=("proofwriter-cwa",),
        milestone="P2.0",
    ),
    Stage(
        name="proofwriter-natlang",
        about="the crowdsourced register -- the perception ceiling",
        check=_natlang_available,
        run=_run_natlang,
        after=("proofwriter-cwa",),
        milestone="Phase One",
    ),
    Stage(
        name="mixed-specialists",
        about="logic, numbers, units and paths in one proof",
        check=_specialists_available,
        run=_run_mixed,
        after=("proofwriter-cwa",),
        milestone="P2.1",
    ),
    Stage(
        name="kernel-fuzz",
        about="hostile input must never reach the core",
        check=_always,
        run=_run_fuzz,
        after=("proofwriter-cwa",),
        milestone="P2.2",
    ),
    Stage(
        name="growth-by-asking",
        about="learning a definition with the fewest questions",
        check=_always,
        run=_run_growth,
        after=("proofwriter-cwa", "kernel-fuzz"),
        milestone="P2.3",
    ),
    Stage(
        name="service-intents",
        about="reading an utterance, and saying when it is none of yours",
        check=_intents_available,
        run=_run_intents,
        milestone="P2.4",
    ),
    Stage(
        name="where-am-i",
        about="nine unlabelled hosts, read from their shape",
        check=_always,
        run=_run_where_am_i,
        after=("mixed-specialists",),
        milestone="P2.4",
    ),
    Stage(
        name="context-drift",
        about="noticing the host change underneath",
        check=_always,
        run=_run_drift,
        after=("where-am-i",),
        milestone="P2.4",
    ),
    Stage(
        name="mnist-weak-supervision",
        about="digits learned through the rules, with no digit labels",
        check=_mnist_available,
        run=_run_mnist,
        milestone="Phase One",
    ),
    # Stages the roadmap asks for that cannot run from here. Listed rather than
    # omitted: a curriculum that hides what it cannot reach reports a smaller,
    # better-looking mind than the one that exists.
    Stage(
        name="babi",
        about="twenty basic reasoning skills, as a fast smoke test",
        check=_never("no reachable mirror of the bAbI tasks"),
        run=lambda quick: Result("babi", UNAVAILABLE),
        milestone="P2.13",
    ),
    Stage(
        name="entailment-bank",
        about="multi-step proof trees over science facts",
        check=_never("not downloaded; no importer written yet"),
        run=lambda quick: Result("entailment-bank", UNAVAILABLE),
        after=("proofwriter-natlang",),
        milestone="P2.6",
    ),
    Stage(
        name="babyai",
        about="English instructions carried out, with an explained plan",
        check=_needs("minigrid", "games"),
        run=_run_babyai,
        after=("where-am-i",),
        milestone="P2.7",
    ),
    Stage(
        name="textworld",
        about="a world unlike the grid, and a wrong action model corrected",
        check=_needs("textworld", "games"),
        run=_run_textworld,
        after=("babyai",),
        milestone="P2.7",
    ),
    Stage(
        name="build-a-module",
        about="a domain no pack covers, drafted from observations",
        check=_never("the module builder is P2.8 and is not built"),
        run=lambda quick: Result("build-a-module", UNAVAILABLE),
        after=("where-am-i",),
        milestone="P2.8",
    ),
)
