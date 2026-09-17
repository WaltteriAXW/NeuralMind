"""What a rule has to survive before it is allowed into a layer.

Four checks, in increasing order of cost, so the cheap ones reject first:

1. **Safety.** Every variable in the head appears in a positive body literal.
   An unsafe rule has an infinite grounding; the engine already refuses them,
   and catching it here means the refusal happens before anything was changed.
2. **Stratification.** Negation must not be recursive. The engine falls back to
   atom-level local stratification, so the check has to allow the same
   programs, not a smaller set -- a firewall that rejects what the engine can
   run is a bug, and one this project has shipped once already.
3. **Consistency with the core.** The rule is solved together with the core and
   the layers below it. If it violates an integrity constraint the core states,
   or derives an atom alongside its own strong negation, it contradicts
   something the mind was built knowing, and that is not a thing a learned rule
   gets to do.
4. **A dry run under a hard budget.** Solve it and see. A rule can be safe,
   stratified and consistent and still not terminate in any useful time --
   runaway recursion over a large universe is the usual way -- and the only
   reliable way to find that out is to try it with a limit.

The dry run happens against a *copy*, never the live stack. That is what makes
this a firewall rather than a test: nothing being checked can affect anything
while it is being checked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..core.parser import ParseError, parse_program
from ..core.program import Program, ProgramError, SafetyError, StratificationError
from ..knowledge.base import KnowledgeBase
from .layers import CORE, DEFAULT_LAYERS, LayerStack

__all__ = ["Firewall", "Verdict", "REJECTED", "ADMITTED"]

ADMITTED = "admitted"
REJECTED = "rejected"


@dataclass
class Verdict:
    """Whether something may enter, and what it failed on if not."""

    outcome: str
    source: str
    #: Which check rejected it: ``parse``, ``safety``, ``stratification``,
    #: ``consistency``, ``budget``. Empty when admitted.
    check: str = ""
    reason: str = ""
    #: How long the dry run took, in milliseconds.
    elapsed_ms: float = 0.0
    #: The model the dry run produced, when it got that far. It is the model
    #: of exactly the program that will exist if this rule is admitted, so a
    #: caller about to check canaries can use it instead of solving again.
    model: object = None

    def __bool__(self) -> bool:
        return self.outcome == ADMITTED

    def describe(self) -> str:
        if self:
            return f"admitted ({self.elapsed_ms:.1f}ms dry run)"
        return f"rejected by the {self.check} check: {self.reason}"

    def to_dict(self) -> dict:
        payload = {"outcome": self.outcome, "source": self.source}
        if not self:
            payload["check"] = self.check
            payload["reason"] = self.reason
        payload["elapsed_ms"] = round(self.elapsed_ms, 2)
        return payload  # the model is deliberately not serialised

    def __str__(self) -> str:
        return self.describe()


class Firewall:
    """Admission control for rules, checked against a copy of the stack."""

    def __init__(
        self,
        stack: LayerStack,
        max_atoms: int = 50_000,
        max_iterations: int = 200,
        layers: Sequence[str] = DEFAULT_LAYERS,
    ) -> None:
        self.stack = stack
        #: Hard limits for the dry run. A rule that needs more than this is
        #: not necessarily wrong, but it is not something to admit unexamined.
        self.max_atoms = max_atoms
        self.max_iterations = max_iterations
        self.layers = tuple(layers)
        self.log: list[Verdict] = []

    def check(self, source: str, name: str = "candidate") -> Verdict:
        """Run every check against a copy. Nothing live is touched."""
        import time

        started = time.perf_counter()

        try:
            candidate = parse_program(source, name, check=False)
        except (ParseError, SyntaxError) as exc:
            return self._record(Verdict(REJECTED, source, "parse", _brief(exc)))

        # 1 and 2: safety and stratification, over the rule together with what
        # it will sit on top of -- a rule is only stratified relative to the
        # program it joins.
        combined = self.stack.view(self.layers)
        merged = combined.rules.merge(candidate)
        for record in combined.facts:
            merged.add(_fact_rule(record))
        try:
            merged.check()
        except SafetyError as exc:
            return self._record(Verdict(REJECTED, source, "safety", _brief(exc)))
        except StratificationError as exc:
            return self._record(Verdict(REJECTED, source, "stratification", _brief(exc)))
        except ProgramError as exc:
            return self._record(Verdict(REJECTED, source, "safety", _brief(exc)))
        except Exception as exc:
            # A firewall that can be made to raise is not one. Whatever went
            # wrong while checking, the rule does not get in, and the reason
            # is recorded rather than reaching the caller as a crash.
            return self._record(
                Verdict(REJECTED, source, "safety", f"could not be checked: {_brief(exc)}")
            )

        # 3 and 4: solve it, under a limit, and look at what came out.
        from ..inference.forward import ForwardChainer, ReasoningLimit

        try:
            model = ForwardChainer(
                merged, max_atoms=self.max_atoms, max_iterations=self.max_iterations
            ).run()
        except ReasoningLimit as exc:
            return self._record(
                Verdict(REJECTED, source, "budget", f"did not settle: {_brief(exc)}")
            )
        except Exception as exc:
            return self._record(Verdict(REJECTED, source, "budget", _brief(exc)))

        elapsed = (time.perf_counter() - started) * 1000.0
        if model.truncated:
            return self._record(
                Verdict(
                    REJECTED, source, "budget",
                    f"stopped early: {model.truncated}", elapsed,
                )
            )
        if model.contradictions:
            clash = ", ".join(str(a) for a in model.contradictions[:3])
            return self._record(
                Verdict(
                    REJECTED, source, "consistency",
                    f"derives both polarities of {clash}", elapsed,
                )
            )
        if model.violations:
            broken = "; ".join(v.describe() for v in model.violations[:3])
            return self._record(
                Verdict(
                    REJECTED, source, "consistency",
                    f"breaks a rule the core states: {broken}", elapsed,
                )
            )
        return self._record(
            Verdict(ADMITTED, source, elapsed_ms=elapsed, model=model)
        )

    def admit(self, source: str, layer: str, name: str = "candidate") -> Verdict:
        """Check, and add to ``layer`` only if it passed."""
        verdict = self.check(source, name)
        if verdict:
            self.stack.add_rules(source, layer, name)
        return verdict

    def _record(self, verdict: Verdict) -> Verdict:
        self.log.append(verdict)
        return verdict

    def to_dict(self) -> dict:
        rejected = [v for v in self.log if not v]
        return {
            "checked": len(self.log),
            "rejected": len(rejected),
            "by_check": {
                check: sum(1 for v in rejected if v.check == check)
                for check in sorted({v.check for v in rejected})
            },
        }


def _fact_rule(record):
    from ..core.program import Rule

    return Rule(record.atom, (), source=record.provenance, label="given")


def _brief(exc: Exception) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__
