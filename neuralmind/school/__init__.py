"""The school: one command that reproduces every number this project claims.

The roadmap has this "running throughout", and it was the item most overdue.
By P2.4 the measurements lived in five different places -- a fuzz harness, a
growth benchmark, two eval invocations, an intent training script -- and a set
of numbers you have to reassemble by hand is a set of numbers that quietly
stops being true.

So: an ordered curriculum, each stage gated on the ones before it, every stage
carrying the target it has to beat, and a dashboard file that makes a
regression fail **by name** rather than by someone noticing a table has
changed.

Three things it does that a plain test suite does not.

**It gates.** There is no information in a NatLang score from a mind that
cannot do generated ProofWriter, so that stage does not run until the earlier
one passes. A blocked stage says which stage blocked it.

**It says what it cannot do.** A stage that cannot run reports *why* -- a
dataset not downloaded, a dependency missing, an environment not reachable
from this machine -- rather than being absent. Five of the roadmap's stages
are in that state today and all five are listed, because a curriculum that hides
what it cannot reach reports a smaller, better-looking mind than the one that
exists.

**Every failure has exactly one attribution.** Seven categories in priority
order; see :mod:`neuralmind.school.attribution` for why the order is the
design and not the taxonomy.

    neuralmind school            # the quick pass, about a minute
    neuralmind school --full     # every question and 10,000 fuzz inputs
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from .attribution import (
    CATEGORIES,
    REMEDY,
    UNATTRIBUTED,
    Attribution,
    Evidence,
    attribute,
    breakdown,
)
from .stages import (
    BLOCKED,
    CURRICULUM,
    FAILED,
    PASSED,
    UNAVAILABLE,
    Result,
    Stage,
)

__all__ = [
    "School",
    "Report",
    "CURRICULUM",
    "Stage",
    "Result",
    "Attribution",
    "Evidence",
    "attribute",
    "breakdown",
    "CATEGORIES",
    "REMEDY",
    "UNATTRIBUTED",
    "PASSED",
    "FAILED",
    "UNAVAILABLE",
    "BLOCKED",
    "DASHBOARD",
]

#: Where the dashboard is written. Tracked over time, so a stage that was
#: passing and is not any more is visible without reading the diff.
DASHBOARD = Path("reports") / "school.json"


@dataclass
class Report:
    """What a whole run of the curriculum found."""

    results: list[Result] = field(default_factory=list)
    quick: bool = True
    seconds: float = 0.0
    at: float = field(default_factory=time.time)

    # -- counting -----------------------------------------------------------

    @property
    def ran(self) -> list[Result]:
        return [r for r in self.results if r.ran]

    @property
    def passed(self) -> list[Result]:
        return [r for r in self.results if r.status == PASSED]

    @property
    def failed(self) -> list[Result]:
        return [r for r in self.results if r.status == FAILED]

    @property
    def unavailable(self) -> list[Result]:
        return [r for r in self.results if r.status == UNAVAILABLE]

    @property
    def blocked(self) -> list[Result]:
        return [r for r in self.results if r.status == BLOCKED]

    @property
    def healthy(self) -> bool:
        """True when every stage that *could* run did pass.

        Deliberately not "every stage passed": a stage that cannot run here is
        not a failure of the mind, and conflating the two would make the
        dashboard red for reasons nobody can act on.
        """
        return not self.failed

    @property
    def attributions(self) -> list[Attribution]:
        return [a for result in self.results for a in result.attributions]

    # -- reporting ----------------------------------------------------------

    def describe(self) -> str:
        lines = [
            f"curriculum: {len(self.passed)} passed, {len(self.failed)} failed, "
            f"{len(self.unavailable)} unavailable, {len(self.blocked)} blocked "
            f"({'quick' if self.quick else 'full'}, {self.seconds:.0f}s)",
            "",
        ]
        lines.extend(result.describe() for result in self.results)
        counts = breakdown(self.attributions)
        if counts:
            lines.append("")
            lines.append("failures by cause, most upstream first:")
            for category in list(CATEGORIES) + [UNATTRIBUTED]:
                if category in counts:
                    lines.append(
                        f"  {category:20} {counts[category]:6}   {REMEDY[category]}"
                    )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "at": self.at,
            "quick": self.quick,
            "seconds": round(self.seconds, 2),
            "healthy": self.healthy,
            "python": platform.python_version(),
            "counts": {
                "passed": len(self.passed),
                "failed": len(self.failed),
                "unavailable": len(self.unavailable),
                "blocked": len(self.blocked),
            },
            "failures_by_cause": breakdown(self.attributions),
            "stages": [result.to_dict() for result in self.results],
        }

    def __str__(self) -> str:
        return self.describe()


class School:
    """Runs the curriculum in order, gating each stage on the ones before it."""

    def __init__(self, curriculum: Sequence[Stage] = CURRICULUM) -> None:
        self.curriculum = list(curriculum)

    def stages(self, only: Optional[Iterable[str]] = None) -> list[Stage]:
        if only is None:
            return list(self.curriculum)
        wanted = set(only)
        unknown = wanted - {stage.name for stage in self.curriculum}
        if unknown:
            raise KeyError(
                f"unknown stage(s) {', '.join(sorted(unknown))}; choose from "
                + ", ".join(stage.name for stage in self.curriculum)
            )
        return [stage for stage in self.curriculum if stage.name in wanted]

    def run(
        self,
        quick: bool = True,
        only: Optional[Iterable[str]] = None,
        on_stage: Optional[Callable[[Stage], None]] = None,
    ) -> Report:
        """Work through the curriculum. Returns the whole report either way."""
        started = time.perf_counter()
        report = Report(quick=quick)
        outcomes: dict[str, str] = {}

        for stage in self.stages(only):
            if on_stage is not None:
                on_stage(stage)

            # Gating comes before availability: a stage whose prerequisite
            # failed tells you nothing, so there is no point finding out
            # whether it could have run.
            unmet = [
                name
                for name in stage.after
                if outcomes.get(name, PASSED) not in (PASSED, UNAVAILABLE)
            ]
            if unmet:
                result = Result(
                    stage=stage.name,
                    status=BLOCKED,
                    detail=f"{', '.join(unmet)} did not pass",
                )
            else:
                why_not = stage.why_not()
                if why_not:
                    result = Result(
                        stage=stage.name, status=UNAVAILABLE, detail=why_not
                    )
                elif stage.slow and quick:
                    result = Result(
                        stage=stage.name,
                        status=UNAVAILABLE,
                        detail="skipped in the quick pass; run with --full",
                    )
                else:
                    result = stage.run(quick)
            outcomes[stage.name] = result.status
            report.results.append(result)

        report.seconds = time.perf_counter() - started
        return report

    # -- the dashboard ------------------------------------------------------

    @staticmethod
    def write(report: Report, path: Path = DASHBOARD, keep: int = 20) -> Path:
        """Append this run to the dashboard, keeping the recent history.

        History is what makes a regression visible: one run says what the
        numbers are, a series says which of them moved.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        history = []
        if path.exists():
            try:
                history = json.loads(path.read_text(encoding="utf-8")).get("runs", [])
            except (ValueError, OSError):
                history = []
        history.append(report.to_dict())
        payload = {
            "latest": history[-1],
            "runs": history[-keep:],
            "regressions": _regressions(history),
        }
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


def _regressions(history: Sequence[dict]) -> list[dict]:
    """Stages that used to pass and now do not, named.

    The whole reason the dashboard keeps history. A stage going from 100% to
    97% is a number nobody notices in a table and a line item here.
    """
    if len(history) < 2:
        return []
    previous = {s["stage"]: s for s in history[-2]["stages"]}
    found = []
    for stage in history[-1]["stages"]:
        was = previous.get(stage["stage"])
        if was is None:
            continue
        if was.get("status") == PASSED and stage.get("status") == FAILED:
            found.append(
                {
                    "stage": stage["stage"],
                    "was": was.get("metric"),
                    "now": stage.get("metric"),
                    "detail": stage.get("detail", ""),
                }
            )
        elif (
            was.get("metric") is not None
            and stage.get("metric") is not None
            and stage["metric"] < was["metric"] - 0.5
        ):
            found.append(
                {
                    "stage": stage["stage"],
                    "was": was["metric"],
                    "now": stage["metric"],
                    "detail": "still passing, but lower than last time",
                }
            )
    return found
