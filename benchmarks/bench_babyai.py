"""BabyAI: instructions in English, carried out with an explained plan.

The roadmap's bar for P2.7 is GoTo, PickUp and Open at 90% or better, each
with a plan you can read. This measures exactly that.

Run with the agent's own 7x7 view by default, not the whole grid. The
difference matters: with the full grid the planner solves a problem the
benchmark does not pose, and the replanning never runs because nothing is
ever a surprise. ``--view full`` is there so the gap can be measured rather
than argued about.

    python benchmarks/bench_babyai.py             # the three required families
    python benchmarks/bench_babyai.py --episodes 200 --view full
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["LEVELS", "Measurement", "measure", "measure_all"]

#: The levels the roadmap names, plus the mixed one that needs both readings.
LEVELS = {
    "goto": ("GoToLocal", "go to an object named by colour and type"),
    "pickup": ("PickupLoc", "pick one up, sometimes named by where it was"),
    "open": ("OpenDoor", "open a door, sometimes named by where it is"),
    "goto-obj-door": ("GoToObjDoor", "objects and doors in the same instruction"),
}


@dataclass
class Measurement:
    """What one level came to."""

    level: str
    episodes: int = 0
    solved: int = 0
    steps: float = 0.0
    replans: float = 0.0
    explorations: float = 0.0
    seconds: float = 0.0
    unread: int = 0
    reasons: dict = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return 100.0 * self.solved / self.episodes if self.episodes else 0.0

    def line(self) -> str:
        return (
            f"{self.level:14} {self.solved:4}/{self.episodes:<4} "
            f"{self.rate:5.1f}%   {self.steps:5.1f} steps  "
            f"{self.replans:4.2f} replans  {self.explorations:4.2f} explore  "
            f"{self.seconds * 1000:5.0f}ms/ep"
        )

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "episodes": self.episodes,
            "solved": self.solved,
            "rate": round(self.rate, 2),
            "steps": round(self.steps, 2),
            "replans": round(self.replans, 3),
            "explorations": round(self.explorations, 3),
            "seconds": round(self.seconds, 4),
            "unread_instructions": self.unread,
            "reasons": dict(self.reasons),
        }


def measure(
    level: str = "goto",
    episodes: int = 100,
    view: str = "agent",
    max_steps: int = 120,
    budget_ms: float = 2_000.0,
    start_seed: int = 0,
) -> Measurement:
    """Run one level and report what happened, failures included."""
    from minigrid.envs import babyai

    from neuralmind.envs import run_episode

    name = LEVELS[level][0]
    warnings.filterwarnings("ignore")
    env = getattr(babyai, name)()

    result = Measurement(level=level)
    started = time.perf_counter()
    for seed in range(start_seed, start_seed + episodes):
        episode = run_episode(
            env, seed=seed, view=view, max_steps=max_steps, budget_ms=budget_ms
        )
        result.episodes += 1
        result.solved += int(episode.solved)
        result.steps += episode.steps
        result.replans += episode.replans
        result.explorations += episode.explorations
        if episode.mission is None:
            result.unread += 1
        if not episode.solved:
            key = episode.reason.split(":")[0][:60] or "unknown"
            result.reasons[key] = result.reasons.get(key, 0) + 1

    for field_name in ("steps", "replans", "explorations"):
        setattr(result, field_name, getattr(result, field_name) / max(episodes, 1))
    result.seconds = (time.perf_counter() - started) / max(episodes, 1)
    return result


def measure_all(
    levels: Sequence[str] = ("goto", "pickup", "open"),
    episodes: int = 100,
    view: str = "agent",
    **kwargs,
) -> list[Measurement]:
    return [measure(level, episodes, view, **kwargs) for level in levels]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--view", choices=("agent", "full"), default="agent")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument(
        "--level", action="append", default=[], choices=sorted(LEVELS)
    )
    args = parser.parse_args(argv)

    levels = args.level or ["goto", "pickup", "open"]
    print(f"BabyAI, {args.view} view, {args.episodes} episode(s) per level")
    print("-" * 92)
    results = []
    for level in levels:
        result = measure(
            level, args.episodes, args.view, max_steps=args.max_steps
        )
        results.append(result)
        print(result.line())
        for reason, count in sorted(result.reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:4}  {reason}")

    worst = min(r.rate for r in results)
    print("-" * 92)
    print(f"lowest level: {worst:.1f}%  (the bar is 90%)")
    return 0 if worst >= 90.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
