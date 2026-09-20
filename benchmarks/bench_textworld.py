"""TextWorld: the same planner, a world that is nothing like a grid.

Two runs of the same games. The first gives the mind a correct action model
and asks whether it can play. The second removes one condition -- that a
container has to be open before you can take what is inside it -- and asks
whether it can work that out from being refused.

The second is the interesting number. Winning with a correct model shows the
planner works; winning with a wrong one, and coming out of the game knowing
what was wrong with it, is the thing P2.7 is actually for.

    python benchmarks/bench_textworld.py --games 30
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

__all__ = ["Measurement", "measure"]


@dataclass
class Measurement:
    model: str = "given"
    games: int = 0
    won: int = 0
    commands: float = 0.0
    rejected: float = 0.0
    experiments: float = 0.0
    learned: int = 0
    correct_lessons: int = 0
    seconds: float = 0.0
    reasons: dict = field(default_factory=dict)

    @property
    def rate(self) -> float:
        return 100.0 * self.won / self.games if self.games else 0.0

    def line(self) -> str:
        return (
            f"{self.model:8} {self.won:3}/{self.games:<3} {self.rate:5.1f}%   "
            f"{self.commands:4.1f} commands  {self.rejected:4.2f} rejected  "
            f"{self.experiments:4.2f} experiments  "
            f"{self.learned:3} lesson(s), {self.correct_lessons} correct"
        )

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "games": self.games,
            "won": self.won,
            "rate": round(self.rate, 2),
            "commands": round(self.commands, 2),
            "rejected": round(self.rejected, 3),
            "experiments": round(self.experiments, 3),
            "lessons": self.learned,
            "correct_lessons": self.correct_lessons,
            "reasons": dict(self.reasons),
        }


#: The condition removed from the naive model, and so the lesson we hope for.
EXPECTED_LESSON = "opened(C)"


def measure(
    games: int = 30,
    naive: bool = False,
    start_seed: int = 1,
    rooms: int = 3,
    objects: int = 4,
    quest_length: int = 3,
    max_steps: int = 40,
) -> Measurement:
    from neuralmind.envs import run_text_episode

    warnings.filterwarnings("ignore")
    result = Measurement(model="naive" if naive else "given")
    started = time.perf_counter()

    for seed in range(start_seed, start_seed + games):
        episode = run_text_episode(
            seed=seed,
            rooms=rooms,
            objects=objects,
            quest_length=quest_length,
            naive=naive,
            max_steps=max_steps,
        )
        result.games += 1
        result.won += int(episode.won)
        result.commands += episode.steps
        result.rejected += episode.rejected
        result.experiments += episode.experiments
        result.learned += len(episode.learned)
        result.correct_lessons += sum(
            1 for lesson in episode.learned if EXPECTED_LESSON in lesson
        )
        if not episode.won:
            key = episode.reason.split(";")[0][:60] or "unknown"
            result.reasons[key] = result.reasons.get(key, 0) + 1

    for name in ("commands", "rejected", "experiments"):
        setattr(result, name, getattr(result, name) / max(games, 1))
    result.seconds = (time.perf_counter() - started) / max(games, 1)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=30)
    parser.add_argument("--rooms", type=int, default=3)
    parser.add_argument("--objects", type=int, default=4)
    parser.add_argument("--quest-length", type=int, default=3)
    args = parser.parse_args(argv)

    print(
        f"TextWorld, {args.games} generated game(s), "
        f"{args.rooms} rooms / {args.objects} objects / quest {args.quest_length}"
    )
    print("-" * 104)
    results = []
    for naive in (False, True):
        result = measure(
            args.games, naive=naive, rooms=args.rooms,
            objects=args.objects, quest_length=args.quest_length,
        )
        results.append(result)
        print(result.line())
        for reason, count in sorted(result.reasons.items(), key=lambda kv: -kv[1]):
            print(f"    {count:4}  {reason}")
    print("-" * 104)
    naive_result = results[1]
    print(
        f"with a condition missing: {naive_result.rate:.0f}% won, "
        f"{naive_result.correct_lessons} game(s) ended knowing what was missing"
    )
    return 0 if all(r.rate >= 90.0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
