"""Reading an utterance well enough to say what *kind* of place this is.

A single sentence carries almost no structure, so the signature reader gets
little from it -- "free text, asks a question, first person" and not much else.
That is enough to see a conversation and not enough to see what the
conversation is about, which is most of what a context is.

So this classifies. Not generates: it scores an utterance against intents it
has seen and returns the closest, or says none is close enough. That last part
is the point and is why CLINC150 is here -- a service assistant meets inputs
belonging to no intent it knows all day long, and "not something I handle" is
a better answer than a confident wrong one.

The model is a TF-IDF nearest-centroid classifier. Small, exactly inspectable,
and trained in a second on a laptop:

* each intent becomes the mean of its examples' TF-IDF vectors;
* an utterance is assigned to the nearest centroid by cosine similarity;
* below a threshold it is **out of scope**, and the threshold is chosen on
  held-out in-scope and out-of-scope data rather than picked.

Why not embeddings: the roadmap wants multilingual-e5, and that is the right
answer for the multilingual half of the problem. This runs without a download,
without a GPU, and without a dependency the core does not already have -- and
it makes the out-of-scope decision *legible*, since a cosine against a
centroid of counted words is something you can take apart by hand.

Optional, like everything neural here. Without NumPy the classifier reports
that it is unavailable and the mind falls back to what shape alone can see.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence

__all__ = ["IntentClassifier", "Prediction", "numpy_available", "OUT_OF_SCOPE"]

#: What the classifier says when nothing is close enough.
OUT_OF_SCOPE = "out_of_scope"

_TOKEN = re.compile(r"[a-z0-9']+")


def numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass(frozen=True)
class Prediction:
    """One utterance's nearest intent, and how near it was."""

    intent: str
    #: Cosine similarity to that intent's centroid, 0..1.
    score: float
    #: Which training set the intent came from: ``banking``, ``assistant``.
    family: str = ""
    #: The runner-up, which is what makes a close call visible.
    second: str = ""
    second_score: float = 0.0

    @property
    def in_scope(self) -> bool:
        return self.intent != OUT_OF_SCOPE

    @property
    def margin(self) -> float:
        """How much better the winner was. A small margin is a coin toss."""
        return self.score - self.second_score

    def describe(self) -> str:
        if not self.in_scope:
            return f"not something I handle (closest was {self.second} at {self.second_score:.2f})"
        return f"{self.intent} ({self.score:.2f}, margin {self.margin:.2f})"

    def __str__(self) -> str:
        return self.describe()


class IntentClassifier:
    """TF-IDF centroids per intent, with an out-of-scope threshold."""

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.vocabulary: dict[str, int] = {}
        self.idf: list[float] = []
        self.intents: list[str] = []
        self.families: list[str] = []
        self._centroids = None

    # -- training -----------------------------------------------------------

    def fit(
        self,
        examples: Sequence[tuple[str, str]],
        families: Optional[dict[str, str]] = None,
        min_df: int = 2,
    ) -> "IntentClassifier":
        """Learn centroids from ``(utterance, intent)`` pairs."""
        if not numpy_available():
            raise RuntimeError("the intent classifier needs NumPy")
        import numpy as np

        families = families or {}
        documents = [_tokens(text) for text, _ in examples]
        counts: dict[str, int] = {}
        for tokens in documents:
            for token in set(tokens):
                counts[token] = counts.get(token, 0) + 1
        self.vocabulary = {
            token: index
            for index, token in enumerate(
                sorted(t for t, n in counts.items() if n >= min_df)
            )
        }
        total = len(documents)
        self.idf = [
            math.log((1 + total) / (1 + counts[token])) + 1.0
            for token in sorted(self.vocabulary)
        ]

        self.intents = sorted({intent for _, intent in examples})
        self.families = [families.get(intent, "") for intent in self.intents]
        index_of = {intent: i for i, intent in enumerate(self.intents)}

        centroids = np.zeros((len(self.intents), len(self.vocabulary)), dtype="float32")
        seen = np.zeros(len(self.intents), dtype="float32")
        for tokens, (_text, intent) in zip(documents, examples):
            vector = self._vector(tokens, np)
            centroids[index_of[intent]] += vector
            seen[index_of[intent]] += 1
        centroids /= np.maximum(seen, 1.0)[:, None]
        norms = np.linalg.norm(centroids, axis=1, keepdims=True)
        self._centroids = centroids / np.maximum(norms, 1e-9)
        return self

    def calibrate(
        self,
        in_scope: Sequence[str],
        out_of_scope: Sequence[str],
        target_recall: float = 0.9,
    ) -> float:
        """Pick the out-of-scope threshold from data rather than by taste.

        Chosen to keep ``target_recall`` of genuinely in-scope utterances while
        rejecting as many out-of-scope ones as that allows. Held-out, and both
        halves matter: a threshold tuned only on in-scope data rejects nothing,
        and one tuned only on out-of-scope data rejects everything.
        """
        if self._centroids is None:
            raise RuntimeError("fit the classifier before calibrating it")
        inside = sorted(self._best(text)[1] for text in in_scope)
        if not inside:
            return self.threshold
        cut = inside[int(len(inside) * (1.0 - target_recall))]
        self.threshold = float(cut)
        return self.threshold

    # -- using --------------------------------------------------------------

    def predict(self, text: str) -> Prediction:
        if self._centroids is None:
            raise RuntimeError("the classifier has not been trained")
        (best, score), (second, second_score) = self._top_two(text)
        family = self.families[self.intents.index(best)] if best in self.intents else ""
        if score < self.threshold:
            return Prediction(
                intent=OUT_OF_SCOPE,
                score=float(score),
                second=best,
                second_score=float(score),
            )
        return Prediction(
            intent=best,
            score=float(score),
            family=family,
            second=second,
            second_score=float(second_score),
        )

    def _vector(self, tokens: Sequence[str], np):
        vector = np.zeros(len(self.vocabulary), dtype="float32")
        for token in tokens:
            index = self.vocabulary.get(token)
            if index is not None:
                vector[index] += 1.0
        if vector.any():
            vector *= np.asarray(self.idf, dtype="float32")
            vector /= max(float(np.linalg.norm(vector)), 1e-9)
        return vector

    def _best(self, text: str) -> tuple[str, float]:
        return self._top_two(text)[0]

    def _top_two(self, text: str):
        import numpy as np

        vector = self._vector(_tokens(text), np)
        scores = self._centroids @ vector
        if scores.size == 0:
            return ("", 0.0), ("", 0.0)
        order = np.argsort(-scores)
        first = (self.intents[int(order[0])], float(scores[int(order[0])]))
        if order.size < 2:
            return first, ("", 0.0)
        second = (self.intents[int(order[1])], float(scores[int(order[1])]))
        return first, second

    # -- persistence --------------------------------------------------------

    def save(self, path) -> None:
        import numpy as np

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, centroids=self._centroids)
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "threshold": self.threshold,
                    "vocabulary": self.vocabulary,
                    "idf": self.idf,
                    "intents": self.intents,
                    "families": self.families,
                },
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "IntentClassifier":
        import numpy as np

        path = Path(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        model = cls(threshold=meta["threshold"])
        model.vocabulary = meta["vocabulary"]
        model.idf = meta["idf"]
        model.intents = meta["intents"]
        model.families = meta["families"]
        model._centroids = np.load(path)["centroids"]
        return model

    def summary(self) -> dict:
        return {
            "intents": len(self.intents),
            "vocabulary": len(self.vocabulary),
            "threshold": round(self.threshold, 4),
            "families": sorted({f for f in self.families if f}),
        }

    def __repr__(self) -> str:
        return f"IntentClassifier({len(self.intents)} intent(s))"


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(str(text).lower())
