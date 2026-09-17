"""What the mind remembers between runs, on SQLite from the standard library.

A mind that learns and then forgets on restart has not learned anything. This
is the store: facts, rules, their belief state, where each came from, and a
version history so a change can be undone after the session that made it has
ended.

SQLite because it ships with Python. The roadmap is explicit that the core
stays dependency-free, and a persistence layer that needs a database server is
the kind of thing that quietly makes a library un-embeddable.

**Belief state is the point.** Everything stored carries one of three states:

``proposed``
    Learned, not yet approved. Stored, visible, and *not* used to answer --
    design rule 5. Nothing here can change an answer.
``confirmed``
    Approved. This is what the mind actually reasons with.
``retired``
    Was confirmed, since withdrawn. Kept rather than deleted, because "we used
    to believe this and stopped" is information, and because an answer given
    last month has to remain explicable.

Versions are cheap and append-only: every change writes a row saying what
happened, so :meth:`undo` is a matter of reading the history rather than
keeping a parallel copy of the world.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence, Union

__all__ = ["Memory", "Belief", "PROPOSED", "CONFIRMED", "RETIRED", "STATES"]

PROPOSED = "proposed"
CONFIRMED = "confirmed"
RETIRED = "retired"
STATES = (PROPOSED, CONFIRMED, RETIRED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    id         INTEGER PRIMARY KEY,
    kind       TEXT NOT NULL,          -- 'fact' or 'rule'
    body       TEXT NOT NULL,          -- the ASP text
    state      TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    evidence   TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE (kind, body)
);
CREATE TABLE IF NOT EXISTS history (
    id        INTEGER PRIMARY KEY,
    belief_id INTEGER NOT NULL,
    was       TEXT,
    became    TEXT NOT NULL,
    reason    TEXT NOT NULL DEFAULT '',
    at        REAL NOT NULL,
    FOREIGN KEY (belief_id) REFERENCES beliefs (id)
);
CREATE INDEX IF NOT EXISTS beliefs_state ON beliefs (state);
"""


@dataclass(frozen=True)
class Belief:
    """One remembered thing and what the mind currently makes of it."""

    id: int
    kind: str
    body: str
    state: str
    provenance: str = ""
    confidence: float = 1.0
    evidence: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def usable(self) -> bool:
        """Whether this may take part in answering. Only confirmed things may."""
        return self.state == CONFIRMED

    def describe(self) -> str:
        mark = {PROPOSED: "?", CONFIRMED: " ", RETIRED: "-"}[self.state]
        return f"{mark} {self.body}" + (f"   [{self.provenance}]" if self.provenance else "")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "body": self.body,
            "state": self.state,
            "provenance": self.provenance,
            "confidence": self.confidence,
        }

    def __str__(self) -> str:
        return self.describe()


class Memory:
    """A persistent store of beliefs, with a history and an undo."""

    def __init__(self, path: Union[str, Path] = ":memory:") -> None:
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(_SCHEMA)
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "Memory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- writing ------------------------------------------------------------

    def remember(
        self,
        body: str,
        kind: str = "rule",
        state: str = PROPOSED,
        provenance: str = "",
        confidence: float = 1.0,
        evidence: str = "",
        reason: str = "",
    ) -> Belief:
        """Store something. Re-remembering keeps the existing state.

        Deliberately: re-learning a rule that a person already retired must not
        quietly resurrect it, which is what "latest write wins" would do.
        """
        if state not in STATES:
            raise ValueError(f"state must be one of {', '.join(STATES)}")
        now = time.time()
        existing = self.find(body, kind)
        if existing is not None:
            return existing
        cursor = self.connection.execute(
            "INSERT INTO beliefs (kind, body, state, provenance, confidence, "
            "evidence, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (kind, body, state, provenance, confidence, evidence, now, now),
        )
        self._record(cursor.lastrowid, None, state, reason or "remembered")
        self.connection.commit()
        return self.get(cursor.lastrowid)

    def set_state(self, belief: Union[Belief, int], state: str, reason: str = "") -> Belief:
        """Move a belief between states, recording what happened."""
        if state not in STATES:
            raise ValueError(f"state must be one of {', '.join(STATES)}")
        belief_id = belief.id if isinstance(belief, Belief) else int(belief)
        current = self.get(belief_id)
        if current is None:
            raise KeyError(f"no belief with id {belief_id}")
        if current.state == state:
            return current
        self.connection.execute(
            "UPDATE beliefs SET state = ?, updated_at = ? WHERE id = ?",
            (state, time.time(), belief_id),
        )
        self._record(belief_id, current.state, state, reason)
        self.connection.commit()
        return self.get(belief_id)

    def confirm(self, belief, reason: str = "approved") -> Belief:
        return self.set_state(belief, CONFIRMED, reason)

    def retire(self, belief, reason: str = "withdrawn") -> Belief:
        return self.set_state(belief, RETIRED, reason)

    def _record(self, belief_id: int, was: Optional[str], became: str, reason: str) -> None:
        self.connection.execute(
            "INSERT INTO history (belief_id, was, became, reason, at) VALUES (?, ?, ?, ?, ?)",
            (belief_id, was, became, reason, time.time()),
        )

    # -- reading ------------------------------------------------------------

    def get(self, belief_id: int) -> Optional[Belief]:
        row = self.connection.execute(
            "SELECT * FROM beliefs WHERE id = ?", (belief_id,)
        ).fetchone()
        return _belief(row) if row else None

    def find(self, body: str, kind: str = "rule") -> Optional[Belief]:
        row = self.connection.execute(
            "SELECT * FROM beliefs WHERE kind = ? AND body = ?", (kind, body)
        ).fetchone()
        return _belief(row) if row else None

    def beliefs(
        self, state: Optional[str] = None, kind: Optional[str] = None
    ) -> list[Belief]:
        clauses, values = [], []
        if state is not None:
            clauses.append("state = ?")
            values.append(state)
        if kind is not None:
            clauses.append("kind = ?")
            values.append(kind)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.connection.execute(
            f"SELECT * FROM beliefs{where} ORDER BY id", values
        ).fetchall()
        return [_belief(row) for row in rows]

    @property
    def proposed(self) -> list[Belief]:
        return self.beliefs(PROPOSED)

    @property
    def confirmed(self) -> list[Belief]:
        return self.beliefs(CONFIRMED)

    def to_asp(self, state: str = CONFIRMED) -> str:
        """The confirmed knowledge as a program.

        Only confirmed by default, and that default is the design rule: a
        proposal that could answer a question is not a proposal.
        """
        return "\n".join(b.body for b in self.beliefs(state))

    # -- history ------------------------------------------------------------

    def history(self, belief_id: Optional[int] = None, limit: int = 50) -> list[dict]:
        where = " WHERE belief_id = ?" if belief_id is not None else ""
        values = (belief_id,) if belief_id is not None else ()
        rows = self.connection.execute(
            f"SELECT h.*, b.body FROM history h JOIN beliefs b ON b.id = h.belief_id"
            f"{where} ORDER BY h.id DESC LIMIT ?",
            (*values, limit),
        ).fetchall()
        return [
            {
                "belief": row["body"],
                "was": row["was"],
                "became": row["became"],
                "reason": row["reason"],
                "at": row["at"],
            }
            for row in rows
        ]

    def undo(self, reason: str = "undone") -> Optional[dict]:
        """Reverse the most recent state change. Returns what was undone.

        A change with no previous state was the belief's first appearance;
        undoing that retires it rather than deleting the row, so the record of
        having believed it survives.
        """
        row = self.connection.execute(
            "SELECT * FROM history ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        target = RETIRED if row["was"] is None else row["was"]
        belief = self.get(row["belief_id"])
        if belief is None:  # pragma: no cover - referential integrity
            return None
        self.set_state(belief, target, reason)
        return {
            "belief": belief.body,
            "from": row["became"],
            "to": target,
            "reason": reason,
        }

    # -- reporting ----------------------------------------------------------

    def summary(self) -> dict:
        counts = {
            row["state"]: row["n"]
            for row in self.connection.execute(
                "SELECT state, COUNT(*) AS n FROM beliefs GROUP BY state"
            )
        }
        return {
            "path": self.path,
            "beliefs": sum(counts.values()),
            "by_state": {state: counts.get(state, 0) for state in STATES},
            "changes": self.connection.execute(
                "SELECT COUNT(*) FROM history"
            ).fetchone()[0],
        }

    def __len__(self) -> int:
        return self.connection.execute("SELECT COUNT(*) FROM beliefs").fetchone()[0]

    def __repr__(self) -> str:
        summary = self.summary()
        return f"Memory({summary['path']!r}, {summary['beliefs']} belief(s))"


def _belief(row) -> Belief:
    return Belief(
        id=row["id"],
        kind=row["kind"],
        body=row["body"],
        state=row["state"],
        provenance=row["provenance"],
        confidence=row["confidence"],
        evidence=row["evidence"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )
