"""Session and slot memory, persisted to SQLite.

Follow-ups work because the *active intent* (the structured object the user
is iterating on) is stored per session and handed to the intent parser as
context — not raw chat history.

SQLite keeps the MVP zero-ops; the repository interface is narrow enough to
swap for Redis/Postgres later.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone

from cae.models import QueryIntent, Session, Turn

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    created_at    TEXT NOT NULL,
    active_intent TEXT
);
CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    payload    TEXT NOT NULL,
    PRIMARY KEY (session_id, turn_index)
);
"""


class SessionStore:
    def __init__(self, path: str = "sessions.db") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # -- sessions ------------------------------------------------------------

    def create_session(self) -> Session:
        session = Session(session_id=uuid.uuid4().hex[:12])
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (session_id, created_at) VALUES (?, ?)",
                (session.session_id, session.created_at.isoformat()),
            )
            self._conn.commit()
        return session

    def get_session(self, session_id: str) -> Session | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT session_id, created_at, active_intent FROM sessions "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            turn_rows = self._conn.execute(
                "SELECT payload FROM turns WHERE session_id = ? ORDER BY turn_index",
                (session_id,),
            ).fetchall()
        active = QueryIntent.model_validate_json(row[2]) if row[2] else None
        turns = [Turn.model_validate_json(r[0]) for r in turn_rows]
        return Session(
            session_id=row[0],
            created_at=datetime.fromisoformat(row[1]),
            active_intent=active,
            turns=turns,
        )

    # -- turns / slot memory --------------------------------------------------

    def append_turn(self, session_id: str, turn: Turn) -> None:
        """Persist a turn and promote its intent to the session's active slot."""
        with self._lock:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM turns WHERE session_id = ?", (session_id,)
            ).fetchone()[0]
            self._conn.execute(
                "INSERT INTO turns (session_id, turn_index, payload) VALUES (?, ?, ?)",
                (session_id, count, turn.model_dump_json()),
            )
            self._conn.execute(
                "UPDATE sessions SET active_intent = ? WHERE session_id = ?",
                (turn.intent.model_dump_json(), session_id),
            )
            self._conn.commit()

    def set_active_intent(self, session_id: str, intent: QueryIntent | None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET active_intent = ? WHERE session_id = ?",
                (intent.model_dump_json() if intent else None, session_id),
            )
            self._conn.commit()

    def list_sessions(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT session_id FROM sessions ORDER BY created_at DESC"
            ).fetchall()
        return [r[0] for r in rows]

    def close(self) -> None:
        self._conn.close()
