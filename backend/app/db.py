"""SQLite persistence for chat sessions (stdlib sqlite3, no external services)."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- 'user' or 'assistant'
    content     TEXT NOT NULL,
    metadata    TEXT,                   -- JSON string, assistant messages only
    created_at  TEXT NOT NULL           -- server time, ISO-8601
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages (session_id, id);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at    TEXT NOT NULL
);
"""


def sqlite_path_from_url(url: str) -> str:
    """'sqlite:////data/app.db' -> '/data/app.db'; 'sqlite:///./app.db' -> './app.db'."""
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError("DATABASE_URL must start with 'sqlite:///' (SQLite only)")
    return url[len(prefix):]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, database_url: str):
        self.path = sqlite_path_from_url(database_url)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        """Create the database file, parent directory and schema if missing."""
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with closing(self._connect()) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            conn.commit()

    def ping(self) -> bool:
        try:
            with closing(self._connect()) as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def get_system_prompt(self, session_id: str) -> str | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row["system_prompt"] if row else None

    def get_history(self, session_id: str) -> list[dict]:
        """All stored messages of a session, oldest first."""
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT role, content, metadata FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        history = []
        for row in rows:
            item = {"role": row["role"], "content": row["content"]}
            if row["role"] == "assistant" and row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            history.append(item)
        return history

    def session_exists(self, session_id: str) -> bool:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT 1 FROM messages WHERE session_id = ? LIMIT 1", (session_id,)
            ).fetchone()
        return row is not None

    def save_exchange(
        self,
        session_id: str,
        system_prompt: str | None,
        user_message: str,
        reply: str,
        metadata: dict,
    ) -> None:
        """Persist the user turn and assistant reply atomically (all or nothing)."""
        now = _now()
        with closing(self._connect()) as conn, conn:  # inner `with conn` = transaction
            if system_prompt is not None:
                conn.execute(
                    "INSERT INTO sessions (session_id, system_prompt, created_at) VALUES (?, ?, ?) "
                    "ON CONFLICT(session_id) DO UPDATE SET system_prompt = excluded.system_prompt",
                    (session_id, system_prompt, now),
                )
            else:
                conn.execute(
                    "INSERT OR IGNORE INTO sessions (session_id, system_prompt, created_at) "
                    "VALUES (?, NULL, ?)",
                    (session_id, now),
                )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_message, now),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) "
                "VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, reply, json.dumps(metadata), _now()),
            )
