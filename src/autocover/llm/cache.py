"""SQLite literal response cache: identical (model, messages, params) -> stored reply.

Parallel generation makes exact-match hits rarer than in a chat app, but re-runs of the
same target (benchmarks, retries after a crash) hit it heavily and cost nothing.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


def cache_key(model: str, messages: list[dict[str, Any]], params: dict[str, Any]) -> str:
    payload = json.dumps({"model": model, "messages": messages, "params": params}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: str | Path):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS responses ("
                " key TEXT PRIMARY KEY, model TEXT, payload TEXT, created REAL)"
            )
            self._conn.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM responses WHERE key = ?", (key,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, payload: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO responses VALUES (?, ?, ?, ?)",
                (key, model, json.dumps(payload), time.time()),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()


class UsageLedger:
    """Requests sent per (quota day, model), and per role, persisted so daily caps and
    role reservations survive restarts.

    Every HTTP attempt counts (including failed ones), because providers count them too.
    """

    def __init__(self, path: str | Path = ":memory:"):
        path = Path(path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS usage ("
                " day TEXT, model TEXT, requests INTEGER, tokens INTEGER,"
                " PRIMARY KEY (day, model))"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS role_usage ("
                " day TEXT, model TEXT, role TEXT, requests INTEGER,"
                " PRIMARY KEY (day, model, role))"
            )
            self._conn.commit()

    def record(self, day: str, model: str, requests: int = 1, tokens: int = 0,
               role: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage VALUES (?, ?, ?, ?) ON CONFLICT(day, model) DO UPDATE SET"
                " requests = requests + excluded.requests, tokens = tokens + excluded.tokens",
                (day, model, requests, tokens),
            )
            if role and requests:
                self._conn.execute(
                    "INSERT INTO role_usage VALUES (?, ?, ?, ?) ON CONFLICT(day, model, role)"
                    " DO UPDATE SET requests = requests + excluded.requests",
                    (day, model, role, requests),
                )
            self._conn.commit()

    def role_requests(self, day: str, model: str) -> dict[str, int]:
        """Requests to `model` on `day`, per role (requests recorded without a role are
        not included)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, requests FROM role_usage WHERE day = ? AND model = ?", (day, model)
            ).fetchall()
        return dict(rows)

    def requests(self, day: str, model: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT requests FROM usage WHERE day = ? AND model = ?", (day, model)
            ).fetchone()
        return row[0] if row else 0

    def day_summary(self, day: str) -> dict[str, tuple[int, int]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT model, requests, tokens FROM usage WHERE day = ? ORDER BY model", (day,)
            ).fetchall()
        return {model: (req, tok) for model, req, tok in rows}

    def close(self) -> None:
        with self._lock:
            self._conn.close()
