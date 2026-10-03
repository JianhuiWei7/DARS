"""Durable, restart-safe judge cache with cross-process single-flight.

- Entries live in one SQLite file (WAL mode), keyed by :func:`cache_key`, a SHA-256 over a canonical
  JSON of everything that determines the answer: messages (after redaction), output schema, prompt /
  schema / parser versions, backend model id + revision, decoding, reasoning, structured-output mode
  and ``sample_index``. The rollout ``policy_version``, tags, priority and request id are NOT in the key.
- Only successful results are stored, together with the raw provider response for provenance.
- Cross-process single-flight uses a ``leases`` table: the first process to insert a lease for a key
  calls the model; others poll until the entry appears, or the lease is released (failure) or
  expires (crashed holder), and then try to take it themselves. The holder renews its lease every
  ``lease_ttl_s / 3``.
- Every lease grant gets a fencing token from a monotonic counter. Renew, write and release require
  the matching owner AND token, so a holder that was paused past its TTL and lost the lease to
  another process can neither extend it, nor overwrite the newer result, nor release the new lease.
- In-process single-flight is done by the client (one in-flight future per key).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import sqlite3
import threading
import time
import uuid
from typing import Any, Callable, Dict, Optional

from judgerl.judge.schema import schema_to_dict

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entries (
    key TEXT PRIMARY KEY,
    created REAL NOT NULL,
    expires REAL,
    model TEXT,
    result TEXT NOT NULL,
    raw TEXT
);
CREATE TABLE IF NOT EXISTS leases (
    key TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    expires REAL NOT NULL,
    token INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS fence (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    token INTEGER NOT NULL
);
INSERT OR IGNORE INTO fence(id, token) VALUES (1, 0);
"""


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def cache_key(*, messages, output_schema, prompt_version, schema_version, parser_version, model: str,
              revision: Optional[str], decoding: Dict[str, Any], reasoning: Dict[str, Any],
              structured: str, sample_index: int = 0, extra_params: Optional[Dict[str, Any]] = None) -> str:
    """Canonical SHA-256 key of a judge request as sent to one model."""
    payload = {
        "v": 1, "messages": messages, "schema": schema_to_dict(output_schema),
        "prompt_version": str(prompt_version), "schema_version": str(schema_version),
        "parser_version": str(parser_version), "model": model, "revision": revision,
        "decoding": decoding, "reasoning": reasoning, "structured": structured,
        "sample_index": int(sample_index), "extra_params": extra_params or {},
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class JudgeCache:
    """SQLite-backed cache. Blocking operations run in a worker thread via the async wrappers."""

    def __init__(self, path: str, ttl_s: Optional[float] = None, lease_ttl_s: float = 300.0,
                 poll_s: float = 0.05, max_poll_s: float = 1.0):
        if not lease_ttl_s or lease_ttl_s <= 0:
            raise ValueError("lease_ttl_s must be > 0")
        self.path = path
        self.ttl_s = ttl_s
        self.lease_ttl_s = lease_ttl_s
        #: renewal period, well below the TTL so one late renewal does not lose the lease
        self.renew_interval_s = lease_ttl_s / 3.0
        self.poll_s = poll_s
        self.max_poll_s = max_poll_s
        self.owner = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, timeout=60.0, check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA busy_timeout=60000")
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:  # e.g. network filesystems without shared memory
            pass
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA_SQL)
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(leases)")}
        if "token" not in cols:  # cache files created before fencing tokens
            self._conn.execute("ALTER TABLE leases ADD COLUMN token INTEGER NOT NULL DEFAULT 0")
        self.hits = 0
        self.misses = 0
        self.writes = 0
        self.lease_waits = 0
        self.fenced_writes = 0

    # ------------------------------------------------------------------ sync API
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        now = time.time()
        with self._lock:
            row = self._conn.execute("SELECT result, raw, expires FROM entries WHERE key=?", (key,)).fetchone()
        if row is None or (row[2] is not None and row[2] < now):
            self.misses += 1
            return None
        self.hits += 1
        rec = json.loads(row[0])
        rec["raw"] = json.loads(row[1]) if row[1] else None
        return rec

    def put(self, key: str, result: Dict[str, Any], raw: Optional[Dict[str, Any]] = None,
            model: Optional[str] = None, token: Optional[int] = None, lease_key: Optional[str] = None) -> bool:
        """Store a result. With ``token`` the write happens only if this cache still holds the lease on
        ``lease_key`` (default ``key``) with that fencing token, checked atomically with the write;
        returns whether it was written."""
        now = time.time()
        expires = now + self.ttl_s if self.ttl_s else None
        body = canonical_json({k: v for k, v in result.items() if k != "raw"})
        raw_s = canonical_json(raw) if raw is not None else None
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                if token is not None:
                    row = conn.execute("SELECT owner, token FROM leases WHERE key=?",
                                       (lease_key or key,)).fetchone()
                    if row is None or row[0] != self.owner or row[1] != token:
                        conn.execute("COMMIT")
                        self.fenced_writes += 1
                        return False
                conn.execute("INSERT OR REPLACE INTO entries(key, created, expires, model, result, raw) "
                             "VALUES (?,?,?,?,?,?)", (key, now, expires, model, body, raw_s))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        self.writes += 1
        return True

    def acquire_lease(self, key: str) -> Optional[int]:
        """Try to become the single process computing ``key``. Returns a fencing token (> 0) or ``None``."""
        now = time.time()
        with self._lock:
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                row = conn.execute("SELECT owner, expires FROM leases WHERE key=?", (key,)).fetchone()
                token: Optional[int] = None
                if row is None or row[1] < now:
                    conn.execute("UPDATE fence SET token = token + 1 WHERE id = 1")
                    token = conn.execute("SELECT token FROM fence WHERE id = 1").fetchone()[0]
                    conn.execute("INSERT OR REPLACE INTO leases(key, owner, expires, token) VALUES (?,?,?,?)",
                                 (key, self.owner, now + self.lease_ttl_s, token))
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise
        return token

    def renew_lease(self, key: str, token: int) -> bool:
        """Extend a held lease. ``False`` means the lease was lost (expired and taken over, or gone):
        the caller must not write its result."""
        with self._lock:
            cur = self._conn.execute("UPDATE leases SET expires=? WHERE key=? AND owner=? AND token=?",
                                     (time.time() + self.lease_ttl_s, key, self.owner, token))
        return cur.rowcount == 1

    def release_lease(self, key: str, token: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM leases WHERE key=? AND owner=? AND token=?", (key, self.owner, token))

    def lease_holder(self, key: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute("SELECT owner, expires FROM leases WHERE key=?", (key,)).fetchone()
        if row is None or row[1] < time.time():
            return None
        return row[0]

    def purge_expired(self) -> int:
        now = time.time()
        with self._lock:
            cur = self._conn.execute("DELETE FROM entries WHERE expires IS NOT NULL AND expires < ?", (now,))
            self._conn.execute("DELETE FROM leases WHERE expires < ?", (now,))
        return cur.rowcount

    def __len__(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]

    def stats(self) -> Dict[str, Any]:
        return {"hits": self.hits, "misses": self.misses, "writes": self.writes, "lease_waits": self.lease_waits,
                "fenced_writes": self.fenced_writes, "entries": len(self)}

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ------------------------------------------------------------------ async wrappers
    async def aget(self, key: str) -> Optional[Dict[str, Any]]:
        return await asyncio.to_thread(self.get, key)

    async def aput(self, key: str, result: Dict[str, Any], raw=None, model=None, token: Optional[int] = None,
                   lease_key: Optional[str] = None) -> bool:
        return await asyncio.to_thread(self.put, key, result, raw, model, token, lease_key)

    async def aacquire_lease(self, key: str) -> Optional[int]:
        return await asyncio.to_thread(self.acquire_lease, key)

    async def arenew_lease(self, key: str, token: int) -> bool:
        return await asyncio.to_thread(self.renew_lease, key, token)

    async def arelease_lease(self, key: str, token: int) -> None:
        await asyncio.to_thread(self.release_lease, key, token)

    async def wait_for_other(self, key: str, deadline: Optional[float],
                             now: Callable[[], float] = time.monotonic) -> Optional[Dict[str, Any]]:
        """Another process holds the lease: poll until its entry appears (returned), or the lease is
        gone (returns ``None``: the caller should try to acquire it), or ``deadline`` passes (``None``)."""
        self.lease_waits += 1
        delay = self.poll_s
        while True:
            hit = await self.aget(key)
            if hit is not None:
                return hit
            holder = await asyncio.to_thread(self.lease_holder, key)
            if holder is None:
                return None
            if deadline is not None and now() >= deadline:
                return None
            await asyncio.sleep(delay)
            delay = min(self.max_poll_s, delay * 1.5)
