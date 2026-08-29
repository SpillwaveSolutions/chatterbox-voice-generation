"""SQLite persistence for chatterbox-tts jobs (quick-260602-b7l).

Stdlib sqlite3 + WAL mode + check_same_thread=False — the worker thread and
the FastAPI handlers share one Connection without lock pain (single writer,
multiple readers; WAL allows readers in parallel with the writer).

Pragmas locked per RESEARCH.md Pitfall 3:
    PRAGMA journal_mode = WAL       # readers don't block on the writer
    PRAGMA synchronous  = NORMAL    # ~ms write durability is fine — WAV-on-disk
                                    # is the real source of truth
    PRAGMA busy_timeout = 5000      # absorb incidental contention

Schema is fixed; bootstrapped idempotently via CREATE ... IF NOT EXISTS so
restarts are safe. No migration framework is needed (one table, terminal rows
expire by retention policy).
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


def _default_db_path() -> Path:
    return Path(os.environ.get(
        "CHATTERBOX_JOBS_DB_PATH", "/app/jobs/jobs.sqlite"
    ))


SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id            TEXT PRIMARY KEY,
  state         TEXT NOT NULL CHECK (state IN ('queued','running','completed','failed')),
  text          TEXT NOT NULL,
  voice_ref     TEXT,
  params_json   TEXT NOT NULL,
  wav_path      TEXT,
  error_message TEXT,
  created_at    TEXT NOT NULL,
  started_at    TEXT,
  completed_at  TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
CREATE INDEX IF NOT EXISTS idx_jobs_completed_at ON jobs(completed_at);
"""


def _utcnow_iso() -> str:
    """ISO-8601 UTC timestamp with trailing Z (matches the schema's TEXT col)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open the jobs SQLite with WAL + cross-thread + Row factory.

    Auto-creates the parent dir so the named volume's first-boot is safe.
    """
    path = db_path or _default_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(path),
        check_same_thread=False,  # single writer (worker), readers from handlers
        isolation_level=None,     # autocommit; we never want a stuck txn
    )
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.row_factory = sqlite3.Row
    return conn


def create_schema_if_missing(conn: sqlite3.Connection) -> None:
    """Idempotent bootstrap. Safe to call on every lifespan startup."""
    conn.executescript(SCHEMA)


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------

def insert_queued(
    conn: sqlite3.Connection,
    job_id: str,
    text: str,
    voice_ref: Optional[str],
    params_json: str,
) -> bool:
    """INSERT OR IGNORE — returns True if row was inserted, False if duplicate.

    This is the idempotency primitive: duplicate POST /jobs with the same
    request_id is collapsed to "no enqueue" by the False return.
    """
    cur = conn.execute(
        "INSERT OR IGNORE INTO jobs (id, state, text, voice_ref, params_json, created_at) "
        "VALUES (?, 'queued', ?, ?, ?, ?)",
        (job_id, text, voice_ref, params_json, _utcnow_iso()),
    )
    return cur.rowcount == 1


def fetch_job(
    conn: sqlite3.Connection, job_id: str
) -> Optional[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM jobs WHERE id = ?", (job_id,)
    ).fetchone()


def fetch_by_state(
    conn: sqlite3.Connection, state: str
) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM jobs WHERE state = ? ORDER BY created_at ASC",
        (state,),
    ).fetchall()


def mark_running(conn: sqlite3.Connection, job_id: str) -> None:
    conn.execute(
        "UPDATE jobs SET state = 'running', started_at = ? WHERE id = ?",
        (_utcnow_iso(), job_id),
    )


def mark_completed(
    conn: sqlite3.Connection, job_id: str, wav_path: str
) -> None:
    conn.execute(
        "UPDATE jobs SET state = 'completed', wav_path = ?, completed_at = ? WHERE id = ?",
        (wav_path, _utcnow_iso(), job_id),
    )


def mark_failed(
    conn: sqlite3.Connection, job_id: str, error_message: str
) -> None:
    conn.execute(
        "UPDATE jobs SET state = 'failed', error_message = ?, completed_at = ? WHERE id = ?",
        (error_message, _utcnow_iso(), job_id),
    )


def delete_job(conn: sqlite3.Connection, job_id: str) -> int:
    """Returns rows affected — caller uses 0 to detect idempotent re-deletes."""
    cur = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    return cur.rowcount


def prune_terminal_over_cap(
    conn: sqlite3.Connection, cap: int = 100
) -> None:
    """Keep only the newest ``cap`` terminal rows; never touch queued/running.

    Single DELETE with a self-subquery: select the IDs of the newest ``cap``
    terminal rows by completed_at DESC, then delete every other terminal row.
    """
    conn.execute(
        """
        DELETE FROM jobs
         WHERE state IN ('completed','failed')
           AND id NOT IN (
                SELECT id FROM jobs
                 WHERE state IN ('completed','failed')
              ORDER BY completed_at DESC
                 LIMIT ?
           )
        """,
        (cap,),
    )
