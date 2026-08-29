"""Boot-time state reconciliation — the import-2 orphan-WAV fix.

When chatterbox-tts restarts in the middle of a 5-hour synth, the WAV finishes
on disk after lifespan teardown but the DB row is stuck at 'running'. Before
this module existed, the operator had to hand-salvage via infra/salvage_import_2.py.

reconcile_boot_state() runs BEFORE the worker task starts (Pitfall 5
ordering invariant). For every 'running' row it probes the WAV on disk:

  - valid (file exists + size >= 100_000 + stdlib wave.open parses + getnframes>0)
        -> mark_completed (the import-2 auto-recover)
  - invalid -> mark_failed with reason 'recovered_from_crash_no_valid_wav'

For every 'queued' row, the id is returned in ReconcileResult.requeue_ids so the
lifespan can populate the in-process queue before yielding.
"""
from __future__ import annotations

import os
import sqlite3
import wave
from dataclasses import dataclass, field
from pathlib import Path

from chatterbox_tts.jobs import db as jobs_db


WAV_MIN_BYTES = 100_000  # ~1.1s of 44.1kHz s16 mono — sanity floor


@dataclass
class ReconcileResult:
    """Output of boot recovery — ids the worker should re-pick-up."""

    requeue_ids: list[str] = field(default_factory=list)


def _is_valid_wav(path: Path) -> bool:
    """True iff the file exists, is big enough, AND wave.open parses cleanly."""
    try:
        if not path.is_file():
            return False
        if path.stat().st_size < WAV_MIN_BYTES:
            return False
        with wave.open(str(path), "rb") as w:
            return w.getnframes() > 0
    except (wave.Error, EOFError, OSError):
        return False


def _default_output_dir() -> Path:
    return Path(os.environ.get(
        "CHATTERBOX_OUTPUT_DIR", "/app/output/tts_chatterbox"
    ))


def reconcile_boot_state(
    conn: sqlite3.Connection, output_dir: Path | None = None
) -> ReconcileResult:
    """Reconcile every non-terminal row against on-disk reality.

    Args:
        conn: open SQLite connection (schema must already exist).
        output_dir: where to look for <id>.wav when a row has no explicit
            wav_path. Defaults to CHATTERBOX_OUTPUT_DIR (or
            /app/output/tts_chatterbox).

    Returns:
        ReconcileResult with the queued ids (in created_at ASC order) the
        caller should re-enqueue.
    """
    out_dir = output_dir or _default_output_dir()

    # 1) Reconcile every 'running' row against WAV-on-disk.
    for row in jobs_db.fetch_by_state(conn, "running"):
        wav_path_field = row["wav_path"]
        candidate = Path(wav_path_field) if wav_path_field else (
            out_dir / f"{row['id']}.wav"
        )
        if _is_valid_wav(candidate):
            jobs_db.mark_completed(conn, row["id"], str(candidate))
        else:
            jobs_db.mark_failed(
                conn, row["id"], "recovered_from_crash_no_valid_wav"
            )

    # 2) Gather queued rows for re-enqueue.
    requeue_ids = [r["id"] for r in jobs_db.fetch_by_state(conn, "queued")]
    return ReconcileResult(requeue_ids=requeue_ids)
