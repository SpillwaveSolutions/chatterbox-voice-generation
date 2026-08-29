"""Single-consumer FIFO worker — replaces the old asyncio.Lock.

Invariant: ``uvicorn --workers 1`` (enforced in Dockerfile) + a single
``asyncio.create_task(run_worker(queue, conn))`` started at lifespan ⇒ exactly
one synth at a time, exactly one SQLite writer.

Why no in-flight cancellation on shutdown? ``asyncio.to_thread`` schedules the
work on the default ThreadPoolExecutor; the awaiting task can be cancelled but
the underlying THREAD keeps running until ``_synthesize_sync`` returns. We
treat this as a feature (per RESEARCH §Pitfall 2): the synth keeps grinding,
the WAV lands on disk, and the next boot's ``reconcile_boot_state`` finds the
valid WAV → ``completed``. Trying to drain the queue on SIGTERM is the wrong
shape — Docker's 10s grace window can't accommodate hours of synth.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from pathlib import Path
from typing import Optional

import structlog

from chatterbox_tts.jobs import db as jobs_db
from chatterbox_tts.model import OutOfMemoryError, synthesize_to_wav

log = structlog.get_logger()


def _voice_refs_dir() -> Path:
    return Path(os.environ.get(
        "CHATTERBOX_VOICE_REFS_DIR", "/app/voice_refs"
    ))


def _output_dir() -> Path:
    return Path(os.environ.get(
        "CHATTERBOX_OUTPUT_DIR", "/app/output/tts_chatterbox"
    ))


async def run_worker(
    queue: "asyncio.Queue[str]", conn: sqlite3.Connection
) -> None:
    """Pop job ids, run synth, persist terminal state. Loops forever.

    The loop is robust:
      - If a popped id is unknown or non-queued (pruned, or somehow already
        terminal), it is silently skipped — duplicate enqueue is benign.
      - OOM -> mark_failed('oom') so the client sees a clean reason.
      - Any other exception -> mark_failed(str(exc)). Logged with traceback.
      - CancelledError -> re-raise (lifespan teardown). The in-flight thread
        keeps running; next boot's recovery handles the orphan WAV.
    """
    while True:
        job_id = await queue.get()
        try:
            row = jobs_db.fetch_job(conn, job_id)
            if row is None or row["state"] != "queued":
                # Pruned or already terminal — duplicate enqueue, benign.
                log.info(
                    "worker_skip_non_queued",
                    job_id=job_id,
                    state=(row["state"] if row else None),
                )
                continue

            jobs_db.mark_running(conn, job_id)
            try:
                params = json.loads(row["params_json"] or "{}")
            except json.JSONDecodeError:
                params = {}
            voice_ref = row["voice_ref"]
            voice_path: Optional[str] = (
                str(_voice_refs_dir() / voice_ref) if voice_ref else None
            )
            output_path = _output_dir() / f"{job_id}.wav"

            log.info("worker_job_start", job_id=job_id)
            wav_path, _dur = await synthesize_to_wav(
                text=row["text"],
                voice_ref_path=voice_path,
                exaggeration=params.get("exaggeration", 0.5),
                cfg_weight=params.get("cfg_weight", 0.5),
                temperature=params.get("temperature", 0.8),
                language_id=params.get("language_id", "en"),
                seed=params.get("seed"),
                sentence_chunk_size=params.get("sentence_chunk_size", 3),
                output_path=output_path,
            )
            jobs_db.mark_completed(conn, job_id, str(wav_path))
            log.info("worker_job_complete", job_id=job_id, wav_path=str(wav_path))

        except asyncio.CancelledError:
            # Lifespan teardown — see module docstring. Re-raise so the task
            # actually exits; do NOT try to interrupt the in-flight thread.
            log.warning("worker_cancelled_mid_job", job_id=job_id)
            raise
        except OutOfMemoryError as exc:
            log.error("worker_job_oom", job_id=job_id, error=str(exc))
            try:
                jobs_db.mark_failed(conn, job_id, "oom")
            except Exception:  # pragma: no cover
                log.exception("mark_failed_oom_write_failed", job_id=job_id)
        except Exception as exc:
            log.exception("worker_job_failed", job_id=job_id)
            try:
                jobs_db.mark_failed(
                    conn, job_id, str(exc) or type(exc).__name__
                )
            except Exception:  # pragma: no cover
                log.exception("mark_failed_write_failed", job_id=job_id)
