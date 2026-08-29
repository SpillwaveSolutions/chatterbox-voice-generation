"""chatterbox_tts.jobs — SQLite-backed async job queue (quick-260602-b7l).

Replaces the old asyncio.Lock-serialised /synthesize sync endpoint with:

  POST   /jobs            (idempotent enqueue; INSERT OR IGNORE on request_id PK)
  GET    /jobs/{id}       (current state lookup)
  DELETE /jobs/{id}       (terminal-only hard delete; 409 for queued/running)

Architecture (RESEARCH.md):
  - Single in-process asyncio.Queue + single worker task started at lifespan.
  - SQLite at /app/jobs/jobs.sqlite (chatterbox_jobs named volume; survives
    container restart). WAL mode + check_same_thread=False so the worker thread
    + HTTP handlers can share one connection without locking pain.
  - Boot recovery (recovery.reconcile_boot_state) probes every 'running' job's
    WAV on disk: valid (>=100KB + stdlib wave.open parses) -> completed;
    invalid -> failed. Queued jobs are re-enqueued into the in-memory queue.
    This is the import-2 orphan-WAV fix — WAV-on-disk IS the source of truth.
"""
from chatterbox_tts.jobs import db, recovery, worker  # noqa: F401

__all__ = ["db", "recovery", "worker"]
