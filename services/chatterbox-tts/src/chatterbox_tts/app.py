"""chatterbox-tts FastAPI app (quick-260602-b7l: async-job-queue refactor).

Routes:
- GET    /health           -> {status, model_loaded, language_id}
- POST   /jobs             -> JobStateResponse (200 idempotent / 201 new)
- GET    /jobs/{id}        -> JobStateResponse (200) or 404
- DELETE /jobs/{id}        -> 204 (idempotent); 409 if state in queued/running

The old POST /synthesize is GONE — no back-compat (CONTEXT.md D-02). The only
caller is libs/tts_chatterbox.client which is being rewritten in the same task.

Lifespan ordering (RESEARCH §Pitfall 5):
    connect SQLite
 -> create schema
 -> reconcile boot state (running -> completed|failed by WAV-on-disk)
 -> populate in-memory queue with requeue_ids
 -> eager-load English model
 -> start asyncio.create_task(run_worker)
 -> yield
 -> task.cancel() + await (suppressing CancelledError)
 -> conn.close()

Boot recovery MUST run BEFORE the worker starts and BEFORE the queue is
populated. Otherwise the worker pulls an empty queue and races recovery's
mark_completed writes.

Watermark (M8): Resemble's Perth Watermarker is mandatory and NOT
opt-out-able. Output always carries it.
"""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Response

from chatterbox_tts.jobs import db as jobs_db
from chatterbox_tts.jobs import recovery as jobs_recovery
from chatterbox_tts.jobs.worker import run_worker
from chatterbox_tts.model import _load_model, _state
from chatterbox_tts.pronunciations import (
    apply_pronunciations,
    compile_pronunciations,
)
from chatterbox_tts.schemas import JobCreateRequest, JobStateResponse

log = structlog.get_logger()

VOICE_REFS_DIR = Path(
    os.environ.get("CHATTERBOX_VOICE_REFS_DIR", "/app/voice_refs")
)
OUTPUT_DIR = Path(
    os.environ.get("CHATTERBOX_OUTPUT_DIR", "/app/output/tts_chatterbox")
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: SQLite + boot recovery + model load + worker task. Teardown reverses."""
    # 1) SQLite connection + schema (idempotent).
    conn = jobs_db.connect()
    jobs_db.create_schema_if_missing(conn)
    log.info("chatterbox_db_ready", db_path=str(jobs_db._default_db_path()))

    # 2) Boot recovery — running-with-valid-WAV -> completed; otherwise -> failed.
    try:
        result = jobs_recovery.reconcile_boot_state(conn, output_dir=OUTPUT_DIR)
        log.info(
            "chatterbox_boot_recovery",
            requeue_count=len(result.requeue_ids),
        )
    except Exception as exc:  # pragma: no cover - exercised only on corrupt DB
        log.exception("chatterbox_boot_recovery_failed", error=str(exc))
        result = jobs_recovery.ReconcileResult(requeue_ids=[])

    # 3) Populate the in-process FIFO queue BEFORE starting the worker.
    queue: asyncio.Queue[str] = asyncio.Queue()
    for job_id in result.requeue_ids:
        queue.put_nowait(job_id)

    # 4) Eager-load English model (existing behavior).
    try:
        _load_model("en")
        log.info("chatterbox_model_loaded", language_id="en")
    except Exception as exc:  # pragma: no cover - exercised via docker run
        log.error("chatterbox_model_load_failed", error=str(exc))

    # 5) Persist conn + queue on app.state for the route handlers.
    app.state.db = conn
    app.state.queue = queue

    # 6) Start the single-consumer FIFO worker.
    worker_task = asyncio.create_task(run_worker(queue, conn))

    try:
        yield
    finally:
        # See worker.run_worker docstring re: in-flight thread NOT being cancelled.
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass
        try:
            conn.close()
        except Exception:  # pragma: no cover
            log.exception("chatterbox_db_close_failed")


app = FastAPI(lifespan=lifespan)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok" if _state["model_loaded"] else "loading",
        "model_loaded": _state["model_loaded"],
        "language_id": _state["language_id"],
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _row_to_state_response(row: sqlite3.Row) -> JobStateResponse:
    """Map a SQLite row to the wire-shape JobStateResponse."""
    return JobStateResponse(
        id=row["id"],
        state=row["state"],
        wav_path=row["wav_path"],
        error=row["error_message"],
        queue_position=None,  # not surfaced today; reserved for future
    )


# ---------------------------------------------------------------------------
# POST /jobs (idempotent on request_id)
# ---------------------------------------------------------------------------

@app.post("/jobs", response_model=JobStateResponse)
async def create_job(req: JobCreateRequest, response: Response):
    """Idempotent enqueue. 200 if id already known, 201 if newly inserted."""
    conn: sqlite3.Connection = app.state.db
    queue: asyncio.Queue[str] = app.state.queue

    request_id = req.request_id or uuid.uuid4().hex

    # voice_ref validation — mirrors the old /synthesize behavior so the
    # caller gets a clear 400 instead of an opaque worker failure.
    if req.voice_ref:
        candidate = VOICE_REFS_DIR / req.voice_ref
        if not candidate.is_file():
            raise HTTPException(
                status_code=400,
                detail=f"voice_ref not found: {req.voice_ref}",
            )

    # Apply pronunciations server-side BEFORE persisting (so the worker
    # synthesises the substituted text). Schema caps already validated shape.
    text = req.text
    if req.pronunciations:
        patterns = compile_pronunciations(req.pronunciations)
        text = apply_pronunciations(text, patterns)

    params_json = json.dumps({
        "exaggeration": req.exaggeration,
        "cfg_weight": req.cfg_weight,
        "temperature": req.temperature,
        "language_id": req.language_id,
        "seed": req.seed,
        "sentence_chunk_size": req.sentence_chunk_size,
    })

    inserted = jobs_db.insert_queued(
        conn,
        request_id,
        text,
        req.voice_ref,
        params_json,
    )
    if inserted:
        # Apply retention AFTER successful enqueue (never block enqueue on prune).
        try:
            jobs_db.prune_terminal_over_cap(conn, cap=100)
        except Exception:  # pragma: no cover - prune is best-effort
            log.exception("prune_terminal_failed")
        queue.put_nowait(request_id)
        response.status_code = 201
        log.info("job_created", request_id=request_id)
    else:
        log.info("job_idempotent_return", request_id=request_id)

    row = jobs_db.fetch_job(conn, request_id)
    if row is None:  # pragma: no cover — defensive
        raise HTTPException(
            status_code=500,
            detail=f"insert_queued reported success but row missing: {request_id}",
        )
    return _row_to_state_response(row)


# ---------------------------------------------------------------------------
# GET /jobs/{request_id}
# ---------------------------------------------------------------------------

@app.get("/jobs/{request_id}", response_model=JobStateResponse)
async def get_job(request_id: str):
    conn: sqlite3.Connection = app.state.db
    row = jobs_db.fetch_job(conn, request_id)
    if row is None:
        raise HTTPException(
            status_code=404, detail=f"job not found: {request_id}"
        )
    return _row_to_state_response(row)


# ---------------------------------------------------------------------------
# DELETE /jobs/{request_id}
# ---------------------------------------------------------------------------

@app.delete("/jobs/{request_id}")
async def delete_job(request_id: str):
    """Terminal-only hard delete. 204 always (idempotent); 409 if non-terminal.

    Use case: Regenerate-after-failure. When an operator clicks Regenerate
    on a 'failed' import, the new request_id collides with the existing
    'failed' row and the idempotent POST returns 'failed' immediately. The
    Regenerate handler can call DELETE first to reset the state and force
    a real re-synth.
    """
    conn: sqlite3.Connection = app.state.db
    row = jobs_db.fetch_job(conn, request_id)
    if row is not None and row["state"] in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"cannot delete non-terminal job: state={row['state']}",
        )
    jobs_db.delete_job(conn, request_id)
    return Response(status_code=204)
