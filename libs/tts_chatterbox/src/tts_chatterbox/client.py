"""HTTP client for the async-job-queue chatterbox-tts service (quick-260602-b7l).

Wire shape:
    POST   {base_url}/jobs           -> {"id", "state", "wav_path"?, "error"?, ...}
    GET    {base_url}/jobs/{id}      -> same body shape; 404 if pruned/unknown
    DELETE {base_url}/jobs/{id}      -> 204 (idempotent on terminal/missing);
                                        409 if state in (queued, running)

Client behaviour:
    1. POST once (3-attempt retry on transient errors, short 30s timeout).
    2. If the POST response is 200/201 with state="failed", the server's
       idempotency layer just returned a pre-existing 'failed' row keyed by
       ``request_id`` (the classic poison-row case after a chatterbox-tts
       crash or a deploy that killed an in-flight job). Auto-DELETE the row
       and re-POST exactly once so Regenerate works without operator
       intervention. See chatterbox app.py docstring on DELETE for the
       designed contract. The retry POST's result is treated normally —
       a second 'failed' surfaces as a RuntimeError without further retry.
    3. Poll GET every 5s until state in (completed, failed) OR 404 raises.
    4. completed -> shutil.copyfile(server_wav_path, output_path), return output_path.
    5. failed    -> RuntimeError with the server's error detail.
    6. 404       -> RuntimeError("chatterbox-tts job not found: ...").
    7. Transient httpx.ConnectError during a poll -> retry with 5s backoff
       up to ``poll_max_attempts`` (see below). This tolerates a chatterbox-tts
       container restart mid-poll: after the restart, boot recovery either
       marks the job 'completed' (valid WAV on disk) or 'failed', and the
       next successful GET surfaces that state without raising on the gap.

POST vs POLL retry budgets (quick-260610-kuo):
    The POST-side budget (``max_attempts``, default 3) and the GET/poll-side
    budget (``poll_max_attempts``, default 60 via env-or-constant) are SPLIT.
    POST failures should fast-fail (Prefect retries the whole task); the poll
    side needs to ride out the chatterbox-tts container's ~3.5 min cold-start
    (quick-260602-b7l), so 60 × 5s = 5 min wall-clock tolerance by default.
    Override per-env via ``CHATTERBOX_TTS_POLL_MAX_ATTEMPTS``.

CROSS-CONTAINER WAV PATH CONTRACT (key_link from quick-260528-njt):
The /jobs/{id} response's ``wav_path`` (on state=completed) is an absolute
path INSIDE the chatterbox-tts container's filesystem. It is reachable from
the flow-runner ONLY because BOTH containers mount ``./output:/app/output``
identically in docker-compose.yml. If the host-side mount is missing or
differs, ``shutil.copyfile`` below raises FileNotFoundError after a healthy
200. See docker-compose.yml chatterbox-tts entry.
"""
from __future__ import annotations

import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import structlog

from tts_chatterbox.config import ChatterboxConfig

log = structlog.get_logger()

DEFAULT_URL = "http://chatterbox-tts:8090"
# Short read/connect timeouts — every HTTP call now returns instantly
# (POST /jobs is an enqueue; GET /jobs/{id} is a state lookup). Hours-long
# synth runs in the background on the server; client polls.
SHORT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
POLL_INTERVAL_SEC = 5.0

# Poll-side GET retry budget — separate from POST's `max_attempts` so a long
# chatterbox-tts cold-start (~3.5 min — quick-260602-b7l) doesn't blow through
# the small POST budget. 60 × 5s = 5 min wall-clock tolerance by default.
# Override per-env via CHATTERBOX_TTS_POLL_MAX_ATTEMPTS (e.g. "120" for 10 min).
POLL_MAX_ATTEMPTS_DEFAULT = 60
POLL_MAX_ATTEMPTS_ENV = "CHATTERBOX_TTS_POLL_MAX_ATTEMPTS"


def synthesize(
    text: str,
    config: ChatterboxConfig,
    output_path: str,
    request_id: Optional[str] = None,
    base_url: Optional[str] = None,
    max_attempts: int = 3,
    backoff_sec: float = 5.0,
    poll_max_attempts: Optional[int] = None,
) -> str:
    """POST a job, poll until terminal, copy WAV to ``output_path``.

    Args:
        text: Narration text to synthesise.
        config: ChatterboxConfig — its ``to_request_body(text, request_id)``
            is sent as the JSON body.
        output_path: Where to deposit the WAV after the server returns. The
            server writes to ``/app/output/tts_chatterbox/<request_id>.wav``
            inside its container; this function copies that to ``output_path``
            via shutil.copyfile through the SHARED ``./output`` mount.
        request_id: Used as the server-side row PK AND the output filename
            AND the log correlation id. Defaults to a uuid4 hex if missing
            (caller is encouraged to always pass one for traceability).
        base_url: Override the CHATTERBOX_TTS_URL env var / default.
        max_attempts: POST-side retry budget only (default 3 — UNCHANGED).
            POST failures fast-fail so Prefect's task retry handles the gap.
        backoff_sec: Linear backoff between retries (default 5s).
        poll_max_attempts: GET/poll-side retry budget (quick-260610-kuo).
            Resolution: explicit kwarg > env ``CHATTERBOX_TTS_POLL_MAX_ATTEMPTS``
            > module default ``POLL_MAX_ATTEMPTS_DEFAULT`` (60). With 5s backoff
            the default gives ~5 min wall-clock tolerance — enough to ride out
            a chatterbox-tts container cold-start (~3.5 min per quick-260602-b7l)
            without surfacing as a job failure.

    Returns:
        ``output_path`` on success.

    Raises:
        ValueError: if ``text`` is empty or whitespace-only.
        RuntimeError: on
            - 4xx from POST or GET (immediate, with server detail)
            - 404 from GET (pruned or never created)
            - state='failed' (with server's error_message)
            - 5xx or transport-error exhaustion after ``max_attempts`` (POST)
              or ``poll_max_attempts`` (GET).
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    # --- Backend dispatch (phase 07). Local is the default; runpod delegates.
    # CHATTERBOX_BACKEND env wins; config.backend is the fallback selector
    # (env > config.backend > "local"). The runpod_client import is LAZY (inside
    # the branch) so importing this module on the local path does not pull in
    # artifact_store/boto3 — matching the lazy-import discipline in tasks.py.
    # Any other non-empty backend value fails loud (ValueError below) rather
    # than silently synthesizing via the local container; empty/unset still
    # falls through to the local path.
    # Everything BELOW this branch is the byte-unchanged local path (SC-4).
    backend = os.environ.get("CHATTERBOX_BACKEND", getattr(config, "backend", "local"))
    if backend == "runpod":
        from tts_chatterbox.runpod_client import synthesize as runpod_synthesize
        return runpod_synthesize(
            text=text,
            config=config,
            output_path=output_path,
            request_id=request_id,
            max_attempts=max_attempts,
            backoff_sec=backoff_sec,
            poll_max_attempts=poll_max_attempts,
        )
    if backend and backend != "local":
        raise ValueError(
            f"Unknown CHATTERBOX_BACKEND {backend!r}: "
            "valid backends are 'local' and 'runpod'"
        )
    # --- LOCAL PATH BELOW IS UNCHANGED (existing POST/poll/copyfile/DELETE) ---

    url = (
        base_url or os.environ.get("CHATTERBOX_BASE_URL", DEFAULT_URL)
    ).rstrip("/")
    rid = request_id or uuid.uuid4().hex
    body = config.to_request_body(text, request_id=rid)

    # Resolve the GET/poll-side budget once per call: kwarg > env > default.
    # See module docstring "POST vs POLL retry budgets" (quick-260610-kuo).
    effective_poll_max = (
        poll_max_attempts
        if poll_max_attempts is not None
        else int(os.environ.get(POLL_MAX_ATTEMPTS_ENV, POLL_MAX_ATTEMPTS_DEFAULT))
    )

    log.info("chatterbox_post_jobs", request_id=rid, url=url)
    with httpx.Client(timeout=SHORT_TIMEOUT) as cli:
        state = _post_with_retry(
            cli, f"{url}/jobs", body, max_attempts, backoff_sec
        )

        # Idempotent-return-of-failed recovery (260610-chatterbox-poison):
        # The server's request_id-keyed idempotency means a pre-existing
        # 'failed' row (e.g. from a crashed prior attempt) is returned as
        # state='failed' on POST, never starting a fresh synth. Detect that
        # exact case — first POST returned terminal 'failed' without ever
        # going through queued/running — and auto-DELETE + re-POST once.
        # A second 'failed' after the retry is treated as genuine and
        # surfaces normally below.
        if state.get("state") == "failed":
            prior_error = state.get("error")
            log.warning(
                "chatterbox_idempotent_failed_retry",
                request_id=rid,
                prior_error=prior_error,
            )
            _delete_with_retry(
                cli, f"{url}/jobs/{rid}", max_attempts, backoff_sec
            )
            state = _post_with_retry(
                cli, f"{url}/jobs", body, max_attempts, backoff_sec
            )

        while state["state"] in ("queued", "running"):
            time.sleep(POLL_INTERVAL_SEC)
            state = _get_with_retry(
                cli, f"{url}/jobs/{rid}", effective_poll_max, backoff_sec
            )

        if state["state"] == "failed":
            raise RuntimeError(
                f"chatterbox-tts job failed: {state.get('error')}"
            )
        if state["state"] != "completed":
            raise RuntimeError(
                f"chatterbox-tts unexpected state: {state['state']}"
            )

        src = state.get("wav_path")
        if not src:
            raise RuntimeError(
                f"chatterbox-tts completed without wav_path: {state}"
            )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if src != output_path:
            shutil.copyfile(src, output_path)
        log.info("chatterbox_synthesize_ok", request_id=rid)
        return output_path


# ---------------------------------------------------------------------------
# Retry helpers
# ---------------------------------------------------------------------------

def _post_with_retry(
    cli: httpx.Client,
    url: str,
    body: dict,
    max_attempts: int,
    backoff_sec: float,
) -> dict:
    """POST with 3-attempt retry on 5xx + transport errors; 4xx raises immediately."""
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            resp = cli.post(url, json=body)
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning(
                "chatterbox_post_transport_error",
                attempt=attempt + 1, error=str(exc),
            )
            if attempt < max_attempts - 1:
                time.sleep(backoff_sec)
                continue
            raise RuntimeError(
                f"chatterbox-tts POST /jobs failed after {max_attempts} attempts: {exc}"
            ) from exc

        if resp.status_code in (200, 201):
            return resp.json()
        if 400 <= resp.status_code < 500:
            raise RuntimeError(
                f"chatterbox-tts {resp.status_code}: {_extract_detail(resp)}"
            )
        # 5xx
        log.warning(
            "chatterbox_post_5xx",
            attempt=attempt + 1,
            status=resp.status_code,
            body=resp.text[:200],
        )
        last_exc = RuntimeError(
            f"chatterbox-tts {resp.status_code}: {resp.text[:200]}"
        )
        if attempt < max_attempts - 1:
            time.sleep(backoff_sec)
    raise RuntimeError(
        f"chatterbox-tts POST /jobs failed after {max_attempts} attempts"
    ) from last_exc


def _get_with_retry(
    cli: httpx.Client,
    url: str,
    max_attempts: int,
    backoff_sec: float,
) -> dict:
    """GET with restart-tolerance: transient httpx errors retry with backoff.

    Tolerates a chatterbox-tts container restart mid-poll. 404 is terminal
    (raises immediately) — the row was pruned or never created.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            resp = cli.get(url)
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning(
                "chatterbox_poll_transport_error",
                attempt=attempt + 1, error=str(exc),
            )
            if attempt < max_attempts - 1:
                time.sleep(backoff_sec)
                continue
            raise RuntimeError(
                f"chatterbox-tts GET poll failed after {max_attempts} attempts: {exc}"
            ) from exc

        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            raise RuntimeError(
                f"chatterbox-tts job not found: {_extract_detail(resp)}"
            )
        if 400 <= resp.status_code < 500:
            raise RuntimeError(
                f"chatterbox-tts {resp.status_code}: {_extract_detail(resp)}"
            )
        # 5xx
        log.warning(
            "chatterbox_poll_5xx",
            attempt=attempt + 1,
            status=resp.status_code,
            body=resp.text[:200],
        )
        last_exc = RuntimeError(
            f"chatterbox-tts {resp.status_code}: {resp.text[:200]}"
        )
        if attempt < max_attempts - 1:
            time.sleep(backoff_sec)
    raise RuntimeError(
        f"chatterbox-tts GET /jobs failed after {max_attempts} attempts"
    ) from last_exc


def _extract_detail(resp: httpx.Response) -> str:
    """Pull JSON ``detail``, falling back to body snippet."""
    try:
        data = resp.json()
        if isinstance(data, dict) and "detail" in data:
            return str(data["detail"])
        return resp.text[:200]
    except Exception:
        return resp.text[:200]


def _delete_with_retry(
    cli: httpx.Client,
    url: str,
    max_attempts: int,
    backoff_sec: float,
) -> None:
    """DELETE /jobs/{id} with retry on transient + 5xx; raise on 4xx (incl 409).

    Used by synthesize() to clear a poisoned 'failed' row before re-POSTing
    a fresh job (260610-chatterbox-poison). The server contract is:

      204 -> deleted (or already missing — idempotent)
      409 -> row is still queued/running; cannot delete. Caller must abort.
      4xx -> raise immediately
      5xx / transport -> retry up to ``max_attempts`` with linear backoff

    Returns None on success; raises RuntimeError on terminal failure. The
    retry budget mirrors POST's so a single transient blip during the
    DELETE doesn't kill the larger recovery attempt.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            resp = cli.delete(url)
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning(
                "chatterbox_delete_transport_error",
                attempt=attempt + 1, error=str(exc),
            )
            if attempt < max_attempts - 1:
                time.sleep(backoff_sec)
                continue
            raise RuntimeError(
                f"chatterbox-tts DELETE /jobs failed after {max_attempts} attempts: {exc}"
            ) from exc

        if resp.status_code == 204:
            return
        if resp.status_code == 409:
            # Concurrent client is actively running the same request_id.
            # Don't loop — there's another producer in flight; the caller
            # should abort the recovery attempt rather than racing.
            raise RuntimeError(
                f"chatterbox-tts DELETE /jobs rejected (409): {_extract_detail(resp)}"
            )
        if 400 <= resp.status_code < 500:
            raise RuntimeError(
                f"chatterbox-tts DELETE {resp.status_code}: {_extract_detail(resp)}"
            )
        # 5xx
        log.warning(
            "chatterbox_delete_5xx",
            attempt=attempt + 1,
            status=resp.status_code,
            body=resp.text[:200],
        )
        last_exc = RuntimeError(
            f"chatterbox-tts DELETE {resp.status_code}: {resp.text[:200]}"
        )
        if attempt < max_attempts - 1:
            time.sleep(backoff_sec)
    raise RuntimeError(
        f"chatterbox-tts DELETE /jobs failed after {max_attempts} attempts"
    ) from last_exc
