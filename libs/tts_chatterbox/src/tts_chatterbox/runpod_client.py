"""RunPod serverless transport for tts_chatterbox (phase 07).

Mirrors the local POST->poll->fetch client (client.py) but talks to the RunPod
serverless API and moves artifacts over presigned object-storage URLs (via
libs/artifact_store) instead of the cross-container ``./output:/app/output``
shared mount (which does not exist across clouds).

Wire shape (07-RESEARCH §"Pattern 2"):
    POST  https://api.runpod.ai/v2/{endpoint_id}/run
        headers: Authorization: Bearer {api_key}, Content-Type: application/json
        body:    {"input": {text, voice_ref_url?, ...knobs..., output_put_url,
                            request_id}, "policy": {"executionTimeout": 1800000}}
        -> {"id": "<job_id>", "status": "IN_QUEUE"}
    GET   https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}   (same auth)
        status in {IN_QUEUE, IN_PROGRESS, RUNNING}  -> keep polling (5s)
        COMPLETED                                   -> download via artifact_store
        {FAILED, CANCELLED, TIMED_OUT}              -> RuntimeError(worker error)

Differences vs the local client (intentional):
    - Artifacts move via artifact_store presigned GET/PUT, not shutil.copyfile.
    - The voice ref uploads ONCE (content-hash dedupe in ensure_uploaded) and is
      passed as a presigned ``voice_ref_url`` in the job input.
    - The POST body carries ``policy.executionTimeout=1800000`` (override RunPod's
      10-min default — Pitfall 3).
    - The default poll budget is HIGHER than local (240 × 5s = 20 min) to cover
      RunPod queue wait + first-pull cold start.
    - NO poison-row DELETE+re-POST recovery is ported: RunPod has no
      request_id-keyed server-side idempotency, so a fresh POST is always a fresh
      job (RESEARCH §Anti-Patterns). There is no DELETE path at all.

``synthesize()`` keeps the local client's public signature and contract: it
returns a local ``output_path`` (byte-identical handoff to the rest of the
pipeline).
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Optional

import httpx
import structlog

from artifact_store import ArtifactStore, ArtifactStoreConfig
from tts_chatterbox.config import ChatterboxConfig

log = structlog.get_logger()

# Short read/connect timeouts — every HTTP call is an enqueue (POST /run) or a
# state lookup (GET /status/{id}); the synth runs in the background on the GPU
# worker and we poll.
SHORT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
POLL_INTERVAL_SEC = 5.0

# RunPod job-status state map (Pitfall 2). Anything not in TERMINAL_ERROR and
# not COMPLETED is treated as "keep polling" (default-branch unknown states).
NON_TERMINAL = {"IN_QUEUE", "IN_PROGRESS", "RUNNING"}
TERMINAL_ERROR = {"FAILED", "CANCELLED", "TIMED_OUT"}
SUCCESS = "COMPLETED"

# Poll-side GET retry budget — default HIGHER than the local client (60) to
# cover RunPod queue wait + cold first-pull. 240 × 5s = 20 min. The SAME env
# override as the local client is reused (kwarg > env > default).
POLL_MAX_ATTEMPTS_DEFAULT = 240
POLL_MAX_ATTEMPTS_ENV = "CHATTERBOX_TTS_POLL_MAX_ATTEMPTS"

# Where the bare-filename voice ref resolves on the flow-runner. The flow-runner
# mounts ``./voices:/app/voice_refs:ro``, so a bare ``narrator.wav`` lives at
# ``/app/voice_refs/narrator.wav`` (Pitfall 6). Overridable for tests/non-Docker.
VOICE_REFS_DIR_ENV = "CHATTERBOX_VOICE_REFS_DIR"
VOICE_REFS_DIR_DEFAULT = "/app/voice_refs"


def synthesize(
    text: str,
    config: ChatterboxConfig,
    output_path: str,
    request_id: Optional[str] = None,
    max_attempts: int = 3,
    backoff_sec: float = 5.0,
    poll_max_attempts: Optional[int] = None,
) -> str:
    """POST a RunPod job, poll until terminal, download the WAV to ``output_path``.

    Same contract as ``tts_chatterbox.client.synthesize``: returns a local
    ``output_path`` on success.

    Args:
        text: Narration text to synthesise.
        config: ChatterboxConfig — supplies the synth knobs AND the transport
            fields ``runpod_endpoint_id`` / ``runpod_api_key``.
        output_path: Local path the WAV is downloaded to from object storage.
        request_id: Job correlation id (traceability only — RunPod has no
            server-side idempotency). Defaults to a uuid4 hex if missing.
        max_attempts: POST-side retry budget (default 3 — fast-fail; Prefect
            retries the whole task).
        backoff_sec: Linear backoff between retries (default 5s).
        poll_max_attempts: GET/poll-side budget. Resolution: explicit kwarg >
            env ``CHATTERBOX_TTS_POLL_MAX_ATTEMPTS`` > module default (240).

    Returns:
        ``output_path`` on success.

    Raises:
        ValueError: if ``text`` is empty or whitespace-only.
        RuntimeError: on terminal-error status (FAILED/CANCELLED/TIMED_OUT),
            on POST exhaustion after ``max_attempts``, or on poll exhaustion
            after the effective poll budget.
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")

    rid = request_id or uuid.uuid4().hex
    endpoint = config.runpod_endpoint_id
    headers = {
        "Authorization": f"Bearer {config.runpod_api_key}",
        "Content-Type": "application/json",
    }
    run_url = f"https://api.runpod.ai/v2/{endpoint}/run"
    status_url = f"https://api.runpod.ai/v2/{endpoint}/status"

    store = ArtifactStore(ArtifactStoreConfig.from_env())

    # 1. Voice ref: resolve the bare filename to a local path under the mounted
    #    voice-refs dir, upload once (content-hash dedupe), presign a GET URL.
    voice_ref_url: Optional[str] = None
    if config.voice_ref:
        refs_dir = os.environ.get(VOICE_REFS_DIR_ENV, VOICE_REFS_DIR_DEFAULT)
        local_ref_path = str(Path(refs_dir) / config.voice_ref)
        ref_key = store.ensure_uploaded(local_ref_path)
        voice_ref_url = store.presign_get(ref_key)

    # 2. Reserve an output key and presign a PUT URL the worker uploads to.
    output_key = f"output/{rid}.wav"
    output_put_url = store.presign_put(output_key)

    # 3. Build the job input + policy (executionTimeout override — Pitfall 3).
    body = {
        "input": {
            "text": text,
            "voice_ref_url": voice_ref_url,
            "exaggeration": config.exaggeration,
            "cfg_weight": config.cfg_weight,
            "temperature": config.temperature,
            "language_id": config.language_id,
            "seed": config.seed,
            "sentence_chunk_size": config.sentence_chunk_size,
            "pronunciations": config.pronunciations,
            "output_put_url": output_put_url,
            "request_id": rid,
        },
        "policy": {"executionTimeout": 1800000},
    }

    effective_poll_max = (
        poll_max_attempts
        if poll_max_attempts is not None
        else int(os.environ.get(POLL_MAX_ATTEMPTS_ENV, POLL_MAX_ATTEMPTS_DEFAULT))
    )

    log.info("runpod_post_run", request_id=rid, endpoint=endpoint)
    with httpx.Client(timeout=SHORT_TIMEOUT) as cli:
        # 4. POST once (no DELETE re-POST dance — RunPod has no idempotency).
        run_resp = _post_with_retry(cli, run_url, body, headers, max_attempts, backoff_sec)
        job_id = run_resp.get("id")
        if not job_id:
            raise RuntimeError(f"runpod POST /run returned no job id: {run_resp}")

        # 5. Poll until terminal.
        status = run_resp.get("status", "IN_QUEUE")
        resp = run_resp
        attempts = 0
        job_status_url = f"{status_url}/{job_id}"
        while status not in TERMINAL_ERROR and status != SUCCESS:
            # Non-terminal AND unknown states both keep polling (capped budget).
            time.sleep(POLL_INTERVAL_SEC)
            resp = _get_with_retry(
                cli, job_status_url, headers, effective_poll_max, backoff_sec
            )
            status = resp.get("status", "")

        # 6. Terminal-error states -> RuntimeError with worker detail.
        if status in TERMINAL_ERROR:
            detail = resp.get("error") or (resp.get("output") or {}).get("error")
            raise RuntimeError(f"runpod job {status}: {detail}")

        # 7. COMPLETED -> download the WAV from object storage (NOT from the
        #    response body — the worker wrote it to output_put_url).
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        store.download_file(output_key, output_path)
        log.info("runpod_synthesize_ok", request_id=rid, job_id=job_id)
        return output_path


# ---------------------------------------------------------------------------
# Retry helpers (mirror client.py posture: POST fast-fails, GET rides it out)
# ---------------------------------------------------------------------------

def _post_with_retry(
    cli: httpx.Client,
    url: str,
    body: dict,
    headers: dict,
    max_attempts: int,
    backoff_sec: float,
) -> dict:
    """POST with retry on 5xx + transport errors; 4xx raises immediately."""
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            resp = cli.post(url, json=body, headers=headers)
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning("runpod_post_transport_error", attempt=attempt + 1, error=str(exc))
            if attempt < max_attempts - 1:
                time.sleep(backoff_sec)
                continue
            raise RuntimeError(
                f"runpod POST /run failed after {max_attempts} attempts: {exc}"
            ) from exc

        if resp.status_code in (200, 201):
            return resp.json()
        if 400 <= resp.status_code < 500:
            raise RuntimeError(
                f"runpod POST /run {resp.status_code}: {_extract_detail(resp)}"
            )
        # 5xx
        log.warning(
            "runpod_post_5xx", attempt=attempt + 1,
            status=resp.status_code, body=resp.text[:200],
        )
        last_exc = RuntimeError(f"runpod {resp.status_code}: {resp.text[:200]}")
        if attempt < max_attempts - 1:
            time.sleep(backoff_sec)
    raise RuntimeError(
        f"runpod POST /run failed after {max_attempts} attempts"
    ) from last_exc


def _get_with_retry(
    cli: httpx.Client,
    url: str,
    headers: dict,
    max_attempts: int,
    backoff_sec: float,
) -> dict:
    """GET /status with ride-out: transient httpx errors retry with backoff.

    Tolerates queue-wait + cold-start network blips mid-poll. 4xx raises
    immediately.
    """
    last_exc: Optional[BaseException] = None
    for attempt in range(max_attempts):
        try:
            resp = cli.get(url, headers=headers)
        except httpx.HTTPError as exc:
            last_exc = exc
            log.warning("runpod_poll_transport_error", attempt=attempt + 1, error=str(exc))
            if attempt < max_attempts - 1:
                time.sleep(backoff_sec)
                continue
            raise RuntimeError(
                f"runpod GET /status failed after {max_attempts} attempts: {exc}"
            ) from exc

        if resp.status_code == 200:
            return resp.json()
        if 400 <= resp.status_code < 500:
            raise RuntimeError(
                f"runpod GET /status {resp.status_code}: {_extract_detail(resp)}"
            )
        # 5xx
        log.warning(
            "runpod_poll_5xx", attempt=attempt + 1,
            status=resp.status_code, body=resp.text[:200],
        )
        last_exc = RuntimeError(f"runpod {resp.status_code}: {resp.text[:200]}")
        if attempt < max_attempts - 1:
            time.sleep(backoff_sec)
    raise RuntimeError(
        f"runpod GET /status failed after {max_attempts} attempts"
    ) from last_exc


def _extract_detail(resp: httpx.Response) -> str:
    """Pull JSON ``error``/``detail``, falling back to a body snippet."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            for key in ("error", "detail"):
                if key in data:
                    return str(data[key])
        return resp.text[:200]
    except Exception:
        return resp.text[:200]
