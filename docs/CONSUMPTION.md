# Consumption contract

This document is the complete adoption contract for the `tts-chatterbox` library and its two
services. It is written for a consumer pipeline that has never seen this repository's origin
project: everything you need to install, configure, and call the library is stated here, with
the numeric boundaries the implementation actually enforces.

**Consent notice: voice references are biometric data.** A voice-reference WAV is a cloneable
biometric identifier of a real person. You must have the speaker's explicit permission before
cloning their voice with this library. Never commit voice references to a repository, never
bake them into images, and never treat this library as a consent-free voice-cloning tool.

## Install

One line, one distribution, two import packages (`tts_chatterbox` and `artifact_store`):

```toml
# pyproject.toml — [tool.uv.sources]
tts-chatterbox = { git = "https://github.com/SpillwaveSolutions/chatterbox-voice-generation", subdirectory = "libs/tts_chatterbox", tag = "v0.1.0" }
```

with `tts-chatterbox` in your dependencies. Pin by tag; your `uv.lock` records the git URL and
rev, which shadows the unrelated `tts-chatterbox` name on PyPI (do NOT install from PyPI — a
squatter package exists under this name). Commit the lock.

## 1. Python API

```python
from tts_chatterbox.client import synthesize
from tts_chatterbox.config import ChatterboxConfig

synthesize(
    text,                     # str — narration text
    config,                   # ChatterboxConfig
    output_path,              # str — where the WAV lands locally
    request_id=None,          # str — ALWAYS pass one (see §6); uuid4 hex if omitted
    base_url=None,            # local backend only — overrides CHATTERBOX_BASE_URL
    max_attempts=3,           # POST-side retry budget
    backoff_sec=5.0,          # linear backoff between retries
    poll_max_attempts=None,   # poll-side budget; None -> env -> backend default
) -> str                      # returns output_path on success
```

**Errors:**

- `ValueError` — empty or whitespace-only `text`; unknown non-empty `CHATTERBOX_BACKEND`.
- `RuntimeError` — 4xx from POST or GET (immediate, with server detail); GET 404 (job pruned
  or never created); terminal `failed` state (with the server's error message); retry
  exhaustion after `max_attempts` (POST side) or the effective poll budget (GET side). On the
  RunPod backend: terminal `FAILED`/`CANCELLED`/`TIMED_OUT` status.

**`ChatterboxConfig` fields and defaults:**

| Field | Default | Notes |
|---|---|---|
| `voice_ref` | `None` | bare filename, resolved server/client-side (§8) |
| `exaggeration` | `0.5` | |
| `cfg_weight` | `0.5` | |
| `temperature` | `0.8` | |
| `device` | `"cpu"` | the GPU worker overrides to cuda internally |
| `language_id` | `"en"` | |
| `seed` | `None` | set it for reproducibility (§9) |
| `sentence_chunk_size` | `3` | |
| `pronunciations` | `{}` | curated per-request map only (§10) |
| `backend` | env `CHATTERBOX_BACKEND` or `"local"` | transport field |
| `runpod_endpoint_id` | env `RUNPOD_CHATTERBOX_ENDPOINT_ID` | transport field |
| `runpod_api_key` | env `RUNPOD_API_KEY` | transport field |

The three transport fields (`backend`, `runpod_endpoint_id`, `runpod_api_key`) are read from
the environment at construct time and are deliberately **absent from `to_request_body()`** —
the local `POST /jobs` body is byte-identical regardless of transport configuration.

## 2. Backend dispatch

Precedence: **`CHATTERBOX_BACKEND` env > `config.backend` > `"local"`**.

- Any unknown non-empty value raises `ValueError` (fail loud, never silently local).
- An empty string falls through to the local backend.
- The runpod transport import is lazy: the local path never imports `artifact_store`/boto3.

## 3. Retry budgets (POST vs poll — deliberately split)

| Budget | Value | Rationale |
|---|---|---|
| POST retries (`max_attempts`) | **3** | fast-fail so your orchestrator's task-level retry owns the gap |
| Poll retries, local backend | **60** (× 5s = 5 min) | rides out the CPU container's ~3.5 min cold start |
| Poll retries, runpod backend | **240** (× 5s = 20 min) | covers RunPod queue wait + cold image pull |
| Poll interval | **5s** | fixed |

Poll budget resolution: explicit `poll_max_attempts` kwarg > env
`CHATTERBOX_TTS_POLL_MAX_ATTEMPTS` > backend default (60 local / 240 runpod).

## 4. Local backend: the jobs HTTP API

The CPU service (`ghcr.io/spillwavesolutions/chatterbox-tts`, port 8090) exposes:

- `GET /health` — `{status, model_loaded, language_id}`
- `POST /jobs` — idempotent enqueue keyed by `request_id`: 200 returns the existing row, 201
  inserts a new one
- `GET /jobs/{id}` — job state, or 404 if pruned/unknown
- `DELETE /jobs/{id}` — 204 (idempotent on terminal/missing); **409** if state is
  queued/running (a concurrent producer owns the row — abort, do not race)

**Poison-row recovery (local only):** if `POST /jobs` returns a terminal `failed` row (the
idempotency layer replaying a crashed prior attempt), the client auto-DELETEs and re-POSTs
exactly once. A second `failed` surfaces as `RuntimeError`.

**Single-worker invariant (part of the contract):** the service MUST run with `--workers 1`.
The job queue is in-process (asyncio) and SQLite has a single writer; N workers would create N
queues fighting over one database and violate FIFO. Consequence: **the local CPU service
serializes all jobs through one worker** — parallel throughput requires the runpod backend.

Client base URL: `base_url` kwarg > env `CHATTERBOX_BASE_URL` > `http://chatterbox-tts:8090`.

## 5. The `wav_path` KEY LINK (local transport)

On `completed`, `GET /jobs/{id}` returns `wav_path` — an absolute path **inside the service
container** (`/app/output/tts_chatterbox/<request_id>.wav` under `CHATTERBOX_OUTPUT_DIR`). The
client copies it to your `output_path` via the filesystem. This only works because both your
caller container and the service mount the same host directory identically (e.g.
`./output:/app/output` on both). **The shared output mount contract IS the local transport
API** — without an identical mount, a healthy 200 is followed by `FileNotFoundError` at copy
time.

## 6. `request_id` semantics

`request_id` is simultaneously: the server-side row primary key, the output filename
(`<request_id>.wav`), and the log correlation id. It **MUST be unique per synthesis call**.
Failure mode of sharing one id: the idempotency layer returns the FIRST job's WAV for every
subsequent call with that id — silently, with no error. Always generate a fresh id per call.

## 7. RunPod backend (GPU)

Set `CHATTERBOX_BACKEND=runpod` plus `RUNPOD_CHATTERBOX_ENDPOINT_ID` and `RUNPOD_API_KEY`.
The transport mirrors POST-then-poll but over RunPod's serverless API:

- `POST https://api.runpod.ai/v2/{endpoint}/run` with
  `policy.executionTimeout=1800000` (30 min hard cap per job, overriding RunPod's 10-min
  default).
- Poll `GET /status/{job_id}` every 5s. State map: `IN_QUEUE`/`IN_PROGRESS`/`RUNNING` keep
  polling; `COMPLETED` fetches the artifact; `FAILED`/`CANCELLED`/`TIMED_OUT` raise.
- **Artifacts always move via presigned URLs, never inline** (RunPod payloads cap ~10MB):
  - the voice ref uploads once to your S3-compatible bucket under a content-hash key
    (`ensure_uploaded` dedupes via HEAD), passed as a presigned GET `voice_ref_url`;
  - the output WAV is uploaded by the worker to a presigned PUT `output_put_url`
    (`output/<request_id>.wav`), then downloaded to your `output_path`.
- Presigned URL TTL is **7200s** (2h) — deliberately outliving the 1800s execution timeout
  plus queue wait.
- **No poison-row recovery, by design**: RunPod has no request_id-keyed server-side
  idempotency, so every POST is a fresh job and there is no DELETE path.

**`ARTIFACT_S3_*` env contract** (caller-side only — the worker receives presigned URLs and
never builds an S3 client):

```
ARTIFACT_S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com
ARTIFACT_S3_BUCKET=<bucket>
ARTIFACT_S3_ACCESS_KEY=...
ARTIFACT_S3_SECRET_KEY=...
ARTIFACT_S3_REGION=auto        # "auto" is the correct value for Cloudflare R2
```

The boto3 client is built with a checksum-disabling `Config`
(`request_checksum_calculation="when_required"`, `response_checksum_validation="when_required"`,
`signature_version="s3v4"`). This is CRITICAL: since boto3 1.36, presigned PUTs otherwise add
an `x-amz-checksum-crc32` trailer that R2/Spaces reject with `403 SignatureDoesNotMatch`.

## 8. Voice references travel by reference only

`config.voice_ref` is a **bare filename**, never a path and never file content:

- Local backend: the service resolves it under its `CHATTERBOX_VOICE_REFS_DIR`
  (default `/app/voice_refs`) — mount your voice-refs directory there, read-only.
- RunPod backend: the client resolves the same bare filename under the caller-side
  `CHATTERBOX_VOICE_REFS_DIR`, uploads it once (content-hash dedupe), and passes a presigned
  URL.

Voice references are never bundled in this repository or in the published images (see the
consent notice above).

## 9. Reproducibility

Identical `text` + `config` + `seed` on the **same backend and same image version** produces
**byte-identical WAV output**. Consumer determinism gates can rely on this. The pins that make
it hold are frozen: `chatterbox-tts==0.1.6` and `torch==2.6.0` (CPU wheels in the service
image, cu124 wheels in the worker image). Do not bump either without re-validating seed
determinism. Note: output always carries Resemble's Perth watermark (not opt-out-able).

## 10. Pronunciations wire rule

The server engine applies `pronunciations` with `re.IGNORECASE` and **no word boundaries** —
a short acronym entry like `"AI" -> "ay eye"` corrupts ordinary words mid-word ("maintain"
becomes "m-ay-eye-ntain"). Therefore:

- Send only small, curated, per-request maps that are safe under substring matching.
- Do word-bounded lexicon work client-side, before calling `synthesize`.
- Server-enforced budget: at most **1000** entries, at most **200** chars per key/value
  (`ValueError`/422 beyond that).

## 11. What the library does NOT do (consumer-owned boundary)

The following are your pipeline's job, not the library's:

- Orchestration and task-level retries (the library fast-fails POSTs on purpose).
- Loudness normalization (loudnorm) and any mastering.
- Silence holds / padding between narration segments.
- Lexicon policy (word-bounded pronunciation expansion — see §10).
- Audio/video assembly of any kind.

The library's contract ends at: text + config + seed in, one WAV at `output_path` out.
