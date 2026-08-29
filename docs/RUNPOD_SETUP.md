# Chatterbox GPU offload (RunPod) — setup & operation

Why: Chatterbox CPU synthesis is ~66x slower than realtime and OOMs on full-length
lessons (a 9-min narration took hours and crashed the 8 GB container). Offloading synth
to a rented GPU makes long-form narration usable. The local CPU
container stays as the default/fallback. This is a **transport swap** — same
`synthesize()` contract; the only change is where synth runs and how artifacts move
(S3 presigned URLs instead of the shared `./output` mount).

Architecture (ported from the `notebook-llm-youtube-autoposter` phase 07 design):
```
caller (no GPU)                              RunPod worker (GPU, scale-to-zero)
  PUT voice ref -> S3 (content-hash key)       download voice ref (presigned GET)
  presign GET (ref) + PUT (out wav)            synth on cuda (reuses chatterbox_tts.model)
  POST /run {text, knobs, urls, request_id} -> PUT wav (presigned PUT)
  poll /status/{id} every 5s              <-   return {ok, duration_s, ...}
  download out wav -> output_path (byte-identical to the local backend)
```

What's CODE-COMPLETE in this repo:
- `artifact_store` (shipped inside the `tts-chatterbox` distribution) — S3-compatible
  wrapper (upload/download, content-hash dedupe, presign).
- `libs/tts_chatterbox/src/tts_chatterbox/runpod_client.py` + a `CHATTERBOX_BACKEND`
  dispatch in `client.py`.
- `services/chatterbox-runpod-worker/` — `handler.py` (reuses the local synth code on cuda) + `Dockerfile`.
- `model.py`'s `_load_model(language_id, device="cpu")` takes a device arg (default `cpu` keeps the
  local service unchanged); the worker pre-loads `device="cuda"` so the cached model is on the GPU.
- CI (`.github/workflows/build-images.yml`, tag-triggered) publishes the worker image to GHCR.
- Borrowed from `notebook-llm-youtube-autoposter` (phase 07) with its R2/Spaces 403 fixes.

## Backend: the RunPod GPU worker (`CHATTERBOX_BACKEND=runpod`)

There is one GPU backend — the **self-built worker** image with the full knob set and a
presigned object-storage handoff. It chunks long text server-side (reuses
`chatterbox_tts.model`), so it is the path for real full-length narrations. `local`
(default) stays the CPU fallback for short clips.

> A managed "Chatterbox Turbo" public endpoint was evaluated and **removed**: it silently
> caps usable output at ~500 chars / ~30s (a full lesson came back as a 0.2s stub), so it
> can't narrate long-form content. The worker backend is the only supported GPU path.

**Consumers REUSE a single shared worker endpoint** rather than deploying a duplicate per
project. The worker image is identical for every consumer (all builds come from
`services/chatterbox-runpod-worker`, which vendors `chatterbox_tts`'s source), so one
deployed endpoint serves all of them. **Normal operation:** you only need the existing
endpoint id + an API key for the account that owns it + the R2 artifact bus — do sections
**1** (API key) and **3** (R2), set `RUNPOD_CHATTERBOX_ENDPOINT_ID` to the shared
endpoint, and `CHATTERBOX_BACKEND=runpod`. Sections **2, 4, 5** are only for the rare case
of (re)building/redeploying the shared worker image.

## Operator setup (one-time — done in web consoles + shell)

For normal operation do **1** and **3**, then set `RUNPOD_CHATTERBOX_ENDPOINT_ID` to the
shared worker endpoint and `CHATTERBOX_BACKEND=runpod`. Sections **2, 4, 5** are only for
(re)building/redeploying the shared worker image.

### 1. RunPod
1. Account at https://www.runpod.io, add credit ($10–25), **set a spend limit** (e.g. $50/mo).
2. API key: Settings → API Keys → Create (Serverless run/read). → `RUNPOD_API_KEY`. Use a
   key for the account that owns the shared endpoint.

### 2. Container registry (GHCR) — only when (re)deploying a PRIVATE worker image
Only needed if the GHCR package is private (skip entirely for a public package):
1. GitHub PAT (classic) with `read:packages` (or `write:packages` if pushing locally).
2. `echo $GH_PAT | docker login ghcr.io -u <owner> --password-stdin`
3. RunPod console → Settings → Container Registry Auth → Add (GHCR user + PAT).

### 3. Object storage (Cloudflare R2 recommended — zero egress)
1. Create a bucket for transient artifacts (e.g. `gpu-artifacts`).
2. R2 → Manage API Tokens → Create: **Account API token**, **Object Read & Write**,
   scoped to that bucket. Copy Access Key ID + Secret (shown once) + the S3 endpoint
   `https://<accountid>.r2.cloudflarestorage.com`.
3. Add a lifecycle rule: expire objects after 7 days (artifacts are transient).
4. Into your caller's `.env` — these live **only on the caller side**. Do NOT put them on
   the RunPod endpoint: the worker uses the presigned URLs passed in each job and never
   builds an S3 client.
   ```
   ARTIFACT_S3_ENDPOINT=https://<accountid>.r2.cloudflarestorage.com
   ARTIFACT_S3_BUCKET=<bucket>
   ARTIFACT_S3_ACCESS_KEY=...
   ARTIFACT_S3_SECRET_KEY=...
   ARTIFACT_S3_REGION=auto
   ```

### 4. Build + push the worker image (amd64 — RunPod GPUs are x86_64) — only when (re)deploying
**Preferred — GitHub Actions (native amd64, no QEMU):** push a `v*` tag; the
`build-images.yml` workflow builds and pushes
`ghcr.io/spillwavesolutions/chatterbox-runpod-worker:<tag>` using the built-in
`GITHUB_TOKEN` (no PAT needed):
```
git tag v0.1.0 && git push origin v0.1.0
```

**Local fallback (slow QEMU build on Apple Silicon):**
```
echo $GH_PAT | docker login ghcr.io -u <owner> --password-stdin
docker buildx build --platform linux/amd64 \
  -f services/chatterbox-runpod-worker/Dockerfile \
  -t ghcr.io/spillwavesolutions/chatterbox-runpod-worker:<tag> --push .
```
(Build from the repo ROOT — the Dockerfile COPYs `services/chatterbox-tts/src`.)

### 5. Create the serverless endpoint — only when (re)deploying (else reuse the shared one)
Console → Serverless → New Endpoint:
| Setting | Value |
|---|---|
| Worker image | `ghcr.io/spillwavesolutions/chatterbox-runpod-worker:<tag>` |
| GPU pool | 24GB: RTX 4090 / A5000 / L4 |
| Min / Max workers | 0 / 1 |
| Idle timeout | 60s · FlashBoot ON |
| Execution timeout | 1800s (the client also sends `policy.executionTimeout` per job) |
| Env vars | **none** — the worker gets everything it needs (presigned URLs) in each job |

Record the endpoint id (in the URL `…/v2/<ENDPOINT_ID>/…`) → `RUNPOD_CHATTERBOX_ENDPOINT_ID`.

### 6. Smoke test
```
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" \
  https://api.runpod.ai/v2/$RUNPOD_CHATTERBOX_ENDPOINT_ID/health
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"smoke_test": true}}' \
  https://api.runpod.ai/v2/$RUNPOD_CHATTERBOX_ENDPOINT_ID/run
# poll /status/<id> until COMPLETED
```

## Switch a consumer pipeline to GPU
In the caller's `.env` set `CHATTERBOX_BACKEND=runpod`,
`RUNPOD_CHATTERBOX_ENDPOINT_ID=<shared endpoint>`, `RUNPOD_API_KEY`, and the
`ARTIFACT_S3_*` keys, then restart the caller so it picks up the new env. Per-segment
synthesis means N jobs per narration, so prefer the RunPod GPU backend for multi-segment
work; the CPU service serializes through one worker (see docs/CONSUMPTION.md §4).

Fallback any time: `CHATTERBOX_BACKEND=local` → the CPU jobs-API container (fine for
short clips).

Cost: a 4090 runs Chatterbox ~2–5x realtime → a 10-min narration ≈ $0.05–0.15/job; $0 idle.
