# RunPod chatterbox worker image (Phase 07-02).
#
# CUDA + torch cu124 + vendored chatterbox_tts + baked HF snapshot. There is NO
# prebuilt runpod/pytorch:2.6.0 tag (RESEARCH §3) — we assemble from nvidia/cuda
# + the cu124 torch wheels to match the service's torch==2.6.0 pin exactly.
#
# Build from the REPO ROOT (this Dockerfile COPYs services/chatterbox-tts/src):
#   docker buildx build --platform linux/amd64 \
#     -f services/chatterbox-runpod-worker/Dockerfile \
#     -t ghcr.io/spillwavesolutions/courseware-chatterbox-worker:<tag> --push .
# (see `task chatterbox-worker:push`). Apple Silicon builds this via QEMU which
# is slow for a torch image — GitHub Actions (native amd64) is the escape hatch
# (RESEARCH §Pitfall 7 / 00-RUNPOD-SETUP).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch FIRST from the cu124 index so the GPU wheels match the service's 2.6.0
# pin (chatterbox 0.1.6 was validated against torch 2.6).
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.6.0 torchaudio==2.6.0

# Vendor the chatterbox_tts source (COPY, NOT a uv workspace dep — the
# py3.11/py>=3.12 boundary makes a workspace dep awkward; RESEARCH Open-Q #3).
COPY services/chatterbox-tts/src /app/src
# The worker's own package.
COPY services/chatterbox-runpod-worker/src /app/worker

# Remaining runtime deps (torch already installed above so chatterbox-tts's
# torch pin resolves against the installed wheel — no second download).
RUN pip install --no-cache-dir \
    chatterbox-tts==0.1.6 "runpod>=1.9" "boto3>=1.37" "httpx>=0.27" "structlog>=24.1"

ENV HF_HOME=/app/.cache/huggingface
# Bake the HF snapshot at BUILD time so cold start is a container pull, not an HF
# download. EN-ONLY first cut: ChatterboxTTS (en) pulls ResembleAI/chatterbox.
# The multilingual model is a SEPARATE repo — NOT baked here; the first non-en
# job would pay a one-time HF download (documented in 07-02-SUMMARY.md). Verify
# the exact repo id at build against a local run's HF cache (RESEARCH Open-Q #1).
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('ResembleAI/chatterbox')"

# Both the vendored chatterbox_tts and the worker package must import.
ENV PYTHONPATH=/app/src:/app/worker
CMD ["python3", "/app/worker/chatterbox_runpod_worker/handler.py"]
