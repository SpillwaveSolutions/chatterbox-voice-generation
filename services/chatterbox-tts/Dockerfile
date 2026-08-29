FROM python:3.11-slim
# Multi-arch: builds on linux/amd64 AND linux/arm64 (Apple Silicon Docker Desktop).
# torch CPU wheels exist for both.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
# Install torch FIRST from CPU index (saves ~4 GB vs CUDA), then chatterbox.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    torch==2.6.0 torchaudio==2.6.0
COPY services/chatterbox-tts/pyproject.toml /app/pyproject.toml
COPY services/chatterbox-tts/src /app/src
RUN pip install --no-cache-dir -e .
ENV HF_HOME=/app/.cache/huggingface
ENV CHATTERBOX_VOICE_REFS_DIR=/app/voice_refs
ENV CHATTERBOX_OUTPUT_DIR=/app/output/tts_chatterbox
# quick-260602-b7l: SQLite job-state DB lives at /app/jobs/jobs.sqlite, mounted
# from the chatterbox_jobs named volume (see docker-compose.yml). The dir is
# auto-created by jobs.db.connect() on first boot but pre-creating here keeps
# the image self-documenting.
ENV CHATTERBOX_JOBS_DB_PATH=/app/jobs/jobs.sqlite
RUN mkdir -p /app/jobs
EXPOSE 8090
# --workers 1 is REQUIRED. The async job queue is in-process (asyncio.Queue +
# single worker task) and SQLite has a single writer (the worker). N workers
# would create N independent queues fighting over one SQLite file -- the FIFO
# invariant would be violated AND parallel synth would blow the 8 GiB cap
# (Chatterbox is CPU-only, can't parallelise). Do NOT change. See
# services/chatterbox-tts/src/chatterbox_tts/jobs/worker.py + RESEARCH §Pitfall 1.
CMD ["uvicorn", "chatterbox_tts.app:app", "--host", "0.0.0.0", "--port", "8090", "--workers", "1"]
