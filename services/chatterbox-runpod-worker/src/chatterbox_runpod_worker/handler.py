"""RunPod serverless handler — GPU Chatterbox TTS synthesis (Phase 07-02).

This worker REUSES the existing ``chatterbox_tts`` synth code on a GPU. It
bypasses the local FastAPI app, so it must replicate two app-level
responsibilities itself (RESEARCH §Pitfall 4,5):

  1. device=cuda — the local ``_load_model`` defaults to cpu. The worker forces
     the resident model onto cuda by calling ``_load_model(language_id,
     device="cuda")`` BEFORE ``_synthesize_sync`` (which calls ``_load_model``
     with no device arg and reuses the module-level cache).
  2. Pronunciations — ``app.py::create_job`` applies them before synth; the
     worker replicates that via ``compile_pronunciations`` + ``apply_pronunciations``.

Artifact flow (RESEARCH §Anti-Patterns): the voice ref arrives via a presigned
GET URL; the output WAV is PUT to a presigned URL — NEVER returned inline or
b64-encoded (RunPod ~10MB payload cap). Synth failures RAISE (RunPod marks the
job FAILED) — the handler never swallows them into a graceful error-dict return.

Local test (no GPU): ``python handler.py --rp_serve_api`` serves a FastAPI app on
:8000 exposing /run /runsync /status/{id} (00-RUNPOD-SETUP §7).

Job input contract (job["input"]):
    text, voice_ref_url?, exaggeration, cfg_weight, temperature, language_id,
    seed?, sentence_chunk_size, pronunciations, output_put_url, request_id,
    smoke_test?
Handler return (becomes the RunPod job ``output``):
    {"ok": True, "duration_s", "chars", "elapsed_s"}
    smoke_test -> {"ok": True, "duration_s", "smoke": True, "elapsed_s"}
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import httpx
import runpod

from chatterbox_tts.model import _load_model, _synthesize_sync
from chatterbox_tts.pronunciations import (
    apply_pronunciations,
    compile_pronunciations,
)


def handler(job):
    """RunPod job handler. Synth on cuda, PUT result to the presigned URL.

    Raises on any failure (download, synth, upload) so RunPod surfaces the job
    as FAILED with the exception detail — we deliberately do NOT swallow
    exceptions into a graceful error-dict return.
    """
    inp = job["input"]
    t0 = time.time()

    # Force cuda: pre-load the resident model on cuda so the no-device
    # _load_model call inside _synthesize_sync reuses the cuda model (cache hit).
    _load_model(inp.get("language_id", "en"), device="cuda")

    # Pronunciations are applied in the FastAPI app, NOT in model.py — replicate
    # it here (the worker bypasses the app). RESEARCH §Pitfall 5.
    text = inp["text"]
    if inp.get("pronunciations"):
        text = apply_pronunciations(
            text, compile_pronunciations(inp["pronunciations"])
        )

    if inp.get("smoke_test"):
        # One short sentence, skip upload, return timing only.
        out = Path(tempfile.mktemp(suffix=".wav"))
        _, dur = _synthesize_sync(
            "Hello from RunPod.",
            None,
            inp.get("exaggeration", 0.5),
            inp.get("cfg_weight", 0.5),
            inp.get("temperature", 0.8),
            inp.get("language_id", "en"),
            inp.get("seed"),
            inp.get("sentence_chunk_size", 3),
            out,
        )
        return {
            "ok": True,
            "duration_s": dur,
            "smoke": True,
            "elapsed_s": time.time() - t0,
        }

    # Download the voice ref via presigned GET (None -> default voice, no GET).
    ref_path = None
    if inp.get("voice_ref_url"):
        ref_path = tempfile.mktemp(suffix=".wav")
        with httpx.Client(timeout=120) as c:
            r = c.get(inp["voice_ref_url"])
            r.raise_for_status()
            Path(ref_path).write_bytes(r.content)

    out = Path(tempfile.mktemp(suffix=".wav"))
    _, dur = _synthesize_sync(
        text,
        ref_path,
        inp["exaggeration"],
        inp["cfg_weight"],
        inp["temperature"],
        inp["language_id"],
        inp.get("seed"),
        inp["sentence_chunk_size"],
        out,
    )

    # Output ALWAYS via presigned PUT — NEVER inline/b64 (RunPod ~10MB cap).
    # No extra Content-Type header (RESEARCH §Pitfall 1: unsigned content-type 403s).
    with httpx.Client(timeout=300) as c:
        resp = c.put(inp["output_put_url"], content=out.read_bytes())
        resp.raise_for_status()

    return {
        "ok": True,
        "duration_s": dur,
        "chars": len(text),
        "elapsed_s": time.time() - t0,
    }


# runpod.serverless.start is the SDK entrypoint. Guarded so unit tests can import
# this module (to patch handler internals) without the SDK launching a server.
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
