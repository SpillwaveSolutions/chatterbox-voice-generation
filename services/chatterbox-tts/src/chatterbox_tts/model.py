"""Chatterbox model loader + synth wrapper.

Mirrors deck2video/tts.py one-to-one for the loader semantics:
- torch.load monkey-patch in the multilingual branch (RESEARCH §3): the
  ChatterboxMultilingualTTS deserialiser doesn't honour ``device='cpu'`` and
  tries to map_location to CUDA on CPU-only hosts. We patch torch.load for
  the duration of from_pretrained to force the correct map_location.
- ``ChatterboxTTS`` (English) vs ``ChatterboxMultilingualTTS`` (everything
  else). Constrained-VM policy: only one model resident at a time. Switching
  language evicts the prior model (RESEARCH §7, Section 14 Q2).
- 24 kHz model output is resampled IN-SERVICE to 44.1 kHz s16 mono WAV so
  the service contract is a stable downstream-friendly format.

Serialisation invariant (quick-260602-b7l): the old module-level ``asyncio.Lock``
was DELETED. The new async-job-queue architecture serialises through a single
in-process worker task (``chatterbox_tts.jobs.worker.run_worker``) consuming a
single ``asyncio.Queue``. The worker is the ONLY caller of ``synthesize_to_wav``,
so there is at most one synth in flight by construction. uvicorn ``--workers 1``
makes the in-process single-consumer property total. Adding a lock back is a
no-op at best and a deadlock risk at worst — leave it out.

OOM detection (M5): extends deck2video's GPU-string check to also catch
``MemoryError`` (cgroup OOM-kill on CPU). On OOM we raise
``OutOfMemoryError`` so the worker writes ``state='failed', error='oom'``.
The flow-runner client surfaces that as a clean RuntimeError; the outer
flow marks the import 'error'. We do NOT substitute a silent WAV.
"""
from __future__ import annotations

import asyncio
import gc
import re
from pathlib import Path
from typing import Optional


# Module-level state. With ``uvicorn --workers 1`` and the single-consumer
# async-job-queue worker, this is single-process / single-flight by construction.
_model = None
_current_language: str = "en"
_state = {"model_loaded": False, "language_id": "en"}


class OutOfMemoryError(RuntimeError):
    """OOM signal — raised by synth so the worker writes state='failed'.

    Subclass of RuntimeError so a generic ``except RuntimeError`` in callers
    still catches it; the dedicated type lets the worker emit
    ``error_message='oom'`` (NOT a silent WAV — the flow-runner needs to mark
    the import 'error').
    """


def _is_oom(exc: BaseException) -> bool:
    """Recognise both Python-level MemoryError and torch's GPU-OOM strings."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "mps backend out of memory" in msg


def _load_model(language_id: str = "en", device: str = "cpu"):
    """Load (and cache) the Chatterbox model for ``language_id`` on ``device``.

    ``device`` defaults to ``"cpu"`` so the local CPU service (app.py eager-load
    ``_load_model("en")`` and ``_synthesize_sync`` which calls ``_load_model``
    with no device arg) is byte-equivalent. The RunPod GPU worker
    (services/chatterbox-runpod-worker) forces the resident model onto cuda by
    calling ``_load_model(language_id, device="cuda")`` ITSELF before invoking
    ``_synthesize_sync`` — the module-level cache means that prior cuda load is
    reused by the no-device ``_synthesize_sync`` call.

    Eviction policy: if a different model is currently loaded, drop the
    reference, gc, then load the requested one. First request after a
    language switch takes 30-90s on CPU (operator-aware via SUMMARY.md).
    """
    import torch

    global _model, _current_language
    if _model is not None and _current_language == language_id:
        return _model

    if _model is not None:
        # Eviction on language switch — constrained VM can't hold both.
        del _model
        _model = None
        gc.collect()

    # torch.load monkey-patch (deck2video tts.py 153-162; RESEARCH §3).
    # ChatterboxMultilingualTTS.from_pretrained ignores device='cpu' when
    # deserialising checkpoints and may try to map to CUDA; force the requested
    # device (cpu default keeps the local service identical; cuda for the worker).
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("map_location", device)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        if language_id == "en":
            from chatterbox.tts import ChatterboxTTS

            _model = ChatterboxTTS.from_pretrained(device=device)
        else:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS

            _model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    finally:
        torch.load = original_load

    _current_language = language_id
    _state["model_loaded"] = True
    _state["language_id"] = language_id
    return _model


def _chunk_sentences(text: str, n: int) -> list[str]:
    """Group sentences ``n``-at-a-time for chunked synthesis.

    Mirrors deck2video/tts.py _split_sentences + the 3-sentence-group join.
    Falls back to the whole text as one chunk if there are no sentence
    delimiters at all (still produces output rather than an empty list).

    Trailing-remainder merge (chatterbox-tick-jitter-260611): when the last
    group is a short leftover (a single fragment, OR n>1 with strictly fewer
    than n sentences), Chatterbox tends to babble/warble on the isolated short
    final chunk — perceived as the audio "getting worse at the end". We fold
    that remainder into the PREVIOUS group so the final chunk always has enough
    context. Only applies when there are 2+ groups (nothing to merge into
    otherwise).
    """
    parts = [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p]
    if not parts:
        return [text]
    groups = [" ".join(parts[i : i + n]) for i in range(0, len(parts), n)]
    remainder = len(parts) % n
    # A short trailing remainder (1..n-1 sentences) synthesized alone is the
    # warble risk. Merge it into the prior group. (remainder==0 -> last group
    # is already full; nothing to do.)
    if len(groups) >= 2 and remainder != 0:
        tail = groups.pop()
        groups[-1] = f"{groups[-1]} {tail}"
    return groups


def _combine_with_crossfade(wavs: list, crossfade_samples: int):
    """Concatenate per-chunk waveforms with a short equal-power crossfade.

    Chatterbox synthesizes each sentence-group independently, so consecutive
    chunks start/end at unrelated signal levels and phases. A hard
    ``torch.cat`` joins them with a step discontinuity at every seam — a
    low-power but spectrally wideband transient the ear hears as a periodic
    "tick" across the whole track (chatterbox-tick-jitter-260611: 84 seams over
    a 22-min job produced a tick roughly every ~10s; the seam splatter showed
    up as energy above the 12 kHz model-Nyquist that a clean band-limited
    resample cannot produce).

    Fix: overlap-add the boundary with an equal-power (sin/cos) crossfade so the
    transition is continuous. ``wavs`` are ``(channels, samples)`` CPU tensors
    (the model's native output shape). Returns a single ``(channels, samples)``
    tensor. A ~5 ms crossfade at 24 kHz is inaudible as a blend but removes the
    discontinuity entirely.
    """
    import torch

    if not wavs:
        raise ValueError("no waveforms to combine")
    if len(wavs) == 1:
        return wavs[0]

    cf = max(0, int(crossfade_samples))
    if cf == 0:
        return torch.cat(wavs, dim=1)

    # Equal-power fade curves (constant perceived loudness through the overlap).
    ramp = torch.linspace(0.0, 1.0, cf, dtype=wavs[0].dtype)
    fade_in = torch.sin(ramp * (torch.pi / 2.0))
    fade_out = torch.cos(ramp * (torch.pi / 2.0))

    out = wavs[0]
    for nxt in wavs[1:]:
        # Clamp the crossfade to what both sides can supply so a very short
        # chunk never over-runs its own length.
        k = min(cf, out.shape[1], nxt.shape[1])
        if k <= 0:
            out = torch.cat([out, nxt], dim=1)
            continue
        fi = fade_in[-k:]
        fo = fade_out[-k:]
        head = out[:, :-k]
        blended = out[:, -k:] * fo + nxt[:, :k] * fi
        tail = nxt[:, k:]
        out = torch.cat([head, blended, tail], dim=1)
    return out


async def synthesize_to_wav(
    text: str,
    voice_ref_path: Optional[str],
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    language_id: str,
    seed: Optional[int],
    sentence_chunk_size: int,
    output_path: Path,
) -> tuple[Path, float]:
    """Async entry: run the sync synth body in a thread.

    Serialisation is provided by the single-consumer worker (quick-260602-b7l),
    NOT by a lock here. See module docstring.
    """
    return await asyncio.to_thread(
        _synthesize_sync,
        text,
        voice_ref_path,
        exaggeration,
        cfg_weight,
        temperature,
        language_id,
        seed,
        sentence_chunk_size,
        output_path,
    )


def _synthesize_sync(
    text: str,
    voice_ref_path: Optional[str],
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    language_id: str,
    seed: Optional[int],
    sentence_chunk_size: int,
    output_path: Path,
) -> tuple[Path, float]:
    """Sync synth body — runs in a worker thread.

    Returns (wav_path, duration_sec). Raises OutOfMemoryError on OOM.
    """
    import warnings

    import torch
    import torchaudio

    model = _load_model(language_id)

    if seed is not None:
        torch.manual_seed(seed)

    chunks = _chunk_sentences(text, sentence_chunk_size)
    wavs: list = []
    try:
        with torch.no_grad(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*sdp_kernel.*", category=FutureWarning
            )
            for group in chunks:
                kw = dict(
                    audio_prompt_path=voice_ref_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    temperature=temperature,
                )
                if language_id != "en":
                    kw["language_id"] = language_id
                wav = model.generate(group, **kw)
                # Move to CPU immediately to free any GPU allocator slot
                # (no-op on CPU-only deploy but keeps the deck2video shape).
                wavs.append(wav.cpu())
                del wav
                gc.collect()

        # Equal-power crossfade at every chunk seam instead of a hard
        # torch.cat (chatterbox-tick-jitter-260611): the hard join left a
        # step discontinuity at each of the ~84 seams in a full-length job,
        # audible as a periodic "tick". ~5 ms at the model's native sample
        # rate is an inaudible blend but removes the discontinuity. Falls back
        # to a plain concat when there is a single chunk.
        crossfade_samples = int(0.005 * model.sr)
        combined = _combine_with_crossfade(wavs, crossfade_samples)
        # In-service resample 24kHz -> 44.1kHz (M6 / RESEARCH §9 option 1).
        resampler = torchaudio.transforms.Resample(
            orig_freq=model.sr, new_freq=44100
        )
        resampled = resampler(combined)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(str(output_path), resampled, 44100)
        # shape is (channels, samples); duration = samples / sr.
        duration_sec = float(resampled.shape[1]) / 44100.0
        return output_path, duration_sec
    except Exception as exc:
        if _is_oom(exc):
            raise OutOfMemoryError(
                "chatterbox-tts OOM during synthesis"
            ) from exc
        raise
