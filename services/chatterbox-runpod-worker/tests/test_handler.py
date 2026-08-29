"""Unit tests for the RunPod worker handler (Phase 07-02 Task 2).

The handler's branching logic is proven on the Mac with NO GPU, NO torch, NO
network: ``runpod`` and the real ``chatterbox_tts`` synth chain are injected as
lightweight ``sys.modules`` stubs at import time, then ``_synthesize_sync`` /
``_load_model`` / ``httpx`` are patched on the handler module per test.

Run command (dedicated venv — worker is excluded from the host workspace):
    PYTHONPATH=services/chatterbox-runpod-worker/src \
      uv run --no-project --with pytest --with pytest-mock --with httpx \
      python -m pytest services/chatterbox-runpod-worker/tests -q
or inside the worker image:
    python -m pytest /app/worker/.../tests
"""
from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inject stubs for the heavy modules BEFORE importing the handler. The real
# chatterbox_tts.pronunciations is pure-python (no torch) so we import it for
# real to prove the pronunciation substitution actually happens; model + runpod
# are stubbed.
# ---------------------------------------------------------------------------

def _install_stubs():
    # runpod stub: handler module references runpod.serverless.start under
    # __main__ only, but the import must resolve.
    runpod_stub = types.ModuleType("runpod")
    runpod_stub.serverless = types.SimpleNamespace(start=MagicMock(name="start"))
    sys.modules["runpod"] = runpod_stub

    # chatterbox_tts.model stub with patchable _synthesize_sync + _load_model.
    model_stub = types.ModuleType("chatterbox_tts.model")
    model_stub._synthesize_sync = MagicMock(name="_synthesize_sync")
    model_stub._load_model = MagicMock(name="_load_model")
    sys.modules["chatterbox_tts.model"] = model_stub

    # Use the REAL pronunciations module (pure python) for a true substitution
    # assertion. If it can't be imported (path not set), fall back to a stub.
    try:
        import chatterbox_tts.pronunciations  # noqa: F401
    except Exception:
        pron_stub = types.ModuleType("chatterbox_tts.pronunciations")
        pron_stub.compile_pronunciations = lambda m: list(m.items())
        pron_stub.apply_pronunciations = lambda t, p: t
        sys.modules["chatterbox_tts.pronunciations"] = pron_stub

    # Ensure parent package exists.
    if "chatterbox_tts" not in sys.modules:
        pkg = types.ModuleType("chatterbox_tts")
        pkg.__path__ = []  # mark as package
        sys.modules["chatterbox_tts"] = pkg


@pytest.fixture
def handler_mod(monkeypatch):
    _install_stubs()
    sys.modules.pop("chatterbox_runpod_worker.handler", None)
    mod = importlib.import_module("chatterbox_runpod_worker.handler")

    # Default: _synthesize_sync returns (path, duration). Override side_effect in
    # specific tests.
    mod._synthesize_sync.reset_mock(return_value=True, side_effect=True)
    mod._load_model.reset_mock(return_value=True, side_effect=True)

    def _synth_ok(text, ref, *args):
        out = args[-1]
        try:
            out.write_bytes(b"WAVDATA")
        except Exception:
            pass
        return out, 1.23

    mod._synthesize_sync.side_effect = _synth_ok

    # httpx stub on the handler module — record GET/PUT calls.
    fake_httpx = MagicMock(name="httpx")
    monkeypatch.setattr(mod, "httpx", fake_httpx)
    return mod, fake_httpx


def _ctx_client(fake_httpx):
    """Return the MagicMock client yielded by ``with httpx.Client(...) as c``."""
    return fake_httpx.Client.return_value.__enter__.return_value


def test_smoke_test_skips_upload_returns_timing(handler_mod):
    mod, fake_httpx = handler_mod
    res = mod.handler({"input": {"smoke_test": True, "text": "hi"}})
    assert res["ok"] is True
    assert res["smoke"] is True
    assert "duration_s" in res
    assert "elapsed_s" in res
    # synth called once, no PUT (no upload on smoke).
    assert mod._synthesize_sync.call_count == 1
    client = _ctx_client(fake_httpx)
    client.put.assert_not_called()


def test_forces_cuda_load(handler_mod):
    mod, _ = handler_mod
    mod.handler({"input": {"smoke_test": True, "text": "hi", "language_id": "en"}})
    mod._load_model.assert_called_once_with("en", device="cuda")


def test_voice_ref_url_downloads_and_passes_temp_path(handler_mod):
    mod, fake_httpx = handler_mod
    client = _ctx_client(fake_httpx)
    client.get.return_value = MagicMock(content=b"REFBYTES")

    mod.handler(
        {
            "input": {
                "text": "hello",
                "voice_ref_url": "https://store/get/ref.wav",
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "language_id": "en",
                "sentence_chunk_size": 3,
                "output_put_url": "https://store/put/out.wav",
            }
        }
    )
    client.get.assert_called_once_with("https://store/get/ref.wav")
    # The ref temp path is the 2nd positional arg to _synthesize_sync.
    call = mod._synthesize_sync.call_args
    ref_path = call.args[1]
    assert ref_path is not None
    assert ref_path.endswith(".wav")


def test_voice_ref_none_passes_none(handler_mod):
    mod, fake_httpx = handler_mod
    client = _ctx_client(fake_httpx)
    mod.handler(
        {
            "input": {
                "text": "hello",
                "voice_ref_url": None,
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "language_id": "en",
                "sentence_chunk_size": 3,
                "output_put_url": "https://store/put/out.wav",
            }
        }
    )
    # No GET when voice_ref_url is None.
    client.get.assert_not_called()
    ref_path = mod._synthesize_sync.call_args.args[1]
    assert ref_path is None


def test_applies_pronunciations(handler_mod):
    mod, _ = handler_mod
    mod.handler(
        {
            "input": {
                "text": "run kubectl now",
                "voice_ref_url": None,
                "pronunciations": {"kubectl": "cube control"},
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "language_id": "en",
                "sentence_chunk_size": 3,
                "output_put_url": "https://store/put/out.wav",
            }
        }
    )
    synthesized_text = mod._synthesize_sync.call_args.args[0]
    assert "cube control" in synthesized_text
    assert "kubectl" not in synthesized_text


def test_success_puts_wav_and_returns_metadata(handler_mod):
    mod, fake_httpx = handler_mod
    client = _ctx_client(fake_httpx)
    res = mod.handler(
        {
            "input": {
                "text": "hello world",
                "voice_ref_url": None,
                "exaggeration": 0.5,
                "cfg_weight": 0.5,
                "temperature": 0.8,
                "language_id": "en",
                "sentence_chunk_size": 3,
                "output_put_url": "https://store/put/out.wav",
            }
        }
    )
    # WAV PUT to the presigned output URL with the synthesized bytes.
    client.put.assert_called_once()
    _, kwargs = client.put.call_args
    assert client.put.call_args.args[0] == "https://store/put/out.wav"
    assert kwargs["content"] == b"WAVDATA"
    assert res == {
        "ok": True,
        "duration_s": 1.23,
        "chars": len("hello world"),
        "elapsed_s": res["elapsed_s"],
    }


def test_synth_failure_propagates_not_graceful(handler_mod):
    mod, _ = handler_mod
    mod._synthesize_sync.side_effect = RuntimeError("cuda blew up")
    with pytest.raises(RuntimeError, match="cuda blew up"):
        mod.handler(
            {
                "input": {
                    "text": "hello",
                    "voice_ref_url": None,
                    "exaggeration": 0.5,
                    "cfg_weight": 0.5,
                    "temperature": 0.8,
                    "language_id": "en",
                    "sentence_chunk_size": 3,
                    "output_put_url": "https://store/put/out.wav",
                }
            }
        )
