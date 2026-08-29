"""Dispatch tests for client.synthesize() backend selection (07-03 Task 3).

client.synthesize() branches at the TOP on CHATTERBOX_BACKEND:
    unset / "local"  -> the existing local POST->poll->fetch path (UNCHANGED)
    "runpod"         -> delegate to runpod_client.synthesize (lazy import)

Precedence is env > config.backend > "local". The local path's full regression
suite lives in test_client_polling.py and must stay green; these tests only
prove the routing decision (local default does NOT call runpod_client; the
runpod backend delegates with forwarded args and issues no local POST).

The runpod target is patched via tts_chatterbox.runpod_client.synthesize, since
the dispatch does a LAZY `from tts_chatterbox.runpod_client import synthesize`
(which resolves the attribute at call time).
"""
from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest

from tts_chatterbox.client import synthesize
from tts_chatterbox.config import ChatterboxConfig


# ---------------------------------------------------------------------------
# Local-path harness (a trimmed _FakeClient, enough to drive one completed flow)
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, post_responses=None, get_responses=None):
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.posts: list[tuple] = []
        self.gets: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json):
        self.posts.append((url, json))
        return self.post_responses.pop(0)

    def get(self, url):
        self.gets.append(url)
        return self.get_responses.pop(0)


def _resp(status_code: int, json_body: dict | None = None) -> httpx.Response:
    return httpx.Response(status_code, json=json_body or {})


# ---------------------------------------------------------------------------
# T-D1: CHATTERBOX_BACKEND unset -> local path runs, runpod NOT called
# ---------------------------------------------------------------------------

def test_backend_unset_runs_local_path(tmp_path, monkeypatch):
    monkeypatch.delenv("CHATTERBOX_BACKEND", raising=False)
    out_path = tmp_path / "out.wav"
    server_wav = tmp_path / "rid.wav"
    server_wav.write_bytes(b"WAV")

    fake = _FakeClient(
        post_responses=[_resp(201, {"id": "rid", "state": "queued"})],
        get_responses=[
            _resp(200, {"id": "rid", "state": "completed",
                        "wav_path": str(server_wav)}),
        ],
    )
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with patch("tts_chatterbox.client.time.sleep"):
            with patch("tts_chatterbox.runpod_client.synthesize") as m_runpod:
                # Force config.backend to "local" too so neither selector trips.
                cfg = ChatterboxConfig(backend="local")
                result = synthesize("hello", cfg, str(out_path), request_id="rid")

    assert result == str(out_path)
    # The local POST was issued; runpod was never touched.
    assert len(fake.posts) == 1
    m_runpod.assert_not_called()
    assert out_path.read_bytes() == b"WAV"


# ---------------------------------------------------------------------------
# T-D2: CHATTERBOX_BACKEND="runpod" -> delegate to runpod_client, no local POST
# ---------------------------------------------------------------------------

def test_backend_runpod_delegates_to_runpod_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHATTERBOX_BACKEND", "runpod")
    out_path = tmp_path / "out.wav"

    fake = _FakeClient()  # if the local path runs, .post pops from [] -> IndexError
    cfg = ChatterboxConfig()
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with patch("tts_chatterbox.runpod_client.synthesize",
                   return_value=str(out_path)) as m_runpod:
            result = synthesize(
                "hello world", cfg, str(out_path),
                request_id="rid", max_attempts=3, backoff_sec=5.0,
                poll_max_attempts=42,
            )

    assert result == str(out_path)
    # Local POST never issued.
    assert fake.posts == []
    # Delegated exactly once with the forwarded args.
    m_runpod.assert_called_once_with(
        text="hello world",
        config=cfg,
        output_path=str(out_path),
        request_id="rid",
        max_attempts=3,
        backoff_sec=5.0,
        poll_max_attempts=42,
    )


# ---------------------------------------------------------------------------
# T-D3: env beats config.backend (precedence: env > config.backend > local)
# ---------------------------------------------------------------------------

def test_env_overrides_config_backend(tmp_path, monkeypatch):
    """config.backend='local' but CHATTERBOX_BACKEND='runpod' -> runpod wins."""
    monkeypatch.setenv("CHATTERBOX_BACKEND", "runpod")
    out_path = tmp_path / "out.wav"
    cfg = ChatterboxConfig(backend="local")  # env must still win

    fake = _FakeClient()
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with patch("tts_chatterbox.runpod_client.synthesize",
                   return_value=str(out_path)) as m_runpod:
            synthesize("hi", cfg, str(out_path), request_id="rid")

    m_runpod.assert_called_once()
    assert fake.posts == []


def test_config_backend_selects_runpod_when_env_unset(tmp_path, monkeypatch):
    """No env -> config.backend='runpod' routes to runpod (the fallback selector)."""
    monkeypatch.delenv("CHATTERBOX_BACKEND", raising=False)
    out_path = tmp_path / "out.wav"
    cfg = ChatterboxConfig(backend="runpod")

    fake = _FakeClient()
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with patch("tts_chatterbox.runpod_client.synthesize",
                   return_value=str(out_path)) as m_runpod:
            synthesize("hi", cfg, str(out_path), request_id="rid")

    m_runpod.assert_called_once()
    assert fake.posts == []


# ---------------------------------------------------------------------------
# T-D4: empty text raises ValueError before any dispatch (both backends)
# ---------------------------------------------------------------------------

def test_empty_text_raises_before_dispatch(monkeypatch):
    monkeypatch.setenv("CHATTERBOX_BACKEND", "runpod")
    with patch("tts_chatterbox.runpod_client.synthesize") as m_runpod:
        with pytest.raises(ValueError, match="text must not be empty"):
            synthesize("   ", ChatterboxConfig(), "out.wav")
    m_runpod.assert_not_called()


# ---------------------------------------------------------------------------
# T-D5: unknown non-empty backend fails loud (never falls through to local)
# ---------------------------------------------------------------------------

def test_unknown_env_backend_raises_value_error(tmp_path, monkeypatch):
    """CHATTERBOX_BACKEND=bogus must raise, not silently synth locally.

    Non-empty text so the empty-text guard cannot mask the dispatch; the
    _FakeClient would IndexError on any local POST, so no HTTP call happens.
    """
    monkeypatch.setenv("CHATTERBOX_BACKEND", "bogus")
    fake = _FakeClient()
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with pytest.raises(
            ValueError,
            match=r"Unknown CHATTERBOX_BACKEND.*valid backends",
        ):
            synthesize("hello", ChatterboxConfig(), str(tmp_path / "out.wav"))
    assert fake.posts == []


def test_unknown_config_backend_raises_when_env_unset(tmp_path, monkeypatch):
    """Env unset -> config.backend is the selector; an unknown value there
    also fails loud (env > config.backend > "local" resolution unchanged)."""
    monkeypatch.delenv("CHATTERBOX_BACKEND", raising=False)
    fake = _FakeClient()
    cfg = ChatterboxConfig(backend="cloudtts")
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with pytest.raises(
            ValueError,
            match=r"Unknown CHATTERBOX_BACKEND.*valid backends",
        ):
            synthesize("hello", cfg, str(tmp_path / "out.wav"))
    assert fake.posts == []


def test_empty_string_env_backend_falls_through_to_local(tmp_path, monkeypatch):
    """CHATTERBOX_BACKEND="" is falsy, not unknown -> local path still runs."""
    monkeypatch.setenv("CHATTERBOX_BACKEND", "")
    out_path = tmp_path / "out.wav"
    server_wav = tmp_path / "rid.wav"
    server_wav.write_bytes(b"WAV")

    fake = _FakeClient(
        post_responses=[_resp(201, {"id": "rid", "state": "queued"})],
        get_responses=[
            _resp(200, {"id": "rid", "state": "completed",
                        "wav_path": str(server_wav)}),
        ],
    )
    with patch("tts_chatterbox.client.httpx.Client", return_value=fake):
        with patch("tts_chatterbox.client.time.sleep"):
            with patch("tts_chatterbox.runpod_client.synthesize") as m_runpod:
                cfg = ChatterboxConfig(backend="local")
                result = synthesize("hello", cfg, str(out_path), request_id="rid")

    assert result == str(out_path)
    assert len(fake.posts) == 1
    m_runpod.assert_not_called()
    assert out_path.read_bytes() == b"WAV"
