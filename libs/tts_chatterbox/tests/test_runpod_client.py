"""Tests for the RunPod serverless transport (07-03 Task 2).

The runpod client mirrors the local POST->poll->fetch shape but:
    POST  https://api.runpod.ai/v2/{ep}/run     -> {"id", "status": "IN_QUEUE"}
    GET   https://api.runpod.ai/v2/{ep}/status/{id}
        IN_QUEUE / IN_PROGRESS / RUNNING -> keep polling (5s)
        COMPLETED                        -> store.download_file(output_key, output_path)
        FAILED / CANCELLED / TIMED_OUT   -> RuntimeError(worker error)
    body carries policy.executionTimeout=1800000 (override the 10-min default).
    voice ref uploads ONCE via ensure_uploaded -> presign_get -> voice_ref_url.
    output reserved via presign_put -> output_put_url.
    NO DELETE (RunPod has no request_id idempotency; poison-row dance not ported).

httpx + ArtifactStore are mocked: no network, no boto3, no GPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from tts_chatterbox.config import ChatterboxConfig


# ---------------------------------------------------------------------------
# Harness (mirrors test_client_polling._FakeClient + _resp)
# ---------------------------------------------------------------------------

class _FakeClient:
    def __init__(self, post_responses=None, get_responses=None):
        self.post_responses = list(post_responses or [])
        self.get_responses = list(get_responses or [])
        self.posts: list[tuple] = []
        self.gets: list[tuple] = []
        self.deletes: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, json=None, headers=None):
        self.posts.append((url, json, headers))
        nxt = self.post_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def get(self, url, headers=None):
        self.gets.append((url, headers))
        nxt = self.get_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def delete(self, url, headers=None):  # pragma: no cover - must never be called
        self.deletes.append(url)
        raise AssertionError("runpod client must never issue DELETE")


def _resp(status_code: int, json_body: dict | None = None, text: str = "") -> httpx.Response:
    if json_body is not None:
        return httpx.Response(status_code, json=json_body)
    return httpx.Response(status_code, text=text)


def _store_mock():
    """A MagicMock ArtifactStore with sensible default return values."""
    store = MagicMock()
    store.ensure_uploaded.return_value = "abc123def4567890/narrator.wav"
    store.presign_get.return_value = "https://s3.example/get?sig=ref"
    store.presign_put.return_value = "https://s3.example/put?sig=out"
    store.download_file.return_value = None
    return store


def _runpod_config(**overrides):
    base = dict(runpod_endpoint_id="ep_test", runpod_api_key="key_test")
    base.update(overrides)
    return ChatterboxConfig(**base)


def _patch_all(fake, store):
    """Patch httpx.Client, time.sleep, ArtifactStore, and ArtifactStoreConfig.from_env."""
    p_client = patch("tts_chatterbox.runpod_client.httpx.Client", return_value=fake)
    p_sleep = patch("tts_chatterbox.runpod_client.time.sleep")
    p_store = patch("tts_chatterbox.runpod_client.ArtifactStore", return_value=store)
    p_cfg = patch(
        "tts_chatterbox.runpod_client.ArtifactStoreConfig.from_env",
        return_value=MagicMock(),
    )
    return p_client, p_sleep, p_store, p_cfg


# ---------------------------------------------------------------------------
# T-R1: POST shape — URL, Bearer header, input object, policy.executionTimeout
# ---------------------------------------------------------------------------

def test_post_shape_carries_policy_execution_timeout(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "job1", "status": "IN_QUEUE"})],
        get_responses=[_resp(200, {"status": "COMPLETED"})],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        result = synthesize(
            "hello world", _runpod_config(), str(out), request_id="job1",
        )

    assert result == str(out)
    assert len(fake.posts) == 1
    url, body, headers = fake.posts[0]
    assert url == "https://api.runpod.ai/v2/ep_test/run"
    assert headers["Authorization"] == "Bearer key_test"
    assert "input" in body
    assert body["input"]["text"] == "hello world"
    assert body["policy"]["executionTimeout"] == 1800000


# ---------------------------------------------------------------------------
# T-R2: poll loop treats IN_QUEUE/IN_PROGRESS/RUNNING all as keep-polling
# ---------------------------------------------------------------------------

def test_poll_through_all_non_terminal_states(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[
            _resp(200, {"status": "IN_QUEUE"}),
            _resp(200, {"status": "IN_PROGRESS"}),
            _resp(200, {"status": "RUNNING"}),
            _resp(200, {"status": "COMPLETED"}),
        ],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        result = synthesize("hi", _runpod_config(), str(out), request_id="j")

    assert result == str(out)
    assert len(fake.gets) == 4
    for url, headers in fake.gets:
        assert url == "https://api.runpod.ai/v2/ep_test/status/j"
        assert headers["Authorization"] == "Bearer key_test"


# ---------------------------------------------------------------------------
# T-R3: COMPLETED downloads via artifact_store, returns output_path
# ---------------------------------------------------------------------------

def test_completed_downloads_via_store_not_response_body(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[_resp(200, {"status": "COMPLETED"})],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        result = synthesize("hi", _runpod_config(), str(out), request_id="j")

    assert result == str(out)
    store.download_file.assert_called_once()
    args, _ = store.download_file.call_args
    # download_file(output_key, output_path)
    assert args[1] == str(out)
    assert args[0] == "output/j.wav"


# ---------------------------------------------------------------------------
# T-R4: terminal-error states raise RuntimeError with worker detail
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("state", ["FAILED", "CANCELLED", "TIMED_OUT"])
def test_terminal_error_states_raise(tmp_path, state):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[
            _resp(200, {"status": state, "error": f"boom_{state.lower()}"}),
        ],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        with pytest.raises(RuntimeError, match=f"boom_{state.lower()}"):
            synthesize("hi", _runpod_config(), str(out), request_id="j")


def test_terminal_error_reads_output_error_fallback(tmp_path):
    """When top-level 'error' is absent, fall back to output.error."""
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[
            _resp(200, {"status": "FAILED", "output": {"error": "worker_traceback"}}),
        ],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        with pytest.raises(RuntimeError, match="worker_traceback"):
            synthesize("hi", _runpod_config(), str(out), request_id="j")


# ---------------------------------------------------------------------------
# T-R5: voice ref uploaded once -> presign_get -> voice_ref_url in POST input
# ---------------------------------------------------------------------------

def test_voice_ref_uploaded_once_and_presigned(tmp_path, monkeypatch):
    from tts_chatterbox.runpod_client import synthesize

    # Point the voice-refs dir at tmp so the bare filename resolves locally.
    voices = tmp_path / "voice_refs"
    voices.mkdir()
    (voices / "narrator.wav").write_bytes(b"REF")
    monkeypatch.setenv("CHATTERBOX_VOICE_REFS_DIR", str(voices))

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[_resp(200, {"status": "COMPLETED"})],
    )
    cfg = _runpod_config(voice_ref="narrator.wav")
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        synthesize("hi", cfg, str(out), request_id="j")

    store.ensure_uploaded.assert_called_once_with(str(voices / "narrator.wav"))
    # presign_get called with the ensure_uploaded key for the ref URL.
    store.presign_get.assert_called_once_with("abc123def4567890/narrator.wav")
    _, body, _ = fake.posts[0]
    assert body["input"]["voice_ref_url"] == "https://s3.example/get?sig=ref"


def test_no_voice_ref_means_no_upload_and_null_url(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[_resp(200, {"status": "COMPLETED"})],
    )
    cfg = _runpod_config(voice_ref=None)
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        synthesize("hi", cfg, str(out), request_id="j")

    store.ensure_uploaded.assert_not_called()
    store.presign_get.assert_not_called()
    _, body, _ = fake.posts[0]
    assert body["input"]["voice_ref_url"] is None


# ---------------------------------------------------------------------------
# T-R6: output key reserved -> presign_put -> output_put_url in POST input
# ---------------------------------------------------------------------------

def test_output_put_url_reserved_via_presign_put(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[_resp(200, {"status": "COMPLETED"})],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        synthesize("hi", _runpod_config(), str(out), request_id="j")

    store.presign_put.assert_called_once_with("output/j.wav")
    _, body, _ = fake.posts[0]
    assert body["input"]["output_put_url"] == "https://s3.example/put?sig=out"


# ---------------------------------------------------------------------------
# T-R7: poll budget defaults to 240 + env override
# ---------------------------------------------------------------------------

def test_poll_budget_default_is_240():
    from tts_chatterbox import runpod_client
    assert runpod_client.POLL_MAX_ATTEMPTS_DEFAULT == 240


def test_poll_budget_default_240_exhausts(tmp_path, monkeypatch):
    from tts_chatterbox.runpod_client import synthesize

    monkeypatch.delenv("CHATTERBOX_TTS_POLL_MAX_ATTEMPTS", raising=False)
    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[httpx.ConnectError("refused") for _ in range(240)],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        with pytest.raises(RuntimeError, match="240 attempts"):
            synthesize("hi", _runpod_config(), str(out), request_id="j",
                       backoff_sec=0.0)
    assert len(fake.gets) == 240


def test_poll_budget_env_override(tmp_path, monkeypatch):
    from tts_chatterbox.runpod_client import synthesize

    monkeypatch.setenv("CHATTERBOX_TTS_POLL_MAX_ATTEMPTS", "5")
    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[httpx.ConnectError("refused") for _ in range(5)],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        with pytest.raises(RuntimeError, match="5 attempts"):
            synthesize("hi", _runpod_config(), str(out), request_id="j",
                       backoff_sec=0.0)
    assert len(fake.gets) == 5


def test_transient_poll_error_rides_out_within_budget(tmp_path):
    """A transient httpx error during poll retries, then succeeds."""
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[
            httpx.ConnectError("blip"),
            _resp(200, {"status": "RUNNING"}),
            _resp(200, {"status": "COMPLETED"}),
        ],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        result = synthesize("hi", _runpod_config(), str(out), request_id="j",
                            poll_max_attempts=5, backoff_sec=0.0)
    assert result == str(out)


# ---------------------------------------------------------------------------
# T-R8: unknown state default-branches to keep-polling (does not crash)
# ---------------------------------------------------------------------------

def test_unknown_state_keeps_polling(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[
            _resp(200, {"status": "SOMETHING_NEW"}),
            _resp(200, {"status": "COMPLETED"}),
        ],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        result = synthesize("hi", _runpod_config(), str(out), request_id="j")
    assert result == str(out)


# ---------------------------------------------------------------------------
# T-R9: NO DELETE ever issued (poison-row dance not ported)
# ---------------------------------------------------------------------------

def test_no_delete_ever_issued(tmp_path):
    from tts_chatterbox.runpod_client import synthesize

    out = tmp_path / "out.wav"
    store = _store_mock()
    fake = _FakeClient(
        post_responses=[_resp(200, {"id": "j", "status": "IN_QUEUE"})],
        get_responses=[_resp(200, {"status": "COMPLETED"})],
    )
    pc, ps, pst, pcfg = _patch_all(fake, store)
    with pc, ps, pst, pcfg:
        synthesize("hi", _runpod_config(), str(out), request_id="j")
    assert fake.deletes == []


# ---------------------------------------------------------------------------
# T-R10: empty text raises ValueError (same as local)
# ---------------------------------------------------------------------------

def test_empty_text_raises_value_error(tmp_path):
    from tts_chatterbox.runpod_client import synthesize
    with pytest.raises(ValueError, match="text must not be empty"):
        synthesize("   ", _runpod_config(), str(tmp_path / "out.wav"))
