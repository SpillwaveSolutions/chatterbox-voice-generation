"""Tests for the phase-07 backend fields on ChatterboxConfig (07-03 Task 1).

ChatterboxConfig gains three env-driven transport fields:
    backend             <- CHATTERBOX_BACKEND (default "local")
    runpod_endpoint_id  <- RUNPOD_CHATTERBOX_ENDPOINT_ID (default "")
    runpod_api_key      <- RUNPOD_API_KEY (default "")

These are TRANSPORT config, not synth params: they must NOT leak into
``to_request_body()`` (the local /jobs body must stay byte-identical — SC-4).
"""
from __future__ import annotations

import importlib

import pytest

from tts_chatterbox.config import ChatterboxConfig


def test_backend_defaults_to_local(monkeypatch):
    """With no CHATTERBOX_BACKEND set, backend defaults to 'local'."""
    monkeypatch.delenv("CHATTERBOX_BACKEND", raising=False)
    c = ChatterboxConfig()
    assert c.backend == "local"


def test_runpod_fields_default_empty(monkeypatch):
    monkeypatch.delenv("RUNPOD_CHATTERBOX_ENDPOINT_ID", raising=False)
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    c = ChatterboxConfig()
    assert c.runpod_endpoint_id == ""
    assert c.runpod_api_key == ""


def test_backend_fields_read_from_env_at_construct(monkeypatch):
    """field(default_factory=...) reads env at construct time (RevoiceConfig pattern)."""
    monkeypatch.setenv("CHATTERBOX_BACKEND", "runpod")
    monkeypatch.setenv("RUNPOD_CHATTERBOX_ENDPOINT_ID", "ep_abc123")
    monkeypatch.setenv("RUNPOD_API_KEY", "key_xyz")
    c = ChatterboxConfig()
    assert c.backend == "runpod"
    assert c.runpod_endpoint_id == "ep_abc123"
    assert c.runpod_api_key == "key_xyz"


def test_to_request_body_excludes_transport_fields(monkeypatch):
    """The local /jobs body must NOT carry backend/runpod_* keys (SC-4)."""
    monkeypatch.setenv("CHATTERBOX_BACKEND", "runpod")
    monkeypatch.setenv("RUNPOD_CHATTERBOX_ENDPOINT_ID", "ep_abc123")
    monkeypatch.setenv("RUNPOD_API_KEY", "key_xyz")
    body = ChatterboxConfig().to_request_body("hello", request_id="r1")
    assert "backend" not in body
    assert "runpod_endpoint_id" not in body
    assert "runpod_api_key" not in body
    # And the canonical local body shape is unchanged.
    assert set(body.keys()) == {
        "text", "voice_ref", "exaggeration", "cfg_weight", "temperature",
        "device", "language_id", "seed", "sentence_chunk_size",
        "pronunciations", "request_id",
    }
