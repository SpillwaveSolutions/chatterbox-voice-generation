"""Unit tests for artifact_store — config + moto-backed store, no network.

Task 1 covers ArtifactStoreConfig.from_env(); Task 2 adds the ArtifactStore
tests (checksum-safe client, HEAD dedupe, presign round-trip) under @mock_aws.
"""
from __future__ import annotations

import requests
from moto import mock_aws

from artifact_store import ArtifactStore, ArtifactStoreConfig

_BUCKET = "test-artifacts"


def _config() -> ArtifactStoreConfig:
    # moto intercepts boto3 at the botocore layer regardless of endpoint_url;
    # an https endpoint keeps presigned URLs well-formed for the requests round-trip.
    return ArtifactStoreConfig(
        endpoint="https://s3.amazonaws.com",
        bucket=_BUCKET,
        access_key="testing",
        secret_key="testing",
        region="us-east-1",
    )


def _make_store_with_bucket() -> ArtifactStore:
    store = ArtifactStore(_config())
    store.client.create_bucket(Bucket=_BUCKET)
    return store


# --- Task 1: ArtifactStoreConfig.from_env ---------------------------------


def test_from_env_reads_all_five_vars(monkeypatch):
    """from_env() with all five vars set yields matching attribute values."""
    monkeypatch.setenv("ARTIFACT_S3_ENDPOINT", "https://acct.r2.cloudflarestorage.com")
    monkeypatch.setenv("ARTIFACT_S3_BUCKET", "courseware-gpu-artifacts")
    monkeypatch.setenv("ARTIFACT_S3_ACCESS_KEY", "AKID")
    monkeypatch.setenv("ARTIFACT_S3_SECRET_KEY", "SECRET")
    monkeypatch.setenv("ARTIFACT_S3_REGION", "nyc3")

    cfg = ArtifactStoreConfig.from_env()

    assert cfg.endpoint == "https://acct.r2.cloudflarestorage.com"
    assert cfg.bucket == "courseware-gpu-artifacts"
    assert cfg.access_key == "AKID"
    assert cfg.secret_key == "SECRET"
    assert cfg.region == "nyc3"


def test_region_defaults_to_auto_when_unset(monkeypatch):
    """region defaults to "auto" (R2) when ARTIFACT_S3_REGION is unset."""
    monkeypatch.delenv("ARTIFACT_S3_REGION", raising=False)
    monkeypatch.setenv("ARTIFACT_S3_ENDPOINT", "https://acct.r2.cloudflarestorage.com")

    cfg = ArtifactStoreConfig.from_env()

    assert cfg.region == "auto"
    assert cfg.endpoint == "https://acct.r2.cloudflarestorage.com"


# --- Task 2: ArtifactStore -------------------------------------------------


def test_client_built_with_checksum_safe_config():
    """The boto3 client uses the R2/Spaces-safe checksum + signing Config.

    This is the highest-priority pitfall: without these flags, boto3>=1.36
    emits an x-amz-checksum-crc32 trailer that R2/Spaces 403 on presigned PUT.
    """
    store = ArtifactStore(_config())
    client_config = store.client.meta.config

    assert client_config.request_checksum_calculation == "when_required"
    assert client_config.response_checksum_validation == "when_required"
    assert client_config.signature_version == "s3v4"


@mock_aws
def test_upload_download_round_trip(tmp_path):
    """upload_file + download_file round-trips a file's bytes through moto."""
    store = _make_store_with_bucket()
    src = tmp_path / "in.wav"
    payload = b"RIFF....fake-wav-bytes"
    src.write_bytes(payload)

    store.upload_file(str(src), "some/key.wav")
    dst = tmp_path / "nested" / "out.wav"
    store.download_file("some/key.wav", str(dst))

    assert dst.read_bytes() == payload


@mock_aws
def test_ensure_uploaded_returns_content_hash_key(tmp_path):
    """ensure_uploaded(path) returns key == sha256(bytes)[:16]/filename."""
    import hashlib

    store = _make_store_with_bucket()
    f = tmp_path / "voice.wav"
    content = b"voice-reference-audio"
    f.write_bytes(content)
    expected = f"{hashlib.sha256(content).hexdigest()[:16]}/voice.wav"

    assert store.ensure_uploaded(str(f)) == expected


@mock_aws
def test_ensure_uploaded_dedupes_on_second_call(tmp_path, mocker):
    """Second call with the same content does NOT re-upload (HEAD short-circuit)."""
    store = _make_store_with_bucket()
    f = tmp_path / "voice.wav"
    f.write_bytes(b"identical-content")

    key1 = store.ensure_uploaded(str(f))  # uploads (HEAD 404)

    spy = mocker.spy(store.client, "upload_file")
    key2 = store.ensure_uploaded(str(f))  # HEAD hit -> skip upload

    assert key1 == key2
    spy.assert_not_called()
    # exactly one object in the bucket
    listing = store.client.list_objects_v2(Bucket=_BUCKET)
    assert listing["KeyCount"] == 1


@mock_aws
def test_presign_put_then_get_round_trip(tmp_path):
    """presign_put -> HTTP PUT bytes -> presign_get -> HTTP GET same bytes."""
    store = _make_store_with_bucket()
    key = "round/trip.wav"
    payload = b"presigned-round-trip-bytes"

    put_url = store.presign_put(key)
    put_resp = requests.put(put_url, data=payload)
    put_resp.raise_for_status()

    get_url = store.presign_get(key)
    get_resp = requests.get(get_url)
    get_resp.raise_for_status()

    assert get_resp.content == payload


def test_presign_default_ttl_is_two_hours():
    """Default presign TTL is 7200s (2h)."""
    from artifact_store.store import _PRESIGN_TTL_DEFAULT

    assert _PRESIGN_TTL_DEFAULT == 7200
