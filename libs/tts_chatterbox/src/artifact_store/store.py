"""ArtifactStore — checksum-safe boto3 wrapper for any S3-compatible endpoint.

This is the cross-cloud artifact bus for the RunPod GPU worker. It provides
upload/download, content-hash dedupe (``ensure_uploaded``), and presigned
GET/PUT URL generation.

CRITICAL: the boto3 client is built with the checksum-disabling Config. Since
boto3 1.36.0, presigned PUT uploads add an ``x-amz-checksum-crc32`` trailer by
default that Cloudflare R2 / DigitalOcean Spaces reject with
``403 SignatureDoesNotMatch``. ``request_checksum_calculation="when_required"``
(plus ``response_checksum_validation="when_required"`` and
``signature_version="s3v4"``) suppresses the trailer. Do NOT remove these flags.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from artifact_store.config import ArtifactStoreConfig

# 2h — must outlive queue wait + synth (RunPod endpoint execution timeout 1800s).
_PRESIGN_TTL_DEFAULT = 7200


class ArtifactStore:
    """Sync boto3 wrapper around an S3-compatible endpoint (R2 / Spaces / S3)."""

    def __init__(self, config: ArtifactStoreConfig):
        self.config = config
        self.bucket = config.bucket
        self.client = boto3.client(
            "s3",
            endpoint_url=config.endpoint,
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key,
            region_name=config.region,
            config=Config(
                signature_version="s3v4",
                # CRITICAL: stops the x-amz-checksum-crc32 trailer (R2/Spaces 403).
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )

    def upload_file(self, path: str, key: str) -> None:
        """Upload a local file to ``key`` in the bucket."""
        self.client.upload_file(str(path), self.bucket, key)

    def download_file(self, key: str, path: str) -> None:
        """Download ``key`` to a local path, creating parent dirs as needed."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))

    def ensure_uploaded(self, path: str) -> str:
        """Upload ``path`` under a content-hash key, only if absent (HEAD-first).

        Returns ``f"{sha256(file)[:16]}/{filename}"``. Voice refs upload once,
        ever: a HEAD hit short-circuits the upload. Any non-404 ClientError is
        re-raised.
        """
        p = Path(path)
        h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        key = f"{h}/{p.name}"
        try:
            self.client.head_object(Bucket=self.bucket, Key=key)
            return key  # already present — voice refs upload once, ever
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey", "NotFound"):
                raise
        self.client.upload_file(str(p), self.bucket, key)
        return key

    def presign_get(self, key: str, ttl: int = _PRESIGN_TTL_DEFAULT) -> str:
        """Generate a presigned GET URL for ``key`` (default TTL 2h)."""
        return self.client.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl
        )

    def presign_put(self, key: str, ttl: int = _PRESIGN_TTL_DEFAULT) -> str:
        """Generate a presigned PUT URL for ``key`` (default TTL 2h).

        Deliberately signs only Bucket + Key: a mismatched signed content-type
        header 403s on R2/Spaces, so the uploader sends bytes with whatever
        default content-type the HTTP client emits.
        """
        return self.client.generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl
        )
