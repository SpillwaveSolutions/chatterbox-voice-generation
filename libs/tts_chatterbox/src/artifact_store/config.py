"""Configuration for the artifact_store S3-compatible client.

Settings are environment-driven via the ``ARTIFACT_S3_*`` env vars (set in
``.env`` on the flow-runner and mirrored on each RunPod endpoint). The dataclass
provides typed values; ``from_env`` reads them with sensible defaults.

The five vars (from .planning/gpu-offload/00-RUNPOD-SETUP.md §3):
  ARTIFACT_S3_ENDPOINT    e.g. https://<acct>.r2.cloudflarestorage.com
  ARTIFACT_S3_BUCKET      e.g. autoposter-gpu-artifacts
  ARTIFACT_S3_ACCESS_KEY
  ARTIFACT_S3_SECRET_KEY
  ARTIFACT_S3_REGION      "auto" for R2; the bucket region (e.g. "nyc3") for Spaces
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ArtifactStoreConfig:
    """Connection settings for any S3-compatible endpoint (R2 / Spaces / S3)."""

    endpoint: str
    bucket: str
    access_key: str
    secret_key: str
    region: str = "auto"

    @classmethod
    def from_env(cls) -> "ArtifactStoreConfig":
        """Build a config from the ARTIFACT_S3_* environment variables.

        Missing string vars default to "" so callers fail fast at first use
        rather than at import time. ``region`` defaults to "auto" (R2).
        """
        return cls(
            endpoint=os.environ.get("ARTIFACT_S3_ENDPOINT", ""),
            bucket=os.environ.get("ARTIFACT_S3_BUCKET", ""),
            access_key=os.environ.get("ARTIFACT_S3_ACCESS_KEY", ""),
            secret_key=os.environ.get("ARTIFACT_S3_SECRET_KEY", ""),
            region=os.environ.get("ARTIFACT_S3_REGION", "auto"),
        )
