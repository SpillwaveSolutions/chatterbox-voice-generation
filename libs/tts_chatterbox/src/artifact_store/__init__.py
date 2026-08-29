"""artifact_store — cross-cloud S3-compatible artifact bus for GPU offload.

A thin sync boto3 wrapper around any S3-compatible endpoint (Cloudflare R2 /
DigitalOcean Spaces / AWS S3). Provides upload/download, content-hash dedupe
(``ensure_uploaded``), and presigned GET/PUT URL generation — with the
checksum-disabling boto3 Config that keeps presigned PUTs from 403-ing against
R2/Spaces.
"""
from __future__ import annotations

from artifact_store.config import ArtifactStoreConfig
from artifact_store.store import ArtifactStore

__all__ = ["ArtifactStoreConfig", "ArtifactStore"]
