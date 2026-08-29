"""ChatterboxConfig dataclass — mirrors the chatterbox-tts POST /jobs request body."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ChatterboxConfig:
    """Mirrors the POST /jobs request body fields for the chatterbox-tts service."""

    voice_ref: Optional[str] = None
    exaggeration: float = 0.5
    cfg_weight: float = 0.5
    temperature: float = 0.8
    device: str = "cpu"
    language_id: str = "en"
    seed: Optional[int] = None
    sentence_chunk_size: int = 3
    pronunciations: Dict[str, str] = field(default_factory=dict)

    # --- Transport backend selection (phase 07; env-driven at construct time).
    # TRANSPORT config only — deliberately ABSENT from to_request_body() so the
    # local /jobs body stays byte-identical. The runpod backend reads endpoint+key
    # off these; the local backend ignores them. ---
    backend: str = field(
        default_factory=lambda: os.environ.get("CHATTERBOX_BACKEND", "local")
    )
    runpod_endpoint_id: str = field(
        default_factory=lambda: os.environ.get("RUNPOD_CHATTERBOX_ENDPOINT_ID", "")
    )
    runpod_api_key: str = field(
        default_factory=lambda: os.environ.get("RUNPOD_API_KEY", "")
    )

    def to_request_body(self, text: str, request_id: Optional[str] = None) -> dict:
        """Render to the JSON shape POST /jobs expects."""
        return {
            "text": text,
            "voice_ref": self.voice_ref,
            "exaggeration": self.exaggeration,
            "cfg_weight": self.cfg_weight,
            "temperature": self.temperature,
            "device": self.device,
            "language_id": self.language_id,
            "seed": self.seed,
            "sentence_chunk_size": self.sentence_chunk_size,
            "pronunciations": self.pronunciations,
            "request_id": request_id,
        }
