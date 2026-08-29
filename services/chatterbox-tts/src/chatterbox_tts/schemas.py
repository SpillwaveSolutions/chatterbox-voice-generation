"""Pydantic request/response schemas for the chatterbox-tts service.

quick-260602-b7l: SynthesizeRequest / SynthesizeResponse are GONE. The async
job-queue refactor replaces them with JobCreateRequest (POST /jobs body) +
JobStateResponse (GET /jobs/{id} and POST /jobs response body).

Field set is preserved 1:1 with the old SynthesizeRequest so the client
wire shape only changes URL + method, not body schema.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class JobCreateRequest(BaseModel):
    """POST /jobs request body. Caps mirror deck2video defaults."""

    text: str = Field(..., min_length=1, max_length=200_000)
    voice_ref: Optional[str] = None  # filename only; resolved server-side
    exaggeration: float = Field(0.5, ge=0.0, le=2.0)
    cfg_weight: float = Field(0.5, ge=0.0, le=2.0)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    device: str = "cpu"  # accepted but pinned to cpu in container
    language_id: str = "en"
    seed: Optional[int] = None
    sentence_chunk_size: int = Field(3, ge=1, le=10)
    pronunciations: dict[str, str] = Field(default_factory=dict)
    request_id: Optional[str] = None  # row PK; uuid4 hex if missing

    @field_validator("pronunciations")
    @classmethod
    def _cap_pronunciations(cls, v: dict) -> dict:
        # Hard caps mirror deck2video/tts.py PRONUNCIATIONS_MAX_*.
        if len(v) > 1000:
            raise ValueError("pronunciations: max 1000 entries")
        for k, val in v.items():
            if not isinstance(k, str) or not isinstance(val, str):
                raise ValueError(
                    "pronunciations: keys and values must be strings"
                )
            if not k:
                raise ValueError("pronunciations: empty keys not allowed")
            if len(k) > 200 or len(val) > 200:
                raise ValueError(
                    "pronunciations: max 200 chars per key/value"
                )
        return v


class JobStateResponse(BaseModel):
    """Response body for POST /jobs and GET /jobs/{id}.

    ``wav_path`` is set only on state='completed'. It is an ABSOLUTE path
    inside the chatterbox-tts container's filesystem; the flow-runner-side
    client reads it via shutil.copyfile (the ``./output`` mount is identical
    on both containers — quick-260528-njt key_link).
    """

    id: str
    state: str  # queued | running | completed | failed
    wav_path: Optional[str] = None
    error: Optional[str] = None
    queue_position: Optional[int] = None  # null unless state='queued'
