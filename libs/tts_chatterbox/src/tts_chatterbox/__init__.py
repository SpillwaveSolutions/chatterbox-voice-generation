"""tts_chatterbox: HTTP client + config dataclass for the chatterbox-tts service."""
from tts_chatterbox.client import synthesize  # noqa: F401
from tts_chatterbox.config import ChatterboxConfig  # noqa: F401

__all__ = ["synthesize", "ChatterboxConfig"]
