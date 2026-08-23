"""
Automated YouTube Long-Form Video Generator
============================================

Pipeline: parse script → whisper transcription → deterministic scene/
timestamp alignment → per-scene FFmpeg processing (varied Ken Burns
motion) → zero-RAM concat mux → Telegram delivery via MTProto.

Optional Sarvam AI-based sync (--sync-method sarvam) replaces the
original Gemini-based approach.
"""

from .config import (
    EFFECTS,
    FPS,
    GIF_HEIGHT_FRAC,
    VIDEO_H,
    VIDEO_W,
)
from .models import Scene

__version__ = "2.0.0"
__all__ = [
    "Scene",
    "EFFECTS",
    "FPS",
    "VIDEO_W",
    "VIDEO_H",
    "GIF_HEIGHT_FRAC",
    "__version__",
]
