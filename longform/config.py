"""
Configuration constants for the long-form video generator.

All tunable parameters live here so they can be adjusted in one place
and easily overridden in tests.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sarvam AI API (replaces Gemini)
# ---------------------------------------------------------------------------

SARVAM_MODEL = "sarvam-105b"
SARVAM_URL = "https://api.sarvam.ai/v1/chat/completions"
SARVAM_MAX_RETRIES = 5
SARVAM_RETRY_WAIT_SECONDS = 20  # handle 503 / 429 "server unavailable"

# Environment variable name for the API key — user will set this later.
SARVAM_API_KEY_ENV = "SARVAM_API_KEY"

# ---------------------------------------------------------------------------
# Video rendering constants
# ---------------------------------------------------------------------------

VIDEO_W, VIDEO_H, FPS = 1920, 1080, 30

# Intermediate render size before zoompan — gives the zoom/pan headroom to
# work with and reduces the jitter/pixelation zoompan produces on 1:1 sources.
ZOOM_SRC_W, ZOOM_SRC_H = 2560, 1440

ZOOM_MAX = 1.15             # zoom_in / zoom_out / fade effects: 1.0 <-> 1.15x
PAN_ZOOM = 1.08            # fixed zoom used by pure pan effects
ZOOM_PAN_COMBO_MAX = 1.12   # zoom level used by combined zoom+pan effects
FADE_SECONDS = 0.4
GIF_HEIGHT_FRAC = 0.48     # Meme/GIF overlay height as a fraction of frame height

# Rotated per-scene so the whole video isn't the same slow zoom-in over and
# over. assign_effects() cycles through these and avoids back-to-back repeats.
EFFECTS = [
    "zoom_in", "zoom_out",
    "pan_left", "pan_right", "pan_up", "pan_down",
    "zoom_in_pan_left", "zoom_in_pan_right",
    "fade_in", "fade_out",
]

# ---------------------------------------------------------------------------
# Network / download
# ---------------------------------------------------------------------------

DOWNLOAD_TIMEOUT = 120
DOWNLOAD_CHUNK = 1024 * 1024  # 1MB streaming chunks
DOWNLOAD_RETRIES = 3
DOWNLOAD_RETRY_WAIT = 3

# ---------------------------------------------------------------------------
# Effect names (kept here so tests can import and verify the full set)
# ---------------------------------------------------------------------------

VALID_EFFECTS = frozenset(EFFECTS)
