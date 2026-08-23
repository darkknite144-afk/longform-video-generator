"""
Data models for the long-form video generator.

The Scene dataclass is the single shared data structure that every
downstream module depends on — parsing, sync, rendering, muxing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class Scene:
    """A single narrated scene with an associated stock-video clip.

    Attributes:
        scene_id:  Human-readable scene identifier (e.g. "001").
        text:      The narration text spoken during this scene.
        video_url: URL of the HD stock-footage clip for this scene.
        gif_url:   Optional Meme/GIF URL overlaid as picture-in-picture.
        start_ms:  Start time in milliseconds (assigned during sync).
        end_ms:    End time in milliseconds (assigned during sync).
    """
    scene_id: str
    text: str
    video_url: str
    gif_url: Optional[str] = None
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    @property
    def duration_sec(self) -> float:
        """Duration of this scene in seconds. Raises if timing not yet set."""
        if self.start_ms is None or self.end_ms is None:
            raise ValueError(f"Scene {self.scene_id} has no timing yet")
        return max(0.05, (self.end_ms - self.start_ms) / 1000.0)

    @property
    def duration_ms(self) -> int:
        """Duration of this scene in milliseconds. Raises if timing not yet set."""
        if self.start_ms is None or self.end_ms is None:
            raise ValueError(f"Scene {self.scene_id} has no timing yet")
        return max(50, self.end_ms - self.start_ms)

    @property
    def safe_id(self) -> str:
        """Filesystem-safe version of scene_id (non-word chars → '_')."""
        return re.sub(r"[^\w-]", "_", self.scene_id)

    @property
    def has_timing(self) -> bool:
        """True if both start_ms and end_ms have been assigned."""
        return self.start_ms is not None and self.end_ms is not None

    @property
    def word_count(self) -> int:
        """Number of whitespace-separated words in the narration text."""
        return len(self.text.split()) if self.text else 0

    def __repr__(self) -> str:  # pragma: no cover
        timing = f", {self.start_ms}->{self.end_ms}ms" if self.has_timing else ""
        gif = ", +gif" if self.gif_url else ""
        return f"Scene({self.scene_id}, {self.word_count}w{timing}{gif})"
