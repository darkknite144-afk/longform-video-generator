"""
Parsing module: reads the emoji-labelled "smart-paced" assets file and
produces a list of Scene objects.

Expected format (one block per scene)::

    SCENE 001 | [00:00 -> 00:07]
    Line: "The narration text spoken during this scene."
    Hero Word: [SOME VISUAL CUE]
       Stock Video : https://videos.pexels.com/video-files/xxxx/xxxx.mp4
       Meme/GIF   : https://media.giphy.com/media/xxxx/giphy.mp4
    -------------------------------------------------------------------

Matching is done on literal "SCENE", "Line:", "Stock Video :" and
"Meme/GIF :" text rather than the emoji glyphs themselves, so it's robust
to minor emoji/encoding differences between editors. Parsing is
scene-block-scoped (split on each "SCENE N |" header) so a scene missing
a field never accidentally picks up its neighbour's.
"""

from __future__ import annotations

import re
from pathlib import Path

from .models import Scene

# Split the file into per-scene blocks first, then pull each field out of
# its own block. Matching fields against the whole file with one big lazy
# regex risks a field from scene N+1 leaking into scene N whenever a field
# is missing/reordered — scoping to the block rules that out entirely.
SCENE_BLOCK_RE = re.compile(r"SCENE\s+(?P<id>\d+)\s*\|.*?(?=SCENE\s+\d+\s*\||\Z)", re.DOTALL)
LINE_RE = re.compile(r'Line:\s*"(?P<text>.*?)"', re.DOTALL)
STOCK_VIDEO_RE = re.compile(r"Stock Video\s*:\s*(?P<url>\S+)")
GIF_RE = re.compile(r"Meme/GIF\s*:\s*(?P<url>\S+)")


def parse_assets_file(path: str) -> list[Scene]:
    """Parse the assets file at *path* into a list of Scene objects.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no scenes are parsed (empty file or wrong format).
    """
    raw = Path(path).read_text(encoding="utf-8")
    return parse_assets_text(raw)


def parse_assets_text(raw: str) -> list[Scene]:
    """Parse a raw text string into a list of Scene objects.

    Separated from parse_assets_file so tests can pass strings directly
    without creating temp files.
    """
    scenes: list[Scene] = []
    for block_m in SCENE_BLOCK_RE.finditer(raw):
        block = block_m.group(0)
        line_m = LINE_RE.search(block)
        stock_m = STOCK_VIDEO_RE.search(block)
        if not (line_m and stock_m):
            continue
        gif_m = GIF_RE.search(block)
        scenes.append(
            Scene(
                scene_id=block_m.group("id"),
                text=line_m.group("text").strip(),
                video_url=stock_m.group("url").strip(),
                gif_url=gif_m.group("url").strip() if gif_m else None,
            )
        )
    if not scenes:
        raise ValueError(
            "No scenes parsed from input. Check that it matches the "
            "emoji-labelled scene format documented in the parser module."
        )
    return scenes


def count_scenes(raw: str) -> int:
    """Count the number of scene blocks in *raw* without parsing them."""
    return len(SCENE_BLOCK_RE.findall(raw))


def validate_scene(scene: Scene) -> list[str]:
    """Return a list of validation error messages for *scene* (empty if OK)."""
    errors: list[str] = []
    if not scene.scene_id:
        errors.append("scene_id is empty")
    if not scene.text:
        errors.append("text is empty")
    if not scene.video_url:
        errors.append("video_url is empty")
    elif not scene.video_url.startswith(("http://", "https://")):
        errors.append(f"video_url is not a valid URL: {scene.video_url}")
    if scene.gif_url is not None and not scene.gif_url.startswith(("http://", "https://")):
        errors.append(f"gif_url is not a valid URL: {scene.gif_url}")
    if scene.has_timing:
        if scene.start_ms < 0:
            errors.append(f"start_ms is negative: {scene.start_ms}")
        if scene.end_ms < 0:
            errors.append(f"end_ms is negative: {scene.end_ms}")
        if scene.end_ms < scene.start_ms:
            errors.append(f"end_ms ({scene.end_ms}) < start_ms ({scene.start_ms})")
    return errors
