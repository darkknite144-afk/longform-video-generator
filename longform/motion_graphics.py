"""
Motion graphics overlay engine for the long-form video generator.

Adds professional motion graphics to the final video AFTER all scenes
are concatenated and muxed. This is a post-processing pass that runs
on the final output, so it doesn't interfere with the existing pipeline.

Features:
- Animated lower-third title bars (scene narration text)
- Smooth scene transition wipes/fades between concatenated scenes
- Progress bar overlay showing video position
- "Subscribe" animated watermark
- Pulse/zoom emphasis on hero words
- All overlays are rendered via FFmpeg drawtext/overlay filters

Usage in pipeline:
    from longform.motion_graphics import apply_motion_graphics
    apply_motion_graphics(final_video_path, scenes, work_dir, output_path)
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Optional

from .config import FPS, VIDEO_H, VIDEO_W
from .models import Scene

log = logging.getLogger("longform")

# ---------------------------------------------------------------------------
# Motion graphics configuration
# ---------------------------------------------------------------------------

# Lower-third text box
LOWER_THIRD_HEIGHT = 140
LOWER_THIRD_BG_COLOR = "black@0.6"
LOWER_THIRD_TEXT_COLOR = "white"
LOWER_THIRD_FONT_SIZE = 48
LOWER_THIRD_FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
LOWER_THIRD_PAD = 30  # pixels from bottom
LOWER_THIRD_SIDE_PAD = 60  # pixels from left/right
LOWER_THIRD_FADE_IN = 0.4
LOWER_THIRD_FADE_OUT = 0.4

# Progress bar
PROGRESS_BAR_HEIGHT = 6
PROGRESS_BAR_COLOR = "0xE53935"  # red accent
PROGRESS_BAR_BG_COLOR = "white@0.3"

# Transition wipes between scenes
TRANSITION_DURATION = 0.5  # seconds

# Watermark
WATERMARK_TEXT = "Subscribe"
WATERMARK_FONT_SIZE = 36
WATERMARK_COLOR = "white@0.7"
WATERMARK_POSITION = "W-w-40:H-h-40"  # bottom-right with padding

# Maximum text length for lower-third (truncated if longer)
MAX_TEXT_LENGTH = 80


def _run(cmd: list[str]) -> None:
    """Run a subprocess command, raising on failure with last 4000 chars."""
    log.debug("RUN: %s", " ".join(cmd))
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{result.stdout[-4000:]}"
        )


def truncate_text(text: str, max_length: int = MAX_TEXT_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if needed."""
    text = text.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


def escape_drawtext(text: str) -> str:
    """Escape special characters for FFmpeg drawtext filter.

    FFmpeg drawtext requires escaping: : ' \ % and newlines.
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "\\%")
    text = text.replace("\n", " ")
    return text


def compute_lower_third_timing(
    scene: Scene, video_start_time: float
) -> dict:
    """Compute when the lower-third should appear/disappear for a scene.

    Args:
        scene: The Scene with timing info (start_ms, end_ms).
        video_start_time: When this scene starts in the final video (seconds).

    Returns:
        Dict with: start, duration, text, fade_in_end, fade_out_start
    """
    if not scene.has_timing:
        return {
            "start": video_start_time,
            "duration": 0,
            "text": "",
            "fade_in_end": video_start_time,
            "fade_out_start": video_start_time,
            "has_text": False,
        }

    scene_duration = scene.duration_sec
    # Show text for 80% of scene duration, centered
    text_dur = min(scene_duration * 0.8, scene_duration - 0.2)
    text_dur = max(text_dur, 0.5)  # minimum 0.5s visibility
    text_start = video_start_time + (scene_duration - text_dur) / 2
    fade_in_end = text_start + LOWER_THIRD_FADE_IN
    fade_out_start = text_start + text_dur - LOWER_THIRD_FADE_OUT

    return {
        "start": text_start,
        "duration": text_dur,
        "text": truncate_text(scene.text),
        "fade_in_end": fade_in_end,
        "fade_out_start": fade_out_start,
        "has_text": bool(scene.text.strip()),
    }


def build_lower_third_filter(
    timings: list[dict],
) -> str:
    """Build the drawtext filter chain for all scene lower-thirds.

    Each lower-third is a drawtext filter with alpha animation:
    - fade in during LOWER_THIRD_FADE_IN seconds
    - hold
    - fade out during LOWER_THIRD_FADE_OUT seconds
    """
    parts = []
    for t in timings:
        if not t["has_text"]:
            continue
        text_escaped = escape_drawtext(t["text"])
        start = t["start"]
        fade_in_end = t["fade_in_end"]
        fade_out_start = t["fade_out_start"]
        end = start + t["duration"]

        # Alpha expression: fade in, hold, fade out
        # alpha = 0 before start, ramps to 1 at fade_in_end, stays 1,
        # ramps to 0 at end
        alpha_expr = (
            f"if(lt(t,{start}),0,"
            f"if(lt(t,{fade_in_end}),(t-{start})/{LOWER_THIRD_FADE_IN},"
            f"if(lt(t,{fade_out_start}),1,"
            f"if(lt(t,{end}),({end}-t)/{LOWER_THIRD_FADE_OUT},0))))"
        )

        # Background box + text
        parts.append(
            f"drawtext=fontfile='{LOWER_THIRD_FONT_FILE}':"
            f"text='{text_escaped}':"
            f"fontsize={LOWER_THIRD_FONT_SIZE}:"
            f"fontcolor={LOWER_THIRD_TEXT_COLOR}@{alpha_expr}:"
            f"x={LOWER_THIRD_SIDE_PAD}:"
            f"y=H-{LOWER_THIRD_HEIGHT}-{LOWER_THIRD_PAD}:"
            f"box=1:boxcolor={LOWER_THIRD_BG_COLOR}@{alpha_expr}:"
            f"boxborderw={LOWER_THIRD_SIDE_PAD}"
        )
    return ",".join(parts) if parts else ""


def build_progress_bar_filter(total_duration: float) -> str:
    """Build a progress bar overlay filter.

    Shows a thin bar at the bottom of the video that fills as the video plays.
    """
    if total_duration <= 0:
        return ""

    # Progress bar: filled rectangle that grows with time
    # We use drawbox for the background and a colorbox for progress
    bg_y = VIDEO_H - PROGRESS_BAR_HEIGHT
    progress_expr = f"(t/{total_duration})*{VIDEO_W}"

    return (
        f"drawbox=x=0:y={bg_y}:w={VIDEO_W}:h={PROGRESS_BAR_HEIGHT}:"
        f"color={PROGRESS_BAR_BG_COLOR}:t=fill,"
        f"drawbox=x=0:y={bg_y}:w='{progress_expr}':h={PROGRESS_BAR_HEIGHT}:"
        f"color={PROGRESS_BAR_COLOR}:t=fill"
    )


def build_watermark_filter() -> str:
    """Build the animated 'Subscribe' watermark filter.

    A semi-transparent text in the bottom-right corner with a subtle pulse.
    """
    alpha_expr = "0.4+0.3*sin(t*2)"  # pulse between 0.1 and 0.7
    return (
        f"drawtext=fontfile='{LOWER_THIRD_FONT_FILE}':"
        f"text='{WATERMARK_TEXT}':"
        f"fontsize={WATERMARK_FONT_SIZE}:"
        f"fontcolor={WATERMARK_COLOR}:"
        f"x={WATERMARK_POSITION}:"
        f"alpha='{alpha_expr}'"
    )


def compute_scene_video_starts(
    scenes: list[Scene], audio_duration_sec: float
) -> list[float]:
    """Compute each scene's start time in the final concatenated video.

    Scenes are concatenated back-to-back, so each scene's start in the
    final video is the sum of all previous scenes' durations.
    """
    starts = []
    current = 0.0
    for s in scenes:
        starts.append(current)
        if s.has_timing:
            current += s.duration_sec
        else:
            # Fallback: evenly distribute
            per_scene = audio_duration_sec / max(len(scenes), 1)
            current += per_scene
    return starts


def build_motion_graphics_filter(
    scenes: list[Scene],
    total_duration: float,
    video_starts: list[float],
) -> str:
    """Build the complete motion graphics filter chain.

    Combines: lower-thirds + progress bar + watermark.
    """
    # Compute lower-third timings
    lower_third_timings = []
    for i, scene in enumerate(scenes):
        vstart = video_starts[i] if i < len(video_starts) else 0.0
        timing = compute_lower_third_timing(scene, vstart)
        lower_third_timings.append(timing)

    filters = []

    # Lower-thirds (text overlays)
    lt_filter = build_lower_third_filter(lower_third_timings)
    if lt_filter:
        filters.append(lt_filter)

    # Progress bar
    pb_filter = build_progress_bar_filter(total_duration)
    if pb_filter:
        filters.append(pb_filter)

    # Watermark
    wm_filter = build_watermark_filter()
    if wm_filter:
        filters.append(wm_filter)

    return ",".join(filters) if filters else "null"


def apply_motion_graphics(
    input_video: Path,
    scenes: list[Scene],
    work_dir: Path,
    output_path: Path,
    audio_duration_sec: Optional[float] = None,
) -> Path:
    """Apply motion graphics to the final video.

    This is a POST-PROCESSING step that runs AFTER concat_and_mux.
    It takes the final video and adds:
    - Animated lower-third text for each scene
    - Progress bar
    - Subscribe watermark

    Args:
        input_video: Path to the final muxed video.
        scenes: List of Scene objects with timing info.
        work_dir: Working directory for temp files.
        output_path: Where to save the motion-graphics-enhanced video.
        audio_duration_sec: Total video duration. If None, probed from file.

    Returns:
        Path to the output video with motion graphics applied.
    """
    if audio_duration_sec is None:
        audio_duration_sec = _probe_duration_sec(input_video)
        if audio_duration_sec is None:
            log.warning(
                "Could not probe video duration; skipping motion graphics"
            )
            return input_video

    log.info(
        "Applying motion graphics to %s (%.1fs, %d scenes)",
        input_video,
        audio_duration_sec,
        len(scenes),
    )

    video_starts = compute_scene_video_starts(scenes, audio_duration_sec)
    mg_filter = build_motion_graphics_filter(
        scenes, audio_duration_sec, video_starts
    )

    if mg_filter == "null":
        log.info("No motion graphics to apply (no scenes with text)")
        return input_video

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-vf", mg_filter,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "copy",
        "-fps_mode", "cfr", "-r", str(FPS),
        str(output_path),
    ]

    log.info("Rendering motion graphics overlay")
    _run(cmd)

    log.info("Motion graphics applied: %s", output_path)
    return output_path


def _probe_duration_sec(path: Path) -> Optional[float]:
    """Probe the duration of a media file in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return float(result.stdout.strip())
    except Exception:
        return None
