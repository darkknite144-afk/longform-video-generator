"""
Zero-RAM muxing: concatenate scene clips (stream copy) and mux the
master audio track over the concatenated visuals.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

from .video_processor import _run

log = logging.getLogger("longform")


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


def write_concat_list(clip_paths: list[Path], work_dir: Path) -> Path:
    """Write the FFmpeg concat list file. Returns the path to the list."""
    concat_list = work_dir / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in clip_paths:
            f.write(f"file '{p.resolve().as_posix()}'\n")
    return concat_list


def concat_and_mux(
    clip_paths: list[Path],
    audio_path: Path,
    output_path: Path,
    work_dir: Path,
) -> None:
    """Concatenate clips (stream copy) and mux the audio track.

    Also logs a warning if the final video and audio durations differ
    by more than 1 second.
    """
    concat_list = write_concat_list(clip_paths, work_dir)
    master_visual = work_dir / "master_visual.mp4"

    log.info("Concatenating %d clips (stream copy, no re-encode)", len(clip_paths))
    _run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-c", "copy",
        str(master_visual),
    ])

    log.info("Muxing master audio track over concatenated visuals")
    _run([
        "ffmpeg", "-y",
        "-i", str(master_visual),
        "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        str(output_path),
    ])

    # Diagnostic safety net
    video_len = _probe_duration_sec(output_path)
    audio_len = _probe_duration_sec(audio_path)
    if video_len is not None and audio_len is not None and abs(video_len - audio_len) > 1.0:
        log.warning(
            "Final video (%.1fs) and source audio (%.1fs) differ by %.1fs — "
            "check the sync stage output if this keeps happening.",
            video_len,
            audio_len,
            abs(video_len - audio_len),
        )
