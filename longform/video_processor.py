"""
Video processing engine: per-scene FFmpeg processing with varied
Ken Burns motion, optional GIF overlay, and exact-frame output.

Each scene is processed in a single FFmpeg pass:
- loops the source indefinitely (-stream_loop -1) so a short stock clip
  always covers the scene
- cuts to an EXACT frame count with -frames:v (not -t, to avoid
  per-scene rounding error accumulation)
- locks a constant frame rate with fps= before zoompan
- scales/crops to a Ken-Burns-friendly intermediate size
- applies the assigned zoompan motion effect
- optionally overlays a Meme/GIF as picture-in-picture near the end
- locks output to 1920x1080 @ 30fps yuv420p, audio stripped (-an)
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

from .config import (
    DOWNLOAD_CHUNK,
    DOWNLOAD_RETRIES,
    DOWNLOAD_RETRY_WAIT,
    DOWNLOAD_TIMEOUT,
    FADE_SECONDS,
    GIF_HEIGHT_FRAC,
    FPS,
    VIDEO_H,
    VIDEO_W,
    ZOOM_SRC_H,
    ZOOM_SRC_W,
)
from .effects import zoompan_expr
from .models import Scene

log = logging.getLogger("longform")


def _run(cmd: list[str]) -> None:
    """Run a subprocess command, raising on failure with last 4000 chars of output."""
    log.debug("RUN: %s", " ".join(cmd))
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\n{result.stdout[-4000:]}")


def download_file(url: str, dest: Path, retries: int = DOWNLOAD_RETRIES) -> None:
    """Download *url* to *dest* with streaming and retry logic.

    Raises RuntimeError if all retries are exhausted.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; LongformBot/1.0)"}
    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT, headers=headers) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=DOWNLOAD_CHUNK):
                        if chunk:
                            f.write(chunk)
            return
        except requests.exceptions.RequestException as e:
            last_err = e
            log.warning(
                "Download failed (attempt %d/%d) for %s: %s",
                attempt,
                retries,
                url,
                e,
            )
            time.sleep(DOWNLOAD_RETRY_WAIT)
    raise RuntimeError(f"Failed to download {url}: {last_err}")


def build_base_chain(z_expr: str, x_expr: str, y_expr: str) -> str:
    """Build the base FFmpeg filter chain string (pre-fade, pre-overlay).

    Separated from process_scene so it can be unit-tested.
    """
    return (
        f"fps={FPS},"
        f"scale={ZOOM_SRC_W}:{ZOOM_SRC_H}:force_original_aspect_ratio=increase,"
        f"crop={ZOOM_SRC_W}:{ZOOM_SRC_H},"
        f"zoompan=z='{z_expr}':d=1:x='{x_expr}':y='{y_expr}':s={VIDEO_W}x{VIDEO_H}:fps={FPS}"
    )


def with_tail_fade(chain: str, effect: str, duration: float) -> str:
    """Add a fade-in or fade-out to *chain* if the effect calls for it
    and the scene is long enough.
    """
    if effect == "fade_in" and duration > FADE_SECONDS * 1.5:
        return f"{chain},fade=t=in:st=0:d={FADE_SECONDS}:color=black"
    if effect == "fade_out" and duration > FADE_SECONDS * 1.5:
        return f"{chain},fade=t=out:st={duration - FADE_SECONDS:.3f}:d={FADE_SECONDS}:color=black"
    return chain


def compute_gif_overlay_params(duration: float) -> dict:
    """Compute GIF overlay parameters for a scene of *duration* seconds.

    Returns a dict with: has_gif (bool), overlay_dur, overlay_start,
    overlay_end, fade_d, gif_h. If has_gif is False the other fields are
    still computed but should not be used.

    Separated from process_scene so it can be unit-tested.
    """
    overlay_dur = min(1.6, duration * 0.4)
    has_gif = overlay_dur >= 0.6
    overlay_start = max(0.0, duration - overlay_dur - 0.2)
    overlay_end = overlay_start + overlay_dur
    fade_d = min(0.2, overlay_dur / 4)
    gif_h = round(VIDEO_H * GIF_HEIGHT_FRAC)
    return {
        "has_gif": has_gif,
        "overlay_dur": overlay_dur,
        "overlay_start": overlay_start,
        "overlay_end": overlay_end,
        "fade_d": fade_d,
        "gif_h": gif_h,
    }


def build_gif_filter_complex(
    base_chain: str,
    effect: str,
    duration: float,
    overlay_params: dict,
) -> str:
    """Build the full filter_complex string for a scene with GIF overlay.

    Separated from process_scene so it can be unit-tested.
    """
    p = overlay_params
    base_seg = f"[0:v]{with_tail_fade(base_chain, effect, duration)},format=yuv420p[base]"
    gif_seg = (
        f"[1:v]fps={FPS},"
        f"scale=-2:{p['gif_h']}:force_original_aspect_ratio=decrease,"
        "format=yuva420p,"
        "pad=iw+10:ih+10:5:5:color=white@1.0,"
        f"fade=t=in:st=0:d={p['fade_d']:.3f}:alpha=1,"
        f"fade=t=out:st={p['overlay_end'] - p['fade_d']:.3f}:d={p['fade_d']:.3f}:alpha=1[gif]"
    )
    overlay_seg = (
        "[base][gif]overlay=x=(W-w)/2:y=(H-h)/2:"
        f"enable='between(t,{p['overlay_start']:.3f},{p['overlay_end']:.3f})',format=yuv420p[outv]"
    )
    return ";".join([base_seg, gif_seg, overlay_seg])


def process_scene(
    scene: Scene, effect: str, work_dir: Path, total_frames: int
) -> Path:
    """Process a single scene: download -> encode -> delete raw.

    Returns the path to the processed scene clip.
    """
    raw_path = work_dir / f"raw_{scene.safe_id}.mp4"
    gif_path = work_dir / f"gif_{scene.safe_id}.mp4"
    out_path = work_dir / f"scene_{scene.safe_id}_proc.mp4"

    log.info("Scene %s: downloading source clip", scene.scene_id)
    download_file(scene.video_url, raw_path)

    duration = total_frames / FPS
    z_expr, x_expr, y_expr = zoompan_expr(effect, total_frames)
    base_chain = build_base_chain(z_expr, x_expr, y_expr)

    overlay_params = compute_gif_overlay_params(duration)
    has_gif = bool(scene.gif_url) and overlay_params["has_gif"]

    if has_gif:
        try:
            download_file(scene.gif_url, gif_path)
        except Exception as e:
            log.warning(
                "Scene %s: GIF download failed (%s), skipping overlay",
                scene.scene_id,
                e,
            )
            has_gif = False

    if has_gif:
        filter_complex = build_gif_filter_complex(
            base_chain, effect, duration, overlay_params
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(raw_path),
            "-stream_loop", "-1", "-i", str(gif_path),
            "-filter_complex", filter_complex,
            "-map", "[outv]",
            "-frames:v", str(total_frames),
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-fps_mode", "cfr", "-r", str(FPS),
            str(out_path),
        ]
    else:
        vf = f"{with_tail_fade(base_chain, effect, duration)},format=yuv420p"
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", str(raw_path),
            "-vf", vf,
            "-frames:v", str(total_frames),
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-fps_mode", "cfr", "-r", str(FPS),
            str(out_path),
        ]

    log.info(
        "Scene %s: encoding (%.2fs, %d frames, effect=%s%s)",
        scene.scene_id,
        duration,
        total_frames,
        effect,
        "+gif" if has_gif else "",
    )
    _run(cmd)

    raw_path.unlink(missing_ok=True)  # free disk immediately
    gif_path.unlink(missing_ok=True)
    return out_path
