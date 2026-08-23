#!/usr/bin/env python3
"""
Automated YouTube Long-Form Video Generator — main entry point.

Pipeline: parse script → whisper transcription → scene/timestamp sync
(deterministic or Sarvam AI) → per-scene FFmpeg processing → concat mux
→ motion graphics overlay → Telegram delivery.

Usage:
  python main.py --assets-file youtube_longform_assets.txt \
                 --audio-file full_audio.mp3 \
                 --sync-method deterministic \
                 --output output/final_video.mp4

For Sarvam AI sync:
  export SARVAM_API_KEY="your-key-here"
  python main.py --sync-method sarvam ...

Motion graphics can be disabled with --no-motion-graphics.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path

from longform.config import FPS
from longform.deterministic_sync import align_scenes_deterministic, compute_scene_frame_counts
from longform.effects import assign_effects
from longform.motion_graphics import apply_motion_graphics
from longform.muxer import concat_and_mux
from longform.parser import parse_assets_file
from longform.sarvam_sync import (
    fallback_proportional_sync,
    get_api_key,
    map_scenes_with_sarvam,
)
from longform.telegram_delivery import upload_to_telegram
from longform.video_processor import process_scene
from longform.whisper_transcribe import save_whisper_words, transcribe_audio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("longform")

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Automated long-form video generator")
    p.add_argument("--assets-file", default="youtube_longform_assets.txt")
    p.add_argument("--audio-file", default="full_audio.mp3")
    p.add_argument("--whisper-model", default="small", choices=["tiny", "base", "small", "medium"])
    p.add_argument(
        "--sync-method",
        default="deterministic",
        choices=["deterministic", "sarvam"],
        help="deterministic (default, recommended) or sarvam (LLM-based)",
    )
    p.add_argument("--output", default="output/final_video.mp4")
    p.add_argument("--work-dir", default="work")
    p.add_argument("--caption", default="Automated Long-Form Video")
    p.add_argument("--no-deliver", action="store_true", help="Skip Telegram upload")
    p.add_argument(
        "--no-motion-graphics",
        action="store_true",
        help="Skip motion graphics overlay (lower-thirds, progress bar, watermark)",
    )
    return p.parse_args()

def run_pipeline(args: argparse.Namespace) -> Path:
    """Run the full video generation pipeline. Returns the output path."""
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg not found on PATH. Install it before running this script.")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Parse
    scenes = parse_assets_file(args.assets_file)

    # 2a. Transcribe
    words, audio_duration_ms = transcribe_audio(args.audio_file, args.whisper_model)
    save_whisper_words(words, str(work_dir / "whisper_words.json"))

    # 2b. Sync
    if args.sync_method == "sarvam":
        try:
            api_key = get_api_key()
            scenes = map_scenes_with_sarvam(scenes, words, api_key)
        except Exception as e:
            log.error("Sarvam mapping failed (%s); falling back to proportional sync", e)
            scenes = fallback_proportional_sync(scenes, words)
    else:
        scenes = align_scenes_deterministic(scenes, words, audio_duration_ms)

    scenes.sort(key=lambda s: s.start_ms or 0)
    (work_dir / "scene_timing.json").write_text(
        json.dumps(
            [
                {"scene_id": s.scene_id, "start_ms": s.start_ms, "end_ms": s.end_ms}
                for s in scenes
            ]
        ),
        encoding="utf-8",
    )

    # 3. Process each scene
    effects = assign_effects(len(scenes))
    (work_dir / "scene_effects.json").write_text(
        json.dumps({s.scene_id: e for s, e in zip(scenes, effects)}),
        encoding="utf-8",
    )

    scene_frames = compute_scene_frame_counts(scenes, FPS)

    clip_paths = [
        process_scene(s, effect, work_dir, n_frames)
        for s, effect, n_frames in zip(scenes, effects, scene_frames)
    ]

    # 4. Concat + mux
    concat_and_mux(clip_paths, Path(args.audio_file), output_path, work_dir)
    log.info("Final video ready: %s (%.1f MB)", output_path, output_path.stat().st_size / 1e6)

    # 5. Motion graphics overlay (post-processing)
    if not args.no_motion_graphics:
        audio_duration_sec = audio_duration_ms / 1000.0
        mg_output = work_dir / "final_with_motion_graphics.mp4"
        try:
            apply_motion_graphics(
                output_path,
                scenes,
                work_dir,
                mg_output,
                audio_duration_sec=audio_duration_sec,
            )
            # Replace the original output with motion-graphics version
            shutil.copy2(mg_output, output_path)
            log.info("Motion graphics applied to final video")
        except Exception as e:
            log.warning("Motion graphics failed (%s); keeping original video", e)
            # Original output_path is still intact
    else:
        log.info("Motion graphics skipped (--no-motion-graphics)")

    return output_path

def main() -> None:
    """Main entry point."""
    args = parse_args()
    output_path = run_pipeline(args)

    # 6. Deliver via Telegram MTProto
    if not args.no_deliver:
        asyncio.run(upload_to_telegram(output_path, caption=args.caption))

if __name__ == "__main__":
    main()
