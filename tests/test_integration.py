"""
Integration tests — end-to-end pipeline tests with mocked external dependencies.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from longform.deterministic_sync import align_scenes_deterministic
from longform.effects import assign_effects
from longform.models import Scene
from longform.parser import parse_assets_text
from longform.sarvam_sync import _build_sarvam_payload, _parse_sarvam_response


# ---------------------------------------------------------------------------
# Full pipeline integration tests
# ---------------------------------------------------------------------------

FULL_ASSETS = '''\
🎬 SCENE 001 | ⏱️ [00:00 ➔ 00:07]
🗣️ Line: "The first scene narration text."
🎯 Hero Word: [FIRST]
   🎥 Stock Video : https://example.com/clip1.mp4
   🤣 Meme/GIF   : https://example.com/gif1.mp4
-------------------------------------------------------------------------------------
🎬 SCENE 002 | ⏱️ [00:07 ➔ 00:14]
🗣️ Line: "The second scene has different words."
🎯 Hero Word: [SECOND]
   🎥 Stock Video : https://example.com/clip2.mp4
-------------------------------------------------------------------------------------
🎬 SCENE 003 | ⏱️ [00:14 ➔ 00:21]
🗣️ Line: "Third scene wraps up the narration."
🎯 Hero Word: [THIRD]
   🎥 Stock Video : https://example.com/clip3.mp4
   🤣 Meme/GIF   : https://example.com/gif3.mp4
-------------------------------------------------------------------------------------
'''


class TestFullPipelineIntegration:
    """Integration tests that exercise multiple modules together."""

    def test_parse_then_sync(self):
        """Parse assets -> deterministic sync -> verify all scenes have timing."""
        scenes = parse_assets_text(FULL_ASSETS)
        assert len(scenes) == 3

        words = [
            {"word": "The", "start": 0, "end": 200},
            {"word": "first", "start": 200, "end": 400},
            {"word": "scene", "start": 400, "end": 600},
            {"word": "narration", "start": 600, "end": 900},
            {"word": "text", "start": 900, "end": 1100},
            {"word": "The", "start": 1100, "end": 1300},
            {"word": "second", "start": 1300, "end": 1600},
            {"word": "scene", "start": 1600, "end": 1800},
            {"word": "has", "start": 1800, "end": 2000},
            {"word": "different", "start": 2000, "end": 2400},
            {"word": "words", "start": 2400, "end": 2700},
            {"word": "Third", "start": 2700, "end": 3000},
            {"word": "scene", "start": 3000, "end": 3200},
            {"word": "wraps", "start": 3200, "end": 3500},
            {"word": "up", "start": 3500, "end": 3700},
            {"word": "the", "start": 3700, "end": 3900},
            {"word": "narration", "start": 3900, "end": 4300},
        ]
        scenes = align_scenes_deterministic(scenes, words, 5000)
        for s in scenes:
            assert s.has_timing
        assert scenes[0].start_ms == 0
        assert scenes[-1].end_ms == 5000

    def test_parse_then_assign_effects(self):
        """Parse -> assign effects -> verify all effects are valid."""
        from longform.config import EFFECTS

        scenes = parse_assets_text(FULL_ASSETS)
        effects = assign_effects(len(scenes))
        assert len(effects) == len(scenes)
        for e in effects:
            assert e in EFFECTS

    def test_parse_then_sarvam_payload(self):
        """Parse -> build Sarvam payload -> verify it contains scene data."""
        scenes = parse_assets_text(FULL_ASSETS)
        words = [{"word": "test", "start": 0, "end": 100}]
        payload = _build_sarvam_payload(scenes, words)
        content = payload["messages"][0]["content"]
        for s in scenes:
            assert s.text in content

    def test_parse_then_validate_all_scenes(self):
        """Parse -> validate all scenes pass validation."""
        from longform.parser import validate_scene

        scenes = parse_assets_text(FULL_ASSETS)
        for s in scenes:
            errors = validate_scene(s)
            assert errors == [], f"Scene {s.scene_id} has errors: {errors}"

    def test_full_sync_coverage(self):
        """Verify deterministic sync covers 0 to total_duration."""
        scenes = parse_assets_text(FULL_ASSETS)
        words = [
            {"word": "first", "start": 0, "end": 500},
            {"word": "second", "start": 500, "end": 1000},
            {"word": "third", "start": 1000, "end": 1500},
        ]
        total = 10000
        scenes = align_scenes_deterministic(scenes, words, total)
        assert scenes[0].start_ms == 0
        assert scenes[-1].end_ms == total

    def test_sarvam_response_roundtrip(self):
        """Build payload -> parse mock response -> verify timing assigned."""
        scenes = parse_assets_text(FULL_ASSETS)
        words = [{"word": "test", "start": 0, "end": 100}]
        payload = _build_sarvam_payload(scenes, words)

        # Simulate Sarvam response
        mapping = [
            {"scene_id": "001", "start_time": 0, "end_time": 3000},
            {"scene_id": "002", "start_time": 3000, "end_time": 6000},
            {"scene_id": "003", "start_time": 6000, "end_time": 10000},
        ]
        mock_response = {
            "choices": [{"message": {"content": json.dumps(mapping)}}]
        }
        result = _parse_sarvam_response(mock_response, scenes)
        assert result[0].start_ms == 0
        assert result[1].start_ms == 3000
        assert result[2].end_ms == 10000

    def test_effect_assignment_matches_scene_count(self):
        """Effects count must match scene count."""
        scenes = parse_assets_text(FULL_ASSETS)
        effects = assign_effects(len(scenes))
        assert len(effects) == len(scenes)

    def test_scenes_are_contiguous_after_sync(self):
        """After sync, adjacent scenes must share boundaries."""
        scenes = parse_assets_text(FULL_ASSETS)
        words = [
            {"word": "first", "start": 0, "end": 500},
            {"word": "second", "start": 500, "end": 1000},
            {"word": "third", "start": 1000, "end": 1500},
        ]
        scenes = align_scenes_deterministic(scenes, words, 5000)
        for i in range(len(scenes) - 1):
            assert scenes[i].end_ms == scenes[i + 1].start_ms

    def test_non_overlapping_scenes(self):
        """No scene should overlap with the next."""
        scenes = parse_assets_text(FULL_ASSETS)
        words = [
            {"word": "first", "start": 0, "end": 500},
            {"word": "second", "start": 500, "end": 1000},
            {"word": "third", "start": 1000, "end": 1500},
        ]
        scenes = align_scenes_deterministic(scenes, words, 5000)
        for i in range(len(scenes) - 1):
            assert scenes[i].end_ms <= scenes[i + 1].start_ms

    def test_all_durations_positive(self):
        """Every scene must have a positive duration after sync."""
        scenes = parse_assets_text(FULL_ASSETS)
        words = [
            {"word": "first", "start": 0, "end": 500},
            {"word": "second", "start": 500, "end": 1000},
            {"word": "third", "start": 1000, "end": 1500},
        ]
        scenes = align_scenes_deterministic(scenes, words, 5000)
        for s in scenes:
            assert s.duration_ms > 0


# ---------------------------------------------------------------------------
# Main entry point tests
# ---------------------------------------------------------------------------

class TestMainEntryPoint:
    """Test the main.py argument parser and pipeline orchestration."""

    def test_parse_args_defaults(self):
        from main import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = ["main.py"]
        try:
            args = parse_args()
            assert args.assets_file == "youtube_longform_assets.txt"
            assert args.audio_file == "full_audio.mp3"
            assert args.sync_method == "deterministic"
            assert args.output == "output/final_video.mp4"
            assert args.no_deliver is False
        finally:
            sys.argv = old_argv

    def test_parse_args_custom(self):
        from main import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = [
            "main.py",
            "--assets-file", "custom.txt",
            "--audio-file", "audio.wav",
            "--sync-method", "sarvam",
            "--output", "out/video.mp4",
            "--no-deliver",
        ]
        try:
            args = parse_args()
            assert args.assets_file == "custom.txt"
            assert args.audio_file == "audio.wav"
            assert args.sync_method == "sarvam"
            assert args.output == "out/video.mp4"
            assert args.no_deliver is True
        finally:
            sys.argv = old_argv

    def test_parse_args_invalid_sync_method(self):
        from main import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = ["main.py", "--sync-method", "invalid"]
        try:
            with pytest.raises(SystemExit):
                parse_args()
        finally:
            sys.argv = old_argv

    def test_parse_args_whisper_model_choices(self):
        from main import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = ["main.py", "--whisper-model", "medium"]
        try:
            args = parse_args()
            assert args.whisper_model == "medium"
        finally:
            sys.argv = old_argv

    def test_parse_args_invalid_whisper_model(self):
        from main import parse_args
        import sys
        old_argv = sys.argv
        sys.argv = ["main.py", "--whisper-model", "huge"]
        try:
            with pytest.raises(SystemExit):
                parse_args()
        finally:
            sys.argv = old_argv
