"""
Tests for longform.motion_graphics — motion graphics overlay engine.

Tests cover:
- Text truncation and escaping
- Lower-third timing computation
- Filter chain building (lower-thirds, progress bar, watermark)
- Scene video start time computation
- Full motion graphics filter assembly
- apply_motion_graphics integration (with mocked subprocess)
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from longform.config import FPS, VIDEO_H, VIDEO_W
from longform.models import Scene
from longform.motion_graphics import (
    MAX_TEXT_LENGTH,
    PROGRESS_BAR_HEIGHT,
    WATERMARK_TEXT,
    apply_motion_graphics,
    build_lower_third_filter,
    build_motion_graphics_filter,
    build_progress_bar_filter,
    build_watermark_filter,
    compute_lower_third_timing,
    compute_scene_video_starts,
    escape_drawtext,
    truncate_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scene(
    sid="001",
    text="Hello world",
    url="https://example.com/v.mp4",
    start_ms=0,
    end_ms=5000,
):
    return Scene(
        scene_id=sid,
        text=text,
        video_url=url,
        start_ms=start_ms,
        end_ms=end_ms,
    )


def make_scenes_no_timing(n=3):
    return [
        Scene(scene_id=f"00{i+1}", text=f"Scene {i+1}", video_url=f"https://example.com/{i+1}.mp4")
        for i in range(n)
    ]


def make_scenes_with_timing():
    return [
        make_scene("001", "First scene narration", start_ms=0, end_ms=5000),
        make_scene("002", "Second scene text", start_ms=5000, end_ms=10000),
        make_scene("003", "Third scene words", start_ms=10000, end_ms=15000),
    ]


# ---------------------------------------------------------------------------
# truncate_text tests
# ---------------------------------------------------------------------------

class TestTruncateText:
    def test_short_text_unchanged(self):
        assert truncate_text("hello") == "hello"

    def test_exact_max_length(self):
        text = "a" * MAX_TEXT_LENGTH
        assert truncate_text(text) == text

    def test_long_text_truncated(self):
        text = "a" * (MAX_TEXT_LENGTH + 10)
        result = truncate_text(text)
        assert len(result) == MAX_TEXT_LENGTH
        assert result.endswith("...")

    def test_custom_max_length(self):
        text = "abcdefghij"
        result = truncate_text(text, max_length=5)
        assert result == "ab..."

    def test_empty_string(self):
        assert truncate_text("") == ""

    def test_whitespace_stripped(self):
        assert truncate_text("  hello  ") == "hello"

    def test_ellipsis_length(self):
        text = "a" * 100
        result = truncate_text(text, max_length=10)
        assert len(result) == 10
        assert result == "aaaaaaa..."


# ---------------------------------------------------------------------------
# escape_drawtext tests
# ---------------------------------------------------------------------------

class TestEscapeDrawtext:
    def test_colon_escaped(self):
        assert escape_drawtext("a:b") == "a\\:b"

    def test_single_quote_escaped(self):
        assert escape_drawtext("it's") == "it\\'s"

    def test_backslash_escaped(self):
        assert escape_drawtext("a\\b") == "a\\\\b"

    def test_percent_escaped(self):
        assert escape_drawtext("100%") == "100\\%"

    def test_newline_replaced(self):
        assert escape_drawtext("a\nb") == "a b"

    def test_no_special_chars(self):
        assert escape_drawtext("hello world") == "hello world"

    def test_multiple_special_chars(self):
        result = escape_drawtext("it's: 100% done\nok")
        assert "\\:" in result
        assert "\\'" in result
        assert "\\%" in result
        assert "\n" not in result


# ---------------------------------------------------------------------------
# compute_lower_third_timing tests
# ---------------------------------------------------------------------------

class TestComputeLowerThirdTiming:
    def test_has_text_true_for_scene_with_text(self):
        scene = make_scene("001", "Hello world", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["has_text"] is True

    def test_has_text_false_for_empty_text(self):
        scene = make_scene("001", "", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["has_text"] is False

    def test_has_text_false_for_no_timing(self):
        scene = Scene(scene_id="001", text="Hello", video_url="url")
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["has_text"] is False

    def test_start_is_video_start_plus_offset(self):
        scene = make_scene("001", "Hello", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["start"] >= 0.0

    def test_duration_positive(self):
        scene = make_scene("001", "Hello", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["duration"] > 0

    def test_duration_minimum_05(self):
        scene = make_scene("001", "Hi", start_ms=0, end_ms=600)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["duration"] >= 0.5

    def test_fade_in_end_after_start(self):
        scene = make_scene("001", "Hello world text", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["fade_in_end"] > timing["start"]

    def test_fade_out_start_before_end(self):
        scene = make_scene("001", "Hello world text", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert timing["fade_out_start"] < timing["start"] + timing["duration"]

    def test_text_is_truncated(self):
        long_text = "word " * 30  # 150 chars
        scene = make_scene("001", long_text, start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 0.0)
        assert len(timing["text"]) <= MAX_TEXT_LENGTH

    def test_video_start_offset_applied(self):
        scene = make_scene("001", "Hello", start_ms=0, end_ms=5000)
        timing = compute_lower_third_timing(scene, 100.0)
        assert timing["start"] >= 100.0


# ---------------------------------------------------------------------------
# build_lower_third_filter tests
# ---------------------------------------------------------------------------

class TestBuildLowerThirdFilter:
    def test_empty_timings_returns_empty(self):
        result = build_lower_third_filter([])
        assert result == ""

    def test_no_text_timings_returns_empty(self):
        timings = [{"has_text": False, "text": "", "start": 0, "fade_in_end": 0,
                      "fade_out_start": 0, "duration": 0}]
        result = build_lower_third_filter(timings)
        assert result == ""

    def test_has_text_returns_drawtext(self):
        timings = [{
            "has_text": True,
            "text": "Hello world",
            "start": 0.0,
            "fade_in_end": 0.4,
            "fade_out_start": 3.6,
            "duration": 4.0,
        }]
        result = build_lower_third_filter(timings)
        assert "drawtext" in result

    def test_contains_fontfile(self):
        timings = [{
            "has_text": True, "text": "Test", "start": 0.0,
            "fade_in_end": 0.4, "fade_out_start": 3.6, "duration": 4.0,
        }]
        result = build_lower_third_filter(timings)
        assert "fontfile" in result

    def test_contains_escaped_text(self):
        timings = [{
            "has_text": True, "text": "Hello: world", "start": 0.0,
            "fade_in_end": 0.4, "fade_out_start": 3.6, "duration": 4.0,
        }]
        result = build_lower_third_filter(timings)
        assert "\\:" in result

    def test_multiple_scenes_joined(self):
        timings = [
            {"has_text": True, "text": "First", "start": 0.0,
             "fade_in_end": 0.4, "fade_out_start": 3.6, "duration": 4.0},
            {"has_text": True, "text": "Second", "start": 5.0,
             "fade_in_end": 5.4, "fade_out_start": 8.6, "duration": 4.0},
        ]
        result = build_lower_third_filter(timings)
        # Should have two drawtext filters joined by comma
        assert result.count("drawtext") == 2

    def test_alpha_expression_present(self):
        timings = [{
            "has_text": True, "text": "Test", "start": 0.0,
            "fade_in_end": 0.4, "fade_out_start": 3.6, "duration": 4.0,
        }]
        result = build_lower_third_filter(timings)
        assert "if(lt(t" in result

    def test_box_parameters_present(self):
        timings = [{
            "has_text": True, "text": "Test", "start": 0.0,
            "fade_in_end": 0.4, "fade_out_start": 3.6, "duration": 4.0,
        }]
        result = build_lower_third_filter(timings)
        assert "box=1" in result
        assert "boxcolor" in result


# ---------------------------------------------------------------------------
# build_progress_bar_filter tests
# ---------------------------------------------------------------------------

class TestBuildProgressBarFilter:
    def test_positive_duration(self):
        result = build_progress_bar_filter(10.0)
        assert "drawbox" in result

    def test_zero_duration_returns_empty(self):
        result = build_progress_bar_filter(0)
        assert result == ""

    def test_negative_duration_returns_empty(self):
        result = build_progress_bar_filter(-1)
        assert result == ""

    def test_contains_video_width(self):
        result = build_progress_bar_filter(10.0)
        assert str(VIDEO_W) in result

    def test_contains_progress_expression(self):
        result = build_progress_bar_filter(10.0)
        assert "t/" in result  # t/duration expression

    def test_has_bg_and_progress(self):
        result = build_progress_bar_filter(10.0)
        assert result.count("drawbox") == 2  # bg + progress

    def test_correct_y_position(self):
        result = build_progress_bar_filter(10.0)
        expected_y = VIDEO_H - PROGRESS_BAR_HEIGHT
        assert str(expected_y) in result


# ---------------------------------------------------------------------------
# build_watermark_filter tests
# ---------------------------------------------------------------------------

class TestBuildWatermarkFilter:
    def test_returns_drawtext(self):
        result = build_watermark_filter()
        assert "drawtext" in result

    def test_contains_subscribe_text(self):
        result = build_watermark_filter()
        assert WATERMARK_TEXT in result

    def test_contains_alpha_pulse(self):
        result = build_watermark_filter()
        assert "sin(t" in result  # pulse expression

    def test_contains_fontfile(self):
        result = build_watermark_filter()
        assert "fontfile" in result


# ---------------------------------------------------------------------------
# compute_scene_video_starts tests
# ---------------------------------------------------------------------------

class TestComputeSceneVideoStarts:
    def test_first_scene_starts_at_zero(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        assert starts[0] == 0.0

    def test_second_scene_starts_after_first(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        assert starts[1] == pytest.approx(5.0)

    def test_third_scene_starts_after_two(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        assert starts[2] == pytest.approx(10.0)

    def test_no_timing_uses_even_distribution(self):
        scenes = make_scenes_no_timing(3)
        starts = compute_scene_video_starts(scenes, 15.0)
        assert starts[0] == 0.0
        assert starts[1] == pytest.approx(5.0)
        assert starts[2] == pytest.approx(10.0)

    def test_single_scene(self):
        scene = [make_scene("001", "Hello", start_ms=0, end_ms=5000)]
        starts = compute_scene_video_starts(scene, 5.0)
        assert starts == [0.0]

    def test_empty_scenes(self):
        starts = compute_scene_video_starts([], 10.0)
        assert starts == []

    def test_starts_are_monotonic_increasing(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        for i in range(1, len(starts)):
            assert starts[i] >= starts[i - 1]


# ---------------------------------------------------------------------------
# build_motion_graphics_filter tests
# ---------------------------------------------------------------------------

class TestBuildMotionGraphicsFilter:
    def test_returns_filter_string(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        result = build_motion_graphics_filter(scenes, 15.0, starts)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_contains_drawtext(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        result = build_motion_graphics_filter(scenes, 15.0, starts)
        assert "drawtext" in result

    def test_contains_drawbox(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        result = build_motion_graphics_filter(scenes, 15.0, starts)
        assert "drawbox" in result

    def test_empty_scenes_still_has_watermark_and_progress(self):
        """Even with no scenes, watermark + progress bar should be present."""
        result = build_motion_graphics_filter([], 10.0, [])
        # Should not be "null" — watermark is always added
        assert "drawtext" in result  # watermark
        assert "drawbox" in result   # progress bar

    def test_scenes_without_timing_uses_fallback(self):
        scenes = make_scenes_no_timing(2)
        starts = compute_scene_video_starts(scenes, 10.0)
        result = build_motion_graphics_filter(scenes, 10.0, starts)
        assert "drawbox" in result  # progress bar

    def test_zero_duration_still_has_watermark(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 0)
        result = build_motion_graphics_filter(scenes, 0, starts)
        # With 0 duration, progress bar is empty but watermark should still be there
        assert "drawtext" in result  # at least watermark


# ---------------------------------------------------------------------------
# apply_motion_graphics tests (mocked subprocess)
# ---------------------------------------------------------------------------

class TestApplyMotionGraphics:
    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_applies_motion_graphics(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 15.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake video")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(
            input_video, scenes, tmp_path, output,
            audio_duration_sec=15.0,
        )

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-vf" in cmd
        assert str(output) in cmd

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_probes_duration_if_not_given(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 12.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(input_video, scenes, tmp_path, output)

        mock_probe.assert_called_once_with(input_video)
        mock_run.assert_called_once()

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_returns_input_if_probe_fails(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = None
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        result = apply_motion_graphics(input_video, scenes, tmp_path, output)
        assert result == input_video
        mock_run.assert_not_called()

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_uses_provided_duration(self, mock_run, mock_probe, tmp_path):
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(
            input_video, scenes, tmp_path, output,
            audio_duration_sec=20.0,
        )
        # Should not probe since duration was given
        mock_probe.assert_not_called()
        mock_run.assert_called_once()

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_cmd_contains_libx264(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 15.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(
            input_video, scenes, tmp_path, output,
            audio_duration_sec=15.0,
        )
        cmd = mock_run.call_args[0][0]
        assert "libx264" in cmd

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_cmd_contains_fps(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 15.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(
            input_video, scenes, tmp_path, output,
            audio_duration_sec=15.0,
        )
        cmd = mock_run.call_args[0][0]
        assert str(FPS) in cmd

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_cmd_copies_audio(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 15.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(
            input_video, scenes, tmp_path, output,
            audio_duration_sec=15.0,
        )
        cmd = mock_run.call_args[0][0]
        assert "-c:a" in cmd
        a_idx = cmd.index("-c:a")
        assert cmd[a_idx + 1] == "copy"

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_empty_scenes_no_filter_skips(self, mock_run, mock_probe, tmp_path):
        # With no scenes, filter would be "null" (just watermark)
        # which should still run ffmpeg but with minimal filter
        mock_probe.return_value = 10.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"

        # Empty scenes → no lower-thirds, but progress bar + watermark
        apply_motion_graphics(
            input_video, [], tmp_path, output,
            audio_duration_sec=10.0,
        )
        # Should still run because watermark is always added
        mock_run.assert_called_once()

    @patch("longform.motion_graphics._probe_duration_sec")
    @patch("longform.motion_graphics._run")
    def test_output_path_in_cmd(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 15.0
        input_video = tmp_path / "input.mp4"
        input_video.write_bytes(b"fake")
        output = tmp_path / "output.mp4"
        scenes = make_scenes_with_timing()

        apply_motion_graphics(
            input_video, scenes, tmp_path, output,
            audio_duration_sec=15.0,
        )
        cmd = mock_run.call_args[0][0]
        assert str(output) == cmd[-1]


# ---------------------------------------------------------------------------
# Integration: full filter chain correctness
# ---------------------------------------------------------------------------

class TestMotionGraphicsIntegration:
    def test_full_filter_chain_has_all_components(self):
        scenes = make_scenes_with_timing()
        starts = compute_scene_video_starts(scenes, 15.0)
        filt = build_motion_graphics_filter(scenes, 15.0, starts)

        # Should have lower-thirds (drawtext), progress bar (drawbox),
        # and watermark (drawtext)
        assert "drawtext" in filt
        assert "drawbox" in filt
        # At least 3 drawtext/drawbox components
        total = filt.count("drawtext") + filt.count("drawbox")
        assert total >= 3

    def test_filter_chain_with_5_scenes(self):
        scenes = [
            make_scene(f"00{i+1}", f"Scene {i+1} text here",
                       start_ms=i * 5000, end_ms=(i + 1) * 5000)
            for i in range(5)
        ]
        starts = compute_scene_video_starts(scenes, 25.0)
        filt = build_motion_graphics_filter(scenes, 25.0, starts)

        # 5 lower-thirds + watermark = 6 drawtext, 2 drawbox
        assert filt.count("drawtext") >= 5
        assert filt.count("drawbox") == 2

    def test_special_chars_in_text_escaped_in_filter(self):
        scene = make_scene("001", "It's: 100% done!",
                            start_ms=0, end_ms=5000)
        starts = compute_scene_video_starts([scene], 5.0)
        filt = build_motion_graphics_filter([scene], 5.0, starts)

        assert "\\:" in filt
        assert "\\'" in filt
        assert "\\%" in filt
