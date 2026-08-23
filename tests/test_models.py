"""
Tests for longform.models — Scene dataclass.
"""

import pytest

from longform.models import Scene


class TestSceneCreation:
    """Test Scene dataclass construction."""

    def test_minimal_creation(self):
        s = Scene(scene_id="001", text="hello world", video_url="https://example.com/v.mp4")
        assert s.scene_id == "001"
        assert s.text == "hello world"
        assert s.video_url == "https://example.com/v.mp4"
        assert s.gif_url is None
        assert s.start_ms is None
        assert s.end_ms is None

    def test_full_creation(self):
        s = Scene(
            scene_id="042",
            text="narration",
            video_url="https://example.com/v.mp4",
            gif_url="https://example.com/g.gif",
            start_ms=1000,
            end_ms=5000,
        )
        assert s.gif_url == "https://example.com/g.gif"
        assert s.start_ms == 1000
        assert s.end_ms == 5000

    def test_empty_text(self):
        s = Scene(scene_id="001", text="", video_url="https://example.com/v.mp4")
        assert s.text == ""

    def test_text_with_unicode(self):
        s = Scene(scene_id="001", text="दुनिया की कुछ jobs", video_url="https://example.com/v.mp4")
        assert "दुनिया" in s.text


class TestSceneDurationSec:
    """Test the duration_sec property."""

    def test_duration_basic(self):
        s = Scene("001", "text", "url", start_ms=1000, end_ms=4000)
        assert s.duration_sec == 3.0

    def test_duration_zero_gap(self):
        s = Scene("001", "text", "url", start_ms=5000, end_ms=5000)
        assert s.duration_sec == 0.05  # min clamp

    def test_duration_negative_gap(self):
        s = Scene("001", "text", "url", start_ms=5000, end_ms=4000)
        assert s.duration_sec == 0.05  # max(0.05, ...)

    def test_duration_no_timing_raises(self):
        s = Scene("001", "text", "url")
        with pytest.raises(ValueError, match="no timing"):
            _ = s.duration_sec

    def test_duration_only_start_raises(self):
        s = Scene("001", "text", "url", start_ms=1000)
        with pytest.raises(ValueError):
            _ = s.duration_sec

    def test_duration_only_end_raises(self):
        s = Scene("001", "text", "url", end_ms=1000)
        with pytest.raises(ValueError):
            _ = s.duration_sec


class TestSceneDurationMs:
    """Test the duration_ms property."""

    def test_duration_ms_basic(self):
        s = Scene("001", "text", "url", start_ms=1000, end_ms=4000)
        assert s.duration_ms == 3000

    def test_duration_ms_zero_gap(self):
        s = Scene("001", "text", "url", start_ms=5000, end_ms=5000)
        assert s.duration_ms == 50  # min clamp

    def test_duration_ms_no_timing_raises(self):
        s = Scene("001", "text", "url")
        with pytest.raises(ValueError):
            _ = s.duration_ms


class TestSceneSafeId:
    """Test the safe_id property — filesystem-safe version of scene_id."""

    def test_safe_id_already_safe(self):
        s = Scene("001", "text", "url")
        assert s.safe_id == "001"

    def test_safe_id_with_spaces(self):
        s = Scene("scene 01", "text", "url")
        assert " " not in s.safe_id
        assert s.safe_id == "scene_01"

    def test_safe_id_with_special_chars(self):
        s = Scene("sc/01:x", "text", "url")
        assert "/" not in s.safe_id
        assert ":" not in s.safe_id

    def test_safe_id_preserves_hyphens(self):
        s = Scene("scene-001", "text", "url")
        assert s.safe_id == "scene-001"

    def test_safe_id_preserves_underscores(self):
        s = Scene("scene_001", "text", "url")
        assert s.safe_id == "scene_001"


class TestSceneHasTiming:
    """Test the has_timing property."""

    def test_has_timing_true(self):
        s = Scene("001", "text", "url", start_ms=100, end_ms=500)
        assert s.has_timing is True

    def test_has_timing_false_no_timing(self):
        s = Scene("001", "text", "url")
        assert s.has_timing is False

    def test_has_timing_false_partial(self):
        s = Scene("001", "text", "url", start_ms=100)
        assert s.has_timing is False


class TestSceneWordCount:
    """Test the word_count property."""

    def test_word_count_basic(self):
        s = Scene("001", "one two three", "url")
        assert s.word_count == 3

    def test_word_count_empty(self):
        s = Scene("001", "", "url")
        assert s.word_count == 0

    def test_word_count_single(self):
        s = Scene("001", "word", "url")
        assert s.word_count == 1

    def test_word_count_unicode(self):
        s = Scene("001", "दुनिया की कुछ jobs", "url")
        assert s.word_count == 4

    def test_word_count_whitespace_only(self):
        s = Scene("001", "   ", "url")
        assert s.word_count == 0

    def test_word_count_multiple_spaces(self):
        s = Scene("001", "one  two   three", "url")
        assert s.word_count == 3
