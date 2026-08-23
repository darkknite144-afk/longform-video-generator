"""
Tests for longform.deterministic_sync — scene/timestamp alignment.
"""

import pytest

from longform.deterministic_sync import (
    _normalize_token,
    _tokenize_scene_text,
    _tokenize_asr_words,
    align_scenes_deterministic,
    compute_scene_frame_counts,
)
from longform.models import Scene


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scene(sid, text, url="https://example.com/v.mp4"):
    return Scene(scene_id=sid, text=text, video_url=url)


def make_words(word_times):
    """word_times: list of (word, start_ms, end_ms)"""
    return [{"word": w, "start": s, "end": e} for w, s, e in word_times]


# ---------------------------------------------------------------------------
# Token normalization tests
# ---------------------------------------------------------------------------

class TestNormalizeToken:
    def test_simple_word(self):
        assert _normalize_token("hello") == "hello"

    def test_uppercase(self):
        assert _normalize_token("HELLO") == "hello"

    def test_with_period(self):
        assert _normalize_token("hello.") == "hello"

    def test_with_comma(self):
        assert _normalize_token("hello,") == "hello"

    def test_with_quotes(self):
        assert _normalize_token('"hello"') == "hello"

    def test_with_parentheses(self):
        assert _normalize_token("(hello)") == "hello"

    def test_with_hindi_punctuation(self):
        assert _normalize_token("नमस्ते।") == "नमस्ते"

    def test_with_dash(self):
        assert _normalize_token("well-known") == "wellknown"

    def test_empty_string(self):
        assert _normalize_token("") == ""

    def test_only_punctuation(self):
        assert _normalize_token("...!!!") == ""


class TestTokenizeSceneText:
    def test_simple(self):
        assert _tokenize_scene_text("hello world") == ["hello", "world"]

    def test_empty(self):
        assert _tokenize_scene_text("") == []

    def test_with_punctuation(self):
        assert _tokenize_scene_text("Hello, world!") == ["hello", "world"]

    def test_hindi(self):
        tokens = _tokenize_scene_text("दुनिया की jobs")
        assert "दुनिया" in tokens
        assert "jobs" in tokens

    def test_multiple_spaces(self):
        assert _tokenize_scene_text("a  b   c") == ["a", "b", "c"]

    def test_only_punctuation_returns_empty(self):
        assert _tokenize_scene_text("... !!! ???") == []


class TestTokenizeAsrWords:
    def test_basic(self):
        words = make_words([("hello", 0, 500), ("world", 500, 1000)])
        tokens = _tokenize_asr_words(words)
        assert tokens == ["hello", "world"]

    def test_preserves_index(self):
        """Tokens list should have same length as words list."""
        words = make_words([("a", 0, 100), ("b", 100, 200), ("c", 200, 300)])
        tokens = _tokenize_asr_words(words)
        assert len(tokens) == len(words)

    def test_normalizes_punctuation(self):
        words = make_words([("hello.", 0, 500)])
        assert _tokenize_asr_words(words) == ["hello"]


# ---------------------------------------------------------------------------
# align_scenes_deterministic tests
# ---------------------------------------------------------------------------

class TestAlignScenesDeterministic:
    def test_empty_words_raises(self):
        scenes = [make_scene("001", "hello world")]
        with pytest.raises(RuntimeError, match="No whisper words"):
            align_scenes_deterministic(scenes, [], 5000)

    def test_single_scene(self):
        scenes = [make_scene("001", "hello world")]
        words = make_words([("hello", 0, 500), ("world", 500, 1000)])
        result = align_scenes_deterministic(scenes, words, 10000)
        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 10000  # pinned to total_duration

    def test_full_coverage_start(self):
        """First scene must start at 0."""
        scenes = [make_scene("001", "a"), make_scene("002", "b")]
        words = make_words([("a", 100, 500), ("b", 600, 1000)])
        result = align_scenes_deterministic(scenes, words, 2000)
        assert result[0].start_ms == 0

    def test_full_coverage_end(self):
        """Last scene must end at total_duration_ms."""
        scenes = [make_scene("001", "a"), make_scene("002", "b")]
        words = make_words([("a", 100, 500), ("b", 600, 1000)])
        result = align_scenes_deterministic(scenes, words, 2000)
        assert result[-1].end_ms == 2000

    def test_non_decreasing_boundaries(self):
        """Scene boundaries must be non-decreasing."""
        scenes = [make_scene(f"{i:03d}", f"word{i}") for i in range(1, 11)]
        words = make_words([(f"word{i}", i * 1000, i * 1000 + 500) for i in range(1, 11)])
        result = align_scenes_deterministic(scenes, words, 20000)
        for i in range(1, len(result)):
            assert result[i].start_ms >= result[i - 1].start_ms

    def test_all_scenes_have_timing(self):
        scenes = [make_scene(f"{i:03d}", f"word{i}") for i in range(1, 6)]
        words = make_words([(f"word{i}", i * 1000, i * 1000 + 800) for i in range(1, 6)])
        result = align_scenes_deterministic(scenes, words, 10000)
        for s in result:
            assert s.has_timing

    def test_minimum_duration_enforced(self):
        """Scenes with end <= start should get at least 50ms."""
        scenes = [make_scene("001", "a"), make_scene("002", "b")]
        words = make_words([("a", 0, 100), ("b", 100, 200)])
        result = align_scenes_deterministic(scenes, words, 200)
        for s in result:
            assert s.duration_ms >= 50

    def test_many_scenes(self):
        """Stress test with 70 scenes (like the real use case)."""
        scenes = [make_scene(f"{i:03d}", f"scene word{i}") for i in range(1, 71)]
        words = [{"word": "scene", "start": i * 100, "end": i * 100 + 50} for i in range(70)]
        words.extend([{"word": f"word{i}", "start": 7000 + i * 100, "end": 7000 + i * 100 + 50} for i in range(1, 71)])
        result = align_scenes_deterministic(scenes, words, 60000)
        assert len(result) == 70
        assert result[0].start_ms == 0
        assert result[-1].end_ms == 60000

    def test_scenes_with_no_matching_words(self):
        """Scenes whose text doesn't match any ASR words should still get timing."""
        scenes = [make_scene("001", "alpha"), make_scene("002", "beta")]
        words = make_words([("alpha", 0, 500), ("gamma", 500, 1000)])
        result = align_scenes_deterministic(scenes, words, 2000)
        assert result[1].has_timing

    def test_returns_same_list_object(self):
        scenes = [make_scene("001", "hello")]
        words = make_words([("hello", 0, 500)])
        result = align_scenes_deterministic(scenes, words, 1000)
        assert result is scenes  # same list, mutated in place

    def test_contiguous_scenes(self):
        """Adjacent scenes should be contiguous (end[i] == start[i+1])."""
        scenes = [make_scene(f"{i:03d}", f"word{i}") for i in range(1, 6)]
        words = make_words([(f"word{i}", i * 1000, i * 1000 + 500) for i in range(1, 6)])
        result = align_scenes_deterministic(scenes, words, 10000)
        for i in range(len(result) - 1):
            assert result[i].end_ms == result[i + 1].start_ms


# ---------------------------------------------------------------------------
# compute_scene_frame_counts tests
# ---------------------------------------------------------------------------

class TestComputeSceneFrameCounts:
    def test_basic(self):
        scenes = [
            make_scene("001", "a", ),
            make_scene("002", "b"),
        ]
        scenes[0].start_ms = 0
        scenes[0].end_ms = 1000
        scenes[1].start_ms = 1000
        scenes[1].end_ms = 2000
        frames = compute_scene_frame_counts(scenes, fps=30)
        assert len(frames) == 2
        assert frames[0] == 30  # 1000ms = 1s = 30 frames
        assert frames[1] == 30

    def test_minimum_one_frame(self):
        scenes = [make_scene("001", "a")]
        scenes[0].start_ms = 100
        scenes[0].end_ms = 100  # same start/end -> 0ms duration
        frames = compute_scene_frame_counts(scenes, fps=30)
        assert frames[0] >= 1

    def test_telescoping_sum(self):
        """Sum of scene frames should equal total frames (telescoping)."""
        scenes = [make_scene(f"{i:03d}", f"w{i}") for i in range(1, 6)]
        for i, s in enumerate(scenes):
            s.start_ms = i * 1000
            s.end_ms = (i + 1) * 1000
        frames = compute_scene_frame_counts(scenes, fps=30)
        assert sum(frames) == 5 * 30  # 5 seconds total = 150 frames

    def test_uneven_durations(self):
        scenes = [
            make_scene("001", "a"),
            make_scene("002", "b"),
            make_scene("003", "c"),
        ]
        scenes[0].start_ms = 0
        scenes[0].end_ms = 500
        scenes[1].start_ms = 500
        scenes[1].end_ms = 2000
        scenes[2].start_ms = 2000
        scenes[2].end_ms = 3500
        frames = compute_scene_frame_counts(scenes, fps=30)
        assert frames[0] == 15   # 500ms
        assert frames[1] == 45   # 1500ms
        assert frames[2] == 45   # 1500ms

    def test_empty_scenes_list(self):
        frames = compute_scene_frame_counts([], fps=30)
        assert frames == []
