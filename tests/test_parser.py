"""
Tests for longform.parser — assets file parsing.
"""

import pytest

from longform.models import Scene
from longform.parser import (
    count_scenes,
    parse_assets_file,
    parse_assets_text,
    validate_scene,
)


# ---------------------------------------------------------------------------
# Sample asset file content for tests
# ---------------------------------------------------------------------------

SAMPLE_SINGLE_SCENE = '''\
🎬 SCENE 001 | ⏱️ [00:00 ➔ 00:07]
🗣️ Line: "दुनिया की कुछ jobs ऐसी हैं जहाँ सिर्फ skill ही नहीं।"
🎯 Hero Word: [WORKER READING RULEBOOK]
   🎥 Stock Video : https://videos.pexels.com/video-files/7841618/7841618-hd_1280_720_30fps.mp4
   🤣 Meme/GIF   : https://media1.giphy.com/media/v1.Y2lk/test/giphy.mp4
-------------------------------------------------------------------------------------
'''

SAMPLE_TWO_SCENES = '''\
🎬 SCENE 001 | ⏱️ [00:00 ➔ 00:07]
🗣️ Line: "First scene narration."
🎯 Hero Word: [FIRST CUE]
   🎥 Stock Video : https://example.com/first.mp4
   🤣 Meme/GIF   : https://example.com/first.gif
-------------------------------------------------------------------------------------
🎬 SCENE 002 | ⏱️ [00:07 ➔ 00:14]
🗣️ Line: "Second scene narration."
🎯 Hero Word: [SECOND CUE]
   🎥 Stock Video : https://example.com/second.mp4
-------------------------------------------------------------------------------------
'''

SAMPLE_SCENE_NO_GIF = '''\
🎬 SCENE 001 | ⏱️ [00:00 ➔ 00:07]
🗣️ Line: "No GIF here."
🎯 Hero Word: [NO GIF]
   🎥 Stock Video : https://example.com/nogif.mp4
-------------------------------------------------------------------------------------
'''

SAMPLE_NO_STOCK_VIDEO = '''\
🎬 SCENE 001 | ⏱️ [00:00 ➔ 00:07]
🗣️ Line: "Missing stock video."
🎯 Hero Word: [MISSING]
-------------------------------------------------------------------------------------
'''

SAMPLE_NO_LINE = '''\
🎬 SCENE 001 | ⏱️ [00:00 ➔ 00:07]
🎯 Hero Word: [NO LINE]
   🎥 Stock Video : https://example.com/noline.mp4
-------------------------------------------------------------------------------------
'''

SAMPLE_EMPTY = ""

SAMPLE_NO_SCENES = "Some random text without any scene headers."


class TestParseAssetsText:
    """Test parse_assets_text with various inputs."""

    def test_parse_single_scene(self):
        scenes = parse_assets_text(SAMPLE_SINGLE_SCENE)
        assert len(scenes) == 1
        s = scenes[0]
        assert s.scene_id == "001"
        assert "दुनिया" in s.text
        assert s.video_url == "https://videos.pexels.com/video-files/7841618/7841618-hd_1280_720_30fps.mp4"
        assert s.gif_url is not None
        assert "giphy.mp4" in s.gif_url

    def test_parse_two_scenes(self):
        scenes = parse_assets_text(SAMPLE_TWO_SCENES)
        assert len(scenes) == 2
        assert scenes[0].scene_id == "001"
        assert scenes[1].scene_id == "002"
        assert scenes[0].text == "First scene narration."
        assert scenes[1].text == "Second scene narration."

    def test_parse_scene_without_gif(self):
        scenes = parse_assets_text(SAMPLE_SCENE_NO_GIF)
        assert len(scenes) == 1
        assert scenes[0].gif_url is None

    def test_parse_second_scene_without_gif(self):
        """Verify scene 2 missing GIF doesn't pick up scene 1's GIF."""
        scenes = parse_assets_text(SAMPLE_TWO_SCENES)
        assert scenes[1].gif_url is None

    def test_skip_scene_missing_stock_video(self):
        with pytest.raises(ValueError, match="No scenes"):
            parse_assets_text(SAMPLE_NO_STOCK_VIDEO)

    def test_skip_scene_missing_line(self):
        with pytest.raises(ValueError, match="No scenes"):
            parse_assets_text(SAMPLE_NO_LINE)

    def test_empty_input_raises(self):
        with pytest.raises(ValueError, match="No scenes"):
            parse_assets_text(SAMPLE_EMPTY)

    def test_no_scenes_raises(self):
        with pytest.raises(ValueError, match="No scenes"):
            parse_assets_text(SAMPLE_NO_SCENES)

    def test_text_stripped(self):
        scenes = parse_assets_text(SAMPLE_SINGLE_SCENE)
        assert scenes[0].text == scenes[0].text.strip()

    def test_url_stripped(self):
        scenes = parse_assets_text(SAMPLE_SINGLE_SCENE)
        assert scenes[0].video_url == scenes[0].video_url.strip()


class TestParseAssetsFile:
    """Test parse_assets_file (reads from disk)."""

    def test_parse_from_file(self, tmp_path):
        f = tmp_path / "assets.txt"
        f.write_text(SAMPLE_SINGLE_SCENE, encoding="utf-8")
        scenes = parse_assets_file(str(f))
        assert len(scenes) == 1
        assert scenes[0].scene_id == "001"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            parse_assets_file("/nonexistent/path/file.txt")


class TestCountScenes:
    """Test count_scenes utility."""

    def test_count_single(self):
        assert count_scenes(SAMPLE_SINGLE_SCENE) == 1

    def test_count_two(self):
        assert count_scenes(SAMPLE_TWO_SCENES) == 2

    def test_count_zero(self):
        assert count_scenes(SAMPLE_NO_SCENES) == 0

    def test_count_empty(self):
        assert count_scenes(SAMPLE_EMPTY) == 0


class TestValidateScene:
    """Test validate_scene utility."""

    def test_valid_scene_no_errors(self):
        s = Scene("001", "some text", "https://example.com/v.mp4")
        assert validate_scene(s) == []

    def test_valid_scene_with_timing(self):
        s = Scene("001", "some text", "https://example.com/v.mp4",
                  start_ms=100, end_ms=500)
        assert validate_scene(s) == []

    def test_empty_scene_id(self):
        s = Scene("", "text", "https://example.com/v.mp4")
        errors = validate_scene(s)
        assert any("scene_id" in e for e in errors)

    def test_empty_text(self):
        s = Scene("001", "", "https://example.com/v.mp4")
        errors = validate_scene(s)
        assert any("text" in e for e in errors)

    def test_empty_video_url(self):
        s = Scene("001", "text", "")
        errors = validate_scene(s)
        assert any("video_url" in e for e in errors)

    def test_invalid_video_url(self):
        s = Scene("001", "text", "not-a-url")
        errors = validate_scene(s)
        assert any("not a valid URL" in e for e in errors)

    def test_invalid_gif_url(self):
        s = Scene("001", "text", "https://example.com/v.mp4", gif_url="bad-url")
        errors = validate_scene(s)
        assert any("gif_url" in e for e in errors)

    def test_negative_start_ms(self):
        s = Scene("001", "text", "https://example.com/v.mp4",
                  start_ms=-100, end_ms=500)
        errors = validate_scene(s)
        assert any("start_ms is negative" in e for e in errors)

    def test_negative_end_ms(self):
        s = Scene("001", "text", "https://example.com/v.mp4",
                  start_ms=100, end_ms=-500)
        errors = validate_scene(s)
        assert any("end_ms is negative" in e for e in errors)

    def test_end_before_start(self):
        s = Scene("001", "text", "https://example.com/v.mp4",
                  start_ms=500, end_ms=100)
        errors = validate_scene(s)
        assert any("end_ms" in e and "start_ms" in e for e in errors)

    def test_valid_gif_url(self):
        s = Scene("001", "text", "https://example.com/v.mp4",
                  gif_url="https://example.com/g.gif")
        errors = validate_scene(s)
        assert errors == []
