"""
Tests for longform.muxer — concat and mux logic.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from longform.muxer import _probe_duration_sec, write_concat_list, concat_and_mux


class TestWriteConcatList:
    def test_writes_file(self, tmp_path):
        clips = [Path("/fake/clip1.mp4"), Path("/fake/clip2.mp4")]
        result = write_concat_list(clips, tmp_path)
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "clip1.mp4" in content
        assert "clip2.mp4" in content

    def test_correct_line_count(self, tmp_path):
        clips = [Path(f"/fake/clip{i}.mp4") for i in range(5)]
        result = write_concat_list(clips, tmp_path)
        lines = result.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 5

    def test_each_line_starts_with_file_keyword(self, tmp_path):
        clips = [Path("/fake/clip1.mp4")]
        result = write_concat_list(clips, tmp_path)
        content = result.read_text(encoding="utf-8")
        assert content.startswith("file '")

    def test_empty_clip_list(self, tmp_path):
        result = write_concat_list([], tmp_path)
        assert result.exists()
        assert result.read_text(encoding="utf-8") == ""

    def test_paths_resolved(self, tmp_path):
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"fake")
        result = write_concat_list([clip], tmp_path)
        content = result.read_text(encoding="utf-8")
        # Should contain the resolved (absolute) path
        assert str(clip.resolve()) in content

    def test_single_clip(self, tmp_path):
        clips = [Path("/fake/only.mp4")]
        result = write_concat_list(clips, tmp_path)
        content = result.read_text(encoding="utf-8")
        assert "only.mp4" in content

    def test_returns_path_object(self, tmp_path):
        result = write_concat_list([Path("/fake/x.mp4")], tmp_path)
        assert isinstance(result, Path)


class TestProbeDurationSec:
    @patch("longform.muxer.subprocess.run")
    def test_successful_probe(self, mock_run):
        mock_run.return_value.stdout = "10.5\n"
        result = _probe_duration_sec(Path("/fake/video.mp4"))
        assert result == 10.5

    @patch("longform.muxer.subprocess.run")
    def test_returns_none_on_exception(self, mock_run):
        mock_run.side_effect = Exception("boom")
        result = _probe_duration_sec(Path("/fake/video.mp4"))
        assert result is None

    @patch("longform.muxer.subprocess.run")
    def test_handles_float_conversion(self, mock_run):
        mock_run.return_value.stdout = "0.033\n"
        result = _probe_duration_sec(Path("/fake/video.mp4"))
        assert result == 0.033

    @patch("longform.muxer.subprocess.run")
    def test_handles_empty_output(self, mock_run):
        mock_run.return_value.stdout = ""
        result = _probe_duration_sec(Path("/fake/video.mp4"))
        assert result is None  # empty string -> exception caught -> None


class TestConcatAndMux:
    @patch("longform.muxer._probe_duration_sec")
    @patch("longform.muxer._run")
    def test_calls_run_twice(self, mock_run, mock_probe, tmp_path):
        mock_probe.return_value = 10.0
        clips = [tmp_path / "clip1.mp4", tmp_path / "clip2.mp4"]
        for c in clips:
            c.write_bytes(b"fake")
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        output = tmp_path / "output.mp4"

        concat_and_mux(clips, audio, output, tmp_path)
        assert mock_run.call_count == 2

    @patch("longform.muxer._probe_duration_sec")
    @patch("longform.muxer._run")
    def test_warning_on_duration_mismatch(self, mock_run, mock_probe, tmp_path, caplog):
        import logging
        mock_probe.side_effect = [15.0, 10.0]  # 5 second difference
        clips = [tmp_path / "clip1.mp4"]
        clips[0].write_bytes(b"fake")
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        output = tmp_path / "output.mp4"

        with caplog.at_level(logging.WARNING):
            concat_and_mux(clips, audio, output, tmp_path)
        assert any("differ" in r.message for r in caplog.records)

    @patch("longform.muxer._probe_duration_sec")
    @patch("longform.muxer._run")
    def test_no_warning_when_durations_match(self, mock_run, mock_probe, tmp_path, caplog):
        import logging
        mock_probe.side_effect = [10.0, 10.0]
        clips = [tmp_path / "clip1.mp4"]
        clips[0].write_bytes(b"fake")
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        output = tmp_path / "output.mp4"

        with caplog.at_level(logging.WARNING):
            concat_and_mux(clips, audio, output, tmp_path)
        assert not any("differ" in r.message for r in caplog.records)

    @patch("longform.muxer._probe_duration_sec")
    @patch("longform.muxer._run")
    def test_no_warning_when_probe_returns_none(self, mock_run, mock_probe, tmp_path, caplog):
        import logging
        mock_probe.return_value = None
        clips = [tmp_path / "clip1.mp4"]
        clips[0].write_bytes(b"fake")
        audio = tmp_path / "audio.mp3"
        audio.write_bytes(b"fake")
        output = tmp_path / "output.mp4"

        with caplog.at_level(logging.WARNING):
            concat_and_mux(clips, audio, output, tmp_path)
        assert not any("differ" in r.message for r in caplog.records)
