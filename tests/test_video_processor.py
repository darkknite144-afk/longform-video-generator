"""
Tests for longform.video_processor — FFmpeg command building and download logic.
"""

from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
import requests

from longform.config import FADE_SECONDS, FPS, GIF_HEIGHT_FRAC, VIDEO_H, VIDEO_W
from longform.models import Scene
from longform.video_processor import (
    build_base_chain,
    build_gif_filter_complex,
    compute_gif_overlay_params,
    download_file,
    with_tail_fade,
)


# ---------------------------------------------------------------------------
# build_base_chain tests
# ---------------------------------------------------------------------------

class TestBuildBaseChain:
    def test_contains_fps(self):
        chain = build_base_chain("1", "0", "0")
        assert f"fps={FPS}" in chain

    def test_contains_scale(self):
        chain = build_base_chain("1", "0", "0")
        assert "scale=" in chain

    def test_contains_crop(self):
        chain = build_base_chain("1", "0", "0")
        assert "crop=" in chain

    def test_contains_zoompan(self):
        chain = build_base_chain("1", "0", "0")
        assert "zoompan=" in chain

    def test_contains_z_expr(self):
        chain = build_base_chain("MY_Z_EXPR", "0", "0")
        assert "MY_Z_EXPR" in chain

    def test_contains_x_y_exprs(self):
        chain = build_base_chain("1", "MY_X", "MY_Y")
        assert "MY_X" in chain
        assert "MY_Y" in chain

    def test_contains_output_size(self):
        chain = build_base_chain("1", "0", "0")
        assert f"{VIDEO_W}x{VIDEO_H}" in chain

    def test_chain_starts_with_fps(self):
        chain = build_base_chain("1", "0", "0")
        assert chain.startswith(f"fps={FPS}")


# ---------------------------------------------------------------------------
# with_tail_fade tests
# ---------------------------------------------------------------------------

class TestWithTailFade:
    def test_fade_in_adds_fade(self):
        chain = "somefilter"
        result = with_tail_fade(chain, "fade_in", duration=2.0)
        assert "fade=t=in" in result

    def test_fade_out_adds_fade(self):
        chain = "somefilter"
        result = with_tail_fade(chain, "fade_out", duration=2.0)
        assert "fade=t=out" in result

    def test_non_fade_effect_no_change(self):
        chain = "somefilter"
        result = with_tail_fade(chain, "zoom_in", duration=2.0)
        assert result == chain

    def test_fade_in_short_duration_no_fade(self):
        """Scene too short for fade should not get fade added."""
        chain = "somefilter"
        result = with_tail_fade(chain, "fade_in", duration=0.1)
        assert result == chain  # no fade added

    def test_fade_out_short_duration_no_fade(self):
        chain = "somefilter"
        result = with_tail_fade(chain, "fade_out", duration=0.1)
        assert result == chain

    def test_fade_in_exact_threshold(self):
        """At exactly FADE_SECONDS * 1.5, fade should be added."""
        chain = "somefilter"
        duration = FADE_SECONDS * 1.5
        result = with_tail_fade(chain, "fade_in", duration)
        # > is used, so at exactly the threshold it should NOT be added
        # (condition is `duration > FADE_SECONDS * 1.5`)
        assert result == chain

    def test_fade_in_just_above_threshold(self):
        chain = "somefilter"
        duration = FADE_SECONDS * 1.5 + 0.01
        result = with_tail_fade(chain, "fade_in", duration)
        assert "fade=t=in" in result

    def test_fade_out_st_calculation(self):
        """fade_out start time should be duration - FADE_SECONDS."""
        chain = "somefilter"
        duration = 5.0
        result = with_tail_fade(chain, "fade_out", duration)
        expected_st = duration - FADE_SECONDS
        assert f"st={expected_st:.3f}" in result

    def test_fade_in_st_is_zero(self):
        chain = "somefilter"
        result = with_tail_fade(chain, "fade_in", duration=2.0)
        assert "st=0" in result


# ---------------------------------------------------------------------------
# compute_gif_overlay_params tests
# ---------------------------------------------------------------------------

class TestComputeGifOverlayParams:
    def test_normal_duration_has_gif(self):
        params = compute_gif_overlay_params(duration=5.0)
        assert params["has_gif"] is True

    def test_short_duration_no_gif(self):
        params = compute_gif_overlay_params(duration=0.5)
        assert params["has_gif"] is False

    def test_overlay_dur_capped(self):
        params = compute_gif_overlay_params(duration=10.0)
        assert params["overlay_dur"] <= 1.6

    def test_overlay_dur_proportional(self):
        params = compute_gif_overlay_params(duration=2.0)
        assert abs(params["overlay_dur"] - 0.8) < 0.01  # 2.0 * 0.4 = 0.8

    def test_overlay_start_non_negative(self):
        params = compute_gif_overlay_params(duration=5.0)
        assert params["overlay_start"] >= 0

    def test_overlay_end_after_start(self):
        params = compute_gif_overlay_params(duration=5.0)
        assert params["overlay_end"] > params["overlay_start"]

    def test_gif_h_correct(self):
        params = compute_gif_overlay_params(duration=5.0)
        assert params["gif_h"] == round(VIDEO_H * GIF_HEIGHT_FRAC)

    def test_fade_d_positive(self):
        params = compute_gif_overlay_params(duration=5.0)
        assert params["fade_d"] > 0

    def test_fade_d_capped(self):
        params = compute_gif_overlay_params(duration=5.0)
        assert params["fade_d"] <= 0.2

    def test_threshold_exactly_0_6(self):
        """At overlay_dur exactly 0.6, has_gif should be True (>=)."""
        # duration * 0.4 = 0.6 -> duration = 1.5
        params = compute_gif_overlay_params(duration=1.5)
        assert abs(params["overlay_dur"] - 0.6) < 1e-9
        assert params["has_gif"] is True

    def test_just_below_threshold(self):
        params = compute_gif_overlay_params(duration=1.49)
        assert params["has_gif"] is False

    def test_all_keys_present(self):
        params = compute_gif_overlay_params(duration=5.0)
        expected_keys = {"has_gif", "overlay_dur", "overlay_start",
                         "overlay_end", "fade_d", "gif_h"}
        assert set(params.keys()) == expected_keys


# ---------------------------------------------------------------------------
# build_gif_filter_complex tests
# ---------------------------------------------------------------------------

class TestBuildGifFilterComplex:
    def test_has_base_label(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "[base]" in fc

    def test_has_gif_label(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "[gif]" in fc

    def test_has_outv_label(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "[outv]" in fc

    def test_has_overlay(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "overlay=" in fc

    def test_has_three_segments(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert fc.count(";") == 2  # 3 segments = 2 semicolons

    def test_has_enable_between(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "between(t," in fc

    def test_contains_base_chain(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("MY_CHAIN", "zoom_in", 5.0, params)
        assert "MY_CHAIN" in fc

    def test_has_fade_in(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "fade=t=in" in fc

    def test_has_fade_out(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "fade=t=out" in fc

    def test_has_yuva420p(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "yuva420p" in fc

    def test_has_white_pad(self):
        params = compute_gif_overlay_params(duration=5.0)
        fc = build_gif_filter_complex("chain", "zoom_in", 5.0, params)
        assert "white@1.0" in fc


# ---------------------------------------------------------------------------
# download_file tests (mocked network)
# ---------------------------------------------------------------------------

class TestDownloadFile:
    @patch("longform.video_processor.requests.get")
    def test_successful_download(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = [b"chunk1", b"chunk2"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_get.return_value = mock_resp

        dest = tmp_path / "video.mp4"
        download_file("https://example.com/video.mp4", dest, retries=1)
        assert dest.exists()

    @patch("longform.video_processor.requests.get")
    @patch("longform.video_processor.time.sleep")
    def test_retry_on_failure(self, mock_sleep, mock_get, tmp_path):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        fail_resp.__enter__ = MagicMock(return_value=fail_resp)
        fail_resp.__exit__ = MagicMock(return_value=False)

        success_resp = MagicMock()
        success_resp.raise_for_status.return_value = None
        success_resp.iter_content.return_value = [b"data"]
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        mock_get.side_effect = [fail_resp, success_resp]

        dest = tmp_path / "video.mp4"
        download_file("https://example.com/video.mp4", dest, retries=3)
        assert dest.exists()
        assert dest.read_bytes() == b"data"

    @patch("longform.video_processor.requests.get")
    @patch("longform.video_processor.time.sleep")
    def test_all_retries_exhausted_raises(self, mock_sleep, mock_get, tmp_path):
        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        fail_resp.__enter__ = MagicMock(return_value=fail_resp)
        fail_resp.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = fail_resp

        dest = tmp_path / "video.mp4"
        with pytest.raises(RuntimeError, match="Failed to download"):
            download_file("https://example.com/video.mp4", dest, retries=2)

    @patch("longform.video_processor.requests.get")
    @patch("longform.video_processor.time.sleep")
    def test_network_error_retried(self, mock_sleep, mock_get, tmp_path):
        success_resp = MagicMock()
        success_resp.raise_for_status.return_value = None
        success_resp.iter_content.return_value = [b"data"]
        success_resp.__enter__ = MagicMock(return_value=success_resp)
        success_resp.__exit__ = MagicMock(return_value=False)

        mock_get.side_effect = [
            requests.exceptions.ConnectionError("down"),
            success_resp,
        ]

        dest = tmp_path / "video.mp4"
        download_file("https://example.com/video.mp4", dest, retries=3)
        assert dest.exists()

    @patch("longform.video_processor.requests.get")
    def test_user_agent_header_sent(self, mock_get, tmp_path):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.iter_content.return_value = [b"data"]
        mock_resp.__enter__ = MagicMock(return_value=mock_resp)
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_get.return_value = mock_resp

        dest = tmp_path / "video.mp4"
        download_file("https://example.com/v.mp4", dest, retries=1)
        call_args = mock_get.call_args
        headers = call_args[1]["headers"]
        assert "User-Agent" in headers
