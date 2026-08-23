"""
Tests for longform.config — configuration constants.
"""

from longform import config


class TestConfigConstants:
    """Verify all config constants have sensible values."""

    def test_video_dimensions(self):
        assert config.VIDEO_W == 1920
        assert config.VIDEO_H == 1080

    def test_fps(self):
        assert config.FPS == 30

    def test_zoom_src_dimensions(self):
        assert config.ZOOM_SRC_W == 2560
        assert config.ZOOM_SRC_H == 1440

    def test_zoom_max(self):
        assert 1.0 < config.ZOOM_MAX <= 1.5

    def test_pan_zoom(self):
        assert 1.0 < config.PAN_ZOOM <= 1.5

    def test_zoom_pan_combo_max(self):
        assert 1.0 < config.ZOOM_PAN_COMBO_MAX <= 1.5

    def test_fade_seconds(self):
        assert 0 < config.FADE_SECONDS <= 1.0

    def test_gif_height_frac(self):
        assert 0 < config.GIF_HEIGHT_FRAC < 1.0

    def test_effects_list_not_empty(self):
        assert len(config.EFFECTS) > 0

    def test_effects_list_has_expected_entries(self):
        expected = {
            "zoom_in", "zoom_out",
            "pan_left", "pan_right", "pan_up", "pan_down",
            "zoom_in_pan_left", "zoom_in_pan_right",
            "fade_in", "fade_out",
        }
        assert set(config.EFFECTS) == expected

    def test_effects_no_duplicates(self):
        assert len(config.EFFECTS) == len(set(config.EFFECTS))

    def test_download_timeout_positive(self):
        assert config.DOWNLOAD_TIMEOUT > 0

    def test_download_chunk_positive(self):
        assert config.DOWNLOAD_CHUNK > 0

    def test_sarvam_model_set(self):
        assert config.SARVAM_MODEL == "sarvam-105b"

    def test_sarvam_url_set(self):
        assert config.SARVAM_URL.startswith("https://")
        assert "/v1/chat/completions" in config.SARVAM_URL

    def test_sarvam_max_retries_positive(self):
        assert config.SARVAM_MAX_RETRIES > 0

    def test_sarvam_retry_wait_positive(self):
        assert config.SARVAM_RETRY_WAIT_SECONDS > 0

    def test_sarvam_api_key_env_name(self):
        assert config.SARVAM_API_KEY_ENV == "SARVAM_API_KEY"

    def test_valid_effects_frozenset(self):
        assert isinstance(config.VALID_EFFECTS, frozenset)
        assert len(config.VALID_EFFECTS) == len(config.EFFECTS)

    def test_download_retries_positive(self):
        assert config.DOWNLOAD_RETRIES > 0

    def test_download_retry_wait_positive(self):
        assert config.DOWNLOAD_RETRY_WAIT > 0
