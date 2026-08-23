"""
Tests for longform.sarvam_sync — Sarvam AI API integration (replaces Gemini).
"""

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from longform.models import Scene
from longform.sarvam_sync import (
    _build_sarvam_payload,
    _call_sarvam_with_retry,
    _extract_json_array,
    _parse_sarvam_response,
    fallback_proportional_sync,
    get_api_key,
    map_scenes_with_sarvam,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_scene(sid, text="hello world", url="https://example.com/v.mp4"):
    return Scene(scene_id=sid, text=text, video_url=url)


def make_words():
    return [
        {"word": "hello", "start": 0, "end": 500},
        {"word": "world", "start": 500, "end": 1000},
    ]


def make_sarvam_response(content_json):
    """Build a mock Sarvam API response with the given content string."""
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps(content_json),
                }
            }
        ]
    }


# ---------------------------------------------------------------------------
# _extract_json_array tests
# ---------------------------------------------------------------------------

class TestExtractJsonArray:
    def test_plain_json(self):
        result = _extract_json_array('[{"a": 1}]')
        assert result == [{"a": 1}]

    def test_with_json_fence(self):
        result = _extract_json_array('```json\n[{"a": 1}]\n```')
        assert result == [{"a": 1}]

    def test_with_plain_fence(self):
        result = _extract_json_array('```\n[{"a": 1}]\n```')
        assert result == [{"a": 1}]

    def test_with_whitespace(self):
        result = _extract_json_array('  \n [{"a": 1}] \n  ')
        assert result == [{"a": 1}]

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _extract_json_array("not json at all")

    def test_empty_array(self):
        result = _extract_json_array('[]')
        assert result == []

    def test_multiple_items(self):
        result = _extract_json_array('[{"a": 1}, {"b": 2}, {"c": 3}]')
        assert len(result) == 3

    def test_fence_with_newlines_inside(self):
        result = _extract_json_array('```json\n[{"scene_id": "001",\n"start_time": 0}]\n```')
        assert result[0]["scene_id"] == "001"


# ---------------------------------------------------------------------------
# _build_sarvam_payload tests
# ---------------------------------------------------------------------------

class TestBuildSarvamPayload:
    def test_payload_has_model(self):
        scenes = [make_scene("001")]
        payload = _build_sarvam_payload(scenes, make_words())
        assert "model" in payload
        assert payload["model"] == "sarvam-105b"

    def test_payload_has_messages(self):
        scenes = [make_scene("001")]
        payload = _build_sarvam_payload(scenes, make_words())
        assert "messages" in payload
        assert len(payload["messages"]) == 1

    def test_message_role_is_user(self):
        payload = _build_sarvam_payload([make_scene("001")], make_words())
        assert payload["messages"][0]["role"] == "user"

    def test_message_has_content(self):
        payload = _build_sarvam_payload([make_scene("001")], make_words())
        assert "content" in payload["messages"][0]
        assert len(payload["messages"][0]["content"]) > 0

    def test_payload_contains_scene_text(self):
        scenes = [make_scene("001", "unique narration text")]
        payload = _build_sarvam_payload(scenes, make_words())
        assert "unique narration text" in payload["messages"][0]["content"]

    def test_payload_contains_word_timestamps(self):
        words = [{"word": "testword", "start": 123, "end": 456}]
        payload = _build_sarvam_payload([make_scene("001")], words)
        assert "testword" in payload["messages"][0]["content"]
        assert "123" in payload["messages"][0]["content"]

    def test_payload_contains_multiple_scenes(self):
        scenes = [make_scene("001", "first"), make_scene("002", "second")]
        payload = _build_sarvam_payload(scenes, make_words())
        assert "first" in payload["messages"][0]["content"]
        assert "second" in payload["messages"][0]["content"]

    def test_payload_contains_instructions(self):
        payload = _build_sarvam_payload([make_scene("001")], make_words())
        content = payload["messages"][0]["content"]
        assert "scene_id" in content
        assert "start_time" in content
        assert "end_time" in content

    def test_payload_with_empty_words(self):
        payload = _build_sarvam_payload([make_scene("001")], [])
        assert "messages" in payload  # should still build


# ---------------------------------------------------------------------------
# _parse_sarvam_response tests
# ---------------------------------------------------------------------------

class TestParseSarvamResponse:
    def test_successful_parse(self):
        scenes = [make_scene("001"), make_scene("002")]
        mapping = [
            {"scene_id": "001", "start_time": 0, "end_time": 5000},
            {"scene_id": "002", "start_time": 5000, "end_time": 10000},
        ]
        data = make_sarvam_response(mapping)
        result = _parse_sarvam_response(data, scenes)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 5000
        assert result[1].start_ms == 5000
        assert result[1].end_ms == 10000

    def test_missing_scene_raises(self):
        scenes = [make_scene("001"), make_scene("002")]
        mapping = [{"scene_id": "001", "start_time": 0, "end_time": 5000}]
        data = make_sarvam_response(mapping)
        with pytest.raises(RuntimeError, match="missing scene"):
            _parse_sarvam_response(data, scenes)

    def test_malformed_response_raises(self):
        scenes = [make_scene("001")]
        bad_data = {"choices": [{"message": {"content": "not json"}}]}
        with pytest.raises(RuntimeError, match="Could not parse"):
            _parse_sarvam_response(bad_data, scenes)

    def test_missing_choices_key_raises(self):
        scenes = [make_scene("001")]
        with pytest.raises(RuntimeError, match="Could not parse"):
            _parse_sarvam_response({}, scenes)

    def test_unknown_scene_id_ignored(self):
        scenes = [make_scene("001")]
        mapping = [
            {"scene_id": "999", "start_time": 0, "end_time": 1000},
            {"scene_id": "001", "start_time": 0, "end_time": 5000},
        ]
        data = make_sarvam_response(mapping)
        result = _parse_sarvam_response(data, scenes)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 5000

    def test_string_times_converted_to_int(self):
        scenes = [make_scene("001")]
        mapping = [{"scene_id": "001", "start_time": "100", "end_time": "5000"}]
        data = make_sarvam_response(mapping)
        result = _parse_sarvam_response(data, scenes)
        assert result[0].start_ms == 100
        assert result[0].end_ms == 5000
        assert isinstance(result[0].start_ms, int)
        assert isinstance(result[0].end_ms, int)

    def test_extra_fields_in_mapping_ignored(self):
        scenes = [make_scene("001")]
        mapping = [{"scene_id": "001", "start_time": 0, "end_time": 5000, "extra": "ignored"}]
        data = make_sarvam_response(mapping)
        result = _parse_sarvam_response(data, scenes)
        assert result[0].start_ms == 0


# ---------------------------------------------------------------------------
# _call_sarvam_with_retry tests (mocked network)
# ---------------------------------------------------------------------------

class TestCallSarvamWithRetry:
    @patch("longform.sarvam_sync.requests.post")
    def test_successful_call(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"choices": [{"message": {"content": "[]"}}]}
        mock_post.return_value = mock_resp

        result = _call_sarvam_with_retry({"model": "test"}, "fake-key")
        assert result == {"choices": [{"message": {"content": "[]"}}]}

    @patch("longform.sarvam_sync.requests.post")
    @patch("longform.sarvam_sync.time.sleep")
    def test_retry_on_503(self, mock_sleep, mock_post):
        fail_resp = MagicMock()
        fail_resp.status_code = 503
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"ok": True}
        mock_post.side_effect = [fail_resp, success_resp]

        result = _call_sarvam_with_retry({"model": "test"}, "fake-key")
        assert result == {"ok": True}
        assert mock_post.call_count == 2

    @patch("longform.sarvam_sync.requests.post")
    @patch("longform.sarvam_sync.time.sleep")
    def test_retry_on_429(self, mock_sleep, mock_post):
        fail_resp = MagicMock()
        fail_resp.status_code = 429
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"ok": True}
        mock_post.side_effect = [fail_resp, success_resp]

        result = _call_sarvam_with_retry({"model": "test"}, "fake-key")
        assert result == {"ok": True}

    @patch("longform.sarvam_sync.requests.post")
    def test_non_retryable_error_raises(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("400 Bad Request")
        mock_post.return_value = mock_resp

        with pytest.raises(requests.exceptions.HTTPError):
            _call_sarvam_with_retry({"model": "test"}, "fake-key")

    @patch("longform.sarvam_sync.requests.post")
    @patch("longform.sarvam_sync.time.sleep")
    def test_network_exception_retried(self, mock_sleep, mock_post):
        success_resp = MagicMock()
        success_resp.status_code = 200
        success_resp.json.return_value = {"ok": True}
        mock_post.side_effect = [
            requests.exceptions.ConnectionError("network down"),
            success_resp,
        ]

        result = _call_sarvam_with_retry({"model": "test"}, "fake-key")
        assert result == {"ok": True}

    @patch("longform.sarvam_sync.requests.post")
    @patch("longform.sarvam_sync.time.sleep")
    def test_all_retries_exhausted_raises(self, mock_sleep, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 503
        mock_post.return_value = mock_resp

        with pytest.raises(RuntimeError, match="Sarvam API failed"):
            _call_sarvam_with_retry({"model": "test"}, "fake-key")

    @patch("longform.sarvam_sync.requests.post")
    def test_headers_include_api_key(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        _call_sarvam_with_retry({"model": "test"}, "my-secret-key")
        call_args = mock_post.call_args
        headers = call_args[1]["headers"]
        assert headers["api-subscription-key"] == "my-secret-key"
        assert headers["Authorization"] == "Bearer my-secret-key"

    @patch("longform.sarvam_sync.requests.post")
    def test_payload_passed_correctly(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        payload = {"model": "sarvam-105b", "messages": [{"role": "user", "content": "hi"}]}
        _call_sarvam_with_retry(payload, "key")
        call_args = mock_post.call_args
        assert call_args[1]["json"] == payload


# ---------------------------------------------------------------------------
# map_scenes_with_sarvam tests (end-to-end with mocked API)
# ---------------------------------------------------------------------------

class TestMapScenesWithSarvam:
    @patch("longform.sarvam_sync._call_sarvam_with_retry")
    def test_successful_mapping(self, mock_call):
        scenes = [make_scene("001"), make_scene("002")]
        mapping = [
            {"scene_id": "001", "start_time": 0, "end_time": 5000},
            {"scene_id": "002", "start_time": 5000, "end_time": 10000},
        ]
        mock_call.return_value = make_sarvam_response(mapping)

        result = map_scenes_with_sarvam(scenes, make_words(), "fake-key")
        assert result[0].start_ms == 0
        assert result[0].end_ms == 5000
        assert result[1].start_ms == 5000
        assert result[1].end_ms == 10000

    @patch("longform.sarvam_sync._call_sarvam_with_retry")
    def test_api_called_once(self, mock_call):
        scenes = [make_scene("001")]
        mapping = [{"scene_id": "001", "start_time": 0, "end_time": 1000}]
        mock_call.return_value = make_sarvam_response(mapping)

        map_scenes_with_sarvam(scenes, make_words(), "key")
        assert mock_call.call_count == 1


# ---------------------------------------------------------------------------
# get_api_key tests
# ---------------------------------------------------------------------------

class TestGetApiKey:
    def test_key_present(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "test-key-123")
        assert get_api_key() == "test-key-123"

    def test_key_missing_raises(self, monkeypatch):
        monkeypatch.delenv("SARVAM_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SARVAM_API_KEY not set"):
            get_api_key()

    def test_empty_key_raises(self, monkeypatch):
        monkeypatch.setenv("SARVAM_API_KEY", "")
        with pytest.raises(RuntimeError, match="SARVAM_API_KEY not set"):
            get_api_key()


# ---------------------------------------------------------------------------
# fallback_proportional_sync tests
# ---------------------------------------------------------------------------

class TestFallbackProportionalSync:
    def test_basic_sync(self):
        scenes = [make_scene("001", "short"), make_scene("002", "longer narration here")]
        words = make_words()
        result = fallback_proportional_sync(scenes, words)
        assert result[0].has_timing
        assert result[1].has_timing

    def test_first_scene_starts_at_first_word(self):
        scenes = [make_scene("001", "a"), make_scene("002", "b")]
        words = make_words()  # starts at 0
        result = fallback_proportional_sync(scenes, words)
        assert result[0].start_ms == 0

    def test_last_scene_ends_at_last_word(self):
        scenes = [make_scene("001", "a"), make_scene("002", "b")]
        words = make_words()  # ends at 1000
        result = fallback_proportional_sync(scenes, words)
        assert result[-1].end_ms == 1000

    def test_empty_words_raises(self):
        scenes = [make_scene("001", "a")]
        with pytest.raises(RuntimeError, match="No whisper words"):
            fallback_proportional_sync(scenes, [])

    def test_word_count_affects_duration(self):
        """Scene with more words should get more time."""
        scenes = [
            make_scene("001", "one"),
            make_scene("002", "one two three four five six seven eight"),
        ]
        words = make_words()
        result = fallback_proportional_sync(scenes, words)
        # Second scene has more words, should have more time
        assert result[1].duration_ms > result[0].duration_ms

    def test_single_scene(self):
        scenes = [make_scene("001", "hello")]
        words = make_words()
        result = fallback_proportional_sync(scenes, words)
        assert result[0].start_ms == 0
        assert result[0].end_ms == 1000

    def test_all_scenes_contiguous(self):
        scenes = [make_scene(f"{i:03d}", f"word{i}") for i in range(1, 6)]
        words = make_words()
        result = fallback_proportional_sync(scenes, words)
        for i in range(len(result) - 1):
            assert result[i].end_ms == result[i + 1].start_ms
