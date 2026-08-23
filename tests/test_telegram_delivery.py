"""
Tests for longform.telegram_delivery — Telegram MTProto delivery.
"""

import pytest

from longform.telegram_delivery import _parse_chat_id


class TestParseChatId:
    def test_numeric_string_returns_int(self):
        assert _parse_chat_id("12345") == 12345

    def test_negative_numeric(self):
        assert _parse_chat_id("-1001234567890") == -1001234567890

    def test_at_mention_returns_string(self):
        assert _parse_chat_id("@mychannel") == "@mychannel"

    def test_strips_whitespace_numeric(self):
        assert _parse_chat_id("  12345  ") == 12345

    def test_strips_whitespace_at_mention(self):
        assert _parse_chat_id("  @mychannel  ") == "@mychannel"

    def test_invalid_string_raises(self):
        with pytest.raises(ValueError):
            _parse_chat_id("not-a-number")

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _parse_chat_id("")

    def test_zero(self):
        assert _parse_chat_id("0") == 0

    def test_large_supergroup_id(self):
        result = _parse_chat_id("-1001234567890")
        assert isinstance(result, int)
        assert result < 0

    def test_at_mention_preserved_exactly(self):
        assert _parse_chat_id("@My_Channel_123") == "@My_Channel_123"
