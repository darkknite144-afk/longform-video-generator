"""
Whisper transcription (offline, CPU).

Uses faster-whisper to produce word-level timestamps needed by the
deterministic sync stage.
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger("longform")


def transcribe_audio(audio_path: str, model_size: str = "small") -> tuple[list[dict], int]:
    """Transcribe *audio_path* and return (word_list, total_audio_duration_ms).

    Each word dict has keys: word (str), start (int ms), end (int ms).

    The duration is needed downstream so the deterministic sync stage can
    anchor the first and last scene boundaries to the true start/end of the
    audio file (not just the first/last detected word), guaranteeing 100%
    coverage.
    """
    from faster_whisper import WhisperModel  # imported here so --help stays fast

    log.info("Loading faster-whisper model '%s' (CPU, int8)", model_size)
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    log.info("Transcribing %s", audio_path)
    segments, info = model.transcribe(audio_path, word_timestamps=True, vad_filter=True)
    log.info("Detected language=%s duration=%.1fs", info.language, info.duration)

    words: list[dict] = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words.append(
                {
                    "word": w.word.strip(),
                    "start": round(w.start * 1000),
                    "end": round(w.end * 1000),
                }
            )
    log.info("Transcribed %d words", len(words))
    duration_ms = round(info.duration * 1000)
    return words, duration_ms


def load_whisper_words(path: str) -> list[dict]:
    """Load previously-saved Whisper word timestamps from a JSON file."""
    import json
    from pathlib import Path

    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_whisper_words(words: list[dict], path: str) -> None:
    """Persist Whisper word timestamps to *path* as JSON."""
    import json
    from pathlib import Path

    Path(path).write_text(json.dumps(words), encoding="utf-8")


def normalize_word(word: str) -> str:
    """Strip punctuation and lower-case a word for matching purposes."""
    import re

    _PUNCT_RE = re.compile(r"[\"'“”‘’.,!?;:()\[\]{}—\-–।॥…]")
    return _PUNCT_RE.sub("", word).strip().lower()
