"""
Sarvam AI scene/timestamp mapping (optional, --sync-method sarvam).

Uses the Sarvam AI Chat Completions API (sarvam-105b model) to map
scenes onto the Whisper word-timestamp timeline. This is the replacement
for the original Gemini-based single-shot LLM JSON-mapping approach.

Sarvam AI API details:
- Endpoint: POST https://api.sarvam.ai/v1/chat/completions
- Auth: api-subscription-key header (or Authorization: Bearer)
- Body: OpenAI-compatible format with model, messages, etc.
- Response: OpenAI-compatible format with choices[0].message.content

Note: The deterministic sync method (deterministic_sync.py) is the
recommended default. This Sarvam-based method is kept for reference /
experimentation, just like the original Gemini path.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Optional

import requests

from .config import (
    SARVAM_API_KEY_ENV,
    SARVAM_MAX_RETRIES,
    SARVAM_MODEL,
    SARVAM_RETRY_WAIT_SECONDS,
    SARVAM_URL,
)
from .models import Scene

log = logging.getLogger("longform")


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from *text*, stripping markdown code fences."""
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def _build_sarvam_payload(scenes: list[Scene], words: list[dict]) -> dict:
    """Build the Sarvam Chat Completions request payload.

    Separated from the API call so it can be unit-tested without network.
    """
    scene_payload = [{"scene_id": s.scene_id, "text": s.text} for s in scenes]

    prompt = f"""You are an expert audio/video alignment engine.

You are given:
1. SCRIPT_SCENES: an ordered list of narration scenes, each with a scene_id and its exact text.
2. WHISPER_WORDS: word-level speech-to-text timestamps (milliseconds) from the full narration audio, in chronological order.

Task: map every scene in SCRIPT_SCENES onto the WHISPER_WORDS timeline by matching each
scene's text to the corresponding contiguous run of words. Output the millisecond start
time of the first word of the scene and the millisecond end time of the last word of the
scene.

Rules:
- Every scene_id in SCRIPT_SCENES must appear exactly once in your output.
- Scenes must be contiguous and non-overlapping, in the same order as SCRIPT_SCENES.
- Base timings strictly on WHISPER_WORDS; do not invent times.
- Output ONLY a raw JSON array, no markdown fences, no commentary, in this exact shape:
[{{"scene_id": "001", "start_time": 0, "end_time": 4120}}, ...]

SCRIPT_SCENES:
{json.dumps(scene_payload, ensure_ascii=False)}

WHISPER_WORDS:
{json.dumps(words, ensure_ascii=False)}"""

    return {
        "model": SARVAM_MODEL,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }


def _call_sarvam_with_retry(
    payload: dict, api_key: str, url: str = SARVAM_URL
) -> dict:
    """Call the Sarvam Chat Completions API with retry logic.

    Retries on 503 (server unavailable) and 429 (rate limited).
    Raises RuntimeError after SARVAM_MAX_RETRIES attempts.
    """
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": api_key,
        # Also send Bearer for OpenAI-compatible tooling compatibility
        "Authorization": f"Bearer {api_key}",
    }
    last_error: Optional[Exception] = None

    for attempt in range(1, SARVAM_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=180)
        except requests.exceptions.RequestException as e:
            last_error = e
            log.warning(
                "Sarvam network error (attempt %d/%d): %s",
                attempt,
                SARVAM_MAX_RETRIES,
                e,
            )
            time.sleep(SARVAM_RETRY_WAIT_SECONDS)
            continue

        if resp.status_code == 200:
            return resp.json()

        if resp.status_code in (503, 429):
            log.warning(
                "Sarvam returned %d (server unavailable/rate limited). "
                "Retrying in %ds (attempt %d/%d)",
                resp.status_code,
                SARVAM_RETRY_WAIT_SECONDS,
                attempt,
                SARVAM_MAX_RETRIES,
            )
            time.sleep(SARVAM_RETRY_WAIT_SECONDS)
            continue

        # Non-retryable error — Sarvam returns 403 for auth failures (not 401)
        resp.raise_for_status()

    raise RuntimeError(
        f"Sarvam API failed after {SARVAM_MAX_RETRIES} attempts: {last_error}"
    )


def _parse_sarvam_response(data: dict, scenes: list[Scene]) -> list[Scene]:
    """Parse the Sarvam API response and assign timings to scenes.

    Separated from the API call so it can be unit-tested with mock data.
    """
    try:
        text = data["choices"][0]["message"]["content"]
        mapping = _extract_json_array(text)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Could not parse Sarvam response: {e}\nRaw: {data}") from e

    by_id = {s.scene_id: s for s in scenes}
    mapped_ids: set[str] = set()
    for item in mapping:
        sid = str(item["scene_id"])
        if sid not in by_id:
            continue
        by_id[sid].start_ms = int(item["start_time"])
        by_id[sid].end_ms = int(item["end_time"])
        mapped_ids.add(sid)

    missing = [s.scene_id for s in scenes if s.scene_id not in mapped_ids]
    if missing:
        raise RuntimeError(f"Sarvam response is missing scene(s): {missing}")

    return scenes


def map_scenes_with_sarvam(
    scenes: list[Scene], words: list[dict], api_key: str
) -> list[Scene]:
    """Map scenes onto the Whisper timeline using the Sarvam AI API.

    Returns the same *scenes* list with start_ms/end_ms assigned.
    """
    payload = _build_sarvam_payload(scenes, words)

    log.info(
        "Calling Sarvam (%s) to map %d scenes against %d words",
        SARVAM_MODEL,
        len(scenes),
        len(words),
    )
    data = _call_sarvam_with_retry(payload, api_key)
    return _parse_sarvam_response(data, scenes)


def get_api_key() -> str:
    """Read the Sarvam API key from the environment variable.

    Raises RuntimeError if not set.
    """
    key = os.environ.get(SARVAM_API_KEY_ENV)
    if not key:
        raise RuntimeError(f"{SARVAM_API_KEY_ENV} not set")
    return key


def fallback_proportional_sync(scenes: list[Scene], words: list[dict]) -> list[Scene]:
    """Last-resort safety net if Sarvam is unavailable after all retries.

    Only used when --sync-method sarvam. Splits total narration duration
    across scenes proportionally to each scene's word count.
    """
    log.warning("Using fallback proportional sync (word-count based)")
    if not words:
        raise RuntimeError("No whisper words available for fallback sync")

    total_start = words[0]["start"]
    total_end = words[-1]["end"]
    total_span = max(1, total_end - total_start)

    word_counts = [max(1, len(s.text.split())) for s in scenes]
    total_words = sum(word_counts)

    cursor = total_start
    for scene, wc in zip(scenes, word_counts):
        share_ms = round(total_span * (wc / total_words))
        scene.start_ms = cursor
        scene.end_ms = cursor + share_ms
        cursor += share_ms
    scenes[-1].end_ms = total_end
    return scenes
