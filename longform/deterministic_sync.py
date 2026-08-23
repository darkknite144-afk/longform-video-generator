"""
Deterministic scene/timestamp alignment (default sync method).

Forced-alignment sync: matches each scene's script text to the Whisper
word-timestamp stream using sequence alignment (difflib), then derives
scene boundaries directly from the matched anchors.

Why this replaces one-shot LLM JSON mapping for timing: handing an LLM
~70+ scenes against a thousand-plus word timestamp table and asking for
exact millisecond boundaries back is a numeric precision task, and
accuracy visibly degrades over the length of the list. Sequence
alignment against the *actual* Whisper timestamps has no such failure
mode — it's deterministic, always accounts for 100% of the audio
(boundaries[0] is pinned to 0ms and boundaries[-1] to the real audio
end), and its only source of error is genuine ASR mistranscription.
"""

from __future__ import annotations

import difflib
import logging
import re
from typing import Optional

from .models import Scene

log = logging.getLogger("longform")

_PUNCT_RE = re.compile(r"[\"'""''.,!?;:()\[\]{}—\-–।॥…]")


def _normalize_token(tok: str) -> str:
    return _PUNCT_RE.sub("", tok).strip().lower()


def _tokenize_scene_text(text: str) -> list[str]:
    out: list[str] = []
    for raw_tok in text.split():
        norm = _normalize_token(raw_tok)
        if norm:
            out.append(norm)
    return out


def _tokenize_asr_words(words: list[dict]) -> list[str]:
    """Return normalized ASR tokens, preserving index correspondence."""
    return [_normalize_token(w["word"]) for w in words]


def align_scenes_deterministic(
    scenes: list[Scene], words: list[dict], total_duration_ms: int
) -> list[Scene]:
    """Align scenes to the Whisper word timeline using sequence alignment.

    Returns the same *scenes* list with start_ms/end_ms assigned.
    Guarantees full audio coverage (0ms to total_duration_ms).
    """
    if not words:
        raise RuntimeError("No whisper words available for sync")

    n = len(scenes)

    # Flat script token stream + which scene owns which token index range.
    script_tokens: list[str] = []
    scene_tok_start = [0] * n
    scene_tok_end = [0] * n
    for i, s in enumerate(scenes):
        scene_tok_start[i] = len(script_tokens)
        script_tokens.extend(_tokenize_scene_text(s.text))
        scene_tok_end[i] = len(script_tokens)

    # ASR token stream — NOT filtered, so index i always corresponds to
    # words[i] (needed to look up real timestamps for matched anchors).
    asr_tokens = _tokenize_asr_words(words)

    matcher = difflib.SequenceMatcher(None, script_tokens, asr_tokens, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size > 0]

    # anchor_ms[script_token_index] = (start_ms, end_ms) of the matched word
    anchor_ms: dict[int, tuple[int, int]] = {}
    for b in blocks:
        for k in range(b.size):
            si, wi = b.a + k, b.b + k
            anchor_ms[si] = (words[wi]["start"], words[wi]["end"])

    scene_anchor_start: list[Optional[int]] = [None] * n
    scene_anchor_end: list[Optional[int]] = [None] * n
    for i in range(n):
        times = [
            anchor_ms[t]
            for t in range(scene_tok_start[i], scene_tok_end[i])
            if t in anchor_ms
        ]
        if times:
            scene_anchor_start[i] = min(t[0] for t in times)
            scene_anchor_end[i] = max(t[1] for t in times)

    # Expected pace — used both to sanity-check anchors and to weight
    # gap interpolation.
    word_counts = [max(1, len(_tokenize_scene_text(s.text))) for s in scenes]
    total_words = max(1, sum(word_counts))
    avg_ms_per_word = total_duration_ms / total_words

    cum_words_before = 0
    expected_mid_ms = [0.0] * n
    for i in range(n):
        expected_mid_ms[i] = (cum_words_before + word_counts[i] / 2) * avg_ms_per_word
        cum_words_before += word_counts[i]

    # Discard implausible anchors before they reach the boundary maths.
    MAX_DRIFT_RATIO = 3.0
    MAX_DRIFT_SLACK_MS = 4000
    n_discarded = 0
    for i in range(n):
        if scene_anchor_start[i] is None:
            continue
        implied_dur = scene_anchor_end[i] - scene_anchor_start[i]
        expected_dur = word_counts[i] * avg_ms_per_word
        implied_mid = (scene_anchor_start[i] + scene_anchor_end[i]) / 2
        drift = abs(implied_mid - expected_mid_ms[i])
        tolerance = expected_dur * MAX_DRIFT_RATIO + MAX_DRIFT_SLACK_MS
        if implied_dur > tolerance or drift > tolerance:
            log.warning(
                "Scene %s: discarding implausible anchor (spans %.1fs, "
                "centred %.1fs from expected). Timing will be interpolated.",
                scenes[i].scene_id,
                implied_dur / 1000,
                drift / 1000,
            )
            scene_anchor_start[i] = None
            scene_anchor_end[i] = None
            n_discarded += 1

    # boundaries[i] = start of scene i = end of scene i-1.
    # N+1 boundaries for N scenes; pinning the ends guarantees full coverage.
    boundaries: list[Optional[int]] = [None] * (n + 1)
    boundaries[0] = 0
    boundaries[n] = total_duration_ms
    for i in range(1, n):
        prev_end, cur_start = scene_anchor_end[i - 1], scene_anchor_start[i]
        if prev_end is not None and cur_start is not None:
            boundaries[i] = round((prev_end + cur_start) / 2)
        elif prev_end is not None:
            boundaries[i] = prev_end
        elif cur_start is not None:
            boundaries[i] = cur_start

    # Fill gaps (scenes with no reliable anchor) by distributing elapsed
    # time proportionally to each scene's own word count.
    i = 1
    while i < n:
        if boundaries[i] is not None:
            i += 1
            continue
        j = i
        while boundaries[j] is None:
            j += 1
        prev_b, next_b = boundaries[i - 1], boundaries[j]
        gap_words = word_counts[i - 1 : j]
        gap_total_words = max(1, sum(gap_words))
        cum = 0
        for k in range(i, j):
            cum += word_counts[k - 1]
            boundaries[k] = round(prev_b + (next_b - prev_b) * (cum / gap_total_words))
        i = j

    # Safety clamp: enforce non-decreasing in case of a rounding tie-flip.
    for k in range(1, n + 1):
        if boundaries[k] < boundaries[k - 1]:
            boundaries[k] = boundaries[k - 1]

    matched = sum(1 for i in range(n) if scene_anchor_start[i] is not None)
    log.info(
        "Deterministic sync: %d/%d scenes matched (%d discarded, rest "
        "interpolated); full coverage %dms-%dms",
        matched,
        n,
        n_discarded,
        boundaries[0],
        boundaries[n],
    )

    for i, s in enumerate(scenes):
        s.start_ms = boundaries[i]
        s.end_ms = boundaries[i + 1]
        if s.end_ms <= s.start_ms:
            s.end_ms = s.start_ms + 50

    return scenes


def compute_scene_frame_counts(
    scenes: list[Scene], fps: int
) -> list[int]:
    """Compute frame-accurate per-scene lengths.

    Rounds each shared *boundary* to a frame index first, then takes
    consecutive differences. Since scenes are contiguous, this makes any
    +/-1 frame rounding error cancel out from one scene to the next
    (telescoping sum) instead of accumulating.
    """
    if not scenes:
        return []
    boundary_ms = [scenes[0].start_ms] + [s.end_ms for s in scenes]
    frame_bounds = [round(ms / 1000 * fps) for ms in boundary_ms]
    scene_frames = [
        max(1, frame_bounds[i + 1] - frame_bounds[i]) for i in range(len(scenes))
    ]
    return scene_frames
