# Automated YouTube Long-Form Video Generator

Automated pipeline that turns a narration script + stock video assets into a
finished long-form video with Ken Burns motion, Meme/GIF overlays, and
synced audio — then delivers it via Telegram.

## What's New (v2.0)

- **Gemini API replaced with Sarvam AI API** — uses `sarvam-105b` model via the
  Sarvam AI Chat Completions endpoint (`https://api.sarvam.ai/v1/chat/completions`).
  Set your API key in the `SARVAM_API_KEY` environment variable.
- **Modular codebase** — the monolithic `main.py` has been split into focused
  modules under `longform/`, each independently testable.
- **Comprehensive test suite** — 271 test cases covering every module.

## Pipeline

```
parse script → whisper transcription → scene/timestamp sync
→ per-scene FFmpeg processing (Ken Burns motion) → concat mux
→ Telegram delivery via MTProto
```

### Sync methods

| Method | Flag | Description |
|--------|------|-------------|
| **deterministic** (default) | `--sync-method deterministic` | Sequence alignment (difflib) against real Whisper word-timestamps. Guaranteed 100% audio coverage, fully reproducible. Recommended. |
| **sarvam** | `--sync-method sarvam` | LLM-based single-shot JSON mapping using Sarvam AI (`sarvam-105b`). Kept for reference/experimentation. Falls back to proportional sync on failure. |

## Setup

```bash
pip install -r requirements.txt

# For Sarvam AI sync (optional)
export SARVAM_API_KEY="your-sarvam-api-key"

# For Telegram delivery
export TELEGRAM_API_ID="your-api-id"
export TELEGRAM_API_HASH="your-api-hash"
export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"
```

## Usage

```bash
# Default (deterministic sync)
python main.py \
  --assets-file youtube_longform_assets.txt \
  --audio-file full_audio.mp3 \
  --output output/final_video.mp4

# Sarvam AI sync
python main.py --sync-method sarvam

# Skip Telegram upload (local testing)
python main.py --no-deliver
```

## Project Structure

```
longform-video-generator/
├── main.py                      # Entry point / CLI
├── requirements.txt
├── longform/
│   ├── __init__.py              # Package exports
│   ├── config.py                # All configuration constants
│   ├── models.py                # Scene dataclass
│   ├── parser.py                # Assets file parsing
│   ├── whisper_transcribe.py    # Whisper transcription
│   ├── deterministic_sync.py    # Deterministic sync (default)
│   ├── sarvam_sync.py           # Sarvam AI sync (replaces Gemini)
│   ├── effects.py               # Ken Burns effects + zoompan expressions
│   ├── video_processor.py       # FFmpeg per-scene processing
│   ├── muxer.py                 # Concat + audio mux
│   └── telegram_delivery.py     # Telegram MTProto upload
└── tests/
    ├── test_config.py
    ├── test_models.py
    ├── test_parser.py
    ├── test_effects.py
    ├── test_deterministic_sync.py
    ├── test_sarvam_sync.py
    ├── test_video_processor.py
    ├── test_muxer.py
    ├── test_telegram_delivery.py
    └── test_integration.py
```

## Testing

```bash
pip install pytest
pytest tests/ -v
```

All 271 tests use synthetic data and mocked network calls — no real API
keys, audio files, or FFmpeg are needed.
