"""
Delivery module: upload the final video to Telegram via MTProto
(Kurigram, a maintained Pyrogram-compatible fork), bypassing the 50MB
Bot API cap.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .config import VIDEO_H, VIDEO_W

log = logging.getLogger("longform")


def _parse_chat_id(chat_id_raw: str):
    """Parse a Telegram chat ID from its raw string form.

    Numeric chat IDs (users, groups, channels incl. "-100..." supergroups)
    MUST be passed as int, not str — pyrogram's peer resolver treats a
    numeric *string* as a phone number, which bots can't resolve.
    """
    chat_id_raw = chat_id_raw.strip()
    if chat_id_raw.startswith("@"):
        return chat_id_raw
    return int(chat_id_raw)


async def upload_to_telegram(video_path: Path, caption: str = "") -> None:
    """Upload *video_path* to Telegram via MTProto (Kurigram/Pyrogram).

    Requires these environment variables:
        TELEGRAM_API_ID, TELEGRAM_API_HASH, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    """
    from pyrogram import Client  # kurigram package, same import path

    api_id = int(os.environ["TELEGRAM_API_ID"])
    api_hash = os.environ["TELEGRAM_API_HASH"]
    bot_token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat_id = _parse_chat_id(os.environ["TELEGRAM_CHAT_ID"])

    app = Client(
        "longform_uploader",
        api_id=api_id,
        api_hash=api_hash,
        bot_token=bot_token,
        in_memory=True,
    )

    def progress(current: int, total: int) -> None:
        pct = (current / total) * 100 if total else 0
        if int(pct) % 10 == 0:
            log.info("Telegram upload: %.1f%% (%d/%d bytes)", pct, current, total)

    async with app:
        log.info(
            "Uploading %s (%.1f MB) to chat %s via MTProto",
            video_path,
            video_path.stat().st_size / 1e6,
            chat_id,
        )
        await app.send_video(
            chat_id=chat_id,
            video=str(video_path),
            caption=caption,
            supports_streaming=True,
            width=VIDEO_W,
            height=VIDEO_H,
            progress=progress,
        )
    log.info("Upload complete")
