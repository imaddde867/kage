"""Telegram bridge — opt-in messaging connector for Kage.

Routes incoming Telegram messages to BrainService and streams replies back.
Lets you message Kage from your iPhone while the local model runs on your Mac.

Setup
-----
1. Create a bot via @BotFather and copy the token.
2. Set in .env:
       TELEGRAM_BOT_TOKEN=<token>
       TELEGRAM_ALLOWED_CHAT_IDS=<your numeric chat id>   # comma-separated
3. Install the optional dependency:
       pip install python-telegram-bot
4. Enable in main.py or app_runner.py by calling TelegramBridge(settings).start()
   in a background thread.

The bridge polls the Telegram API (no webhook needed).  Each message is routed
through the provided think_callback, which should match the signature of
BrainService.think_text_stream.  Replies longer than 4096 characters are
automatically split into Telegram-sized chunks.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterator
from typing import Any

logger = logging.getLogger(__name__)

_TELEGRAM_CHUNK_SIZE = 4096  # Telegram message size limit


try:
    from telegram import Update  # type: ignore[import]
    from telegram.ext import Application, CommandHandler, MessageHandler, filters  # type: ignore[import]
    _TELEGRAM_AVAILABLE = True
except ImportError:
    _TELEGRAM_AVAILABLE = False


class TelegramBridge:
    """Polling-mode Telegram bot that routes messages to a BrainService callback.

    Parameters
    ----------
    settings:
        The Kage Settings object (must have telegram_bot_token and
        telegram_allowed_chat_ids).
    think_callback:
        Callable[[str], Iterator[str]] — typically BrainService.think_text_stream.
        Called on each incoming message; its streamed chunks are concatenated and
        sent back as one or more Telegram messages.
    """

    def __init__(
        self,
        settings: Any,
        think_callback: Callable[[str], Iterator[str]],
    ) -> None:
        self._settings = settings
        self._think = think_callback
        self._thread: threading.Thread | None = None

    def _allowed_chat_ids(self) -> set[int]:
        raw = getattr(self._settings, "telegram_allowed_chat_ids", ())
        result: set[int] = set()
        for item in raw:
            try:
                result.add(int(str(item).strip()))
            except (ValueError, TypeError):
                pass
        return result

    def _bot_token(self) -> str:
        return str(getattr(self._settings, "telegram_bot_token", "")).strip()

    def _split_message(self, text: str) -> list[str]:
        """Split a long reply into Telegram-sized chunks."""
        chunks: list[str] = []
        while text:
            chunks.append(text[:_TELEGRAM_CHUNK_SIZE])
            text = text[_TELEGRAM_CHUNK_SIZE:]
        return chunks or [""]

    def _build_app(self) -> Any:
        if not _TELEGRAM_AVAILABLE:
            raise RuntimeError(
                "python-telegram-bot is not installed. Run: pip install python-telegram-bot"
            )
        token = self._bot_token()
        if not token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is not set in .env")

        allowed = self._allowed_chat_ids()

        async def _handle_message(update: Any, context: Any) -> None:
            if update.message is None or update.message.text is None:
                return
            chat_id = update.effective_chat.id if update.effective_chat else None
            if allowed and chat_id not in allowed:
                logger.warning("Telegram message from unauthorized chat_id=%s — ignored", chat_id)
                await update.message.reply_text("Unauthorized.")
                return

            user_text = update.message.text.strip()
            if not user_text:
                return

            logger.info("Telegram message from chat_id=%s: %r", chat_id, user_text[:80])
            try:
                parts: list[str] = []
                for chunk in self._think(user_text):
                    parts.append(chunk)
                reply = "".join(parts).strip()
            except Exception as exc:
                logger.error("Error generating reply for Telegram message: %s", exc)
                reply = "Sorry, I encountered an error processing your message."

            if not reply:
                reply = "(no response)"

            for chunk in self._split_message(reply):
                await update.message.reply_text(chunk)

        async def _handle_start(update: Any, context: Any) -> None:
            if update.message:
                await update.message.reply_text("Kage is listening.")

        app = Application.builder().token(token).build()
        app.add_handler(CommandHandler("start", _handle_start))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message))
        return app

    def start(self) -> None:
        """Start the Telegram polling loop in a background daemon thread."""
        if not _TELEGRAM_AVAILABLE:
            logger.warning(
                "Telegram bridge disabled: python-telegram-bot is not installed. "
                "Run: pip install python-telegram-bot"
            )
            return

        token = self._bot_token()
        if not token:
            logger.info("Telegram bridge disabled: TELEGRAM_BOT_TOKEN not set")
            return

        def _run() -> None:
            import asyncio
            try:
                app = self._build_app()
                asyncio.run(app.run_polling())
            except Exception as exc:
                logger.error("Telegram bridge crashed: %s", exc)

        self._thread = threading.Thread(target=_run, name="telegram-bridge", daemon=True)
        self._thread.start()
        logger.info("Telegram bridge started (polling)")
