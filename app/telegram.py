import asyncio
import logging

from aiogram import Bot
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.exceptions import TelegramNetworkError
from aiogram.types import User, WebhookInfo

from app.config import Settings

logger = logging.getLogger(__name__)


class TelegramStartupError(RuntimeError):
    """Понятная ошибка ограниченной сетевой проверки, без токена и адреса прокси."""


async def check_telegram_startup(bot: Bot) -> tuple[User, WebhookInfo]:
    """До трёх безопасных попыток чтения, общий бюджет попыток и пауз — 27 секунд."""
    for attempt in range(1, 4):
        try:
            async with asyncio.timeout(8):
                me = await bot.get_me(request_timeout=8)
                webhook = await bot.get_webhook_info(request_timeout=8)
            return me, webhook
        except (TelegramNetworkError, TimeoutError) as error:
            logger.warning(
                "Telegram startup network failure: attempt=%d/3 kind=%s",
                attempt,
                type(error).__name__,
            )
            if attempt == 3:
                raise TelegramStartupError(
                    "Telegram не отвечает после трёх попыток. Проверьте доступ к Telegram "
                    "из WSL и работу VPN/прокси, затем повторите запуск."
                ) from None
            await asyncio.sleep(attempt)
    raise AssertionError("Недостижимая ветка проверки Telegram.")


def create_bot(settings: Settings) -> Bot:
    # Одна сессия для getMe, polling и ответов; прямого fallback нет.
    session = AiohttpSession(proxy=settings.telegram_proxy_url or None, timeout=40)
    return Bot(token=settings.bot_token, session=session)
