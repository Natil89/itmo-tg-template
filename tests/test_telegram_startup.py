import asyncio
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.exceptions import TelegramNetworkError, TelegramUnauthorizedError
from aiogram.methods import GetMe

from app.telegram import TelegramStartupError, check_telegram_startup


async def test_transient_get_me_timeout_retries_without_replacing_connection(monkeypatch):
    me, webhook = Mock(), Mock()
    bot = Mock(
        get_me=AsyncMock(side_effect=[TelegramNetworkError(GetMe(), "private-key"), me]),
        get_webhook_info=AsyncMock(return_value=webhook),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.telegram.asyncio.sleep", sleep)
    assert await check_telegram_startup(bot) == (me, webhook)
    assert bot.get_me.await_count == 2
    bot.get_webhook_info.assert_awaited_once_with(request_timeout=8)
    sleep.assert_awaited_once_with(1)


async def test_exhausted_network_checks_stop_after_three_and_hide_details(monkeypatch, caplog):
    bot = Mock(
        get_me=AsyncMock(side_effect=TelegramNetworkError(GetMe(), "private-key")),
        get_webhook_info=AsyncMock(),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.telegram.asyncio.sleep", sleep)
    with pytest.raises(TelegramStartupError) as error:
        await check_telegram_startup(bot)
    assert bot.get_me.await_count == 3 and sleep.await_count == 2
    bot.get_webhook_info.assert_not_called()
    assert "private-key" not in str(error.value) and "private-key" not in caplog.text
    assert "attempt=3/3" in caplog.text


async def test_webhook_timeout_retries_whole_read_handshake(monkeypatch):
    me, webhook = Mock(), Mock()
    bot = Mock(
        get_me=AsyncMock(return_value=me),
        get_webhook_info=AsyncMock(side_effect=[TimeoutError("private-key"), webhook]),
    )
    monkeypatch.setattr("app.telegram.asyncio.sleep", AsyncMock())
    assert await check_telegram_startup(bot) == (me, webhook)
    assert bot.get_me.await_count == 2 and bot.get_webhook_info.await_count == 2


async def test_invalid_token_is_not_retried(monkeypatch):
    bot = Mock(
        get_me=AsyncMock(side_effect=TelegramUnauthorizedError(GetMe(), "unauthorized")),
        get_webhook_info=AsyncMock(),
    )
    sleep = AsyncMock()
    monkeypatch.setattr("app.telegram.asyncio.sleep", sleep)
    with pytest.raises(TelegramUnauthorizedError):
        await check_telegram_startup(bot)
    assert bot.get_me.await_count == 1
    sleep.assert_not_called()


async def test_cancellation_is_propagated_without_retry(monkeypatch):
    bot = Mock(get_me=AsyncMock(side_effect=asyncio.CancelledError()), get_webhook_info=AsyncMock())
    sleep = AsyncMock()
    monkeypatch.setattr("app.telegram.asyncio.sleep", sleep)
    with pytest.raises(asyncio.CancelledError):
        await check_telegram_startup(bot)
    assert bot.get_me.await_count == 1
    sleep.assert_not_called()
