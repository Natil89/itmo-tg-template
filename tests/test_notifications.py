from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.handlers.assistant import create_router
from app.notifications import (
    CitySelection,
    NotificationError,
    NotificationProfile,
    NotificationStore,
)
from app.weather import GeoCity

CITY = GeoCity(city="Москва", country="Россия", admin1="Москва", latitude=55.75, longitude=37.62)
NOW = datetime(2027, 1, 1, 5, tzinfo=UTC)


def profile(enabled=False):
    return NotificationProfile(101, 101, CITY, "Europe/Moscow", enabled, 1)


async def test_city_coordinates_not_persisted_until_owner_confirms():
    clock = [NOW]
    weather = Mock(find_cities=AsyncMock(return_value=[CITY]))
    store = Mock(confirm_city=AsyncMock(return_value=profile()))
    selection = CitySelection(weather, store, clock=lambda: clock[0])
    proposals = await selection.propose(101, 101, "Москва, Россия")
    store.confirm_city.assert_not_awaited()
    token = str(proposals[0].token)
    with pytest.raises(NotificationError):
        await selection.confirm(token, 202, 202)
    store.confirm_city.assert_not_awaited()
    assert await selection.confirm(token, 101, 101) == profile()
    assert store.confirm_city.call_args.args[:2] == (101, 101)
    assert store.confirm_city.call_args.kwargs["expires_at"] == NOW + timedelta(minutes=5)
    with pytest.raises(NotificationError):
        await selection.confirm(token, 101, 101)
    assert store.confirm_city.await_count == 1


async def test_city_choice_expires_and_cleanup_removes_coordinates():
    clock = [NOW]
    store = Mock(confirm_city=AsyncMock())
    selection = CitySelection(
        Mock(find_cities=AsyncMock(return_value=[CITY])), store, clock=lambda: clock[0]
    )
    proposal = (await selection.propose(101, 101, "Москва"))[0]
    clock[0] += timedelta(minutes=5)
    with pytest.raises(NotificationError):
        await selection.confirm(str(proposal.token), 101, 101)
    assert not selection._proposals
    store.confirm_city.assert_not_awaited()
    await selection.propose(101, 101, "Москва")
    clock[0] += timedelta(minutes=5)
    await selection.purge()
    assert not selection._proposals


async def test_city_expiry_rechecked_after_waiting_for_transaction_locks():
    clock = [NOW]
    connection = Mock(
        execute=AsyncMock(), fetchrow=AsyncMock(return_value=None), fetchval=AsyncMock()
    )

    @asynccontextmanager
    async def transaction():
        yield

    connection.transaction = transaction

    async def lock(*_):
        clock[0] += timedelta(minutes=5)

    connection.execute.side_effect = lock
    store = NotificationStore(Mock())

    @asynccontextmanager
    async def acquire():
        yield connection

    store.connection = acquire
    with pytest.raises(NotificationError, match="истёк"):
        await store.confirm_city(
            101,
            101,
            uuid4(),
            CITY,
            NOW,
            expires_at=NOW + timedelta(minutes=5),
            clock=lambda: clock[0],
        )
    connection.fetchval.assert_not_awaited()  # Даже nonce не записан, не говоря о координатах.


async def dispatch(text, store, selection=None):
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    reminder_store = Mock(get_timezone=AsyncMock(return_value="Europe/Moscow"))
    store.touch_destination = AsyncMock()
    try:
        await dispatcher.feed_update(
            bot,
            Update(
                update_id=1,
                message=Message(
                    message_id=1,
                    date=NOW,
                    chat=Chat(id=101, type="private"),
                    from_user=User(id=101, is_bot=False, first_name="Test"),
                    text=text,
                ),
            ),
            notification_store=store,
            city_selection=selection,
            reminders=reminder_store,
            delivery_enabled=True,
        )
        return [
            call.args[1]
            for call in bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
    finally:
        await bot.session.close()


async def test_city_command_displays_normalized_name_and_confirmation_button():
    store = Mock(confirm_city=AsyncMock())
    selection = CitySelection(Mock(find_cities=AsyncMock(return_value=[CITY])), store)
    sent = await dispatch("/city Москва, Россия", store, selection)
    assert "Москва, Россия" in sent[0].text
    assert sent[0].reply_markup.inline_keyboard[0][0].callback_data.startswith("city:")
    assert "55.75" not in sent[0].text
    store.confirm_city.assert_not_awaited()


async def test_notifications_has_no_internal_ids_or_coordinates():
    store = Mock(get=AsyncMock(return_value=profile(True)))
    sent = await dispatch("/notifications", store)
    assert "Москва, Россия" in sent[0].text and "08:00" in sent[0].text
    assert "Europe/Moscow" in sent[0].text
    assert "101" not in sent[0].text and "55.75" not in sent[0].text


async def test_briefing_on_missing_configuration_does_not_pretend_success():
    store = Mock(set_enabled=AsyncMock(side_effect=NotificationError("Сначала подтвердите город.")))
    sent = await dispatch("/briefing_on", store)
    assert sent[0].text == "Сначала подтвердите город."


async def test_long_city_names_keep_full_text_and_buttons_without_exceeding_telegram_limit():
    large = GeoCity(
        city="🌍" * 200, country="🌎" * 200, admin1="🌏" * 200, latitude=55.0, longitude=37.0
    )
    store = Mock(confirm_city=AsyncMock())
    selection = CitySelection(Mock(find_cities=AsyncMock(return_value=[large] * 5)), store)
    sent = await dispatch("/city город", store, selection)
    assert len(sent) > 1
    assert all(len(part.text.encode("utf-16-le")) <= 4096 * 2 for part in sent)
    assert sent[-1].reply_markup is not None
    assert all(part.reply_markup is None for part in sent[:-1])
    store.confirm_city.assert_not_awaited()
