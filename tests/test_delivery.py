from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError, TelegramRetryAfter
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, User

from app.config import ConfigError, Settings
from app.delivery import (
    DeliveryJob,
    PermanentDeliveryError,
    ReminderDeliveryService,
    TemporaryDeliveryError,
    instant,
)
from app.delivery_telegram import DeliveryDestinationMiddleware, TelegramReminderSender
from tests.test_llm_config import BASE


@pytest.mark.parametrize("flag,expected", [("true", True), ("false", False)])
def test_delivery_configuration_explicit_and_safe(tmp_path, flag, expected):
    settings = Settings.load(
        tmp_path / "missing.env", environ={**BASE, "REMINDER_DELIVERY_ENABLED": flag}
    )
    assert settings.reminder_delivery_enabled is expected


def test_delivery_disabled_by_default_and_unknown_flag_rejected(tmp_path):
    assert not Settings.load(tmp_path / "missing.env", environ=BASE).reminder_delivery_enabled
    with pytest.raises(ConfigError, match="REMINDER_DELIVERY_ENABLED"):
        Settings.load(
            tmp_path / "missing.env", environ={**BASE, "REMINDER_DELIVERY_ENABLED": "yes"}
        )
    with pytest.raises(ValueError):
        instant(datetime(2027, 1, 1))


async def test_recipient_comes_only_from_private_update():
    middleware = DeliveryDestinationMiddleware()
    store, handler = Mock(bind_destination=AsyncMock()), AsyncMock()
    msg = Message(
        message_id=1,
        date=datetime.now(UTC),
        text="destination=999",
        chat=Chat(id=101, type="private"),
        from_user=User(id=101, is_bot=False, first_name="Test"),
    )
    await middleware(handler, msg, {"delivery_store": store})
    store.bind_destination.assert_awaited_once_with(101, 101)
    store.bind_destination.reset_mock()
    await middleware(
        handler,
        msg.model_copy(update={"chat": Chat(id=-1, type="group")}),
        {"delivery_store": store},
    )
    store.bind_destination.assert_not_awaited()


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("network", TemporaryDeliveryError),
        ("limit", TemporaryDeliveryError),
        ("bad", PermanentDeliveryError),
    ],
)
async def test_telegram_error_classification_does_not_expose_text(kind, expected):
    method = SendMessage(chat_id=101, text="test")
    error = (
        TelegramNetworkError(method=method, message="private marker")
        if kind == "network"
        else TelegramRetryAfter(method=method, message="private marker", retry_after=60)
        if kind == "limit"
        else TelegramBadRequest(method=method, message="private marker")
    )
    bot = Mock(send_message=AsyncMock(side_effect=error))
    with pytest.raises(expected) as caught:
        await TelegramReminderSender(bot).send(101, "test")
    assert "private marker" not in str(caught.value)


@pytest.mark.parametrize("boundary", ["late", "lease"])
async def test_clock_crossing_boundary_while_waiting_for_db_never_sends(boundary):
    now = datetime(2027, 1, 1, 12, tzinfo=UTC)
    clock = [now]
    job = DeliveryJob(
        event_id="reminder:1",
        actor_id=101,
        destination_chat_id=9001,
        reminder_id=1,
        due_at=now - timedelta(minutes=15) if boundary == "late" else now,
        text="test",
        attempts=1,
        lease_id=uuid4(),
        lease_until=now + timedelta(seconds=1),
    )

    async def authorize(*_):
        clock[0] += timedelta(seconds=2)
        return True

    store = Mock(
        prepare=AsyncMock(),
        claim=AsyncMock(side_effect=[job, None]),
        authorized=AsyncMock(side_effect=authorize),
        failed=AsyncMock(),
    )
    sender = Mock(send=AsyncMock())
    assert await ReminderDeliveryService(store, sender, clock=lambda: clock[0]).run_once() == 0
    sender.send.assert_not_awaited()
    assert store.failed.call_args.kwargs["code"] == ("late" if boundary == "late" else None)
