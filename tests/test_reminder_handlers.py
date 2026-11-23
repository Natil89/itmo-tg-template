"""Telegram-граница напоминаний: сообщения подменены, внешних вызовов нет."""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.filters import CommandObject
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User
from mcp.types import CallToolResult

from app.handlers.assistant import create_router, reminder_confirmation, timezone
from app.reminders import PendingReminder, ReminderDraft, ReminderError

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)
ACTION_ID = "00000000-0000-0000-0000-000000000001"


def draft():
    return ReminderDraft(text="Сделать домашку", remind_at="2030-01-02T18:00:00+03:00")


def pending(*, user_id=101, status="confirmed"):
    return PendingReminder(
        action_id=ACTION_ID,
        user_id=user_id,
        payload=draft(),
        timezone="Europe/Moscow",
        expires_at=NOW + timedelta(minutes=5),
        status=status,
    )


def callback(*, user_id=101, chat_id=None, action="confirm", chat_type="private"):
    return CallbackQuery(
        id="callback",
        from_user=User(id=user_id, is_bot=False, first_name="Test"),
        chat_instance="test",
        data=f"reminder:{action}:{ACTION_ID}",
        message=Message(
            message_id=5,
            date=NOW,
            text="Подтвердить?",
            chat=Chat(id=user_id if chat_id is None else chat_id, type=chat_type),
        ),
    )


def save_result(**changes):
    return CallToolResult(
        content=[],
        structuredContent={
            "reminder_id": 7,
            "text": "Сделать домашку",
            "remind_at": "2030-01-02T15:00:00+00:00",
            "timezone": "Europe/Moscow",
            "created": True,
            **changes,
        },
    )


@pytest.fixture
def telegram(monkeypatch):
    answer, edit, acknowledge = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(Message, "answer", answer)
    monkeypatch.setattr(Message, "edit_reply_markup", edit)
    monkeypatch.setattr(CallbackQuery, "answer", acknowledge)
    return answer, edit, acknowledge


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 11, 16, 12, 34, 56, tzinfo=UTC).astimezone(tz)


async def test_timezone_saved_for_telegram_owner_and_shows_current_local_time(monkeypatch):
    from app.handlers import assistant

    monkeypatch.setattr(assistant, "datetime", FixedDatetime)
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(set_timezone=AsyncMock(), get_timezone=AsyncMock())
    await timezone(message, CommandObject(command="timezone", args=" Europe/Moscow "), store)
    store.set_timezone.assert_awaited_once_with(101, "Europe/Moscow")
    store.get_timezone.assert_not_awaited()
    assert message.answer.call_args.args[0] == (
        "Часовой пояс сохранён: Europe/Moscow.\nСейчас в нём: 16.11.2026 15:34:56."
    )


@pytest.mark.parametrize(
    "zone,expected",
    [
        ("Europe/Moscow", "Ваш часовой пояс: Europe/Moscow."),
        (None, "Часовой пояс не задан. Укажите, например: /timezone Europe/Moscow."),
    ],
)
async def test_timezone_without_arguments_shows_saved_value_or_setup_hint(zone, expected):
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(get_timezone=AsyncMock(return_value=zone), set_timezone=AsyncMock())
    await timezone(message, CommandObject(command="timezone"), store)
    store.get_timezone.assert_awaited_once_with(101)
    store.set_timezone.assert_not_awaited()
    assert message.answer.call_args.args[0] == expected


async def test_unknown_timezone_has_safe_domain_error():
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(set_timezone=AsyncMock(side_effect=ReminderError("Неизвестный часовой пояс.")))
    await timezone(message, CommandObject(command="timezone", args="Unknown/Zone"), store)
    assert message.answer.call_args.args[0] == "Неизвестный часовой пояс."


async def test_timezone_without_telegram_owner_does_not_access_store():
    message = Mock(from_user=None, answer=AsyncMock())
    store = Mock(get_timezone=AsyncMock(), set_timezone=AsyncMock())
    await timezone(message, CommandObject(command="timezone", args="Europe/Moscow"), store)
    store.get_timezone.assert_not_awaited()
    store.set_timezone.assert_not_awaited()


async def test_confirmation_uses_callback_owner_and_normalizes_success_time(telegram):
    answer, edit, acknowledge = telegram
    store = Mock(confirm=AsyncMock(return_value=pending()))
    mcp = Mock(call_tool=AsyncMock(return_value=save_result()))
    await reminder_confirmation(callback(), store, mcp)
    store.confirm.assert_awaited_once_with(101, ACTION_ID)
    mcp.call_tool.assert_awaited_once_with(
        "add_reminder",
        draft().model_dump(),
        user_id=101,
        action_id=ACTION_ID,
    )
    assert "Напоминание сохранено №7" in answer.call_args.args[0]
    assert "02.01.2030 18:00:00 (Europe/Moscow)" in answer.call_args.args[0]
    assert "не подключена" in answer.call_args.args[0]
    acknowledge.assert_awaited_once()
    edit.assert_awaited_once_with(reply_markup=None)


async def test_duplicate_confirmation_reports_existing_record(telegram):
    answer, _, _ = telegram
    store = Mock(confirm=AsyncMock(return_value=pending(status="executed")))
    mcp = Mock(call_tool=AsyncMock(return_value=save_result(created=False)))
    await reminder_confirmation(callback(), store, mcp)
    assert "Напоминание уже было сохранено №7" in answer.call_args.args[0]


@pytest.mark.parametrize("reason", ["Чужое действие.", "Подтверждение просрочено."])
async def test_foreign_or_expired_confirmation_never_calls_mcp(telegram, reason):
    answer, edit, _ = telegram
    store = Mock(confirm=AsyncMock(side_effect=ReminderError(reason)))
    mcp = Mock(call_tool=AsyncMock())
    await reminder_confirmation(callback(user_id=202), store, mcp)
    store.confirm.assert_awaited_once_with(202, ACTION_ID)
    mcp.call_tool.assert_not_awaited()
    assert answer.call_args.args[0] == reason
    edit.assert_not_awaited()


@pytest.mark.parametrize("prepared", [pending(user_id=202), pending(status="prepared")])
async def test_untrusted_confirmation_result_is_rejected_before_mcp(telegram, prepared):
    answer, _, _ = telegram
    store = Mock(confirm=AsyncMock(return_value=prepared))
    mcp = Mock(call_tool=AsyncMock())
    await reminder_confirmation(callback(), store, mcp)
    assert answer.call_args.args[0] == "Подтверждение недоступно."
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "changes",
    [
        {"text": "Другой текст"},
        {"remind_at": "2030-01-02T16:00:00+00:00"},
        {"timezone": "UTC"},
        {"reminder_id": 0},
    ],
)
async def test_changed_server_result_is_not_reported_as_success(telegram, changes):
    answer, edit, _ = telegram
    store = Mock(confirm=AsyncMock(return_value=pending()))
    mcp = Mock(call_tool=AsyncMock(return_value=save_result(**changes)))
    await reminder_confirmation(callback(), store, mcp)
    assert answer.call_args.args[0] == "Сервер вернул неожиданный результат сохранения."
    edit.assert_not_awaited()


async def test_error_tool_result_does_not_claim_success(telegram):
    answer, edit, _ = telegram
    store = Mock(confirm=AsyncMock(return_value=pending()))
    mcp = Mock(call_tool=AsyncMock(return_value=CallToolResult(content=[], isError=True)))
    await reminder_confirmation(callback(), store, mcp)
    assert "Не удалось сохранить" in answer.call_args.args[0]
    edit.assert_not_awaited()


async def test_mcp_exception_is_safe_and_leaves_confirmation_for_retry(telegram):
    answer, edit, _ = telegram
    store = Mock(confirm=AsyncMock(return_value=pending()))
    mcp = Mock(call_tool=AsyncMock(side_effect=ConnectionError("private-secret")))
    await reminder_confirmation(callback(), store, mcp)
    assert "Не удалось обработать" in answer.call_args.args[0]
    assert "private-secret" not in answer.call_args.args[0]
    edit.assert_not_awaited()


@pytest.mark.parametrize("cancelled", [False, True])
async def test_cancel_never_calls_mcp(telegram, cancelled):
    answer, _, _ = telegram
    store = Mock(cancel=AsyncMock(return_value=cancelled), confirm=AsyncMock())
    mcp = Mock(call_tool=AsyncMock())
    await reminder_confirmation(callback(action="cancel"), store, mcp)
    store.cancel.assert_awaited_once_with(101, ACTION_ID)
    store.confirm.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    assert ("отменено" if cancelled else "недоступно") in answer.call_args.args[0]


@pytest.mark.parametrize(
    "query",
    [
        callback(chat_type="group"),
        callback(user_id=202, chat_id=101),
    ],
)
async def test_non_owner_chat_is_rejected_without_store_or_mcp(telegram, query):
    answer, _, acknowledge = telegram
    store, mcp = Mock(confirm=AsyncMock()), Mock(call_tool=AsyncMock())
    await reminder_confirmation(query, store, mcp)
    acknowledge.assert_awaited_once_with(
        "Действие доступно только в вашем личном чате.",
        show_alert=True,
    )
    store.confirm.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    answer.assert_not_awaited()


async def dispatch(text, *, agent_answer=None, reminders=None):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    llm = Mock(generate=AsyncMock())
    user_state = Mock(
        get_mode=AsyncMock(return_value="study"),
        get_temperature=AsyncMock(return_value=0.3),
        get_history=AsyncMock(return_value=[]),
        save_history=AsyncMock(),
    )
    agent = Mock(answer=AsyncMock(return_value=agent_answer))
    router = create_router()
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    update = Update(
        update_id=1,
        message=Message(
            message_id=1,
            date=NOW,
            chat=Chat(id=101, type="private"),
            from_user=User(id=101, is_bot=False, first_name="Test"),
            text=text,
        ),
    )
    await dispatcher.feed_update(
        bot,
        update,
        llm=llm,
        user_state=user_state,
        schedule_agent=agent,
        reminders=reminders,
    )
    sent = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    return agent, user_state, sent


async def test_timezone_route_precedes_unknown_command_without_llm():
    store = Mock(get_timezone=AsyncMock(return_value="Europe/Moscow"))
    agent, _, sent = await dispatch("/timezone", reminders=store)
    assert [item.text for item in sent] == ["Ваш часовой пояс: Europe/Moscow."]
    agent.answer.assert_not_awaited()


async def test_prepared_reminder_shows_exact_time_and_bound_confirmation_buttons():
    proposal = pending(status="prepared")
    agent, user_state, sent = await dispatch("Напомни сделать домашку", agent_answer=proposal)
    assert agent.answer.await_count == 1
    assert len(sent) == 1
    assert "02.01.2030 18:00:00" in sent[0].text
    assert "Europe/Moscow" in sent[0].text
    assert "Сделать домашку" in sent[0].text
    assert "не более пяти минут" in sent[0].text
    assert [button.callback_data for button in sent[0].reply_markup.inline_keyboard[0]] == [
        f"reminder:confirm:{ACTION_ID}",
        f"reminder:cancel:{ACTION_ID}",
    ]
    assert user_state.save_history.call_args.args[1][-1]["content"] == sent[0].text


async def test_executed_proposal_is_not_reconfirmed():
    _, _, sent = await dispatch("Напомни сделать домашку", agent_answer=pending(status="executed"))
    assert "уже обработано" in sent[0].text
    assert sent[0].reply_markup is None
