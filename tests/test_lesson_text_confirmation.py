from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import Chat, Message, User
from mcp.types import CallToolResult, Tool

from app.handlers.assistant import text_lesson_confirmation
from app.lesson_confirmation import confirm_lesson, is_affirmation
from app.personal_schedule import LessonDraft, PendingLesson, PersonalScheduleError
from tests.test_personal_schedule_flow import DRAFT


@pytest.mark.parametrize(
    "text", ["да", "Да!", "всё верно", "все верно", "Да, всё верно!", "подтверждаю"]
)
def test_exact_positive_answer(text):
    assert is_affirmation(text)


@pytest.mark.parametrize(
    "text",
    [
        "да, но поменяй время",
        "не подтверждаю",
        "нет",
        "наверное",
        "давай обсудим",
        "что значит да?",
    ],
)
def test_other_text_is_not_consent(text):
    assert not is_affirmation(text)


def message(text="Да!", actor=101, chat=101, chat_type="private"):
    return Message(
        message_id=11,
        date=datetime.now(UTC),
        text=text,
        chat=Chat(id=chat, type=chat_type),
        from_user=User(id=actor, is_bot=False, first_name="Test"),
    )


def adapters():
    pending = PendingLesson(
        "00000000-0000-0000-0000-000000000001",
        101,
        DRAFT,
        datetime.now(UTC) + timedelta(minutes=5),
        "confirmed",
    )
    store = Mock(
        text_confirmation_candidate=AsyncMock(return_value=pending),
        confirm=AsyncMock(return_value=pending),
    )
    mcp = Mock(
        tools=[Tool(name="add_lesson", inputSchema=LessonDraft.model_json_schema())],
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[],
                structuredContent={"lesson_id": 1, **DRAFT.model_dump(), "created": True},
            )
        ),
    )
    return store, mcp, pending


async def test_text_confirmation_uses_same_trusted_action_and_no_llm(monkeypatch):
    monkeypatch.setattr(Message, "answer", AsyncMock())
    store, mcp, pending = adapters()
    assert await text_lesson_confirmation(message(), store, mcp)
    store.text_confirmation_candidate.assert_awaited_once_with(101, 101, 11, reply_to=None)
    store.confirm.assert_awaited_once_with(101, pending.action_id)
    mcp.call_tool.assert_awaited_once_with(
        "add_lesson",
        DRAFT.model_dump(),
        user_id=101,
        action_id=pending.action_id,
    )


@pytest.mark.parametrize("error", [None, PersonalScheduleError("Несколько карточек")])
async def test_absent_or_ambiguous_card_never_executes(monkeypatch, error):
    monkeypatch.setattr(Message, "answer", AsyncMock())
    store, mcp, _ = adapters()
    store.text_confirmation_candidate.return_value = None
    store.text_confirmation_candidate.side_effect = error
    assert await text_lesson_confirmation(message(), store, mcp) is (error is not None)
    store.confirm.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "msg", [message(chat=202), message(chat_type="group"), message("да, но поменяй время")]
)
async def test_wrong_chat_or_nonaffirmation_does_not_touch_storage(msg):
    store, mcp, _ = adapters()
    assert not await text_lesson_confirmation(msg, store, mcp)
    store.text_confirmation_candidate.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_nonstudy_unreplied_yes_is_not_confirmation():
    store, mcp, _ = adapters()
    assert not await text_lesson_confirmation(message(), store, mcp, allow_unreplied=False)
    store.text_confirmation_candidate.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_discovery_removal_during_confirmation_blocks_execution():
    store, mcp, pending = adapters()

    async def confirm(*_):
        mcp.tools = []
        return pending

    store.confirm.side_effect = confirm
    with pytest.raises(PersonalScheduleError, match="недоступно"):
        await confirm_lesson(101, pending.action_id, store, mcp)
    mcp.call_tool.assert_not_awaited()


async def test_real_card_has_keyboard_and_is_bound_to_sent_message():
    from aiogram import Bot, Dispatcher
    from aiogram.methods import SendMessage
    from aiogram.types import Update

    from app.handlers.assistant import create_router

    store, mcp, pending = adapters()
    store.bind_prompt = AsyncMock()
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    sent_card = message("Карточка").model_copy(update={"message_id": 100})
    bot.session = AsyncMock(return_value=sent_card)
    state = AsyncMock()
    state.get_mode.return_value = "study"
    state.get_temperature.return_value = 0.3
    state.get_history.return_value = []
    service = Mock(answer=AsyncMock(return_value=pending), timezone="Europe/Moscow")
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    try:
        await dispatcher.feed_update(
            bot,
            Update(update_id=1, message=message("Добавь занятие")),
            llm=AsyncMock(),
            user_state=state,
            assistant_service=service,
            personal_schedule=store,
            schedule_mcp=mcp,
        )
        sent = [
            call.args[1]
            for call in bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        assert len(sent) == 1
        assert (
            sent[0].reply_markup.inline_keyboard[0][0].callback_data
            == f"lesson:confirm:{pending.action_id}"
        )
        assert "ответьте «да»" in sent[0].text
        store.bind_prompt.assert_awaited_once_with(101, pending.action_id, 101, 100)
        mcp.call_tool.assert_not_awaited()
    finally:
        await bot.session.close()
