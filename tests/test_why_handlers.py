"""Аудит Telegram-границы и чтение /why без внешних запросов."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import CallbackQuery, Chat, Message, Update, User
from mcp.types import CallToolResult, Tool

from app.audit import AuditError, AuditRecord, AuditTrace
from app.handlers.assistant import (
    AUDIT_WARNING,
    create_router,
    lesson_confirmation,
    reminder_confirmation,
    why,
)
from app.llm import LlmError
from app.personal_schedule import LessonDraft, PendingLesson, PersonalScheduleError
from app.reminders import PendingReminder, ReminderDraft, ReminderError
from app.user_state import UserStateError

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"
NOW = datetime(2030, 1, 1, 12, tzinfo=UTC)
ACTION_ID = "00000000-0000-0000-0000-000000000001"


def record():
    trace = AuditTrace(started_at=NOW)
    trace.add(
        "tool",
        tool="get_schedule",
        arguments={"date": "2030-01-02"},
        validation="passed",
        execution="success",
        reason="personal_schedule",
    )
    return AuditRecord(started_at=NOW, duration_ms=12, steps=trace.steps)


async def test_why_reads_trusted_owner_in_saved_timezone_without_writing_audit():
    message = Mock(from_user=Mock(id=101), chat=Mock(id=202), answer=AsyncMock())
    store = Mock(latest=AsyncMock(return_value=record()), save=AsyncMock())
    reminders = Mock(get_timezone=AsyncMock(return_value="Europe/Moscow"))
    await why(message, store, reminders)
    store.latest.assert_awaited_once_with(101)
    reminders.get_timezone.assert_awaited_once_with(101)
    store.save.assert_not_awaited()
    text = message.answer.call_args.args[0]
    assert "01.01.2030 15:00:00 (Europe/Moscow)" in text
    assert "get_schedule" in text and "2030-01-02" in text
    assert "Проверка: пройдена" in text
    assert "Выполнение: успешно" in text
    assert "101" not in text and "202" not in text


async def test_why_without_previous_decision_shows_hint_without_loading_timezone():
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(latest=AsyncMock(return_value=None), save=AsyncMock())
    reminders = Mock(get_timezone=AsyncMock())
    await why(message, store, reminders)
    assert "пока нет сохранённых решений" in message.answer.call_args.args[0]
    reminders.get_timezone.assert_not_awaited()
    store.save.assert_not_awaited()


async def test_why_without_owner_does_not_read_audit():
    message = Mock(from_user=None, answer=AsyncMock())
    store = Mock(latest=AsyncMock(), save=AsyncMock())
    await why(message, store)
    store.latest.assert_not_awaited()
    store.save.assert_not_awaited()
    assert "Не удалось определить" in message.answer.call_args.args[0]


async def test_why_default_zone_is_utc():
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(latest=AsyncMock(return_value=record()))
    await why(message, store)
    assert "01.01.2030 12:00:00 (UTC)" in message.answer.call_args.args[0]


async def test_why_can_fall_back_to_utc_when_timezone_storage_fails(caplog):
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(latest=AsyncMock(return_value=record()))
    reminders = Mock(get_timezone=AsyncMock(side_effect=ReminderError("private-secret")))
    await why(message, store, reminders)
    assert "(UTC)" in message.answer.call_args.args[0]
    assert "private-secret" not in caplog.text


async def test_why_storage_failure_has_no_secret_in_response_or_log(caplog):
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(latest=AsyncMock(side_effect=AuditError("private-secret")), save=AsyncMock())
    await why(message, store)
    assert "Диагностика сейчас недоступна" in message.answer.call_args.args[0]
    assert "private-secret" not in caplog.text
    assert "private-secret" not in message.answer.call_args.args[0]
    store.save.assert_not_awaited()


async def dispatch(text, *, audit_store=None, agent=None, user_state=None):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    llm = Mock(generate=AsyncMock(return_value="Обычный ответ"))
    if user_state is None:
        user_state = Mock(
            get_mode=AsyncMock(return_value="study"),
            get_temperature=AsyncMock(return_value=0.3),
            get_history=AsyncMock(return_value=[]),
            save_history=AsyncMock(),
        )
    mcp = Mock(call_tool=AsyncMock(), refresh=AsyncMock())
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
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
        audit_store=audit_store,
        schedule_mcp=mcp,
    )
    sent = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    return sent, llm, user_state, mcp


async def test_why_route_is_read_only_and_does_not_replace_previous_decision():
    store = Mock(latest=AsyncMock(return_value=record()), save=AsyncMock())
    agent = Mock(answer=AsyncMock())
    sent, llm, user_state, mcp = await dispatch("/why", audit_store=store, agent=agent)
    assert "get_schedule" in sent[0].text
    store.latest.assert_awaited_once_with(101)
    store.save.assert_not_awaited()
    agent.answer.assert_not_awaited()
    llm.generate.assert_not_awaited()
    user_state.save_history.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
    mcp.refresh.assert_not_awaited()


async def test_question_passes_trace_and_saves_only_structured_steps_once():
    async def answer(text, *, audit, **kwargs):
        audit.add(
            "tool",
            tool="get_schedule",
            arguments={"date": "2030-01-02"},
            validation="passed",
            execution="success",
            reason="personal_schedule",
        )
        return "Занятий нет."

    agent = Mock(answer=AsyncMock(side_effect=answer))
    store = Mock(save=AsyncMock())
    sent, _, _, _ = await dispatch("private-request-text", audit_store=store, agent=agent)
    assert sent[0].text == "Занятий нет."
    assert store.save.await_count == 1
    owner, key, trace = store.save.call_args.args
    assert (owner, key) == (101, "message:1")
    assert trace is agent.answer.call_args.kwargs["audit"]
    assert trace.steps[0].arguments == {"date": "2030-01-02"}
    assert "private-request-text" not in str(trace.steps)


async def test_no_audit_argument_is_added_without_configured_store():
    agent = Mock(answer=AsyncMock(return_value="Ответ"))
    await dispatch("Вопрос", agent=agent)
    assert "audit" not in agent.answer.call_args.kwargs


async def test_user_state_failure_before_agent_is_saved_as_undetermined_error():
    state = Mock(get_mode=AsyncMock(side_effect=UserStateError("Настройки недоступны.")))
    agent = Mock(answer=AsyncMock())
    store = Mock(save=AsyncMock())
    sent, _, _, _ = await dispatch("Вопрос", audit_store=store, agent=agent, user_state=state)
    assert sent[0].text == "Настройки недоступны."
    agent.answer.assert_not_awaited()
    step = store.save.call_args.args[2].steps[0]
    assert (step.action, step.execution, step.reason) == (
        "undetermined",
        "error",
        "request_failed",
    )


async def test_agent_rejection_trace_is_preserved_when_error_is_returned():
    async def answer(text, *, audit, **kwargs):
        audit.add(
            "tool",
            tool="get_schedule",
            validation="rejected",
            execution="not_run",
            reason="invalid_arguments",
        )
        raise LlmError("Уточните дату.")

    store = Mock(save=AsyncMock())
    sent, _, _, _ = await dispatch(
        "Вопрос",
        audit_store=store,
        agent=Mock(answer=AsyncMock(side_effect=answer)),
    )
    assert sent[0].text == "Уточните дату."
    assert store.save.await_count == 1
    assert store.save.call_args.args[2].steps[0].reason == "invalid_arguments"


async def test_audit_save_failure_does_not_lose_success_or_retry_business(caplog):
    agent = Mock(answer=AsyncMock(return_value="Полезный ответ"))
    store = Mock(save=AsyncMock(side_effect=AuditError("private-secret")))
    sent, _, state, _ = await dispatch("Вопрос", audit_store=store, agent=agent)
    assert sent[0].text.startswith("Полезный ответ")
    assert AUDIT_WARNING in sent[0].text
    assert agent.answer.await_count == 1
    assert state.save_history.await_count == 1
    assert store.save.await_count == 1
    assert "private-secret" not in caplog.text
    assert "101" not in caplog.text


async def test_history_error_after_read_preserves_read_success_and_records_request_failure():
    async def answer(text, *, audit, **kwargs):
        audit.add(
            "tool",
            tool="get_schedule",
            validation="passed",
            execution="success",
            reason="personal_schedule",
        )
        return "Занятий нет."

    state = Mock(
        get_mode=AsyncMock(return_value="study"),
        get_temperature=AsyncMock(return_value=0.3),
        get_history=AsyncMock(return_value=[]),
        save_history=AsyncMock(side_effect=UserStateError("Не удалось сохранить историю.")),
    )
    store = Mock(save=AsyncMock())
    sent, _, _, _ = await dispatch(
        "Вопрос",
        audit_store=store,
        agent=Mock(answer=AsyncMock(side_effect=answer)),
        user_state=state,
    )
    assert "Не удалось сохранить историю" in sent[0].text
    steps = store.save.call_args.args[2].steps
    assert [step.execution for step in steps] == ["success", "error"]
    assert steps[-1].reason == "request_failed"


async def test_fallback_plain_llm_answer_is_not_claimed_as_known_classification():
    store = Mock(save=AsyncMock())
    sent, llm, _, _ = await dispatch("Вопрос", audit_store=store)
    assert sent[0].text == "Обычный ответ"
    llm.generate.assert_awaited_once()
    step = store.save.call_args.args[2].steps[0]
    assert step.action == "undetermined"
    assert step.reason == "classification_unavailable"


async def test_prepared_reminder_is_audited_without_executing_mcp():
    draft = ReminderDraft(text="private-reminder", remind_at="2030-01-02T18:00:00+03:00")
    proposal = PendingReminder(ACTION_ID, 101, draft, "Europe/Moscow", NOW, "prepared")

    async def answer(text, *, audit, **kwargs):
        audit.add(
            "tool",
            tool="add_reminder",
            arguments=draft.model_dump(),
            validation="passed",
            execution="awaiting_confirmation",
            reason="requires_confirmation",
        )
        return proposal

    store = Mock(save=AsyncMock())
    sent, _, _, mcp = await dispatch(
        "Напомни",
        audit_store=store,
        agent=Mock(answer=AsyncMock(side_effect=answer)),
    )
    assert sent[0].reply_markup is not None
    step = store.save.call_args.args[2].steps[0]
    assert step.execution == "awaiting_confirmation"
    assert step.arguments["text_length"] == len("private-reminder")
    assert "private-reminder" not in str(step.arguments)
    mcp.call_tool.assert_not_awaited()


def callback(kind, *, user_id=101, chat_id=None, action="confirm"):
    return CallbackQuery(
        id="opaque-callback",
        from_user=User(id=user_id, is_bot=False, first_name="Test"),
        chat_instance="test",
        data=f"{kind}:{action}:{ACTION_ID}",
        message=Message(
            message_id=5,
            date=NOW,
            chat=Chat(id=user_id if chat_id is None else chat_id, type="private"),
            text="Подтвердить?",
        ),
    )


def callback_tools():
    return [
        Tool(name="add_lesson", inputSchema=LessonDraft.model_json_schema()),
        Tool(name="add_reminder", inputSchema=ReminderDraft.model_json_schema()),
    ]


def confirmation(kind, *, foreign=False, created=True):
    owner = 202 if foreign else 101
    if kind == "lesson":
        draft = LessonDraft(
            date="2030-01-02",
            start="11:00",
            end="12:30",
            title="private-title",
            location="private-location",
        )
        pending = PendingLesson(ACTION_ID, owner, draft, NOW, "confirmed")
        result = {"lesson_id": 1, **draft.model_dump(), "created": created}
        return lesson_confirmation, pending, result, PersonalScheduleError
    draft = ReminderDraft(text="private-reminder", remind_at="2030-01-02T18:00:00+03:00")
    pending = PendingReminder(ACTION_ID, owner, draft, "Europe/Moscow", NOW, "confirmed")
    result = {
        "reminder_id": 1,
        **draft.model_dump(),
        "timezone": "Europe/Moscow",
        "created": created,
    }
    return reminder_confirmation, pending, result, ReminderError


@pytest.fixture
def telegram(monkeypatch):
    answer, edit, acknowledge = AsyncMock(), AsyncMock(), AsyncMock()
    monkeypatch.setattr(Message, "answer", answer)
    monkeypatch.setattr(Message, "edit_reply_markup", edit)
    monkeypatch.setattr(CallbackQuery, "answer", acknowledge)
    return answer, edit, acknowledge


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
@pytest.mark.parametrize("created", [True, False])
async def test_confirmation_audit_is_saved_before_telegram_success(telegram, kind, created):
    answer, _, _ = telegram
    handler, pending, result, _ = confirmation(kind, created=created)
    order = []

    async def save(*args):
        order.append("audit")

    async def reply(*args, **kwargs):
        order.append("reply")

    answer.side_effect = reply
    audit_store = Mock(save=AsyncMock(side_effect=save))
    business = Mock(confirm=AsyncMock(return_value=pending))
    mcp = Mock(
        tools=callback_tools(),
        call_tool=AsyncMock(return_value=CallToolResult(content=[], structuredContent=result)),
    )
    await handler(callback(kind), business, mcp, audit_store)
    assert order == ["audit", "reply"]
    assert mcp.call_tool.await_count == 1
    user_id, key, trace = audit_store.save.call_args.args
    assert (user_id, key) == (101, "callback:opaque-callback")
    step = trace.steps[0]
    assert (step.tool, step.validation, step.execution, step.reason) == (
        f"add_{kind}" if kind == "lesson" else "add_reminder",
        "passed",
        "success",
        "confirmation_accepted",
    )
    assert "private-" not in str(step.arguments)


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
async def test_foreign_callback_does_not_audit_other_users_payload(telegram, kind):
    _, _, _ = telegram
    handler, pending, _, _ = confirmation(kind, foreign=True)
    business = Mock(confirm=AsyncMock(return_value=pending))
    audit_store, mcp = Mock(save=AsyncMock()), Mock(tools=callback_tools(), call_tool=AsyncMock())
    await handler(callback(kind), business, mcp, audit_store)
    step = audit_store.save.call_args.args[2].steps[0]
    assert step.arguments == {}
    assert step.validation == "rejected"
    assert step.reason == "unavailable_confirmation"
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
async def test_expired_or_unavailable_callback_audits_rejection_without_arguments(telegram, kind):
    handler, _, _, error_type = confirmation(kind)
    business = Mock(confirm=AsyncMock(side_effect=error_type("Подтверждение недоступно.")))
    audit_store, mcp = Mock(save=AsyncMock()), Mock(tools=callback_tools(), call_tool=AsyncMock())
    await handler(callback(kind), business, mcp, audit_store)
    step = audit_store.save.call_args.args[2].steps[0]
    assert (step.arguments, step.validation, step.execution, step.reason) == (
        {},
        "rejected",
        "not_run",
        "unavailable_confirmation",
    )
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
@pytest.mark.parametrize("cancelled", [True, False])
async def test_cancellation_audit_never_contains_unread_payload(telegram, kind, cancelled):
    handler, _, _, _ = confirmation(kind)
    business = Mock(cancel=AsyncMock(return_value=cancelled), confirm=AsyncMock())
    audit_store, mcp = Mock(save=AsyncMock()), Mock(tools=callback_tools(), call_tool=AsyncMock())
    await handler(callback(kind, action="cancel"), business, mcp, audit_store)
    step = audit_store.save.call_args.args[2].steps[0]
    assert step.arguments == {}
    assert step.execution == ("cancelled" if cancelled else "not_run")
    assert step.reason == ("cancelled_by_user" if cancelled else "unavailable_confirmation")
    business.confirm.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
async def test_callback_audit_failure_does_not_retry_committed_action(telegram, kind, caplog):
    answer, _, _ = telegram
    handler, pending, result, _ = confirmation(kind)
    business = Mock(confirm=AsyncMock(return_value=pending))
    mcp = Mock(
        tools=callback_tools(),
        call_tool=AsyncMock(return_value=CallToolResult(content=[], structuredContent=result)),
    )
    audit_store = Mock(save=AsyncMock(side_effect=AuditError("private-secret")))
    await handler(callback(kind), business, mcp, audit_store)
    assert business.confirm.await_count == 1 and mcp.call_tool.await_count == 1
    assert audit_store.save.await_count == 1
    assert "добавлено" in answer.call_args.args[0] or "сохранено" in answer.call_args.args[0]
    assert "private-secret" not in caplog.text
    assert ACTION_ID not in caplog.text
    assert "101" not in caplog.text


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
async def test_mcp_failure_is_audited_without_claiming_success(telegram, kind):
    answer, _, _ = telegram
    handler, pending, _, _ = confirmation(kind)
    business = Mock(confirm=AsyncMock(return_value=pending))
    mcp = Mock(
        tools=callback_tools(), call_tool=AsyncMock(side_effect=ConnectionError("private-secret"))
    )
    audit_store = Mock(save=AsyncMock())
    await handler(callback(kind), business, mcp, audit_store)
    step = audit_store.save.call_args.args[2].steps[0]
    assert (step.validation, step.execution, step.reason) == ("passed", "error", "request_failed")
    assert "Не удалось обработать" in answer.call_args.args[0]
    assert "private-secret" not in answer.call_args.args[0]


@pytest.mark.parametrize("kind", ["lesson", "reminder"])
async def test_private_chat_rejection_is_audited_for_current_actor_only(telegram, kind):
    answer, _, acknowledge = telegram
    handler, _, _, _ = confirmation(kind)
    business, mcp = Mock(confirm=AsyncMock()), Mock(call_tool=AsyncMock())
    audit_store = Mock(save=AsyncMock())
    await handler(callback(kind, user_id=202, chat_id=101), business, mcp, audit_store)
    assert audit_store.save.call_args.args[0] == 202
    assert audit_store.save.call_args.args[2].steps[0].arguments == {}
    acknowledge.assert_awaited_once()
    answer.assert_not_awaited()
    business.confirm.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()
