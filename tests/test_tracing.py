import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, Update, User

from app.handlers.assistant import create_router
from app.tracing import current_trace, recorded, span
from tests.test_llm_tools import fixture


async def test_parallel_spans_keep_parent_and_only_provider_usage_is_saved():
    store = Mock(save=AsyncMock())

    async def read_weather():
        with span("briefing", "read_weather"):
            await asyncio.sleep(0)

    llm, session = fixture({"content": "private response"}, finish="stop")
    response = session.post.return_value.__aenter__.return_value
    payload = await response.json()
    payload["usage"] = {"prompt_tokens": 123, "completion_tokens": 7, "reasoning": "hidden"}
    response.json.return_value = payload

    async with recorded(store, 101, "daily_briefing", "briefing", "delivery_attempt") as root:
        await asyncio.gather(read_weather(), llm.generate("private query"))
        with span("mcp", "call_tool", tool="secret as tool name"):
            pass
    trace = store.save.call_args.args[0]
    assert current_trace.get() is None
    assert all(step.parent_id == root.id for step in trace.spans[1:])
    llm_step = next(step for step in trace.spans if step.component == "llm")
    assert (llm_step.input_tokens, llm_step.output_tokens) == (123, 7)
    assert trace.spans[-1].tool_name is None
    assert "private" not in repr(trace) and "hidden" not in repr(trace)


async def test_trace_command_reads_only_owner_and_does_not_create_new_trace():
    identity = uuid4()
    row = {
        "trace_id": identity,
        "kind": "user_query",
        "route": "schedule",
        "confidence": 0.9,
        "ordinal": 0,
        "component": "mcp",
        "operation": "call_tool",
        "status": "ok",
        "duration_ms": 7,
        "tool_name": "get_schedule",
        "input_tokens": None,
        "output_tokens": None,
        "error_code": None,
    }
    store = Mock(get=AsyncMock(side_effect=[[row], []]), save=AsyncMock())
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    try:
        for number, actor, command in ((1, 101, "/trace last"), (2, 202, f"/trace {identity}")):
            message = Message(
                message_id=number,
                date=datetime.now(UTC),
                chat=Chat(id=actor, type="private"),
                from_user=User(id=actor, is_bot=False, first_name="Test"),
                text=command,
            )
            await dispatcher.feed_update(
                bot, Update(update_id=number, message=message), trace_store=store
            )
        assert [call.args for call in store.get.await_args_list] == [(101, None), (202, identity)]
        sent = [
            call.args[1].text
            for call in bot.session.call_args_list
            if isinstance(call.args[1], SendMessage)
        ]
        assert str(identity) in sent[0] and "get_schedule" in sent[0] and "7 мс" in sent[0]
        assert sent[1] == "Трасса не найдена."
        store.save.assert_not_awaited()
    finally:
        await bot.session.close()


async def test_storage_failure_preserves_handler_outcome():
    store = Mock(save=AsyncMock(side_effect=RuntimeError("private failure")))

    async with recorded(store, 101, "user_query", "telegram", "request"):
        assert current_trace.get() is not None
    assert current_trace.get() is None


async def test_normal_telegram_request_records_route_and_send_under_same_trace():
    from app.agent_contracts import RouteDecision
    from app.assistant_service import AssistantService
    from app.supervisor import Supervisor

    route = RouteDecision(target="clarify", reason="Untrusted route explanation", confidence=0.9)
    router = Mock(decide=AsyncMock(return_value=route))
    state = Mock(
        get_mode=AsyncMock(return_value="study"),
        get_temperature=AsyncMock(return_value=0.3),
        get_history=AsyncMock(return_value=[]),
        save_history=AsyncMock(),
    )
    store = Mock(save=AsyncMock())
    service = AssistantService(
        Mock(),
        Supervisor(router, {}, Mock()),
        Mock(get_timezone=AsyncMock(return_value=None)),
        "Europe/Moscow",
    )
    bot = Bot("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk")
    bot.session = AsyncMock()
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    try:
        message = Message(
            message_id=1,
            date=datetime.now(UTC),
            chat=Chat(id=101, type="private"),
            from_user=User(id=101, is_bot=False, first_name="Test"),
            text="private message",
        )
        await dispatcher.feed_update(
            bot,
            Update(update_id=1, message=message),
            trace_store=store,
            user_state=state,
            llm=Mock(),
            assistant_service=service,
        )
        trace = store.save.call_args.args[0]
        assert trace.actor_id == 101 and trace.route == "clarify"
        root, supervisor, router_step, send = trace.spans
        assert router_step.parent_id == supervisor.id
        assert supervisor.parent_id == send.parent_id == root.id
        assert send.operation == "send"
        assert "private message" not in repr(trace) and "Untrusted route" not in repr(trace)
    finally:
        await bot.session.close()
