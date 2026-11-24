import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiogram.types import CallbackQuery, Chat, Message, User
from mcp.types import CallToolResult, Tool, ToolAnnotations

from app.handlers.assistant import lesson_confirmation
from app.llm import LlmClient, LlmError
from app.personal_schedule import LessonDraft, LessonSaved, PendingLesson, PersonalScheduleError
from app.schedule_agent import ScheduleAgent
from app.weather import WeatherResult

NOW = datetime(2026, 11, 16, tzinfo=UTC)
DRAFT = LessonDraft(
    date="2026-11-17", start="11:00", end="12:30", title="Математика", location="305"
)


def proposal(name, arguments):
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments),
                },
            }
        ],
    }


def agent_fixture(tool, *decisions, output=None):
    llm = LlmClient(Mock(), api_key="fake", model="test/model")
    llm.complete_with_tools = AsyncMock(side_effect=decisions)
    mcp = SimpleNamespace(
        tools=[tool],
        session=Mock(),
        request_timeout_seconds=2,
        call_tool=AsyncMock(return_value=output),
    )
    store = Mock(
        prepare=AsyncMock(
            return_value=PendingLesson(
                "00000000-0000-0000-0000-000000000001",
                101,
                DRAFT,
                NOW + timedelta(minutes=5),
                "prepared",
            )
        )
    )
    return ScheduleAgent(llm, mcp, "Europe/Moscow", personal_store=store), llm, mcp, store


async def test_write_is_prepared_with_trusted_owner_and_not_executed_before_confirmation():
    tool = Tool(
        name="add_lesson",
        inputSchema=LessonDraft.model_json_schema(),
        outputSchema=LessonSaved.model_json_schema(),
        annotations=ToolAnnotations(readOnlyHint=False),
    )
    agent, _, mcp, store = agent_fixture(tool, proposal("add_lesson", DRAFT.model_dump()))
    pending = await agent.answer(
        "Добавь занятие", system_prompt="Помогай", now=NOW, user_id=101, update_id=77
    )
    assert isinstance(pending, PendingLesson)
    store.prepare.assert_awaited_once_with(101, 77, DRAFT, now=NOW)
    mcp.call_tool.assert_not_called()


async def test_model_cannot_supply_owner_or_confirm_action():
    tool = Tool(
        name="add_lesson",
        inputSchema=LessonDraft.model_json_schema(),
        outputSchema=LessonSaved.model_json_schema(),
        annotations=ToolAnnotations(readOnlyHint=False),
    )
    agent, _, mcp, store = agent_fixture(
        tool,
        proposal(
            "add_lesson",
            {
                **DRAFT.model_dump(),
                "user_id": 202,
            },
        ),
    )
    with pytest.raises(LlmError):
        await agent.answer("Добавь", system_prompt="Помогай", now=NOW, user_id=101, update_id=77)
    mcp.call_tool.assert_not_called()
    store.prepare.assert_not_called()


@pytest.mark.parametrize(
    "result",
    [
        WeatherResult(status="not_found"),
        WeatherResult(
            status="ambiguous",
            candidates=[
                {"city": "Спрингфилд", "country": "США", "admin1": "Иллинойс"},
                {"city": "Спрингфилд", "country": "США", "admin1": "Массачусетс"},
            ],
        ),
    ],
)
async def test_weather_result_reaches_model_as_data_for_clarification(result):
    tool = Tool(
        name="get_weather",
        inputSchema={
            "type": "object",
            "properties": {"city": {"type": "string", "minLength": 1, "maxLength": 100}},
            "required": ["city"],
            "additionalProperties": False,
        },
        outputSchema=WeatherResult.model_json_schema(),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    agent, llm, mcp, _ = agent_fixture(
        tool,
        proposal("get_weather", {"city": "  Спрингфилд  "}),
        {"role": "assistant", "content": "Уточни город."},
        output=CallToolResult(content=[], structuredContent=result.model_dump()),
    )
    assert await agent.answer("Погода?", system_prompt="Помогай", now=NOW) == "Уточни город."
    mcp.call_tool.assert_awaited_once_with("get_weather", {"city": "Спрингфилд"}, user_id=None)
    messages = llm.complete_with_tools.call_args.args[0]
    assert messages[-1]["role"] == "tool"
    assert json.loads(messages[-1]["content"])["data"]["status"] == result.status


def callback(user_id=101, action="confirm"):
    message = Message(
        message_id=5, date=NOW, chat=Chat(id=user_id, type="private"), text="Подтвердить?"
    )
    query = CallbackQuery(
        id="callback",
        from_user=User(id=user_id, is_bot=False, first_name="Test"),
        chat_instance="test",
        data=f"lesson:{action}:00000000-0000-0000-0000-000000000001",
        message=message,
    )
    return query


@pytest.mark.parametrize("assumed", [False, True])
@pytest.mark.parametrize(
    "model_answer", ["Температура 3 °C, ветер 4 м/с.", "Уточните страну, пожалуйста."]
)
async def test_assumed_weather_always_shows_place_and_correction_without_requiring_confirmation(
    assumed,
    model_answer,
):
    result = WeatherResult.model_validate(
        {
            "status": "ok",
            "assumed_location": assumed,
            "weather": {
                "city": "Москва",
                "country": "Россия",
                "admin1": "Москва",
                "temperature_c": 3.0,
                "weather_code": 2,
                "wind_speed": 4.0,
                "observed_at": "2026-11-16T12:00:00+03:00",
                "timezone": "Europe/Moscow",
            },
        }
    )
    tool = Tool(
        name="get_weather",
        inputSchema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
        outputSchema=WeatherResult.model_json_schema(),
        annotations=ToolAnnotations(readOnlyHint=True),
    )
    agent, _, _, _ = agent_fixture(
        tool,
        proposal("get_weather", {"city": "Москва"}),
        {"role": "assistant", "content": model_answer},
        output=CallToolResult(content=[], structuredContent=result.model_dump()),
    )
    answer = await agent.answer("Погода в Москве?", system_prompt="Помогай", now=NOW)
    assert "3 °C" in answer and "4 м/с" in answer
    assert "16.11.2026 12:00" in answer and "Open-Meteo" in answer
    assert "Уточните страну, пожалуйста." not in answer
    if assumed:
        assert answer.startswith("Выбрано по населению: Москва, Россия.")
        assert answer.endswith("Если вы имели в виду другое место, уточните страну или регион.")
    else:
        assert "Выбрано по населению" not in answer
        assert "Если вы имели в виду" not in answer


async def test_confirmation_uses_callback_owner_and_never_calls_model(monkeypatch):
    # Methods are patched at the Telegram boundary; no real Telegram requests.
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(Message, "answer", AsyncMock())
    monkeypatch.setattr(Message, "edit_reply_markup", AsyncMock())
    query = callback()
    pending = PendingLesson(
        "00000000-0000-0000-0000-000000000001", 101, DRAFT, NOW + timedelta(minutes=5), "confirmed"
    )
    store = Mock(confirm=AsyncMock(return_value=pending))
    mcp = Mock(
        tools=[Tool(name="add_lesson", inputSchema=LessonDraft.model_json_schema())],
        call_tool=AsyncMock(
            return_value=CallToolResult(
                content=[],
                structuredContent={
                    "lesson_id": 1,
                    **DRAFT.model_dump(),
                    "created": True,
                },
            )
        ),
    )
    await lesson_confirmation(query, store, mcp)
    store.confirm.assert_awaited_once_with(101, pending.action_id)
    mcp.call_tool.assert_awaited_once_with(
        "add_lesson", DRAFT.model_dump(), user_id=101, action_id=pending.action_id
    )
    assert "Занятие добавлено" in Message.answer.call_args.args[0]


async def test_foreign_confirmation_fails_before_mcp(monkeypatch):
    monkeypatch.setattr(CallbackQuery, "answer", AsyncMock())
    monkeypatch.setattr(Message, "answer", AsyncMock())
    store = Mock(confirm=AsyncMock(side_effect=PersonalScheduleError("Действие недоступно.")))
    mcp = Mock(
        tools=[Tool(name="add_lesson", inputSchema=LessonDraft.model_json_schema())],
        call_tool=AsyncMock(),
    )
    await lesson_confirmation(callback(user_id=202), store, mcp)
    assert store.confirm.call_args.args[0] == 202
    mcp.call_tool.assert_not_called()
