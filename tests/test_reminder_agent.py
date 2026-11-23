import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from mcp.types import Tool, ToolAnnotations

from app.llm import LlmClient, LlmError
from app.reminders import PendingReminder, ReminderDraft, ReminderError, ReminderSaved
from app.schedule_agent import ScheduleAgent

NOW = datetime(2026, 11, 16, 22, 30, tzinfo=UTC)
DRAFT = ReminderDraft(text="Сделать домашку", remind_at="2026-11-18T18:00:00+03:00")


def fixture(arguments=None, zone="Europe/Moscow"):
    llm = LlmClient(Mock(), api_key="fake", model="test/model")
    llm.complete_with_tools = AsyncMock(
        return_value={
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "reminder-1",
                    "type": "function",
                    "function": {
                        "name": "add_reminder",
                        "arguments": json.dumps(
                            DRAFT.model_dump() if arguments is None else arguments
                        ),
                    },
                }
            ],
        }
    )
    tool = Tool(
        name="add_reminder",
        inputSchema=ReminderDraft.model_json_schema(),
        outputSchema=ReminderSaved.model_json_schema(),
        annotations=ToolAnnotations(readOnlyHint=False),
    )
    mcp = SimpleNamespace(tools=[tool], session=Mock(), call_tool=AsyncMock())
    pending = PendingReminder(
        "test-action", 101, DRAFT, zone, NOW + timedelta(minutes=5), "prepared"
    )
    store = Mock(get_timezone=AsyncMock(return_value=zone), prepare=AsyncMock(return_value=pending))
    return ScheduleAgent(llm, mcp, "UTC", reminders=store), llm, mcp, store


async def test_proposal_uses_trusted_owner_and_personal_timezone_without_mcp_write():
    agent, llm, mcp, store = fixture()
    result = await agent.answer(
        "Напомни завтра в 18:00 сделать домашку",
        system_prompt="Помогай",
        user_id=101,
        update_id=55,
        now=NOW,
    )
    assert isinstance(result, PendingReminder)
    store.get_timezone.assert_awaited_once_with(101)
    store.prepare.assert_awaited_once_with(101, 55, DRAFT, now=NOW)
    mcp.call_tool.assert_not_awaited()
    # В Москве уже 17-е: модель получает личное время, а не только дату UTC.
    assert "2026-11-17T01:30:00+03:00" in llm.complete_with_tools.call_args.args[0][0]["content"]


@pytest.mark.parametrize("extra", ["user_id", "action_id", "confirmed", "timezone"])
async def test_model_cannot_supply_trusted_metadata(extra):
    agent, _, mcp, store = fixture({**DRAFT.model_dump(), extra: "forged"})
    with pytest.raises(LlmError):
        await agent.answer("Напомни", system_prompt="Помогай", user_id=101, update_id=55, now=NOW)
    store.prepare.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_missing_timezone_cannot_be_bypassed_by_model_call():
    agent, _, mcp, store = fixture(zone=None)
    with pytest.raises(ReminderError, match="timezone"):
        await agent.answer("Напомни", system_prompt="Помогай", user_id=101, update_id=55, now=NOW)
    store.prepare.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


async def test_untrusted_request_does_not_expose_write_tool():
    agent, _, _, store = fixture()
    with pytest.raises(LlmError):
        await agent.answer("Напомни", system_prompt="Помогай", now=NOW)
    store.get_timezone.assert_not_awaited()
    store.prepare.assert_not_awaited()
