"""Связь агентного цикла с настоящим локальным MCP; модель подменена."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

from app.llm import LlmClient
from app.mcp_client import connect_schedule
from app.schedule_agent import ScheduleAgent


async def test_free_question_roundtrip_uses_discovered_schema_and_real_schedule():
    # Arrange: реальный stdio-процесс, без Telegram и внешней модели.
    llm = LlmClient(Mock(), api_key="fake-key", model="test/model")

    async def model(messages, tools, *, temperature):
        assert tools[0]["function"]["name"] == "get_schedule"
        assert tools[0]["function"]["parameters"]["additionalProperties"] is False
        if messages[-1]["role"] == "tool":
            data = json.loads(messages[-1]["content"])["data"]
            assert data["date"] == "2026-11-16"
            lesson = data["lessons"][0]
            return {
                "role": "assistant",
                "content": f"Завтра в {lesson['start']} — {lesson['title']}.",
            }
        assert messages[-1] == {"role": "user", "content": "Какие занятия завтра?"}
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "schedule-call",
                    "type": "function",
                    "function": {"name": "get_schedule", "arguments": '{"date":"2026-11-16"}'},
                }
            ],
        }

    llm.complete_with_tools = AsyncMock(side_effect=model)
    async with connect_schedule() as mcp:
        assert mcp.session is not None
        agent = ScheduleAgent(llm, mcp, "Europe/Moscow")
        # Act
        answer = await agent.answer(
            "Какие занятия завтра?",
            system_prompt="Помогай учиться",
            now=datetime(2026, 11, 15, 12, tzinfo=UTC),
        )
    # Assert: финальный текст построен из фактического результата сервера.
    assert answer == "Завтра в 09:00 — Программирование."
    assert llm.complete_with_tools.await_count == 2
