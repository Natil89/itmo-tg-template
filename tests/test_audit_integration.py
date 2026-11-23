"""Аудит проверяется в отдельном PostgreSQL, рабочие диалоги не читаются."""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, Mock

import pytest
from test_integration import database as database
from test_schedule_agent import call, decision

from app.audit import AuditStore, AuditTrace, render_audit
from app.db import create_pool
from app.llm import LlmClient
from app.mcp_client import connect_schedule
from app.schedule_agent import ScheduleAgent
from app.user_state import UserStateStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Нужен RUN_INTEGRATION=1"),
]


async def test_audit_persists_isolates_owners_and_tracks_real_mcp(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    audit = AuditStore(pool, secrets=("SecretCity",))
    started = datetime.now(UTC)
    try:
        await audit.initialize()
        assert await audit.latest(101) is None
        trace = AuditTrace(started_at=started)
        trace.add(
            "tool",
            tool="get_weather",
            arguments={"city": "SecretCity", "user_id": 202},
            validation="passed",
            execution="success",
            reason="live_weather",
        )
        await asyncio.gather(*(audit.save(101, "message:1", trace) for _ in range(2)))
        assert await pool.fetchval("SELECT count(*) FROM assistant_audit_events") == 1
        raw = await pool.fetchval(
            "SELECT steps::text FROM assistant_audit_events WHERE user_id=101"
        )
        assert "SecretCity" not in raw and "user_id" not in raw
        assert await audit.latest(202) is None
        other = AuditTrace(started_at=started + timedelta(seconds=1))
        other.add("respond", execution="success", reason="no_tool_needed")
        await audit.save(202, "message:1", other)
        assert (await audit.latest(101)).steps[0].tool == "get_weather"
        assert (await audit.latest(202)).steps[0].action == "respond"

        # Повторный event key не переписывает исходное решение.
        await audit.save(101, "message:1", other)
        assert (await audit.latest(101)).steps[0].tool == "get_weather"

        # Реальное stdio чтение личного расписания; LLM подменена, Telegram нет.
        async with connect_schedule(database_settings=settings) as mcp:
            llm = LlmClient(Mock(), api_key="unused-test", model="test/model")
            llm.complete_with_tools = AsyncMock(
                side_effect=[
                    decision(call("2026-11-17")),
                    {"role": "assistant", "content": "Занятий нет."},
                ]
            )
            agent = ScheduleAgent(llm, mcp, "UTC")
            read = AuditTrace(started_at=started + timedelta(seconds=2))
            assert (
                await agent.answer(
                    "Что у меня 17 ноября?",
                    system_prompt="private-system-prompt",
                    user_id=101,
                    update_id=2,
                    audit=read,
                )
                == "Занятий нет."
            )
            await audit.save(101, "message:2", read)
        latest = await audit.latest(101)
        assert latest.steps[0].tool == "get_schedule"
        assert latest.steps[0].execution == "success"
        assert latest.steps[0].arguments == {"date": "2026-11-17"}
        rendered = render_audit(latest, timezone="Europe/Moscow")
        assert "get_schedule" in rendered and "2026-11-17" in rendered
        assert "private-system-prompt" not in rendered and "message:2" not in rendered

        state = UserStateStore(pool)
        await state.initialize()
        await state.save_history(101, [])
        await audit.initialize()
        assert (await audit.latest(101)).steps[0].tool == "get_schedule"

        # Порядок — по началу запроса, а не по времени окончания INSERT.
        earlier = AuditTrace(started_at=started - timedelta(seconds=5))
        earlier.add("clarify", execution="success", reason="missing_details")
        await audit.save(101, "message:old", earlier)
        assert (await audit.latest(101)).steps[0].tool == "get_schedule"
    finally:
        await pool.close()
    reopened = await create_pool(settings)
    try:
        restored = AuditStore(reopened)
        assert (await restored.latest(101)).steps[0].tool == "get_schedule"
        assert (await restored.latest(202)).steps[0].action == "respond"
        assert await restored.latest(303) is None
        rows = await reopened.fetch("SELECT steps::text AS steps FROM assistant_audit_events")
        assert all("SecretCity" not in json.dumps(json.loads(row["steps"])) for row in rows)
    finally:
        await reopened.close()
