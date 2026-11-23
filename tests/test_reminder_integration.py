"""Изолированная Docker PostgreSQL; рабочая база и Telegram не используются."""

import asyncio
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from test_integration import database as database

from app.db import create_pool
from app.mcp_client import connect_schedule
from app.reminders import ReminderDraft, ReminderError, ReminderStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Нужен RUN_INTEGRATION=1"),
]


async def test_reminders_timezone_confirmation_retries_and_signed_mcp(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    store = ReminderStore(pool)
    now = datetime.now(UTC)
    draft = ReminderDraft(
        text="Учебное тестовое напоминание", remind_at=(now + timedelta(hours=1)).isoformat()
    )
    try:
        await store.initialize()
        assert await store.get_timezone(101) is None
        with pytest.raises(ReminderError):
            await store.prepare(101, 1, draft)
        await store.set_timezone(101, "UTC")
        await store.set_timezone(202, "Asia/Tokyo")
        assert await store.get_timezone(101) == "UTC"
        assert await store.get_timezone(202) == "Asia/Tokyo"
        proposals = await asyncio.gather(store.prepare(101, 1, draft), store.prepare(101, 1, draft))
        pending = proposals[0]
        assert pending.action_id == proposals[1].action_id
        assert await pool.fetchval("SELECT count(*) FROM reminder_actions") == 1
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 0
        with pytest.raises(ReminderError):
            await store.confirm(202, pending.action_id)
        with pytest.raises(ReminderError):
            await store.execute(101, pending.action_id, pending.payload)
        changed = ReminderDraft(text="Другое действие", remind_at=draft.remind_at)
        with pytest.raises(ReminderError):
            await store.prepare(101, 1, changed)

        async with connect_schedule(database_settings=settings) as mcp:
            assert mcp.session is not None
            assert "add_reminder" in {tool.name for tool in mcp.tools}
            args = pending.payload.model_dump()
            assert (await mcp.session.call_tool("add_reminder", args)).isError
            assert (await mcp.call_tool("add_reminder", args, user_id=101)).isError
            assert (
                await mcp.call_tool("add_reminder", args, user_id=101, action_id=pending.action_id)
            ).isError
            await store.confirm(101, pending.action_id)
            assert (
                await mcp.call_tool("add_reminder", args, user_id=202, action_id=pending.action_id)
            ).isError
            assert (
                await mcp.call_tool(
                    "add_reminder", changed.model_dump(), user_id=101, action_id=pending.action_id
                )
            ).isError
            assert (
                await mcp.call_tool(
                    "add_reminder",
                    {**args, "user_id": 202},
                    user_id=101,
                    action_id=pending.action_id,
                )
            ).isError
            results = await asyncio.gather(
                *(
                    mcp.call_tool("add_reminder", args, user_id=101, action_id=pending.action_id)
                    for _ in range(2)
                )
            )
            assert all(not item.isError for item in results)
            saved = [item.structuredContent for item in results]
            assert saved[0]["reminder_id"] == saved[1]["reminder_id"]
            assert sorted(item["created"] for item in saved) == [False, True]
            assert saved[0]["timezone"] == "UTC"
        assert await pool.fetchval("SELECT count(*) FROM reminders WHERE user_id=101") == 1
        assert await pool.fetchval("SELECT count(*) FROM reminders WHERE user_id=202") == 0
        # Сетевой повтор после наступления времени возвращает прежний результат.
        repeated = await store.execute(
            101, pending.action_id, pending.payload, now=now + timedelta(days=2)
        )
        assert repeated.created is False
        assert not await store.cancel(101, pending.action_id)

        cancelled = await store.prepare(101, 2, draft)
        assert await store.cancel(101, cancelled.action_id)
        with pytest.raises(ReminderError):
            await store.confirm(101, cancelled.action_id)

        expired = await store.prepare(101, 3, draft)
        with pytest.raises(ReminderError):
            await store.confirm(101, expired.action_id, now=now + timedelta(minutes=6))

        near = ReminderDraft(
            text="Слишком поздно", remind_at=(now + timedelta(seconds=10)).isoformat()
        )
        stale = await store.prepare(101, 4, near, now=now)
        await store.confirm(101, stale.action_id, now=now)
        with pytest.raises(ReminderError):
            await store.execute(
                101, stale.action_id, stale.payload, now=now + timedelta(seconds=11)
            )

        waiting = await store.prepare(101, 5, draft)
        await pool.execute(
            "UPDATE reminder_actions SET expires_at=clock_timestamp()+interval '0.2 seconds' "
            "WHERE action_id=$1",
            UUID(waiting.action_id),
        )
        async with pool.acquire() as connection, connection.transaction():
            await connection.fetchrow(
                "SELECT action_id FROM reminder_actions WHERE action_id=$1 FOR UPDATE",
                UUID(waiting.action_id),
            )
            task = asyncio.create_task(store.confirm(101, waiting.action_id))
            await asyncio.sleep(0.3)
        with pytest.raises(ReminderError):
            await task
        await store.initialize()
        assert await pool.fetchval("SELECT count(*) FROM reminders") == 1
    finally:
        await pool.close()
    restored_pool = await create_pool(settings)
    try:
        restored = ReminderStore(restored_pool)
        assert await restored.get_timezone(101) == "UTC"
        assert await restored_pool.fetchval("SELECT count(*) FROM reminders") == 1
    finally:
        await restored_pool.close()
