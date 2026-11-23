"""Проверка изоляции и конкурентных подтверждений на отдельном PostgreSQL."""

import asyncio
import json
import os
from datetime import UTC, datetime, timedelta
from uuid import UUID

import pytest
from mcp.shared.exceptions import McpError
from test_integration import database as database

from app.db import create_pool
from app.personal_schedule import LessonDraft, PersonalScheduleError, PersonalScheduleStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]

NOW = datetime(2026, 11, 16, 9, tzinfo=UTC)
DRAFT = LessonDraft(
    date="2026-11-17", start="09:00", end="10:30", title="Алгебра", location="Аудитория A"
)


async def test_personal_schedule_isolation_confirmation_repeats_and_real_concurrency(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        store = PersonalScheduleStore(pool)
        await store.initialize()
        assert (await store.for_date(101, DRAFT.date)).lessons == []
        assert (await store.for_date(202, DRAFT.date)).lessons == []

        # Двойная доставка Telegram update создаёт ровно одно предложение.
        first, repeated = await asyncio.gather(
            store.prepare(101, 11, DRAFT, now=NOW),
            store.prepare(101, 11, DRAFT, now=NOW),
        )
        assert first.action_id == repeated.action_id
        assert first.payload == DRAFT
        assert first.expires_at == NOW + timedelta(minutes=5)
        assert first.status == "prepared"
        assert await pool.fetchval("SELECT count(*) FROM personal_schedule_actions") == 1
        with pytest.raises(PersonalScheduleError):
            await store.prepare(101, 11, DRAFT.model_copy(update={"title": "Другое"}), now=NOW)

        # Чужой владелец не может прочитать, подтвердить, отменить или выполнить действие.
        assert await store.get_pending(202, first.action_id, now=NOW) is None
        with pytest.raises(PersonalScheduleError):
            await store.confirm(202, first.action_id, now=NOW)
        assert not await store.cancel(202, first.action_id, now=NOW)
        with pytest.raises(PersonalScheduleError):
            await store.execute(202, first.action_id, DRAFT, now=NOW)
        with pytest.raises(PersonalScheduleError):
            await store.execute(101, first.action_id, DRAFT, now=NOW)
        assert await pool.fetchval("SELECT count(*) FROM personal_schedule_lessons") == 0

        # Два конкурентных подтверждения/выполнения сохраняют одно занятие атомарно.
        async def confirm_and_execute():
            await store.confirm(101, first.action_id, now=NOW)
            return await store.execute(101, first.action_id, DRAFT, now=NOW)

        saved = await asyncio.gather(confirm_and_execute(), confirm_and_execute())
        assert saved[0].lesson_id == saved[1].lesson_id
        assert sorted(item.created for item in saved) == [False, True]
        assert await pool.fetchval("SELECT count(*) FROM personal_schedule_lessons") == 1
        assert (await store.get_pending(101, first.action_id, now=NOW)).status == "executed"
        assert (await store.confirm(101, first.action_id, now=NOW)).status == "executed"
        assert not await store.cancel(101, first.action_id, now=NOW)
        retry = await store.execute(101, first.action_id, DRAFT, now=NOW + timedelta(days=1))
        assert retry.lesson_id == saved[0].lesson_id and not retry.created

        expected = [
            {"start": "09:00", "end": "10:30", "title": "Алгебра", "location": "Аудитория A"}
        ]
        assert [
            lesson.model_dump() for lesson in (await store.for_date(101, DRAFT.date)).lessons
        ] == expected
        assert (await store.for_date(202, DRAFT.date)).lessons == []
        week = await store.current_week(101, "Europe/Moscow", now=NOW)
        assert (week.start, week.end, len(week.days)) == ("2026-11-16", "2026-11-22", 7)
        assert week.days[1].lessons[0].title == "Алгебра"
        boundary = await store.current_week(
            101, "Europe/Moscow", now=datetime(2027, 1, 1, tzinfo=UTC)
        )
        assert (boundary.start, boundary.end) == ("2026-12-28", "2027-01-03")

        # Изменённый payload требует отдельного согласия; TTL включает confirmed-состояние.
        changed = await store.prepare(101, 12, DRAFT, now=NOW)
        await store.confirm(101, changed.action_id, now=NOW)
        with pytest.raises(PersonalScheduleError):
            await store.execute(
                101, changed.action_id, DRAFT.model_copy(update={"start": "08:00"}), now=NOW
            )
        with pytest.raises(PersonalScheduleError):
            await store.execute(101, changed.action_id, DRAFT, now=NOW + timedelta(minutes=5))
        assert (
            await store.get_pending(101, changed.action_id, now=NOW + timedelta(minutes=5))
        ).status == "expired"
        with pytest.raises(PersonalScheduleError):
            await store.confirm(101, changed.action_id, now=NOW + timedelta(minutes=5))

        cancelled = await store.prepare(101, 13, DRAFT, now=NOW)
        assert await store.cancel(101, cancelled.action_id, now=NOW)
        with pytest.raises(PersonalScheduleError):
            await store.confirm(101, cancelled.action_id, now=NOW)
        with pytest.raises(PersonalScheduleError):
            await store.execute(101, cancelled.action_id, DRAFT, now=NOW)
        assert await pool.fetchval("SELECT count(*) FROM personal_schedule_lessons") == 1

        # Повторная инициализация не сбрасывает персональные данные.
        await store.initialize()
        assert (await store.for_date(101, DRAFT.date)).lessons[0].title == "Алгебра"

        # Время ожидания row lock не продлевает действительность подтверждения.
        waiting = await store.prepare(303, 14, DRAFT)
        await pool.execute(
            "UPDATE personal_schedule_actions SET "
            "expires_at=clock_timestamp()+interval '0.2 seconds' "
            "WHERE action_id=$1",
            UUID(waiting.action_id),
        )
        async with pool.acquire() as blocker, blocker.transaction():
            await blocker.fetchrow(
                "SELECT action_id FROM personal_schedule_actions WHERE action_id=$1 FOR UPDATE",
                UUID(waiting.action_id),
            )
            confirm_task = asyncio.create_task(store.confirm(303, waiting.action_id))
            await asyncio.sleep(0.3)
        with pytest.raises(PersonalScheduleError):
            await confirm_task
    finally:
        await pool.close()


async def test_signed_mcp_isolates_users_resources_and_confirmed_writes(database):
    from app.mcp_client import connect_schedule
    from app.mcp_identity import sign_context

    settings, _, _, _ = database
    pool = await create_pool(settings)
    store = PersonalScheduleStore(pool)
    try:
        await store.initialize()
        async with connect_schedule(database_settings=settings) as mcp:
            assert mcp.session is not None
            assert {tool.name for tool in mcp.tools} == {
                "get_schedule",
                "get_weather",
                "add_lesson",
                "add_reminder",
            }
            args = {"date": DRAFT.date}
            assert (await mcp.session.call_tool("get_schedule", args)).isError
            assert (await mcp.call_tool("get_schedule", args, user_id=101)).structuredContent == {
                "date": DRAFT.date,
                "lessons": [],
            }
            prepared = await store.prepare(101, 15, DRAFT)
            # Даже подписанный владелец не может записать без действующего согласия.
            assert (
                await mcp.call_tool(
                    "add_lesson", DRAFT.model_dump(), user_id=101, action_id=prepared.action_id
                )
            ).isError
            await store.confirm(101, prepared.action_id)
            assert (
                await mcp.call_tool(
                    "add_lesson", DRAFT.model_dump(), user_id=202, action_id=prepared.action_id
                )
            ).isError
            first, repeated = await asyncio.gather(
                mcp.call_tool(
                    "add_lesson", DRAFT.model_dump(), user_id=101, action_id=prepared.action_id
                ),
                mcp.call_tool(
                    "add_lesson", DRAFT.model_dump(), user_id=101, action_id=prepared.action_id
                ),
            )
            assert not first.isError and not repeated.isError
            assert first.structuredContent["lesson_id"] == repeated.structuredContent["lesson_id"]
            assert sorted(
                [first.structuredContent["created"], repeated.structuredContent["created"]]
            ) == [False, True]
            assert await pool.fetchval("SELECT count(*) FROM personal_schedule_lessons") == 1
            own = await mcp.call_tool("get_schedule", args, user_id=101)
            other = await mcp.call_tool("get_schedule", args, user_id=202)
            assert own.structuredContent["lessons"][0]["title"] == DRAFT.title
            assert other.structuredContent["lessons"] == []
            assert (
                await mcp.call_tool("get_schedule", {**args, "user_id": 101}, user_id=202)
            ).isError
            forged = sign_context("wrong" * 16, 101, "tool:get_schedule", args)
            assert (await mcp.session.call_tool("get_schedule", args, meta=forged)).isError
            wrong_scope = sign_context(mcp._context_secret, 101, "tool:add_lesson", args)
            assert (await mcp.session.call_tool("get_schedule", args, meta=wrong_scope)).isError
            wrong_args = sign_context(
                mcp._context_secret, 101, "tool:get_schedule", {"date": "2027-01-01"}
            )
            assert (await mcp.session.call_tool("get_schedule", args, meta=wrong_args)).isError
            week = await mcp.read_current_week(user_id=202)
            assert all(not day["lessons"] for day in json.loads(week.contents[0].text)["days"])
            with pytest.raises(McpError):
                await mcp.session.read_resource("schedule://current-week")
    finally:
        await pool.close()

    pool = await create_pool(settings)
    try:
        restored = PersonalScheduleStore(pool)
        await restored.initialize()
        assert len((await restored.for_date(101, DRAFT.date)).lessons) == 1
        assert (await restored.for_date(202, DRAFT.date)).lessons == []
    finally:
        await pool.close()
