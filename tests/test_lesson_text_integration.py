import asyncio
import os
from datetime import UTC, datetime, timedelta

import pytest
from test_integration import database as database

from app.db import create_pool
from app.personal_schedule import PersonalScheduleError, PersonalScheduleStore
from tests.test_personal_schedule_flow import DRAFT

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Нужен RUN_INTEGRATION=1"),
]


async def test_display_binding_ambiguity_expiration_and_concurrent_confirmation(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    now = datetime.now(UTC)
    try:
        store = PersonalScheduleStore(pool)
        await store.initialize()
        first = await store.prepare(101, 1, DRAFT, now=now)
        assert await store.text_confirmation_candidate(101, 101, 11, now=now) is None
        await store.bind_prompt(101, first.action_id, 101, 10)
        assert (
            await store.text_confirmation_candidate(101, 101, 11, now=now)
        ).action_id == first.action_id
        assert await store.text_confirmation_candidate(202, 202, 11, now=now) is None
        assert await store.text_confirmation_candidate(101, 101, 10, now=now) is None
        assert await store.text_confirmation_candidate(101, 101, 11, reply_to=9, now=now) is None
        with pytest.raises(PersonalScheduleError):
            await store.text_confirmation_candidate(101, 202, 11, now=now)
        second = await store.prepare(101, 2, DRAFT, now=now)
        await store.bind_prompt(101, second.action_id, 101, 12)
        with pytest.raises(PersonalScheduleError, match="несколько"):
            await store.text_confirmation_candidate(101, 101, 13, now=now)
        assert (
            await store.text_confirmation_candidate(101, 101, 13, reply_to=10, now=now)
        ).action_id == first.action_id
        assert (
            await store.text_confirmation_candidate(101, 101, 13, now=now + timedelta(minutes=5))
            is None
        )

        async def confirm():
            await store.confirm(101, first.action_id, now=now)
            return await store.execute(101, first.action_id, DRAFT, now=now)

        results = await asyncio.gather(confirm(), confirm())
        assert results[0].lesson_id == results[1].lesson_id
        assert sum(result.created for result in results) == 1
        assert await pool.fetchval("SELECT count(*) FROM personal_schedule_lessons") == 1
        # Повторный старый ответ не подтверждает другую карточку.
        assert await store.text_confirmation_candidate(101, 101, 13, reply_to=10, now=now) is None
    finally:
        await pool.close()
