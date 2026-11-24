"""Реальный отдельный PostgreSQL, fake clock и фиктивный Telegram. Отправок в сеть нет."""

import asyncio
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from test_integration import database as database

from app.db import create_pool
from app.delivery import (
    DeliveryStore,
    PermanentDeliveryError,
    ReminderDeliveryService,
    TemporaryDeliveryError,
)
from app.reminders import ReminderDraft, ReminderStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Нужен RUN_INTEGRATION=1"),
]


class Clock:
    def __init__(self):
        self.now = datetime(2027, 1, 1, 12, tzinfo=UTC)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += timedelta(seconds=seconds)


class FakeTelegram:
    def __init__(self):
        self.calls = []
        self.error = None

    async def send(self, chat_id, text):
        self.calls.append((chat_id, text))
        if self.error:
            raise self.error
        return len(self.calls)


async def test_persistent_delivery_concurrency_boundaries_failures_and_restart(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    clock, telegram = Clock(), FakeTelegram()
    try:
        reminders = ReminderStore(pool)
        await reminders.initialize()
        store = DeliveryStore(pool)
        await store.initialize()
        await store.initialize()  # Повтор миграции сохраняет данные.
        await store.bind_destination(101, 9001)  # actor_id и destination_chat_id — разные поля.
        counter = 0

        async def seed(due=None, owner=101):
            nonlocal counter
            counter += 1
            due = due or clock.now
            await reminders.set_timezone(owner, "UTC")
            draft = ReminderDraft(text=f"Учебное напоминание {counter}", remind_at=due.isoformat())
            before = due - timedelta(hours=1)
            proposal = await reminders.prepare(owner, counter, draft, now=before)
            await reminders.confirm(owner, proposal.action_id, now=before)
            saved = await reminders.execute(owner, proposal.action_id, draft, now=before)
            return saved.reminder_id

        async def row(identity):
            return await pool.fetchrow(
                "SELECT * FROM reminder_deliveries WHERE reminder_id=$1", identity
            )

        first = await seed()
        assert await store.prepare(clock()) == 1
        assert await store.prepare(clock()) == 0
        assert await pool.fetchval("SELECT count(*) FROM reminder_deliveries") == 1
        fresh = await seed(clock() + timedelta(hours=2))
        assert sum(await asyncio.gather(store.prepare(clock()), store.prepare(clock()))) == 1
        assert (
            await pool.fetchval(
                "SELECT count(*) FROM reminder_deliveries WHERE reminder_id=$1", fresh
            )
            == 1
        )
        service = ReminderDeliveryService(store, telegram, clock=clock)
        assert sum(await asyncio.gather(service.run_once(), service.run_once())) == 1
        assert await service.run_once() == 0
        assert len(telegram.calls) == 1 and telegram.calls[0][0] == 9001
        assert (await row(first))["status"] == "sent"
        assert (await row(first))["attempts"] == 1

        boundary = await seed(clock() - timedelta(minutes=15))
        late = await seed(clock() - timedelta(minutes=15, microseconds=1))
        future = await seed(clock() + timedelta(seconds=1))
        assert await service.run_once() == 1
        assert (await row(boundary))["status"] == "sent"
        assert telegram.calls[-1][1].startswith("С задержкой.")
        assert (await row(late))["status"] == "missed" and (await row(late))["attempts"] == 0
        assert (await row(future))["status"] == "pending" and (await row(future))["attempts"] == 0
        clock.advance(1)
        assert await service.run_once() == 1

        temporary = await seed()
        telegram.error = TemporaryDeliveryError()
        before_calls = len(telegram.calls)
        assert await service.run_once() == 0
        assert (await row(temporary))["status"] == "retry"
        assert await service.run_once() == 0
        clock.advance(30)
        await service.run_once()
        clock.advance(120)
        await service.run_once()
        assert (await row(temporary))["status"] == "failed"
        assert (await row(temporary))["attempts"] == 3
        assert len(telegram.calls) - before_calls == 3
        clock.advance(10)
        await service.run_once()
        assert len(telegram.calls) - before_calls == 3

        permanent = await seed()
        telegram.error = PermanentDeliveryError()
        await service.run_once()
        assert (await row(permanent))["status"] == "failed"
        assert (await row(permanent))["attempts"] == 1
        telegram.error = None

        restart = await seed()
        await store.prepare(clock())
        interrupted = await store.claim(clock())
        assert interrupted.reminder_id == restart and interrupted.attempts == 1
        assert await store.claim(clock()) is None
        await pool.close()
        pool = await create_pool(settings)
        reminders = ReminderStore(pool)
        store = DeliveryStore(pool)
        await store.initialize()
        clock.advance(60)
        service = ReminderDeliveryService(store, telegram, clock=clock)
        assert await service.run_once() == 1
        assert (await row(restart))["status"] == "sent" and (await row(restart))["attempts"] == 2

        tampered = await seed()
        await store.prepare(clock())
        claimed = await store.claim(clock())
        assert claimed.reminder_id == tampered
        forged = replace(claimed, destination_chat_id=9999)
        assert not await store.authorized(forged, clock())
        original_claim = store.claim
        store.claim = AsyncMock(side_effect=[forged, None])
        before_calls = len(telegram.calls)
        assert await service.run_once() == 0
        assert len(telegram.calls) == before_calls
        store.claim = original_claim

        missing = await seed(owner=202)
        assert await service.run_once() == 0
        assert await row(missing) is None

        changed = await seed()
        await store.prepare(clock())
        await store.bind_destination(101, 9002)
        await service.run_once()
        assert (await row(changed))["status"] == "failed"
        assert (await row(changed))["error_code"] == "destination_changed"
        assert len(telegram.calls) == before_calls
    finally:
        await pool.close()
