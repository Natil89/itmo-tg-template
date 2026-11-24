"""Одна временная PostgreSQL; fake clock/Telegram. Рабочая БД и рассылка не используются."""

import asyncio
import os
from dataclasses import replace
from datetime import timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from test_integration import database as database

from app.briefing import BriefingService, BriefingStore
from app.briefing_sources import Section
from app.db import create_pool
from app.delivery import PermanentDeliveryError, TemporaryDeliveryError
from app.notifications import (
    CitySelection,
    NotificationError,
    NotificationProfile,
    NotificationStore,
)
from app.reminders import ReminderStore
from tests.test_delivery_integration import FakeTelegram
from tests.test_notifications import CITY, NOW

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Нужен RUN_INTEGRATION=1"),
]


async def test_city_consent_daily_uniqueness_partial_failures_and_delivery_recovery(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    clock = [NOW]
    telegram = FakeTelegram()
    sources = Mock(
        **{
            name: AsyncMock(return_value=Section(text=f"Раздел {name}"))
            for name in ("schedule", "reminders", "weather", "focus")
        }
    )
    try:
        reminders = ReminderStore(pool)
        await reminders.initialize()
        profiles = NotificationStore(pool)
        await profiles.initialize()
        store = BriefingStore(profiles)
        await store.initialize()
        await store.initialize()

        async def enabled(actor, chat=None):
            await reminders.set_timezone(actor, "Europe/Moscow")
            await profiles.confirm_city(actor, chat or actor + 9000, uuid4(), CITY, clock[0])
            return await profiles.set_enabled(actor, True)

        async def row(actor):
            return await pool.fetchrow(
                "SELECT * FROM briefing_deliveries WHERE actor_id=$1 "
                "ORDER BY local_date DESC LIMIT 1",
                actor,
            )

        selection = CitySelection(
            Mock(find_cities=AsyncMock(return_value=[CITY])), profiles, clock=lambda: clock[0]
        )
        proposal = (await selection.propose(101, 9101, "Москва, Россия"))[0]
        assert await pool.fetchval("SELECT count(*) FROM notification_profiles") == 0
        with pytest.raises(NotificationError):
            await selection.confirm(str(proposal.token), 202, 9202)
        assert await pool.fetchval("SELECT count(*) FROM notification_profiles") == 0
        initial = await selection.confirm(str(proposal.token), 101, 9101)
        assert initial.city == CITY and not initial.enabled
        with pytest.raises(NotificationError, match="часовой пояс"):
            await profiles.set_enabled(101, True)
        await reminders.set_timezone(101, "Europe/Moscow")
        first = await profiles.set_enabled(101, True)
        same = await profiles.set_enabled(101, True)
        assert first.revision == same.revision  # Повтор /briefing_on не отзывает текущее задание.
        duplicate_confirmation = await profiles.confirm_city(
            101, 9101, proposal.token, CITY, clock[0]
        )
        assert duplicate_confirmation.enabled and duplicate_confirmation.revision == first.revision

        assert await store.reserve(first, clock[0])
        assert not await store.reserve(first, clock[0])
        service = BriefingService(store, sources, telegram, clock=lambda: clock[0])
        assert sum(await asyncio.gather(service.run_once(), service.run_once())) == 1
        assert await service.run_once() == 0
        assert (await row(101))["status"] == "sent" and (await row(101))["attempts"] == 1
        assert telegram.calls[0][0] == 9101
        assert "Europe/Moscow" in telegram.calls[0][1]

        second = await enabled(202)
        assert (
            sum(
                await asyncio.gather(
                    store.reserve(second, clock[0]), store.reserve(second, clock[0])
                )
            )
            == 1
        )
        assert (
            await pool.fetchval("SELECT count(*) FROM briefing_deliveries WHERE actor_id=202") == 1
        )
        assert await service.run_once() == 1

        boundary = await enabled(303)
        clock[0] = NOW + timedelta(hours=1)
        assert await store.reserve(boundary, clock[0])
        assert await service.run_once() == 1
        assert (await row(303))["status"] == "sent"
        late = await enabled(404)
        clock[0] += timedelta(microseconds=1)
        assert not await store.reserve(late, clock[0])
        assert await service.run_once() == 0
        assert await row(404) is None
        for actor in (101, 202, 303, 404):
            await profiles.set_enabled(actor, False)

        clock[0] = NOW + timedelta(days=1) - timedelta(seconds=1)
        future = await enabled(505)
        assert not await store.reserve(future, clock[0])
        assert await service.run_once() == 0
        clock[0] += timedelta(seconds=1)
        assert await service.run_once() == 1
        assert (await row(505))["local_date"].isoformat() == "2027-01-02"

        with pytest.raises(NotificationError, match="город"):
            await profiles.set_enabled(606, True)
        forged_profile = NotificationProfile(606, 9606, CITY, "Europe/Moscow", True, 1)
        assert not await store.reserve(forged_profile, clock[0])
        await profiles.confirm_city(707, 9707, uuid4(), CITY, clock[0])
        with pytest.raises(NotificationError, match="часовой пояс"):
            await profiles.set_enabled(707, True)

        opted_out = await enabled(808)
        await store.reserve(opted_out, clock[0])
        await profiles.set_enabled(808, False)
        before = len(telegram.calls)
        assert await service.run_once() == 0 and len(telegram.calls) == before

        await enabled(909)

        async def disable_during_prepare(*_):
            await profiles.set_enabled(909, False)
            return Section(text="Погода получена")

        sources.weather.side_effect = disable_during_prepare
        assert await service.run_once() == 0
        assert len(telegram.calls) == before
        assert (await row(909))["error_code"] == "profile_changed"
        sources.weather.side_effect = None

        changed = await enabled(1001)
        await store.reserve(changed, clock[0])
        claimed = await store.claim(clock[0])
        assert claimed.profile.actor_id == 1001
        forged = replace(claimed, profile=replace(claimed.profile, destination_chat_id=999999))
        assert not await store.valid(forged, clock[0])
        original_claim = store.claim
        store.claim = AsyncMock(side_effect=[forged, None])
        assert await service.run_once() == 0 and len(telegram.calls) == before
        store.claim = original_claim

        await enabled(1101)
        telegram.error = TemporaryDeliveryError()
        assert await service.run_once() == 0
        assert (await row(1101))["status"] == "retry"
        assert await service.run_once() == 0
        clock[0] += timedelta(seconds=30)
        await service.run_once()
        clock[0] += timedelta(seconds=120)
        await service.run_once()
        assert (await row(1101))["status"] == "failed" and (await row(1101))["attempts"] == 3
        calls = len(telegram.calls)
        await service.run_once()
        assert len(telegram.calls) == calls

        await enabled(1201)
        telegram.error = PermanentDeliveryError()
        await service.run_once()
        assert (await row(1201))["status"] == "failed" and (await row(1201))["attempts"] == 1
        telegram.error = None

        restarted = await enabled(1301)
        await store.reserve(restarted, clock[0])
        pending = await store.claim(clock[0])
        await store.save_body(pending, await service.body(pending))
        source_calls = sources.schedule.await_count
        await pool.close()
        pool = await create_pool(settings)
        profiles = NotificationStore(pool)
        await profiles.initialize()
        store = BriefingStore(profiles)
        await store.initialize()
        clock[0] += timedelta(seconds=60)
        service = BriefingService(store, sources, telegram, clock=lambda: clock[0])
        assert await service.run_once() == 1
        assert (await row(1301))["status"] == "sent" and (await row(1301))["attempts"] == 2
        assert (
            sources.schedule.await_count == source_calls
        )  # Текст пережил рестарт; источники не повторены.
    finally:
        await pool.close()


async def test_read_sources_use_real_owner_scoped_repositories_and_local_corpus(database):
    from zoneinfo import ZoneInfo

    from app.briefing_sources import BriefingSources
    from app.materials import MaterialsService
    from app.personal_schedule import LessonDraft, PersonalScheduleStore
    from app.reminders import ReminderDraft
    from app.weather import CurrentWeather

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        schedule, reminders = PersonalScheduleStore(pool), ReminderStore(pool)
        await schedule.initialize()
        await reminders.initialize()
        for actor, title in ((111, "Учебное занятие"), (222, "Чужое занятие")):
            draft = LessonDraft(
                date="2027-01-01",
                start="09:00",
                end="10:00",
                title=title,
                location="Тестовая аудитория",
            )
            proposal = await schedule.prepare(actor, 1, draft, now=NOW)
            await schedule.confirm(actor, proposal.action_id, now=NOW)
            await schedule.execute(actor, proposal.action_id, draft, now=NOW)
            await reminders.set_timezone(actor, "Europe/Moscow")
            reminder = ReminderDraft(
                text=title + " напоминание",
                remind_at=(NOW + timedelta(hours=1))
                .astimezone(ZoneInfo("Europe/Moscow"))
                .isoformat(),
            )
            pending = await reminders.prepare(actor, 1, reminder, now=NOW)
            await reminders.confirm(actor, pending.action_id, now=NOW)
            await reminders.execute(actor, pending.action_id, reminder, now=NOW)
        weather = Mock(
            weather_at=AsyncMock(
                return_value=CurrentWeather(
                    city=CITY.city,
                    country=CITY.country,
                    admin1=CITY.admin1,
                    temperature_c=2.0,
                    wind_speed=1.0,
                    weather_code=0,
                    observed_at=NOW.astimezone(ZoneInfo("Europe/Moscow")).isoformat(),
                    timezone="Europe/Moscow",
                )
            )
        )
        sources = BriefingSources(pool, schedule, weather, MaterialsService())
        day = NOW.astimezone(ZoneInfo("Europe/Moscow")).date()
        lessons = await sources.schedule(111, day)
        notices = await sources.reminders(111, day, "Europe/Moscow")
        assert "Учебное занятие" in lessons.text and "Чужое" not in lessons.text
        assert "Учебное занятие напоминание" in notices.text and "Чужое" not in notices.text
        assert "09:00" in notices.text
        assert "2 °C" in (await sources.weather(CITY)).text
        assert "Источник:" in (await sources.focus(day)).text
        weather.weather_at.assert_awaited_once_with(CITY)
    finally:
        await pool.close()
