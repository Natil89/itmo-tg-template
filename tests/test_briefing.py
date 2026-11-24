import asyncio
from datetime import UTC, date, datetime, timedelta
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.briefing import BriefingJob, BriefingService, BriefingStore, daily_due
from app.briefing_sources import BriefingSources, Section
from tests.test_notifications import NOW, profile


def job(body=None):
    return BriefingJob(
        "briefing:101:2027-01-01",
        date(2027, 1, 1),
        NOW,
        profile(True),
        1,
        uuid4(),
        NOW + timedelta(seconds=60),
        body,
    )


@pytest.mark.parametrize(
    "zone,utc,expected_day,expected_hour",
    [
        ("Europe/Moscow", "2027-01-01T05:00:00+00:00", date(2027, 1, 1), 5),
        ("Asia/Tokyo", "2027-01-01T23:00:00+00:00", date(2027, 1, 2), 23),
        ("America/New_York", "2027-03-14T12:00:00+00:00", date(2027, 3, 14), 12),
    ],
)
def test_local_calendar_date_and_dst(zone, utc, expected_day, expected_hour):
    day, due = daily_due(datetime.fromisoformat(utc), zone)
    assert day == expected_day and due.hour == expected_hour
    assert due.tzinfo is UTC


@pytest.mark.parametrize("delta", [-1, 3600.000001])
async def test_no_event_before_time_or_after_sixty_minutes(delta):
    store = BriefingStore(Mock())
    assert not await store.reserve(profile(True), NOW + timedelta(seconds=delta))


async def test_partial_failure_timeout_and_empty_data_are_different():
    async def unavailable(*_):
        raise ValueError("private document marker")

    async def slow(*_):
        await asyncio.Event().wait()

    sources = Mock(
        schedule=AsyncMock(return_value=Section(text="Занятий нет.", empty=True)),
        reminders=AsyncMock(return_value=Section(text="09:00 — учебный пример")),
        weather=slow,
        focus=AsyncMock(side_effect=unavailable),
    )
    text = await BriefingService(
        Mock(), sources, Mock(), clock=lambda: NOW, section_timeout=0.01
    ).body(job())
    assert "Занятий нет" in text and "09:00 — учебный пример" in text
    assert "превышено время ожидания" in text and "ошибка получения" in text
    assert "private document marker" not in text
    assert "Europe/Moscow" in text and "08:00:00" in text


async def test_sections_fit_one_telegram_message_even_with_emoji():
    section = Section(text="🌦" * 1000)
    sources = Mock(
        **{
            name: AsyncMock(return_value=section)
            for name in ("schedule", "reminders", "weather", "focus")
        }
    )
    text = await BriefingService(Mock(), sources, Mock(), clock=lambda: NOW).body(job())
    assert len(text.encode("utf-16-le")) <= 7600
    assert "сокращён" in text


@pytest.mark.parametrize("boundary", ["deadline", "lease"])
async def test_waiting_for_db_does_not_allow_late_or_expired_lease_send(boundary):
    current = [NOW]
    candidate = job(body="готовая сводка")

    async def validate(*_):
        current[0] += timedelta(seconds=3600.000001 if boundary == "deadline" else 60)
        return True

    store = Mock(
        profiles=Mock(active=AsyncMock(return_value=[])),
        claim=AsyncMock(side_effect=[candidate, None]),
        valid=AsyncMock(side_effect=validate),
        finish=AsyncMock(),
    )
    sender = Mock(send=AsyncMock())
    assert await BriefingService(store, Mock(), sender, clock=lambda: current[0]).run_once() == 0
    sender.send.assert_not_awaited()


async def test_many_reminders_explicitly_mark_truncated_list():
    rows = [{"text": "x", "remind_at": NOW} for _ in range(11)]
    pool = Mock(fetch=AsyncMock(return_value=rows))
    source = BriefingSources(pool, Mock(), Mock(), Mock())
    section = await source.reminders(101, date(2027, 1, 1), "Europe/Moscow")
    assert "первые 10" in section.text and "есть ещё" in section.text


async def test_city_geocoding_filters_country_without_population_auto_confirmation():
    from tests.test_weather import location, response, service

    weather, session = service(
        response(
            {
                "results": [
                    location(),
                    location(country="США", country_code="US", latitude=46.7, longitude=-117.0),
                ]
            }
        ),
        policy="population",
    )
    candidates = await weather.find_cities("Москва, Россия")
    assert len(candidates) == 1 and candidates[0].country == "Россия"
    assert session.get.call_args.kwargs["params"]["name"] == "Москва"


async def test_briefing_weather_uses_confirmed_coordinates_without_geocoding():
    from app.weather import WEATHER_URL
    from tests.test_notifications import CITY
    from tests.test_weather import response, service
    from tests.test_weather import weather as payload

    client, session = service(response(payload()))
    result = await client.weather_at(CITY)
    assert result.temperature_c == -2.5
    session.get.assert_called_once()
    assert session.get.call_args.args[0] == WEATHER_URL
    assert session.get.call_args.kwargs["params"]["latitude"] == CITY.latitude
    assert session.get.call_args.kwargs["params"]["longitude"] == CITY.longitude


async def test_focus_comes_from_actual_local_materials():
    from app.materials import MaterialsService

    materials = MaterialsService()
    source = BriefingSources(Mock(), Mock(), Mock(), materials)
    section = await source.focus(date(2027, 1, 1))
    assert "Источник:" in section.text
    assert any(document.title in section.text for document in materials.catalog().documents)


async def test_daily_reminder_boundaries_follow_dst_local_day():
    pool = Mock(fetch=AsyncMock(return_value=[]))
    source = BriefingSources(pool, Mock(), Mock(), Mock())
    assert (await source.reminders(101, date(2027, 3, 14), "America/New_York")).empty
    _, actor, start, end = pool.fetch.call_args.args
    assert actor == 101
    assert end.astimezone(UTC) - start.astimezone(UTC) == timedelta(hours=23)
