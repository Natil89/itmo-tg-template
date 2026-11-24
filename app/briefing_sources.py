"""Только чтение для сводки: репозитории, фиксированная погода и локальный корпус."""

import asyncio
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import asyncpg
from pydantic import Field

from app.agent_contracts import Contract
from app.materials import MaterialsService, SearchQuery
from app.personal_schedule import PersonalScheduleStore
from app.weather import GeoCity, WeatherService


def bounded(text: str, units: int = 700) -> str:
    encoded = text.encode("utf-16-le")
    if len(encoded) <= units * 2:
        return text
    return encoded[: (units - 25) * 2].decode("utf-16-le", errors="ignore") + "\n…раздел сокращён."


class Section(Contract):
    text: str = Field(min_length=1, max_length=1000)
    empty: bool = False


class BriefingSources:
    def __init__(
        self,
        pool: asyncpg.Pool,
        schedule: PersonalScheduleStore,
        weather: WeatherService,
        materials: MaterialsService,
    ) -> None:
        self.pool, self.schedule_store, self.weather_service, self.materials = (
            pool,
            schedule,
            weather,
            materials,
        )

    async def schedule(self, actor_id: int, day: date) -> Section:
        result = await self.schedule_store.for_date(actor_id, day.isoformat())
        if not result.lessons:
            return Section(text="Занятий нет.", empty=True)
        return Section(
            text=bounded(
                "\n".join(
                    f"{lesson.start}–{lesson.end} {lesson.title}, {lesson.location}"
                    for lesson in result.lessons
                )
            )
        )

    async def reminders(self, actor_id: int, day: date, timezone: str) -> Section:
        zone = ZoneInfo(timezone)
        start = datetime.combine(day, time.min, zone)
        end = datetime.combine(day + timedelta(days=1), time.min, zone)
        rows = await self.pool.fetch(
            "SELECT text,remind_at FROM reminders WHERE user_id=$1 AND remind_at >= $2 "
            "AND remind_at < $3 ORDER BY remind_at,id LIMIT 11",
            actor_id,
            start,
            end,
        )
        if not rows:
            return Section(text="Напоминаний на сегодня нет.", empty=True)
        text = "\n".join(
            f"{row['remind_at'].astimezone(zone):%H:%M} — {row['text']}" for row in rows[:10]
        )
        if len(rows) > 10:
            text += "\nПоказаны первые 10, есть ещё напоминания."
        return Section(text=bounded(text))

    async def weather(self, city: GeoCity) -> Section:
        result = await self.weather_service.weather_at(city)
        return Section(
            text=bounded(
                f"{city.place}: {result.temperature_c:g} °C, ветер {result.wind_speed:g} м/с; "
                f"код условий WMO {result.weather_code}.\n"
                f"Наблюдение {result.observed_at} ({result.timezone}). Источник: {result.source}."
            )
        )

    async def focus(self, day: date) -> Section:
        catalog = await asyncio.to_thread(self.materials.catalog)
        if not catalog.documents:
            return Section(text="Нет доступных учебных материалов.", empty=True)
        documents = sorted(catalog.documents, key=lambda document: document.document_id)
        chosen = documents[day.toordinal() % len(documents)]
        result = await asyncio.to_thread(
            self.materials.search, SearchQuery(query=chosen.title, limit=1)
        )
        if not result.hits:
            return Section(text="Подходящий учебный фокус не найден.", empty=True)
        hit = result.hits[0]
        return Section(text=bounded(f"Повторите тему «{hit.title}».\nИсточник: {hit.source}"))
