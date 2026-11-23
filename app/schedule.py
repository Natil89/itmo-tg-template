"""Чтение фиксированного расписания; без зависимости от Telegram и MCP."""

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict, Field, model_validator

DEFAULT_SCHEDULE = Path(__file__).parent / "data" / "schedule.json"


class ScheduleError(ValueError):
    """Безопасная ошибка источника или даты."""


def parse_date(value: str) -> date:
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() != value:
            raise ValueError
        return parsed
    except (ValueError, TypeError):
        raise ScheduleError("Укажите существующую дату") from None


class Lesson(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    start: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(pattern=r"^(?:[01]\d|2[0-3]):[0-5]\d$")
    title: str = Field(min_length=1, max_length=200)
    location: str = Field(min_length=1, max_length=200)

    @model_validator(mode="after")
    def check_times(self) -> "Lesson":
        if self.end <= self.start:
            raise ValueError("Занятие должно завершаться после начала.")
        return self


class DaySchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    lessons: list[Lesson]


class WeekSchedule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timezone: str
    start: str
    end: str
    days: list[DaySchedule] = Field(min_length=7, max_length=7)


class ScheduleService:
    def __init__(self, path: Path, timezone: str) -> None:
        self._path = path
        self._zone = ZoneInfo(timezone)

    @property
    def timezone(self) -> str:
        return self._zone.key

    def _load(self) -> dict[str, list[Lesson]]:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError
            result = {}
            for day, lessons in raw.items():
                parse_date(day)
                if not isinstance(lessons, list):
                    raise ValueError
                result[day] = [Lesson.model_validate(item) for item in lessons]
            return result
        except (OSError, ValueError, TypeError):
            raise ScheduleError(
                "Источник расписания недоступен или имеет неверный формат."
            ) from None

    def for_date(self, value: str) -> DaySchedule:
        day = parse_date(value).isoformat()
        return DaySchedule(date=day, lessons=self._load().get(day, []))

    def current_week(self, now: datetime | None = None) -> WeekSchedule:
        if now is not None and (now.tzinfo is None or now.utcoffset() is None):
            raise ValueError("Часы должны возвращать время с часовым поясом.")
        today = (now or datetime.now(self._zone)).astimezone(self._zone).date()
        monday = today - timedelta(days=today.weekday())
        data = self._load()
        days = [
            DaySchedule(
                date=(monday + timedelta(days=i)).isoformat(),
                lessons=data.get((monday + timedelta(days=i)).isoformat(), []),
            )
            for i in range(7)
        ]
        return WeekSchedule(
            timezone=self._zone.key, start=days[0].date, end=days[-1].date, days=days
        )
