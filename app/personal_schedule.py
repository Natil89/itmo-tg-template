"""Личное расписание и подтверждаемые операции в PostgreSQL."""

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app.schedule import DaySchedule, Lesson, WeekSchedule, parse_date


class PersonalScheduleError(ValueError):
    """Безопасное сообщение без SQL, идентификаторов и параметров подключения."""


class LessonDraft(Lesson):
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")

    @field_validator("date")
    @classmethod
    def valid_date(cls, value: str) -> str:
        return parse_date(value).isoformat()

    @field_validator("title", "location", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class LessonSaved(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    lesson_id: int
    date: str
    start: str
    end: str
    title: str
    location: str
    created: bool


ActionStatus = Literal["prepared", "confirmed", "executed", "cancelled", "expired"]


@dataclass(frozen=True)
class PendingLesson:
    action_id: str
    user_id: int
    payload: LessonDraft
    expires_at: datetime
    status: ActionStatus
    lesson_id: int | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS personal_schedule_actions (
    action_id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL CHECK (user_id > 0),
    update_id BIGINT NOT NULL CHECK (update_id > 0),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    status TEXT NOT NULL DEFAULT 'prepared'
        CHECK (status IN ('prepared', 'confirmed', 'executed', 'cancelled', 'expired')),
    expires_at TIMESTAMPTZ NOT NULL,
    lesson_id BIGINT,
    UNIQUE (user_id, update_id),
    UNIQUE (action_id, user_id),
    CHECK (status <> 'executed' OR lesson_id IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS personal_schedule_lessons (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL CHECK (user_id > 0),
    lesson_date DATE NOT NULL,
    start_time TIME NOT NULL,
    end_time TIME NOT NULL CHECK (end_time > start_time),
    title TEXT NOT NULL CHECK (length(btrim(title)) BETWEEN 1 AND 200),
    location TEXT NOT NULL CHECK (length(btrim(location)) BETWEEN 1 AND 200),
    action_id UUID NOT NULL UNIQUE,
    FOREIGN KEY (action_id, user_id)
        REFERENCES personal_schedule_actions (action_id, user_id)
);
CREATE INDEX IF NOT EXISTS personal_schedule_lessons_owner_date
    ON personal_schedule_lessons (user_id, lesson_date, start_time, id);
"""

_DB_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)


def _owner(user_id: int) -> None:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise PersonalScheduleError("Не удалось определить владельца расписания.")


def _now(value: datetime | None) -> datetime:
    current = value if value is not None else datetime.now(UTC)
    if current.tzinfo is None or current.utcoffset() is None:
        raise PersonalScheduleError("Время операции должно содержать часовой пояс.")
    return current.astimezone(UTC)


def _action_id(value: str) -> UUID | None:
    try:
        return UUID(value)
    except (ValueError, TypeError, AttributeError):
        return None


def _draft(value: LessonDraft) -> LessonDraft:
    try:
        return LessonDraft.model_validate(value.model_dump())
    except (ValidationError, AttributeError):
        raise PersonalScheduleError("Проверьте дату, время, название и место занятия.") from None


def _pending(row: asyncpg.Record | dict) -> PendingLesson:
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return PendingLesson(
        action_id=str(row["action_id"]),
        user_id=row["user_id"],
        payload=LessonDraft.model_validate(payload),
        expires_at=row["expires_at"],
        status=row["status"],
        lesson_id=row["lesson_id"],
    )


def _saved(row: asyncpg.Record, *, created: bool) -> LessonSaved:
    return LessonSaved(
        lesson_id=row["id"],
        date=row["lesson_date"].isoformat(),
        start=row["start_time"].strftime("%H:%M"),
        end=row["end_time"].strftime("%H:%M"),
        title=row["title"],
        location=row["location"],
        created=created,
    )


class PersonalScheduleStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        try:
            await self._pool.execute(_SCHEMA)
        except _DB_ERRORS:
            raise PersonalScheduleError("Не удалось подготовить хранилище расписания.") from None

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[asyncpg.Connection]:
        try:
            async with self._pool.acquire() as connection:
                yield connection
        except _DB_ERRORS:
            raise PersonalScheduleError("Хранилище расписания сейчас недоступно.") from None

    async def for_date(self, user_id: int, value: str) -> DaySchedule:
        _owner(user_id)
        day = parse_date(value)
        async with self._connection() as connection:
            rows = await connection.fetch(
                "SELECT start_time, end_time, title, location FROM personal_schedule_lessons "
                "WHERE user_id = $1 AND lesson_date = $2 ORDER BY start_time, end_time, id",
                user_id,
                day,
            )
        return DaySchedule(
            date=day.isoformat(),
            lessons=[
                Lesson(
                    start=row["start_time"].strftime("%H:%M"),
                    end=row["end_time"].strftime("%H:%M"),
                    title=row["title"],
                    location=row["location"],
                )
                for row in rows
            ],
        )

    async def current_week(
        self,
        user_id: int,
        timezone: str,
        now: datetime | None = None,
    ) -> WeekSchedule:
        _owner(user_id)
        try:
            zone = ZoneInfo(timezone)
        except (ZoneInfoNotFoundError, ValueError, TypeError):
            raise PersonalScheduleError("Укажите существующий часовой пояс IANA.") from None
        today = _now(now).astimezone(zone).date()
        monday = today - timedelta(days=today.weekday())
        sunday = monday + timedelta(days=6)
        async with self._connection() as connection:
            rows = await connection.fetch(
                "SELECT lesson_date, start_time, end_time, title, location "
                "FROM personal_schedule_lessons WHERE user_id = $1 "
                "AND lesson_date BETWEEN $2 AND $3 ORDER BY lesson_date, start_time, end_time, id",
                user_id,
                monday,
                sunday,
            )
        grouped: dict[str, list[Lesson]] = {}
        for row in rows:
            grouped.setdefault(row["lesson_date"].isoformat(), []).append(
                Lesson(
                    start=row["start_time"].strftime("%H:%M"),
                    end=row["end_time"].strftime("%H:%M"),
                    title=row["title"],
                    location=row["location"],
                )
            )
        return WeekSchedule(
            timezone=zone.key,
            start=monday.isoformat(),
            end=sunday.isoformat(),
            days=[
                DaySchedule(
                    date=(monday + timedelta(days=i)).isoformat(),
                    lessons=grouped.get((monday + timedelta(days=i)).isoformat(), []),
                )
                for i in range(7)
            ],
        )

    async def prepare(
        self,
        user_id: int,
        update_id: int,
        draft: LessonDraft,
        *,
        now: datetime | None = None,
    ) -> PendingLesson:
        _owner(user_id)
        if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id <= 0:
            raise PersonalScheduleError("Не удалось определить исходный запрос.")
        payload = _draft(draft).model_dump()
        instant = _now(now)
        async with self._connection() as connection, connection.transaction():
            row = await connection.fetchrow(
                "INSERT INTO personal_schedule_actions "
                "(action_id, user_id, update_id, payload, expires_at) VALUES ($1,$2,$3,$4,$5) "
                "ON CONFLICT (user_id, update_id) DO NOTHING RETURNING *",
                uuid4(),
                user_id,
                update_id,
                json.dumps(payload, ensure_ascii=False),
                instant + timedelta(minutes=5),
            )
            if row is None:
                row = await connection.fetchrow(
                    "SELECT * FROM personal_schedule_actions "
                    "WHERE user_id = $1 AND update_id = $2 FOR UPDATE",
                    user_id,
                    update_id,
                )
            pending = _pending(row)
            if pending.payload.model_dump() != payload:
                raise PersonalScheduleError("Этот запрос уже связан с другим занятием.")
            return pending

    async def _locked_action(
        self,
        connection: asyncpg.Connection,
        user_id: int,
        action_id: UUID,
        instant: datetime | None,
    ) -> PendingLesson | None:
        row = await connection.fetchrow(
            "SELECT * FROM personal_schedule_actions WHERE user_id=$1 AND action_id=$2 FOR UPDATE",
            user_id,
            action_id,
        )
        if row is None:
            return None
        # Рабочие часы читаем после ожидания row lock, иначе истёкшее в очереди
        # подтверждение могло бы выполниться с устаревшим временем проверки.
        instant = _now(instant)
        pending = _pending(row)
        if pending.status in {"prepared", "confirmed"} and pending.expires_at <= instant:
            row = await connection.fetchrow(
                "UPDATE personal_schedule_actions SET status='expired' "
                "WHERE user_id=$1 AND action_id=$2 RETURNING *",
                user_id,
                action_id,
            )
            pending = _pending(row)
        return pending

    async def get_pending(
        self,
        user_id: int,
        action_id: str,
        *,
        now: datetime | None = None,
    ) -> PendingLesson | None:
        _owner(user_id)
        instant = _now(now) if now is not None else None
        identity = _action_id(action_id)
        if identity is None:
            return None
        async with self._connection() as connection, connection.transaction():
            return await self._locked_action(connection, user_id, identity, instant)

    async def confirm(
        self,
        user_id: int,
        action_id: str,
        *,
        now: datetime | None = None,
    ) -> PendingLesson:
        _owner(user_id)
        instant = _now(now) if now is not None else None
        identity = _action_id(action_id)
        if identity is None:
            raise PersonalScheduleError("Действие не найдено или недоступно.")
        async with self._connection() as connection, connection.transaction():
            pending = await self._locked_action(connection, user_id, identity, instant)
            if pending is not None and pending.status == "prepared":
                pending = _pending(
                    await connection.fetchrow(
                        "UPDATE personal_schedule_actions SET status='confirmed' "
                        "WHERE user_id=$1 AND action_id=$2 RETURNING *",
                        user_id,
                        identity,
                    )
                )
        if pending is None or pending.status not in {"confirmed", "executed"}:
            raise PersonalScheduleError("Подтверждение недоступно, отменено или просрочено.")
        return pending

    async def cancel(
        self,
        user_id: int,
        action_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        _owner(user_id)
        instant = _now(now) if now is not None else None
        identity = _action_id(action_id)
        if identity is None:
            return False
        async with self._connection() as connection, connection.transaction():
            pending = await self._locked_action(connection, user_id, identity, instant)
            if pending is None or pending.status not in {"prepared", "confirmed"}:
                return False
            await connection.execute(
                "UPDATE personal_schedule_actions SET status='cancelled' "
                "WHERE user_id=$1 AND action_id=$2",
                user_id,
                identity,
            )
            return True

    async def execute(
        self,
        user_id: int,
        action_id: str,
        draft: LessonDraft,
        *,
        now: datetime | None = None,
    ) -> LessonSaved:
        _owner(user_id)
        instant = _now(now) if now is not None else None
        payload = _draft(draft)
        identity = _action_id(action_id)
        if identity is None:
            raise PersonalScheduleError("Действие не найдено или недоступно.")
        result = None
        async with self._connection() as connection, connection.transaction():
            pending = await self._locked_action(connection, user_id, identity, instant)
            if pending is not None and pending.payload == payload:
                if pending.status == "executed":
                    row = await connection.fetchrow(
                        "SELECT * FROM personal_schedule_lessons "
                        "WHERE user_id=$1 AND action_id=$2 AND id=$3",
                        user_id,
                        identity,
                        pending.lesson_id,
                    )
                    if row is not None:
                        result = _saved(row, created=False)
                elif pending.status == "confirmed":
                    row = await connection.fetchrow(
                        "INSERT INTO personal_schedule_lessons "
                        "(user_id,lesson_date,start_time,end_time,title,location,action_id) "
                        "VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING *",
                        user_id,
                        parse_date(payload.date),
                        time.fromisoformat(payload.start),
                        time.fromisoformat(payload.end),
                        payload.title,
                        payload.location,
                        identity,
                    )
                    await connection.execute(
                        "UPDATE personal_schedule_actions SET status='executed',lesson_id=$3 "
                        "WHERE user_id=$1 AND action_id=$2",
                        user_id,
                        identity,
                        row["id"],
                    )
                    result = _saved(row, created=True)
        if result is None:
            raise PersonalScheduleError(
                "Занятие не сохранено: подтвердите действующее предложение без изменения данных."
            )
        return result
