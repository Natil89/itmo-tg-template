"""Подтверждаемые напоминания и пользовательские часовые пояса в PostgreSQL."""

import json
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class ReminderError(ValueError):
    """Безопасная ошибка без секретов, SQL и внутренних идентификаторов."""


_ISO_TIME = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _parse_time(value: str) -> datetime:
    try:
        if not isinstance(value, str) or not _ISO_TIME.fullmatch(value):
            raise ValueError
        result = datetime.fromisoformat(value)
        if result.tzinfo is None or result.utcoffset() is None:
            raise ValueError
        return result
    except ValueError:
        raise ReminderError("Укажите дату и время ISO 8601 с явным часовым поясом.") from None


class ReminderDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    text: str = Field(min_length=1, max_length=500)
    remind_at: str

    @field_validator("text", mode="before")
    @classmethod
    def strip_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("remind_at")
    @classmethod
    def explicit_time(cls, value: str) -> str:
        _parse_time(value)
        return value


class ReminderSaved(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    reminder_id: int
    text: str
    remind_at: str
    timezone: str
    created: bool


ActionStatus = Literal["prepared", "confirmed", "executed", "cancelled", "expired"]


@dataclass(frozen=True)
class PendingReminder:
    action_id: str
    user_id: int
    payload: ReminderDraft
    timezone: str
    expires_at: datetime
    status: ActionStatus
    reminder_id: int | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminder_user_settings (
    user_id BIGINT PRIMARY KEY CHECK (user_id > 0),
    timezone TEXT NOT NULL CHECK (length(timezone) BETWEEN 1 AND 100)
);
CREATE TABLE IF NOT EXISTS reminder_actions (
    action_id UUID PRIMARY KEY,
    user_id BIGINT NOT NULL CHECK (user_id > 0),
    update_id BIGINT NOT NULL CHECK (update_id > 0),
    payload JSONB NOT NULL CHECK (jsonb_typeof(payload) = 'object'),
    timezone TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'prepared'
        CHECK (status IN ('prepared','confirmed','executed','cancelled','expired')),
    expires_at TIMESTAMPTZ NOT NULL,
    reminder_id BIGINT,
    UNIQUE (user_id, update_id),
    UNIQUE (action_id, user_id),
    CHECK (status <> 'executed' OR reminder_id IS NOT NULL)
);
CREATE TABLE IF NOT EXISTS reminders (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL CHECK (user_id > 0),
    text TEXT NOT NULL CHECK (length(btrim(text)) BETWEEN 1 AND 500),
    remind_at TIMESTAMPTZ NOT NULL,
    timezone TEXT NOT NULL,
    action_id UUID NOT NULL UNIQUE,
    FOREIGN KEY (action_id,user_id) REFERENCES reminder_actions (action_id,user_id)
);
CREATE INDEX IF NOT EXISTS reminders_owner_time ON reminders (user_id,remind_at,id);
"""
_DB_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)


def _owner(user_id: int) -> None:
    if isinstance(user_id, bool) or not isinstance(user_id, int) or user_id <= 0:
        raise ReminderError("Не удалось определить владельца напоминания.")


def _now(value: datetime | None) -> datetime:
    current = value if value is not None else datetime.now(UTC)
    if not isinstance(current, datetime) or current.tzinfo is None or current.utcoffset() is None:
        raise ReminderError("Время операции должно содержать часовой пояс.")
    return current.astimezone(UTC)


def _zone(value: str) -> ZoneInfo:
    try:
        return ZoneInfo(value)
    except (ValueError, TypeError, ZoneInfoNotFoundError):
        raise ReminderError(
            "Укажите существующий часовой пояс IANA, например Europe/Moscow."
        ) from None


def _identity(value: str) -> UUID | None:
    try:
        return UUID(value)
    except (ValueError, TypeError, AttributeError):
        return None


def _draft(value: ReminderDraft) -> ReminderDraft:
    try:
        return ReminderDraft.model_validate(value.model_dump())
    except (AttributeError, ValidationError):
        raise ReminderError("Проверьте текст и время напоминания.") from None


def _normalize(value: ReminderDraft, timezone: str) -> ReminderDraft:
    draft = _draft(value)
    timestamp, zone = _parse_time(draft.remind_at), _zone(timezone)
    local = timestamp.replace(tzinfo=None)
    offsets = set()
    try:
        for fold in (0, 1):
            candidate = local.replace(tzinfo=zone, fold=fold)
            restored = candidate.astimezone(UTC).astimezone(zone).replace(tzinfo=None)
            if restored == local:
                offsets.add(candidate.utcoffset())
        actual = timestamp.astimezone(zone)
    except (OverflowError, ValueError):
        raise ReminderError("Дата напоминания выходит за допустимый диапазон.") from None
    if len(offsets) != 1:
        raise ReminderError("Локальное время неоднозначно или не существует. Уточните время.")
    if timestamp.utcoffset() != actual.utcoffset():
        raise ReminderError("Смещение времени не совпадает с сохранённым часовым поясом.")
    return ReminderDraft(text=draft.text, remind_at=actual.isoformat())


def _pending(row: asyncpg.Record | dict) -> PendingReminder:
    payload = row["payload"]
    if isinstance(payload, str):
        payload = json.loads(payload)
    return PendingReminder(
        action_id=str(row["action_id"]),
        user_id=row["user_id"],
        payload=ReminderDraft.model_validate(payload),
        timezone=row["timezone"],
        expires_at=row["expires_at"],
        status=row["status"],
        reminder_id=row["reminder_id"],
    )


def _saved(row: asyncpg.Record, *, created: bool) -> ReminderSaved:
    return ReminderSaved(
        reminder_id=row["id"],
        text=row["text"],
        remind_at=row["remind_at"].astimezone(_zone(row["timezone"])).isoformat(),
        timezone=row["timezone"],
        created=created,
    )


class ReminderStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        try:
            await self._pool.execute(_SCHEMA)
        except _DB_ERRORS:
            raise ReminderError("Не удалось подготовить хранилище напоминаний.") from None

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[asyncpg.Connection]:
        try:
            async with self._pool.acquire() as connection:
                yield connection
        except _DB_ERRORS:
            raise ReminderError("Хранилище напоминаний сейчас недоступно.") from None

    async def get_timezone(self, user_id: int) -> str | None:
        _owner(user_id)
        async with self._connection() as connection:
            return await connection.fetchval(
                "SELECT timezone FROM reminder_user_settings WHERE user_id=$1",
                user_id,
            )

    async def set_timezone(self, user_id: int, zone: str) -> None:
        _owner(user_id)
        validated = _zone(zone).key
        async with self._connection() as connection:
            await connection.execute(
                "INSERT INTO reminder_user_settings (user_id,timezone) VALUES ($1,$2) "
                "ON CONFLICT (user_id) DO UPDATE SET timezone=EXCLUDED.timezone",
                user_id,
                validated,
            )

    async def prepare(
        self,
        user_id: int,
        update_id: int,
        draft: ReminderDraft,
        *,
        now: datetime | None = None,
    ) -> PendingReminder:
        _owner(user_id)
        if isinstance(update_id, bool) or not isinstance(update_id, int) or update_id <= 0:
            raise ReminderError("Не удалось определить исходный запрос.")
        _draft(draft)
        if now is not None:
            _now(now)
        async with self._connection() as connection, connection.transaction():
            existing = await connection.fetchrow(
                "SELECT * FROM reminder_actions WHERE user_id=$1 AND update_id=$2 FOR UPDATE",
                user_id,
                update_id,
            )
            if existing is not None:
                pending = _pending(existing)
                if _normalize(draft, pending.timezone) != pending.payload:
                    raise ReminderError("Этот запрос уже связан с другим напоминанием.")
                return pending
            timezone = await connection.fetchval(
                "SELECT timezone FROM reminder_user_settings WHERE user_id=$1 FOR SHARE",
                user_id,
            )
            if timezone is None:
                raise ReminderError("Сначала настройте часовой пояс командой /timezone.")
            normalized = _normalize(draft, timezone)
            instant = _now(now)
            if _parse_time(normalized.remind_at) <= instant:
                raise ReminderError("Напоминание должно быть назначено на будущее время.")
            row = await connection.fetchrow(
                "INSERT INTO reminder_actions "
                "(action_id,user_id,update_id,payload,timezone,expires_at) "
                "VALUES ($1,$2,$3,$4,$5,$6) ON CONFLICT (user_id,update_id) DO NOTHING RETURNING *",
                uuid4(),
                user_id,
                update_id,
                normalized.model_dump_json(),
                timezone,
                instant + timedelta(minutes=5),
            )
            if row is None:
                row = await connection.fetchrow(
                    "SELECT * FROM reminder_actions WHERE user_id=$1 AND update_id=$2 FOR UPDATE",
                    user_id,
                    update_id,
                )
            pending = _pending(row)
            if _normalize(draft, pending.timezone) != pending.payload:
                raise ReminderError("Этот запрос уже связан с другим напоминанием.")
            return pending

    async def _locked_action(
        self,
        connection: asyncpg.Connection,
        user_id: int,
        identity: UUID,
        now: datetime | None,
    ) -> PendingReminder | None:
        row = await connection.fetchrow(
            "SELECT * FROM reminder_actions WHERE user_id=$1 AND action_id=$2 FOR UPDATE",
            user_id,
            identity,
        )
        if row is None:
            return None
        pending = _pending(row)
        # Часы читаются после ожидания блокировки, не до него.
        if pending.status in {"prepared", "confirmed"} and pending.expires_at <= _now(now):
            pending = _pending(
                await connection.fetchrow(
                    "UPDATE reminder_actions SET status='expired' "
                    "WHERE user_id=$1 AND action_id=$2 RETURNING *",
                    user_id,
                    identity,
                )
            )
        return pending

    async def get_pending(
        self,
        user_id: int,
        action_id: str,
        *,
        now: datetime | None = None,
    ) -> PendingReminder | None:
        _owner(user_id)
        if now is not None:
            _now(now)
        identity = _identity(action_id)
        if identity is None:
            return None
        async with self._connection() as connection, connection.transaction():
            return await self._locked_action(connection, user_id, identity, now)

    async def confirm(
        self,
        user_id: int,
        action_id: str,
        *,
        now: datetime | None = None,
    ) -> PendingReminder:
        _owner(user_id)
        if now is not None:
            _now(now)
        identity = _identity(action_id)
        if identity is None:
            raise ReminderError("Действие не найдено или недоступно.")
        async with self._connection() as connection, connection.transaction():
            pending = await self._locked_action(connection, user_id, identity, now)
            if pending is not None and pending.status == "prepared":
                pending = _pending(
                    await connection.fetchrow(
                        "UPDATE reminder_actions SET status='confirmed' "
                        "WHERE user_id=$1 AND action_id=$2 RETURNING *",
                        user_id,
                        identity,
                    )
                )
        if pending is None or pending.status not in {"confirmed", "executed"}:
            raise ReminderError("Подтверждение недоступно, отменено или просрочено.")
        return pending

    async def cancel(
        self,
        user_id: int,
        action_id: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        _owner(user_id)
        if now is not None:
            _now(now)
        identity = _identity(action_id)
        if identity is None:
            return False
        async with self._connection() as connection, connection.transaction():
            pending = await self._locked_action(connection, user_id, identity, now)
            if pending is None or pending.status not in {"prepared", "confirmed"}:
                return False
            await connection.execute(
                "UPDATE reminder_actions SET status='cancelled' WHERE user_id=$1 AND action_id=$2",
                user_id,
                identity,
            )
            return True

    async def execute(
        self,
        user_id: int,
        action_id: str,
        draft: ReminderDraft,
        *,
        now: datetime | None = None,
    ) -> ReminderSaved:
        _owner(user_id)
        _draft(draft)
        if now is not None:
            _now(now)
        identity = _identity(action_id)
        if identity is None:
            raise ReminderError("Действие не найдено или недоступно.")
        result = None
        async with self._connection() as connection, connection.transaction():
            pending = await self._locked_action(connection, user_id, identity, now)
            if pending is not None:
                normalized = _normalize(draft, pending.timezone)
                if normalized != pending.payload:
                    raise ReminderError("Параметры изменились. Подготовьте новое подтверждение.")
                if pending.status == "executed":
                    row = await connection.fetchrow(
                        "SELECT * FROM reminders WHERE user_id=$1 AND action_id=$2 AND id=$3",
                        user_id,
                        identity,
                        pending.reminder_id,
                    )
                    if row is not None:
                        result = _saved(row, created=False)
                elif pending.status == "confirmed":
                    timestamp = _parse_time(normalized.remind_at)
                    if timestamp <= _now(now):
                        raise ReminderError("Напоминание должно быть назначено на будущее время.")
                    row = await connection.fetchrow(
                        "INSERT INTO reminders (user_id,text,remind_at,timezone,action_id) "
                        "VALUES ($1,$2,$3,$4,$5) RETURNING *",
                        user_id,
                        normalized.text,
                        timestamp,
                        pending.timezone,
                        identity,
                    )
                    await connection.execute(
                        "UPDATE reminder_actions SET status='executed',reminder_id=$3 "
                        "WHERE user_id=$1 AND action_id=$2",
                        user_id,
                        identity,
                        row["id"],
                    )
                    result = _saved(row, created=True)
        if result is None:
            raise ReminderError("Напоминание не сохранено: подтвердите действующее предложение.")
        return result
