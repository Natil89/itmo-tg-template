"""Подтверждённый город и отдельное согласие на ежедневную сводку."""

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

import asyncpg

from app.delivery import DB_ERRORS, instant
from app.weather import GeoCity, WeatherService

SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_profiles (
    actor_id BIGINT PRIMARY KEY CHECK(actor_id > 0),
    destination_chat_id BIGINT NOT NULL CHECK(destination_chat_id > 0),
    city JSONB NOT NULL CHECK(jsonb_typeof(city)='object'),
    city_confirmed_at TIMESTAMPTZ NOT NULL,
    briefing_enabled BOOLEAN NOT NULL DEFAULT FALSE,
    revision BIGINT NOT NULL DEFAULT 1 CHECK(revision > 0)
);
CREATE TABLE IF NOT EXISTS city_confirmations (
    token UUID PRIMARY KEY,
    actor_id BIGINT NOT NULL,
    confirmed_at TIMESTAMPTZ NOT NULL
);
"""


class NotificationError(ValueError):
    pass


@dataclass(frozen=True)
class NotificationProfile:
    actor_id: int
    destination_chat_id: int
    city: GeoCity
    timezone: str | None
    enabled: bool
    revision: int


def profile_from_row(row) -> NotificationProfile:
    city = row["city"]
    if isinstance(city, str):
        city = json.loads(city)
    timezone = row["timezone"]
    if timezone is not None:
        ZoneInfo(timezone)
    return NotificationProfile(
        row["actor_id"],
        row["destination_chat_id"],
        GeoCity.model_validate(city),
        timezone,
        row["briefing_enabled"],
        row["revision"],
    )


PROFILE_SELECT = (
    "SELECT p.*,s.timezone FROM notification_profiles p "
    "LEFT JOIN reminder_user_settings s ON s.user_id=p.actor_id "
)


class NotificationStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        try:
            async with self.pool.acquire() as connection:
                yield connection
        except DB_ERRORS:
            raise NotificationError("Настройки уведомлений временно недоступны.") from None

    async def initialize(self) -> None:
        try:
            await self.pool.execute(SCHEMA)
        except DB_ERRORS:
            raise NotificationError("Не удалось подготовить настройки уведомлений.") from None

    async def get(self, actor_id: int) -> NotificationProfile | None:
        async with self.connection() as connection:
            row = await connection.fetchrow(PROFILE_SELECT + "WHERE p.actor_id=$1", actor_id)
        return profile_from_row(row) if row else None

    async def confirm_city(
        self,
        actor_id: int,
        chat_id: int,
        token: UUID,
        city: GeoCity,
        now: datetime,
        *,
        expires_at: datetime | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> NotificationProfile:
        if any(type(value) is not int or value <= 0 for value in (actor_id, chat_id)):
            raise NotificationError("Не удалось определить владельца и личный чат.")
        city = GeoCity.model_validate(city.model_dump())
        read_clock = clock or (lambda: now)

        def checked_now():
            current = instant(read_clock())
            if expires_at is not None and current >= instant(expires_at):
                raise NotificationError("Выбор истёк. Повторите /city.")
            return current

        async with self.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock($1)", actor_id)
            await connection.fetchrow(
                "SELECT actor_id FROM notification_profiles WHERE actor_id=$1 FOR UPDATE", actor_id
            )
            fresh = await connection.fetchval(
                "INSERT INTO city_confirmations(token,actor_id,confirmed_at) VALUES($1,$2,$3) "
                "ON CONFLICT DO NOTHING RETURNING token",
                token,
                actor_id,
                checked_now(),
            )
            current = checked_now()
            if fresh:
                await connection.execute(
                    "INSERT INTO notification_profiles"
                    "(actor_id,destination_chat_id,city,city_confirmed_at) "
                    "VALUES($1,$2,$3,$4) ON CONFLICT(actor_id) DO UPDATE SET "
                    "destination_chat_id=EXCLUDED.destination_chat_id,city=EXCLUDED.city,"
                    "city_confirmed_at=EXCLUDED.city_confirmed_at,briefing_enabled=FALSE,"
                    "revision=notification_profiles.revision+1",
                    actor_id,
                    chat_id,
                    city.model_dump_json(),
                    current,
                )
            elif (
                await connection.fetchval(
                    "SELECT actor_id FROM city_confirmations WHERE token=$1", token
                )
                != actor_id
            ):
                raise NotificationError("Подтверждение недоступно.")
            row = await connection.fetchrow(PROFILE_SELECT + "WHERE p.actor_id=$1", actor_id)
        if row is None:
            raise NotificationError("Подтверждение недоступно.")
        return profile_from_row(row)

    async def set_enabled(self, actor_id: int, enabled: bool) -> NotificationProfile | None:
        async with self.connection() as connection, connection.transaction():
            row = await connection.fetchrow(
                PROFILE_SELECT + "WHERE p.actor_id=$1 FOR UPDATE OF p", actor_id
            )
            if row is None:
                if not enabled:
                    return None
                raise NotificationError("Сначала выберите и подтвердите город командой /city.")
            profile = profile_from_row(row)
            if enabled and profile.timezone is None:
                raise NotificationError("Сначала настройте часовой пояс командой /timezone.")
            await connection.execute(
                "UPDATE notification_profiles SET briefing_enabled=$2,"
                "revision=revision+CASE WHEN briefing_enabled=$2 THEN 0 ELSE 1 END "
                "WHERE actor_id=$1",
                actor_id,
                enabled,
            )
            row = await connection.fetchrow(PROFILE_SELECT + "WHERE p.actor_id=$1", actor_id)
        return profile_from_row(row)

    async def touch_destination(self, actor_id: int, chat_id: int) -> None:
        async with self.connection() as connection:
            await connection.execute(
                "UPDATE notification_profiles SET destination_chat_id=$2,"
                "revision=revision+CASE WHEN destination_chat_id=$2 THEN 0 ELSE 1 END "
                "WHERE actor_id=$1",
                actor_id,
                chat_id,
            )

    async def active(self) -> list[NotificationProfile]:
        async with self.connection() as connection:
            rows = await connection.fetch(
                PROFILE_SELECT + "WHERE p.briefing_enabled AND s.timezone IS NOT NULL"
            )
        return [profile_from_row(row) for row in rows]


@dataclass(frozen=True)
class CityProposal:
    token: UUID
    actor_id: int
    chat_id: int
    city: GeoCity
    expires_at: datetime


class CitySelection:
    """Координаты до согласия только в RAM. Право подтверждения — пять минут."""

    def __init__(
        self,
        weather: WeatherService,
        store: NotificationStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.weather, self.store = weather, store
        self.clock = clock or (lambda: datetime.now(UTC))
        self._proposals: dict[UUID, CityProposal] = {}
        self._lock = asyncio.Lock()

    async def propose(self, actor_id: int, chat_id: int, query: str) -> list[CityProposal]:
        cities = await self.weather.find_cities(query)
        if not cities:
            raise NotificationError("Город не найден. Уточните название и страну.")
        if len(cities) > 5:
            raise NotificationError("Слишком много вариантов. Уточните страну или регион.")
        cities = [GeoCity.model_validate(city.model_dump()) for city in cities]
        async with self._lock:
            now = instant(self.clock())
            self._proposals = {
                key: item
                for key, item in self._proposals.items()
                if item.expires_at > now and item.actor_id != actor_id
            }
            if len(self._proposals) + len(cities) > 1024:
                raise NotificationError("Выбор города временно недоступен. Повторите позже.")
            result = [
                CityProposal(uuid4(), actor_id, chat_id, city, now + timedelta(minutes=5))
                for city in cities
            ]
            self._proposals.update({item.token: item for item in result})
            return result

    async def confirm(self, token: str, actor_id: int, chat_id: int) -> NotificationProfile:
        try:
            identity = UUID(token)
        except (ValueError, TypeError, AttributeError):
            raise NotificationError("Подтверждение города недоступно.") from None
        async with self._lock:
            proposal = self._proposals.get(identity)
            if proposal is not None and proposal.expires_at <= instant(self.clock()):
                self._proposals.pop(identity, None)
                proposal = None
            if (
                proposal is None
                or proposal.actor_id != actor_id
                or proposal.chat_id != chat_id
                or proposal.expires_at <= instant(self.clock())
            ):
                raise NotificationError("Выбор истёк или недоступен. Повторите /city.")
            profile = await self.store.confirm_city(
                actor_id,
                chat_id,
                identity,
                proposal.city,
                self.clock(),
                expires_at=proposal.expires_at,
                clock=self.clock,
            )
            self._proposals = {
                key: item for key, item in self._proposals.items() if item.actor_id != actor_id
            }
            return profile

    async def purge(self) -> None:
        async with self._lock:
            now = instant(self.clock())
            self._proposals = {
                key: item for key, item in self._proposals.items() if item.expires_at > now
            }

    async def run_cleanup(self) -> None:
        while True:
            await self.purge()
            await asyncio.sleep(15)
