"""Постоянная доставка подтверждённых напоминаний. Модель здесь не вызывается."""

import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

import asyncpg

from app.tracing import TraceStore, recorded, span

logger = logging.getLogger(__name__)
GRACE = timedelta(minutes=15)
LEASE = timedelta(seconds=60)
RETRY_DELAYS = (30, 120)
DB_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)

SCHEMA = """
CREATE TABLE IF NOT EXISTS delivery_destinations (
    actor_id BIGINT PRIMARY KEY CHECK(actor_id > 0),
    destination_chat_id BIGINT NOT NULL CHECK(destination_chat_id > 0)
);
CREATE UNIQUE INDEX IF NOT EXISTS reminders_id_owner ON reminders(id,user_id);
CREATE TABLE IF NOT EXISTS reminder_deliveries (
    event_id TEXT PRIMARY KEY,
    reminder_id BIGINT NOT NULL UNIQUE,
    actor_id BIGINT NOT NULL REFERENCES delivery_destinations(actor_id),
    destination_chat_id BIGINT NOT NULL CHECK(destination_chat_id > 0),
    due_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','sending','retry','sent','missed','failed')),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
    next_attempt_at TIMESTAMPTZ NOT NULL,
    lease_id UUID,
    lease_until TIMESTAMPTZ,
    sent_at TIMESTAMPTZ,
    telegram_message_id BIGINT,
    error_code TEXT CHECK(error_code IN
        ('temporary','permanent','attempts_exhausted','late','destination_changed','internal')),
    FOREIGN KEY(reminder_id,actor_id) REFERENCES reminders(id,user_id),
    CHECK(status <> 'sending' OR (lease_id IS NOT NULL AND lease_until IS NOT NULL)),
    CHECK(status <> 'sent' OR
        (sent_at IS NOT NULL AND telegram_message_id IS NOT NULL AND telegram_message_id > 0))
);
CREATE INDEX IF NOT EXISTS reminder_deliveries_pending
    ON reminder_deliveries(status,due_at,next_attempt_at);
"""


class DeliveryError(ValueError):
    def __init__(self) -> None:
        super().__init__("Хранилище доставки временно недоступно.")


class TemporaryDeliveryError(Exception):
    def __init__(self, retry_after: int = 0) -> None:
        self.retry_after = max(0, min(int(retry_after), 86400))
        super().__init__("temporary")


class PermanentDeliveryError(Exception):
    pass


def instant(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Нужны часы с часовым поясом.")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class DeliveryJob:
    event_id: str
    actor_id: int
    destination_chat_id: int
    reminder_id: int
    due_at: datetime
    text: str
    attempts: int
    lease_id: UUID
    lease_until: datetime


class Sender(Protocol):
    async def send(self, destination_chat_id: int, text: str) -> int: ...


class DeliveryStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    @asynccontextmanager
    async def connection(self) -> AsyncIterator[asyncpg.Connection]:
        try:
            async with self._pool.acquire() as connection:
                yield connection
        except DB_ERRORS:
            raise DeliveryError from None

    async def initialize(self) -> None:
        async with self.connection() as connection:
            await connection.execute(SCHEMA)

    async def bind_destination(self, actor_id: int, destination_chat_id: int) -> None:
        """Только приложение передаёт значения из проверенного private update."""
        if any(type(value) is not int or value <= 0 for value in (actor_id, destination_chat_id)):
            raise ValueError("Нужны доверенные владелец и личный чат.")
        async with self.connection() as connection:
            await connection.execute(
                "INSERT INTO delivery_destinations(actor_id,destination_chat_id) VALUES($1,$2) "
                "ON CONFLICT(actor_id) DO UPDATE "
                "SET destination_chat_id=EXCLUDED.destination_chat_id",
                actor_id,
                destination_chat_id,
            )

    async def prepare(self, now: datetime) -> int:
        instant(now)
        async with self.connection() as connection:
            rows = await connection.fetch(
                "INSERT INTO reminder_deliveries(event_id,reminder_id,actor_id,destination_chat_id,"
                "due_at,next_attempt_at) "
                "SELECT 'reminder:'||r.id::text,r.id,r.user_id,d.destination_chat_id,"
                "r.remind_at,r.remind_at "
                "FROM reminders r JOIN reminder_actions a ON a.action_id=r.action_id "
                "AND a.user_id=r.user_id AND a.status='executed' AND a.reminder_id=r.id "
                "JOIN delivery_destinations d ON d.actor_id=r.user_id "
                "WHERE NOT EXISTS(SELECT 1 FROM reminder_deliveries j WHERE j.reminder_id=r.id) "
                "ORDER BY r.remind_at,r.id LIMIT 1000 ON CONFLICT DO NOTHING RETURNING event_id"
            )
        return len(rows)

    async def claim(self, now: datetime) -> DeliveryJob | None:
        now = instant(now)
        async with self.connection() as connection, connection.transaction():
            await connection.execute(
                "UPDATE reminder_deliveries SET status='missed',error_code='late',"
                "lease_id=NULL,lease_until=NULL WHERE due_at < $1 AND "
                "(status IN ('pending','retry') OR (status='sending' AND lease_until <= $2))",
                now - GRACE,
                now,
            )
            row = await connection.fetchrow(
                "SELECT j.*,r.text,d.destination_chat_id AS current_destination "
                "FROM reminder_deliveries j JOIN reminders r "
                "ON r.id=j.reminder_id AND r.user_id=j.actor_id "
                "JOIN delivery_destinations d ON d.actor_id=j.actor_id "
                "WHERE j.due_at <= $1 AND ("
                "(j.status IN ('pending','retry') AND (j.next_attempt_at <= $1 OR j.due_at < $2)) "
                "OR (j.status='sending' AND j.lease_until <= $1)) "
                "ORDER BY j.due_at,j.event_id LIMIT 1 FOR UPDATE OF j SKIP LOCKED",
                now,
                now - GRACE,
            )
            if row is None:
                return None
            reason = None
            if now > row["due_at"] + GRACE:
                reason = "late"
            elif row["destination_chat_id"] != row["current_destination"]:
                reason = "destination_changed"
            elif row["attempts"] >= 3:
                reason = "attempts_exhausted"
            if reason:
                await connection.execute(
                    "UPDATE reminder_deliveries SET status=$2,error_code=$3,"
                    "lease_id=NULL,lease_until=NULL "
                    "WHERE event_id=$1",
                    row["event_id"],
                    "missed" if reason == "late" else "failed",
                    reason,
                )
                return None
            lease_id = uuid4()
            await connection.execute(
                "UPDATE reminder_deliveries SET status='sending',attempts=attempts+1,"
                "lease_id=$2,lease_until=$3,error_code=NULL WHERE event_id=$1",
                row["event_id"],
                lease_id,
                now + LEASE,
            )
            return DeliveryJob(
                event_id=row["event_id"],
                actor_id=row["actor_id"],
                destination_chat_id=row["destination_chat_id"],
                reminder_id=row["reminder_id"],
                due_at=row["due_at"],
                text=row["text"],
                attempts=row["attempts"] + 1,
                lease_id=lease_id,
                lease_until=now + LEASE,
            )

    async def authorized(self, job: DeliveryJob, now: datetime) -> bool:
        async with self.connection() as connection:
            return bool(
                await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM reminder_deliveries j "
                    "JOIN delivery_destinations d ON d.actor_id=j.actor_id "
                    "JOIN reminders r ON r.id=j.reminder_id AND r.user_id=j.actor_id "
                    "WHERE j.event_id=$1 AND j.lease_id=$2 AND j.status='sending' "
                    "AND j.actor_id=$3 AND j.destination_chat_id=$4 AND d.destination_chat_id=$4 "
                    "AND j.reminder_id=$5 AND r.text=$6 AND j.due_at=$7 AND r.remind_at=j.due_at "
                    "AND j.due_at <= $8 AND j.due_at >= $9 AND j.lease_until > $8)",
                    job.event_id,
                    job.lease_id,
                    job.actor_id,
                    job.destination_chat_id,
                    job.reminder_id,
                    job.text,
                    job.due_at,
                    instant(now),
                    instant(now) - GRACE,
                )
            )

    async def sent(self, job: DeliveryJob, message_id: int, now: datetime) -> None:
        if type(message_id) is not int or message_id <= 0:
            raise ValueError("Некорректный результат Telegram.")
        async with self.connection() as connection:
            await connection.execute(
                "UPDATE reminder_deliveries SET status='sent',sent_at=$3,telegram_message_id=$4,"
                "lease_id=NULL,lease_until=NULL,error_code=NULL "
                "WHERE event_id=$1 AND lease_id=$2 AND status='sending'",
                job.event_id,
                job.lease_id,
                instant(now),
                message_id,
            )

    async def failed(
        self,
        job: DeliveryJob,
        now: datetime,
        *,
        temporary: bool,
        retry_after: int = 0,
        code: str | None = None,
    ) -> None:
        retry = temporary and job.attempts < 3
        error_code = code or (
            "temporary" if retry else "attempts_exhausted" if temporary else "permanent"
        )
        if error_code not in {"temporary", "attempts_exhausted", "permanent", "internal", "late"}:
            raise ValueError("Неизвестный безопасный код ошибки.")
        delay = max(RETRY_DELAYS[min(job.attempts - 1, 1)], retry_after)
        async with self.connection() as connection:
            await connection.execute(
                "UPDATE reminder_deliveries SET status=$3,error_code=$4,next_attempt_at=$5,"
                "lease_id=NULL,lease_until=NULL "
                "WHERE event_id=$1 AND lease_id=$2 AND status='sending'",
                job.event_id,
                job.lease_id,
                "missed" if error_code == "late" else "retry" if retry else "failed",
                error_code,
                instant(now) + timedelta(seconds=delay),
            )


class ReminderDeliveryService:
    def __init__(
        self,
        store: DeliveryStore,
        sender: Sender,
        *,
        clock: Callable[[], datetime] | None = None,
        trace_store: TraceStore | None = None,
    ) -> None:
        self.trace_store = trace_store
        self._store, self._sender = store, sender
        self._clock = clock or (lambda: datetime.now(UTC))

    async def run_once(self) -> int:
        await self._store.prepare(self._clock())
        sent = 0
        for _ in range(100):
            job = await self._store.claim(self._clock())
            if job is None:
                break
            async with recorded(
                self.trace_store,
                job.actor_id,
                "reminder_delivery",
                "reminder_delivery",
                "delivery_attempt",
            ) as root:
                authorized = await self._store.authorized(job, self._clock())
                current = instant(self._clock())  # Часы заново после ожидания БД.
                if (
                    not authorized
                    or current > job.due_at + GRACE
                    or current >= job.lease_until
                    or current < job.due_at
                ):
                    late = current > job.due_at + GRACE
                    expired_lease = current >= job.lease_until
                    await self._store.failed(
                        job,
                        current,
                        temporary=expired_lease and not late,
                        code="late" if late else None if expired_lease else "internal",
                    )
                    root.status, root.error_code = "skipped", "late" if late else "policy"
                    continue
                text = "Напоминание: " + job.text
                if current > job.due_at:
                    text = "С задержкой. " + text
                try:
                    with span("telegram", "send"):
                        async with asyncio.timeout(10):
                            message_id = await self._sender.send(job.destination_chat_id, text)
                except (TemporaryDeliveryError, TimeoutError) as error:
                    root.status = "retry" if job.attempts < 3 else "failed"
                    root.error_code = "temporary"
                    await self._store.failed(
                        job,
                        self._clock(),
                        temporary=True,
                        retry_after=getattr(error, "retry_after", 0),
                    )
                except PermanentDeliveryError:
                    root.status, root.error_code = "failed", "permanent"
                    await self._store.failed(job, self._clock(), temporary=False)
                except Exception:
                    root.status, root.error_code = "failed", "internal"
                    await self._store.failed(job, self._clock(), temporary=False, code="internal")
                else:
                    # Если запись результата не удалась, lease обеспечит восстановление.
                    # Telegram уже мог отправить: при повторе остаётся риск дубля.
                    await self._store.sent(job, message_id, self._clock())
                    sent += 1
        return sent

    async def run(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as error:
                logger.warning("Reminder delivery iteration failed: %s", type(error).__name__)
            await asyncio.sleep(5)
