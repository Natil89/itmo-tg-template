"""Ежедневная сводка: одно постоянное событие на владельца и локальную дату."""

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time, timedelta
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo

from app.briefing_sources import BriefingSources, Section, bounded
from app.delivery import DB_ERRORS, PermanentDeliveryError, Sender, TemporaryDeliveryError, instant
from app.notifications import (
    NotificationError,
    NotificationProfile,
    NotificationStore,
    profile_from_row,
)
from app.tracing import Operation, TraceStore, recorded, span

logger = logging.getLogger(__name__)
GRACE = timedelta(hours=1)
LEASE = timedelta(seconds=60)
SCHEMA = """
CREATE TABLE IF NOT EXISTS briefing_deliveries (
    event_id TEXT PRIMARY KEY,
    actor_id BIGINT NOT NULL REFERENCES notification_profiles(actor_id),
    local_date DATE NOT NULL,
    destination_chat_id BIGINT NOT NULL,
    profile_revision BIGINT NOT NULL,
    timezone TEXT NOT NULL,
    due_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK(status IN ('pending','sending','retry','sent','missed','failed')),
    body TEXT CHECK(length(body) <= 4000),
    attempts INTEGER NOT NULL DEFAULT 0 CHECK(attempts BETWEEN 0 AND 3),
    next_attempt_at TIMESTAMPTZ NOT NULL,
    lease_id UUID, lease_until TIMESTAMPTZ, sent_at TIMESTAMPTZ, telegram_message_id BIGINT,
    error_code TEXT CHECK(error_code IN
        ('late','profile_changed','temporary','permanent','attempts_exhausted','internal')),
    UNIQUE(actor_id,local_date),
    CHECK(status <> 'sending' OR (lease_id IS NOT NULL AND lease_until IS NOT NULL)),
    CHECK(status <> 'sent' OR (sent_at IS NOT NULL AND telegram_message_id IS NOT NULL))
);
"""


def daily_due(now: datetime, timezone: str) -> tuple[date, datetime]:
    local = instant(now).astimezone(ZoneInfo(timezone))
    return local.date(), datetime.combine(local.date(), time(8), local.tzinfo).astimezone(UTC)


@dataclass(frozen=True)
class BriefingJob:
    event_id: str
    day: date
    due_at: datetime
    profile: NotificationProfile
    attempts: int
    lease_id: UUID
    lease_until: datetime
    body: str | None


class BriefingStore:
    def __init__(self, profiles: NotificationStore) -> None:
        self.profiles = profiles

    async def initialize(self) -> None:
        try:
            await self.profiles.pool.execute(SCHEMA)
        except DB_ERRORS:
            raise NotificationError("Не удалось подготовить хранилище сводки.") from None

    async def reserve(self, profile: NotificationProfile, now: datetime) -> bool:
        if not profile.enabled or profile.timezone is None:
            return False
        day, due = daily_due(now, profile.timezone)
        if not due <= instant(now) <= due + GRACE:
            return False
        async with self.profiles.connection() as connection:
            value = await connection.fetchval(
                "INSERT INTO briefing_deliveries(event_id,actor_id,local_date,destination_chat_id,"
                "profile_revision,timezone,due_at,next_attempt_at) "
                "SELECT $1,p.actor_id,$2,p.destination_chat_id,p.revision,s.timezone,$3,$3 "
                "FROM notification_profiles p JOIN reminder_user_settings s "
                "ON s.user_id=p.actor_id "
                "WHERE p.actor_id=$4 AND p.briefing_enabled AND p.revision=$5 AND s.timezone=$6 "
                "AND p.destination_chat_id=$7 ON CONFLICT DO NOTHING RETURNING event_id",
                f"briefing:{profile.actor_id}:{day.isoformat()}",
                day,
                due,
                profile.actor_id,
                profile.revision,
                profile.timezone,
                profile.destination_chat_id,
            )
        return value is not None

    async def claim(self, now: datetime) -> BriefingJob | None:
        now = instant(now)
        async with self.profiles.connection() as connection, connection.transaction():
            await connection.execute(
                "UPDATE briefing_deliveries SET status='missed',error_code='late',"
                "lease_id=NULL,lease_until=NULL WHERE due_at < $1 AND "
                "(status IN ('pending','retry') OR (status='sending' AND lease_until <= $2))",
                now - GRACE,
                now,
            )
            await connection.execute(
                "UPDATE briefing_deliveries SET status='failed',error_code='attempts_exhausted',"
                "lease_id=NULL,lease_until=NULL "
                "WHERE status='sending' AND lease_until <= $1 AND attempts=3",
                now,
            )
            row = await connection.fetchrow(
                "SELECT j.*,p.city,p.briefing_enabled,p.revision AS current_revision,"
                "s.timezone AS current_timezone,p.destination_chat_id AS current_destination "
                "FROM briefing_deliveries j JOIN notification_profiles p ON p.actor_id=j.actor_id "
                "JOIN reminder_user_settings s ON s.user_id=j.actor_id "
                "WHERE p.briefing_enabled AND p.revision=j.profile_revision "
                "AND p.destination_chat_id=j.destination_chat_id AND s.timezone=j.timezone "
                "AND j.due_at <= $1 AND j.due_at >= $2 AND j.attempts < 3 AND "
                "((j.status IN ('pending','retry') AND j.next_attempt_at <= $1) "
                "OR (j.status='sending' AND j.lease_until <= $1)) "
                "ORDER BY j.due_at,j.event_id LIMIT 1 FOR UPDATE OF j SKIP LOCKED",
                now,
                now - GRACE,
            )
            if row is None:
                return None
            lease = uuid4()
            await connection.execute(
                "UPDATE briefing_deliveries SET status='sending',attempts=attempts+1,lease_id=$2,"
                "lease_until=$3,error_code=NULL WHERE event_id=$1",
                row["event_id"],
                lease,
                now + LEASE,
            )
            profile = profile_from_row(
                {
                    "actor_id": row["actor_id"],
                    "destination_chat_id": row["current_destination"],
                    "city": row["city"],
                    "timezone": row["current_timezone"],
                    "briefing_enabled": row["briefing_enabled"],
                    "revision": row["current_revision"],
                }
            )
            return BriefingJob(
                row["event_id"],
                row["local_date"],
                row["due_at"],
                profile,
                row["attempts"] + 1,
                lease,
                now + LEASE,
                row["body"],
            )

    async def valid(self, job: BriefingJob, now: datetime) -> bool:
        profile = job.profile
        async with self.profiles.connection() as connection:
            return bool(
                await connection.fetchval(
                    "SELECT EXISTS(SELECT 1 FROM briefing_deliveries j "
                    "JOIN notification_profiles p ON p.actor_id=j.actor_id "
                    "JOIN reminder_user_settings s ON s.user_id=j.actor_id "
                    "WHERE j.event_id=$1 AND j.actor_id=$2 AND j.destination_chat_id=$3 "
                    "AND j.profile_revision=$4 AND j.timezone=$5 AND p.briefing_enabled "
                    "AND p.revision=j.profile_revision "
                    "AND p.destination_chat_id=j.destination_chat_id "
                    "AND s.timezone=j.timezone AND j.lease_id=$6 AND j.status='sending' "
                    "AND j.lease_until > $7 AND j.due_at <= $7 AND j.due_at >= $8 "
                    "AND j.local_date=$9 AND j.due_at=$10 AND j.lease_until=$11 "
                    "AND p.city=$12::jsonb AND j.body IS NOT DISTINCT FROM $13)",
                    job.event_id,
                    profile.actor_id,
                    profile.destination_chat_id,
                    profile.revision,
                    profile.timezone,
                    job.lease_id,
                    instant(now),
                    instant(now) - GRACE,
                    job.day,
                    job.due_at,
                    job.lease_until,
                    profile.city.model_dump_json(),
                    job.body,
                )
            )

    async def save_body(self, job: BriefingJob, body: str) -> BriefingJob:
        if len(body.encode("utf-16-le")) > 7600:
            raise ValueError("Сводка слишком длинная.")
        async with self.profiles.connection() as connection:
            await connection.execute(
                "UPDATE briefing_deliveries SET body=$3 WHERE event_id=$1 AND lease_id=$2 "
                "AND status='sending' AND body IS NULL",
                job.event_id,
                job.lease_id,
                body,
            )
        return replace(job, body=body)

    async def finish(
        self,
        job: BriefingJob,
        now: datetime,
        *,
        message_id: int | None = None,
        code: str | None = None,
        temporary: bool = False,
        retry_after: int = 0,
    ) -> None:
        if message_id is not None and (type(message_id) is not int or message_id <= 0):
            raise ValueError("Некорректный результат Telegram.")
        if code not in {None, "late", "profile_changed", "temporary", "permanent", "internal"}:
            raise ValueError("Неизвестный безопасный код ошибки.")
        retry = temporary and job.attempts < 3
        status = (
            "sent" if message_id else "missed" if code == "late" else "retry" if retry else "failed"
        )
        if temporary and not retry:
            code = "attempts_exhausted"
        delay = max((30, 120)[min(job.attempts - 1, 1)], retry_after)
        async with self.profiles.connection() as connection:
            await connection.execute(
                "UPDATE briefing_deliveries SET status=$3,error_code=$4,telegram_message_id=$5,"
                "sent_at=$6,next_attempt_at=$7,lease_id=NULL,lease_until=NULL "
                "WHERE event_id=$1 AND lease_id=$2 AND status='sending'",
                job.event_id,
                job.lease_id,
                status,
                code,
                message_id,
                instant(now) if message_id else None,
                instant(now) + timedelta(seconds=delay),
            )


class BriefingService:
    def __init__(
        self,
        store: BriefingStore,
        sources: BriefingSources,
        sender: Sender,
        *,
        clock: Callable[[], datetime] | None = None,
        section_timeout: float = 5,
        trace_store: TraceStore | None = None,
    ) -> None:
        if not 0 < section_timeout <= 10:
            raise ValueError("Нужен таймаут раздела до десяти секунд.")
        self.trace_store = trace_store
        self.store, self.sources, self.sender = store, sources, sender
        self.clock = clock or (lambda: datetime.now(UTC))
        self.section_timeout = section_timeout

    async def body(self, job: BriefingJob) -> str:
        async def section(title, operation: Operation, read):
            try:
                with span("briefing", operation) as step:
                    async with asyncio.timeout(self.section_timeout):
                        raw = await read
                        value = Section.model_validate(raw.model_dump())
                    if value.empty:
                        step.status = "not_found"
                return f"{title}\n{bounded(value.text)}"
            except TimeoutError:
                return f"{title}\nНедоступно: превышено время ожидания."
            except Exception:
                return f"{title}\nНедоступно: ошибка получения данных."

        profile = job.profile
        sections = await asyncio.gather(
            section("Занятия", "read_schedule", self.sources.schedule(profile.actor_id, job.day)),
            section(
                "Напоминания",
                "read_reminders",
                self.sources.reminders(profile.actor_id, job.day, profile.timezone),
            ),
            section("Погода", "read_weather", self.sources.weather(profile.city)),
            section("Учебный фокус", "read_focus", self.sources.focus(job.day)),
        )
        prepared = instant(self.clock()).astimezone(ZoneInfo(profile.timezone))
        return (
            f"Сводка на {job.day:%d.%m.%Y}\n"
            f"Подготовлено {prepared:%H:%M:%S} ({profile.timezone})\n\n" + "\n\n".join(sections)
        )

    async def run_once(self) -> int:
        for profile in await self.store.profiles.active():
            await self.store.reserve(profile, self.clock())
        sent = 0
        for _ in range(100):
            job = await self.store.claim(self.clock())
            if job is None:
                break
            async with recorded(
                self.trace_store,
                job.profile.actor_id,
                "daily_briefing",
                "briefing",
                "delivery_attempt",
            ) as root:
                if job.body is None:
                    job = await self.store.save_body(job, await self.body(job))
                valid = await self.store.valid(job, self.clock())
                current = instant(self.clock())
                if (
                    not valid
                    or current > job.due_at + GRACE
                    or current >= job.lease_until
                    or current < job.due_at
                ):
                    late = current > job.due_at + GRACE
                    expired = current >= job.lease_until
                    await self.store.finish(
                        job,
                        current,
                        code="late" if late else "temporary" if expired else "profile_changed",
                        temporary=expired and not late,
                    )
                    root.status, root.error_code = "skipped", "late" if late else "policy"
                    continue
                try:
                    with span("telegram", "send"):
                        async with asyncio.timeout(10):
                            message_id = await self.sender.send(
                                job.profile.destination_chat_id, job.body
                            )
                except (TemporaryDeliveryError, TimeoutError) as error:
                    root.status = "retry" if job.attempts < 3 else "failed"
                    root.error_code = "temporary"
                    await self.store.finish(
                        job,
                        self.clock(),
                        code="temporary",
                        temporary=True,
                        retry_after=getattr(error, "retry_after", 0),
                    )
                except PermanentDeliveryError:
                    root.status, root.error_code = "failed", "permanent"
                    await self.store.finish(job, self.clock(), code="permanent")
                except Exception:
                    root.status, root.error_code = "failed", "internal"
                    await self.store.finish(job, self.clock(), code="internal")
                else:
                    await self.store.finish(job, self.clock(), message_id=message_id)
                    sent += 1
        return sent

    async def run(self) -> None:
        while True:
            try:
                await self.run_once()
            except Exception as error:
                logger.warning("Briefing iteration failed: %s", type(error).__name__)
            await asyncio.sleep(15)
