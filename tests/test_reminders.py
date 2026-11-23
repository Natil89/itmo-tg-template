from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock
from uuid import UUID
from zoneinfo import ZoneInfo

import asyncpg
import pytest
from pydantic import ValidationError

from app.reminders import ReminderDraft, ReminderError, ReminderStore

NOW = datetime(2026, 11, 16, 9, tzinfo=UTC)
ACTION = UUID("b32d664d-e5c7-40af-8bb7-8884654391c8")


def draft(**changes):
    return ReminderDraft.model_validate(
        {
            "text": "Сдать лабораторную",
            "remind_at": "2026-11-17T09:00:00+03:00",
            **changes,
        }
    )


def record(*, status="prepared", payload=None, expires_at=None, timezone="Europe/Moscow"):
    return {
        "action_id": ACTION,
        "user_id": 42,
        "update_id": 11,
        "payload": (payload or draft()).model_dump_json(),
        "timezone": timezone,
        "expires_at": expires_at or NOW + timedelta(minutes=5),
        "status": status,
        "reminder_id": 7 if status == "executed" else None,
    }


def store_fixture():
    connection = Mock(fetchrow=AsyncMock(), fetchval=AsyncMock(), execute=AsyncMock())
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock()
    transaction.__aexit__ = AsyncMock(return_value=False)
    connection.transaction.return_value = transaction
    acquisition = MagicMock()
    acquisition.__aenter__ = AsyncMock(return_value=connection)
    acquisition.__aexit__ = AsyncMock(return_value=False)
    pool = Mock(acquire=Mock(return_value=acquisition), execute=AsyncMock())
    return ReminderStore(pool), pool, connection


def test_draft_strips_text_and_accepts_explicit_iso_offset():
    assert draft(text="  Сдать лабораторную \n").text == "Сдать лабораторную"
    assert draft(remind_at="2026-11-17T09:00Z").remind_at == "2026-11-17T09:00Z"


@pytest.mark.parametrize(
    "changes",
    [
        {"text": " \n "},
        {"text": "x" * 501},
        {"text": 42},
        {"remind_at": "2026-11-17T09:00"},
        {"remind_at": "2026-02-30T09:00+03:00"},
        {"remind_at": "20261117T090000+0300"},
        {"remind_at": 42},
        {"owner": 42},
    ],
)
def test_draft_rejects_invalid_or_untrusted_fields(changes):
    with pytest.raises(ValidationError):
        draft(**changes)


async def test_timezone_is_validated_before_database_and_never_guessed():
    store, pool, _ = store_fixture()
    with pytest.raises(ReminderError, match="IANA"):
        await store.set_timezone(42, "Unknown/Zone")
    pool.acquire.assert_not_called()


async def test_timezone_query_and_upsert_are_scoped_to_owner():
    store, _, connection = store_fixture()
    connection.fetchval.return_value = None
    assert await store.get_timezone(42) is None
    assert connection.fetchval.call_args.args[1:] == (42,)
    await store.set_timezone(42, "Europe/Moscow")
    assert connection.execute.call_args.args[1:] == (42, "Europe/Moscow")


async def test_prepare_requires_saved_zone_and_does_not_insert_without_it():
    store, _, connection = store_fixture()
    connection.fetchrow.return_value = None
    connection.fetchval.return_value = None
    with pytest.raises(ReminderError, match="/timezone"):
        await store.prepare(42, 11, draft(), now=NOW)
    assert connection.fetchrow.await_count == 1


async def test_prepare_normalizes_time_and_has_five_minute_ttl():
    store, _, connection = store_fixture()
    connection.fetchval.return_value = "Europe/Moscow"
    connection.fetchrow.side_effect = [None, record()]
    pending = await store.prepare(42, 11, draft(remind_at="2026-11-17T09:00+03:00"), now=NOW)
    assert pending.payload.remind_at == "2026-11-17T09:00:00+03:00"
    assert pending.timezone == "Europe/Moscow"
    assert pending.expires_at == NOW + timedelta(minutes=5)
    assert pending.status == "prepared"
    assert connection.fetchrow.call_args.args[-1] == NOW + timedelta(minutes=5)


@pytest.mark.parametrize(
    "zone,when,error",
    [
        ("Europe/Moscow", "2026-11-17T09:00:00+02:00", "Смещение"),
        ("Europe/Moscow", "2026-11-16T09:00:00+03:00", "будущее"),
        ("America/New_York", "2026-03-08T02:30:00-05:00", "неоднозначно или не существует"),
        ("America/New_York", "2026-11-01T01:30:00-04:00", "неоднозначно или не существует"),
        ("America/New_York", "2026-11-01T01:30:00-05:00", "неоднозначно или не существует"),
    ],
)
async def test_prepare_rejects_wrong_offset_past_gap_and_fold(zone, when, error):
    store, _, connection = store_fixture()
    connection.fetchrow.return_value = None
    connection.fetchval.return_value = zone
    clock = NOW if zone == "Europe/Moscow" else datetime(2026, 1, 1, tzinfo=UTC)
    with pytest.raises(ReminderError, match=error):
        await store.prepare(42, 11, draft(remind_at=when), now=clock)
    assert connection.fetchrow.await_count == 1


async def test_duplicate_update_preserves_snapshot_zone_and_payload():
    store, _, connection = store_fixture()
    connection.fetchrow.return_value = record()
    pending = await store.prepare(42, 11, draft(), now=NOW)
    assert pending.action_id == str(ACTION)
    connection.fetchval.assert_not_called()
    with pytest.raises(ReminderError, match="другим напоминанием"):
        await store.prepare(42, 11, draft(text="Изменено"), now=NOW)


async def test_foreign_action_is_not_read_confirmed_cancelled_or_executed():
    store, _, connection = store_fixture()
    connection.fetchrow.return_value = None
    assert await store.get_pending(202, str(ACTION), now=NOW) is None
    with pytest.raises(ReminderError):
        await store.confirm(202, str(ACTION), now=NOW)
    assert not await store.cancel(202, str(ACTION), now=NOW)
    with pytest.raises(ReminderError):
        await store.execute(202, str(ACTION), draft(), now=NOW)
    assert all(call.args[1:] == (202, ACTION) for call in connection.fetchrow.call_args_list)
    connection.execute.assert_not_called()


async def test_execute_requires_confirmation_and_exact_payload():
    store, _, connection = store_fixture()
    connection.fetchrow.return_value = record()
    with pytest.raises(ReminderError, match="подтвердите"):
        await store.execute(42, str(ACTION), draft(), now=NOW)
    connection.execute.assert_not_called()
    connection.fetchrow.return_value = record(status="confirmed")
    with pytest.raises(ReminderError, match="Параметры изменились"):
        await store.execute(42, str(ACTION), draft(text="Другое"), now=NOW)


async def test_expired_confirmation_is_marked_expired_and_rejected():
    store, _, connection = store_fixture()
    connection.fetchrow.side_effect = [record(), record(status="expired")]
    with pytest.raises(ReminderError, match="просрочено"):
        await store.confirm(42, str(ACTION), now=NOW + timedelta(minutes=5))
    assert "status='expired'" in connection.fetchrow.call_args.args[0]


async def test_future_is_rechecked_at_execution_before_insert():
    store, _, connection = store_fixture()
    when = NOW + timedelta(seconds=30)
    payload = draft(remind_at=when.astimezone(ZoneInfo("Europe/Moscow")).isoformat())
    connection.fetchrow.return_value = record(status="confirmed", payload=payload)
    with pytest.raises(ReminderError, match="будущее"):
        await store.execute(42, str(ACTION), payload, now=NOW + timedelta(seconds=31))
    assert connection.fetchrow.await_count == 1
    connection.execute.assert_not_called()


async def test_repeated_execution_returns_prior_result_even_after_target_time():
    store, _, connection = store_fixture()
    saved = {
        "id": 7,
        "text": draft().text,
        "remind_at": datetime(2026, 11, 17, 6, tzinfo=UTC),
        "timezone": "Europe/Moscow",
    }
    connection.fetchrow.side_effect = [record(status="executed"), saved]
    result = await store.execute(42, str(ACTION), draft(), now=NOW + timedelta(days=3))
    assert result.reminder_id == 7
    assert result.remind_at == "2026-11-17T09:00:00+03:00"
    assert result.timezone == "Europe/Moscow"
    assert not result.created
    connection.execute.assert_not_called()


async def test_database_errors_are_safe():
    store, pool, _ = store_fixture()
    pool.execute.side_effect = asyncpg.PostgresError("private-password")
    with pytest.raises(ReminderError) as error:
        await store.initialize()
    assert "private-password" not in str(error.value)


@pytest.mark.parametrize("owner", [None, True, 0, -1, "42"])
async def test_invalid_owner_is_rejected_before_sql(owner):
    store, pool, _ = store_fixture()
    with pytest.raises(ReminderError, match="владельца"):
        await store.get_timezone(owner)
    pool.acquire.assert_not_called()
