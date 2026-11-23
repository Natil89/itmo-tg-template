from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock

import asyncpg
import pytest
from pydantic import ValidationError

from app.personal_schedule import LessonDraft, PersonalScheduleError, PersonalScheduleStore


def draft(**changes):
    return LessonDraft.model_validate(
        {
            "date": "2026-11-17",
            "start": "09:00",
            "end": "10:30",
            "title": "Программирование",
            "location": "Аудитория A",
            **changes,
        }
    )


def test_draft_has_strict_date_and_normalizes_visible_text():
    assert draft(title="  Алгебра  ", location="  A1  ").model_dump() == {
        "date": "2026-11-17",
        "start": "09:00",
        "end": "10:30",
        "title": "Алгебра",
        "location": "A1",
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"date": "2026-02-30"},
        {"date": "20261117"},
        {"date": 20261117},
        {"start": "9:00"},
        {"end": "09:00"},
        {"end": "08:00"},
        {"title": " \n "},
        {"location": " \t "},
        {"owner": 123},
    ],
)
def test_draft_rejects_invalid_dates_times_empty_fields_and_extra_owner(changes):
    with pytest.raises(ValidationError):
        draft(**changes)


@pytest.mark.parametrize("owner", [None, True, 0, -1, "42"])
async def test_untrusted_owner_is_rejected_before_database(owner):
    pool = Mock()
    store = PersonalScheduleStore(pool)
    with pytest.raises(PersonalScheduleError, match="владельца"):
        await store.for_date(owner, "2026-11-17")
    pool.acquire.assert_not_called()


async def test_naive_clock_and_unknown_zone_are_rejected_before_database():
    pool = Mock()
    store = PersonalScheduleStore(pool)
    with pytest.raises(PersonalScheduleError, match="часовой пояс"):
        await store.prepare(42, 1, draft(), now=datetime(2026, 11, 16))
    with pytest.raises(PersonalScheduleError, match="IANA"):
        await store.current_week(42, "Unknown/Zone", now=datetime(2026, 11, 16, tzinfo=UTC))
    pool.acquire.assert_not_called()


async def test_invalid_action_identifiers_do_not_query_or_expose_other_users():
    pool = Mock()
    store = PersonalScheduleStore(pool)
    assert await store.get_pending(42, "not-a-uuid") is None
    assert not await store.cancel(42, "not-a-uuid")
    with pytest.raises(PersonalScheduleError, match="не найдено"):
        await store.confirm(42, "not-a-uuid")
    with pytest.raises(PersonalScheduleError, match="не найдено"):
        await store.execute(42, "not-a-uuid", draft())
    pool.acquire.assert_not_called()


async def test_database_error_has_safe_message_without_connection_details():
    pool = Mock(execute=AsyncMock(side_effect=asyncpg.PostgresError("private-password")))
    with pytest.raises(PersonalScheduleError) as error:
        await PersonalScheduleStore(pool).initialize()
    assert "private-password" not in str(error.value)


async def test_constructed_unvalidated_draft_is_revalidated_before_sql():
    pool = Mock()
    invalid = LessonDraft.model_construct(
        date="2026-02-30",
        start="09:00",
        end="10:00",
        title="A",
        location="B",
    )
    with pytest.raises(PersonalScheduleError):
        await PersonalScheduleStore(pool).prepare(42, 1, invalid)
    pool.acquire.assert_not_called()
