import json
from datetime import UTC, datetime, timedelta, timezone
from unittest.mock import AsyncMock, Mock

import asyncpg
import pytest
from pydantic import ValidationError

from app.audit import (
    AuditError,
    AuditRecord,
    AuditStep,
    AuditStore,
    AuditTrace,
    render_audit,
    safe_arguments,
)

NOW = datetime(2026, 11, 17, 9, 30, tzinfo=UTC)


def test_summaries_keep_only_public_fields_and_lengths_without_private_text():
    assert safe_arguments(
        "get_schedule",
        {
            "date": "2026-11-18",
            "owner": 42,
            "path": "private-path",
        },
    ) == {"date": "2026-11-18"}
    assert safe_arguments(
        "add_lesson",
        {
            "date": "2026-11-18",
            "start": "09:00",
            "end": "10:30",
            "title": "Секретное название",
            "location": "Закрытая аудитория",
            "action_id": "private",
        },
    ) == {
        "date": "2026-11-18",
        "start": "09:00",
        "end": "10:30",
        "title_length": len("Секретное название"),
        "location_length": len("Закрытая аудитория"),
    }
    summary = safe_arguments(
        "add_reminder",
        {
            "text": "Секретная заметка",
            "remind_at": "2026-11-18T09:00+03:00",
            "owner": 42,
        },
    )
    assert summary == {
        "text_length": len("Секретная заметка"),
        "remind_at": "2026-11-18T09:00:00+03:00",
    }
    assert safe_arguments("add_reminder", summary) == summary


@pytest.mark.parametrize(
    "city",
    [
        "https://user:password@host",
        "Москва\nprivate-secret",
        "x" * 40,
        "sk-" + "s" * 50,
        "Москва; SELECT",
        "Москва\u202eprivate",
        42,
    ],
)
def test_weather_summary_rejects_urls_controls_and_token_like_strings(city):
    assert safe_arguments("get_weather", {"city": city, "user_id": 42}) == {}


@pytest.mark.parametrize(
    "city", ["Москва", "Санкт-Петербург, Россия", "St. Petersburg", "L’Haÿ-les-Roses"]
)
def test_weather_city_has_bounded_human_readable_summary(city):
    assert safe_arguments("get_weather", {"city": city}) == {"city": city}


def test_invalid_dates_times_and_unknown_fields_are_dropped():
    assert safe_arguments("get_schedule", {"date": "2026-02-30"}) == {}
    assert safe_arguments("get_schedule", {"date": "20261118"}) == {}
    assert safe_arguments("add_lesson", {"start": "24:00", "end": "private-secret"}) == {}
    assert safe_arguments("add_reminder", {"remind_at": "2026-11-18T09:00"}) == {}
    assert safe_arguments("shell", {"command": "private-secret"}) == {}
    assert safe_arguments("get_weather", "private-secret") == {}


def test_trace_strict_codes_unknown_tool_limit_and_updates():
    trace = AuditTrace(started_at=NOW)
    first = trace.add(
        "tool",
        tool="shell",
        arguments={"command": "private-secret"},
        validation="rejected",
        execution="not_run",
        reason="invalid_arguments",
    )
    assert first.tool is None and first.arguments == {}
    trace.update(execution="error")
    assert trace.steps[0].execution == "error"
    for _ in range(20):
        trace.add("respond", reason="no_tool_needed")
    assert len(trace.steps) == 10
    with pytest.raises(AuditError):
        trace.update(reason="private system prompt")
    with pytest.raises(ValidationError):
        AuditStep(action="respond", hidden_prompt="private-secret")


def test_step_references_can_be_updated_independently_for_multi_call_audit():
    trace = AuditTrace(started_at=NOW)
    first = trace.add("tool", tool="get_schedule", arguments={"date": "2026-11-18"})
    second = trace.add("tool", tool="get_weather", arguments={"city": "Москва"})
    first.validation = "passed"
    first.execution = "success"
    first.reason = "personal_schedule"
    second.validation = "rejected"
    second.execution = "error"
    second.reason = "city_not_found"
    assert [step.execution for step in trace.steps] == ["success", "error"]
    with pytest.raises(ValidationError):
        first.reason = "private-secret"


def test_duration_uses_monotonic_clock_and_is_nonnegative(monkeypatch):
    trace = AuditTrace(started_at=NOW)
    monkeypatch.setattr("app.audit.monotonic", lambda: trace._monotonic_start + 1.25)
    assert trace.duration_ms == 1250
    monkeypatch.setattr("app.audit.monotonic", lambda: trace._monotonic_start - 1)
    assert trace.duration_ms == 0


def test_render_all_steps_timezone_statuses_and_privacy():
    trace = AuditTrace(started_at=NOW)
    trace.add(
        "tool",
        tool="get_weather",
        arguments={"city": "Москва", "owner": 123},
        validation="passed",
        execution="success",
        reason="live_weather",
    )
    trace.add(
        "tool",
        tool="add_reminder",
        arguments={
            "text": "private-secret",
            "remind_at": "2026-11-18T09:00:00+03:00",
            "action_id": "hidden",
        },
        validation="passed",
        execution="awaiting_confirmation",
        reason="requires_confirmation",
    )
    rendered = render_audit(AuditRecord(NOW, 1250, trace.steps), timezone="Europe/Moscow")
    assert "17.11.2026 12:30:00 (Europe/Moscow)" in rendered
    assert "1250 мс" in rendered
    assert "get_weather" in rendered and "get_schedule" not in rendered
    assert "актуальные погодные данные" in rendered
    assert "ожидает подтверждения" in rendered
    assert "private-secret" not in rendered and "hidden" not in rendered
    assert "owner" not in rendered and "123" not in rendered


async def test_store_persists_only_safe_summary_and_redacts_values_not_keys():
    pool = Mock(execute=AsyncMock())
    store = AuditStore(pool, secrets=("Москва", "private-secret", "tool"))
    trace = AuditTrace(started_at=NOW)
    step = trace.add(
        "tool",
        tool="get_weather",
        arguments={"city": "Москва"},
        validation="passed",
        execution="success",
        reason="live_weather",
    )
    # Прямое изменение словаря тоже повторно проверяется перед записью.
    step.arguments["user_id"] = 123
    step.arguments["system_prompt"] = "private-secret"
    await store.save(42, "update:1", trace)
    arguments = pool.execute.call_args.args
    assert arguments[1:3] == (42, "update:1")
    assert "ON CONFLICT (user_id,request_key) DO NOTHING" in arguments[0]
    saved = json.loads(arguments[-1])
    assert saved[0]["action"] == "tool"
    assert saved[0]["tool"] == "get_weather"
    assert saved[0]["arguments"] == {}
    assert saved[0]["reason"] == "live_weather"
    assert "private-secret" not in arguments[-1] and "Москва" not in arguments[-1]


async def test_latest_is_owner_filtered_and_validates_safe_summary():
    step = AuditStep(
        action="tool",
        tool="get_schedule",
        arguments={"date": "2026-11-18"},
        validation="passed",
        execution="success",
        reason="personal_schedule",
    )
    pool = Mock(
        fetchrow=AsyncMock(
            return_value={
                "started_at": NOW,
                "duration_ms": 150,
                "steps": json.dumps([step.model_dump()]),
            }
        )
    )
    record = await AuditStore(pool).latest(42)
    assert pool.fetchrow.call_args.args[1:] == (42,)
    assert "WHERE user_id=$1" in pool.fetchrow.call_args.args[0]
    assert "ORDER BY started_at DESC,id DESC" in pool.fetchrow.call_args.args[0]
    assert record.started_at == NOW and record.duration_ms == 150
    assert record.steps[0].arguments == {"date": "2026-11-18"}
    pool.fetchrow.return_value = None
    assert await AuditStore(pool).latest(43) is None


async def test_latest_rejects_raw_reason_text_and_corrupt_data_safely():
    pool = Mock(
        fetchrow=AsyncMock(
            return_value={
                "started_at": NOW,
                "duration_ms": 10,
                "steps": [{"action": "respond", "reason": "private-secret"}],
            }
        )
    )
    with pytest.raises(AuditError) as error:
        await AuditStore(pool).latest(42)
    assert "private-secret" not in str(error.value)


@pytest.mark.parametrize("owner", [None, True, 0, -1, "42"])
async def test_owner_is_required_before_any_database_call(owner):
    pool = Mock()
    store = AuditStore(pool)
    with pytest.raises(AuditError):
        await store.latest(owner)
    with pytest.raises(AuditError):
        await store.save(owner, "update:1", AuditTrace(started_at=NOW))
    pool.fetchrow.assert_not_called()
    pool.execute.assert_not_called()


async def test_database_failure_does_not_leak_credentials():
    pool = Mock(
        execute=AsyncMock(side_effect=asyncpg.PostgresError("private-password")),
        fetchrow=AsyncMock(side_effect=asyncpg.PostgresError("private-password")),
    )
    store = AuditStore(pool)
    trace = AuditTrace(started_at=NOW)
    for operation in (store.initialize(), store.save(42, "update:1", trace), store.latest(42)):
        with pytest.raises(AuditError) as error:
            await operation
        assert "private-password" not in str(error.value)


def test_naive_timestamps_unknown_zone_and_negative_duration_are_rejected():
    with pytest.raises(AuditError):
        AuditTrace(started_at=datetime(2026, 11, 17))
    with pytest.raises(AuditError):
        AuditRecord(NOW, -1, [])
    with pytest.raises(AuditError):
        render_audit(AuditRecord(NOW, 0, []), timezone="Unknown/Zone")
    shifted = NOW.astimezone(timezone(timedelta(hours=3)))
    assert AuditTrace(started_at=shifted).started_at == NOW
