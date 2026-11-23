import json
from datetime import datetime

import pytest

from app.reminders import ReminderDraft
from scripts.evaluate_lab2 import DATASET, FixtureMcp, MemoryReminders, arguments_correct, metrics


def cases():
    return json.loads(DATASET.read_text(encoding="utf-8"))["cases"]


def test_expected_arguments_pass_and_tool_denominator_is_nine():
    rows = []
    for case in cases():
        result = arguments_correct(case, case["expected_action"], case["expected_arguments"])
        assert result is (True if case["expected_arguments"] else None)
        rows.append({"routing_correct": True, "arguments_correct": result})
    assert metrics(rows) == {
        "routing_correct": 12,
        "routing_total": 12,
        "arguments_correct": 9,
        "arguments_total": 9,
    }


def test_missing_tool_does_not_shrink_argument_denominator():
    rows = [
        {"routing_correct": False, "arguments_correct": arguments_correct(case, "undetermined", {})}
        for case in cases()
    ]
    assert metrics(rows) == {
        "routing_correct": 0,
        "routing_total": 12,
        "arguments_correct": 0,
        "arguments_total": 9,
    }


@pytest.mark.parametrize("value", ["Тбилиси Грузия", "Тбилиси, Грузия.", "Тбилиси"])
def test_country_separator_and_qualifier_are_significant(value):
    case = cases()[2]
    assert arguments_correct(case, "get_weather", {"city": value}) is False


def test_case_and_spacing_do_not_change_city_identity():
    assert arguments_correct(cases()[2], "get_weather", {"city": " тбилиси ,   грузия "}) is True


def test_reminder_text_case_yo_and_punctuation_are_insignificant():
    case = cases()[6]
    assert (
        arguments_correct(
            case,
            "add_reminder",
            {
                **case["expected_arguments"],
                "text": "Отправить отчет.",
            },
        )
        is True
    )


@pytest.mark.parametrize(
    "value",
    [
        "2026-10-13T15:30:00+00:00",
        "2026-10-13 18:30:00+03:00",
        "2026-10-13T18:30:00",
    ],
)
def test_reminder_requires_saved_zone_offset_and_strict_iso(value):
    case = cases()[6]
    assert (
        arguments_correct(
            case,
            "add_reminder",
            {
                **case["expected_arguments"],
                "remind_at": value,
            },
        )
        is False
    )


def test_extra_fields_or_different_action_fail():
    case = cases()[3]
    assert arguments_correct(case, "get_weather", case["expected_arguments"]) is False
    assert arguments_correct(case, "get_schedule", {"date": "2026-10-12", "user_id": 1}) is False
    assert arguments_correct(case, "get_schedule", {"date": "2026-10-12."}) is False


async def test_harness_never_executes_write_tools():
    mcp = FixtureMcp([], datetime.fromisoformat("2026-10-12T09:00:00+03:00"))
    with pytest.raises(RuntimeError, match="запрещена"):
        await mcp.call_tool("add_reminder", {"text": "test"})


async def test_all_three_reminders_are_only_prepared_in_memory_at_fixed_time():
    store = MemoryReminders("Europe/Moscow")
    now = datetime.fromisoformat("2026-10-12T09:00:00+03:00")
    for index, case in enumerate(cases()[6:9], 1):
        pending = await store.prepare(
            1, index, ReminderDraft(**case["expected_arguments"]), now=now
        )
        assert pending.status == "prepared"
    assert len(store.prepared) == 3
