import json
from unittest.mock import AsyncMock, MagicMock, Mock

import aiohttp
import pytest
from aiogram.filters import CommandObject

from app.handlers.assistant import timezone
from app.reminders import ReminderError
from app.timezones import GEOCODING_URL, TimezoneChoice, TimezoneResolver


def city(name="Москва", zone="Europe/Moscow", population=1_000_000, **extra):
    return {"name": name, "timezone": zone, "country": "Россия", "population": population, **extra}


def response(data, status=200, body=None):
    payload = json.dumps(data).encode() if body is None else body
    result = MagicMock(status=status, content_length=len(payload))
    result.content.read = AsyncMock(side_effect=[payload, b""])
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=result)
    context.__aexit__ = AsyncMock(return_value=False)
    return context


@pytest.mark.parametrize("zone", ["Europe/Moscow", "Asia/Yerevan", "Asia/Tokyo", "UTC"])
async def test_iana_zone_works_offline(zone):
    session = Mock()
    assert await TimezoneResolver(session).resolve(zone) == TimezoneChoice(zone)
    session.get.assert_not_called()


@pytest.mark.parametrize(
    "name,zone", [("Казань", "Europe/Moscow"), ("Ереван", "Asia/Yerevan"), ("Токио", "Asia/Tokyo")]
)
async def test_general_lookup_does_not_use_hardcoded_city_list(name, zone):
    session = Mock(get=Mock(return_value=response({"results": [city(name, zone)]})))
    selected = await TimezoneResolver(session).resolve(name)
    assert selected.timezone == zone and name in selected.place
    assert session.get.call_args.args == (GEOCODING_URL,)
    assert session.get.call_args.kwargs["params"]["name"] == name
    assert session.get.call_args.kwargs["allow_redirects"] is False


async def test_exact_city_wins_over_prefix_and_largest_known_leader_is_shown():
    data = {
        "results": [
            city("Еревани", "Europe/Moscow", 4_000_000),
            city("Ереван", "Asia/Yerevan", 1_000_000),
            city("Ереван", "Asia/Baku", 100),
        ]
    }
    selected = TimezoneResolver._choose(data, "Ереван")
    assert selected.timezone == "Asia/Yerevan"


async def test_missing_population_of_small_namesake_does_not_block_large_city():
    selected = TimezoneResolver._choose(
        {"results": [city("Казань"), city("Казань", "Europe/Kirov", None)]}, "Казань"
    )
    assert selected.timezone == "Europe/Moscow"


@pytest.mark.parametrize("populations", [(100_000, 90_000), (0, 0), (5000, 0)])
async def test_real_ambiguity_requests_country_or_region(populations):
    with pytest.raises(ReminderError, match="Уточните страну или регион"):
        TimezoneResolver._choose(
            {
                "results": [
                    city("Город", "Europe/Moscow", populations[0]),
                    city("Город", "Asia/Tokyo", populations[1]),
                ]
            },
            "Город",
        )


def test_same_zone_for_different_places_is_unambiguous_for_timezone():
    selected = TimezoneResolver._choose(
        {"results": [city("Город", population=0), city("Город", population=0, admin1="Область")]},
        "Город",
    )
    assert selected.timezone == "Europe/Moscow"


async def test_country_qualifier_is_sent_to_source():
    session = Mock(get=Mock(return_value=response({"results": [city("Париж", "Europe/Paris")]})))
    assert (await TimezoneResolver(session).resolve("Париж, Франция")).timezone == "Europe/Paris"
    assert session.get.call_args.kwargs["params"]["name"] == "Париж, Франция"


@pytest.mark.parametrize(
    "data",
    [
        {"results": []},
        {"results": "invalid"},
        {"results": [city(zone="Invalid/Zone")]},
        {"results": [None]},
        {"error": True},
    ],
)
async def test_invalid_or_missing_results_have_safe_error(data):
    session = Mock(get=Mock(return_value=response(data)))
    with pytest.raises(ReminderError):
        await TimezoneResolver(session).resolve("Город")


async def test_network_failure_is_safe_and_does_not_use_arbitrary_urls():
    session = Mock(get=Mock(side_effect=aiohttp.ClientConnectionError("private-proxy")))
    with pytest.raises(ReminderError) as error:
        await TimezoneResolver(session).resolve("Город")
    assert "private-proxy" not in str(error.value)
    assert session.get.call_count == 2


async def test_oversized_response_is_rejected():
    session = Mock(get=Mock(return_value=response({}, body=b"x" * 512_001)))
    with pytest.raises(ReminderError):
        await TimezoneResolver(session).resolve("Город")


async def test_city_command_saves_only_resolved_iana_and_shows_selected_place():
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(set_timezone=AsyncMock())
    resolver = Mock(resolve=AsyncMock(return_value=TimezoneChoice("Asia/Tokyo", "Токио, Япония")))
    await timezone(message, CommandObject(command="timezone", args="Токио"), store, resolver)
    store.set_timezone.assert_awaited_once_with(101, "Asia/Tokyo")
    answer = message.answer.call_args.args[0]
    assert "Токио, Япония" in answer and "Asia/Tokyo" in answer and "Сейчас в нём" in answer


async def test_ambiguous_command_does_not_overwrite_saved_timezone():
    message = Mock(from_user=Mock(id=101), answer=AsyncMock())
    store = Mock(set_timezone=AsyncMock())
    resolver = Mock(resolve=AsyncMock(side_effect=ReminderError("Уточните страну или регион.")))
    await timezone(message, CommandObject(command="timezone", args="Город"), store, resolver)
    store.set_timezone.assert_not_awaited()
    assert "Уточните" in message.answer.call_args.args[0]
