import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest

from app.weather import (
    GEOCODING_URL,
    WEATHER_URL,
    WeatherError,
    WeatherService,
)


def location(**changes):
    return {
        "name": "Москва",
        "country": "Россия",
        "admin1": "Москва",
        "latitude": 55.75,
        "longitude": 37.62,
        "timezone": "Europe/Moscow",
        **changes,
    }


def weather(**changes):
    return {
        "timezone": "Europe/Moscow",
        "utc_offset_seconds": 10800,
        "current_units": {
            "temperature_2m": "°C",
            "weather_code": "wmo code",
            "wind_speed_10m": "m/s",
            "time": "iso8601",
        },
        "current": {
            "time": "2026-11-16T12:00",
            "temperature_2m": -2.5,
            "weather_code": 3,
            "wind_speed_10m": 4.2,
        },
        **changes,
    }


def response(payload=None, *, body=None, status=200, failure=None, content_length=None):
    body = body if body is not None else json.dumps(payload, ensure_ascii=False).encode()
    result = MagicMock(status=status)
    result.content_length = len(body) if content_length is None else content_length
    result.content.read = AsyncMock(side_effect=[body, b""])
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=result, side_effect=failure)
    context.__aexit__ = AsyncMock(return_value=False)
    return context


def service(*responses, policy="strict"):
    session = MagicMock()
    session.get.side_effect = responses
    return WeatherService(session, location_policy=policy), session


async def test_current_weather_from_fixed_endpoints_and_normalized_city():
    # Arrange
    client, session = service(response({"results": [location()]}), response(weather()))
    # Act
    result = await client.get_weather("  Москва, Россия  ")
    # Assert
    assert result.model_dump() == {
        "status": "ok",
        "assumed_location": False,
        "candidates": [],
        "weather": {
            "city": "Москва",
            "country": "Россия",
            "admin1": "Москва",
            "temperature_c": -2.5,
            "weather_code": 3,
            "wind_speed": 4.2,
            "wind_unit": "m/s",
            "observed_at": "2026-11-16T12:00:00+03:00",
            "timezone": "Europe/Moscow",
            "source": "Open-Meteo",
        },
    }
    calls = session.get.call_args_list
    assert [call.args[0] for call in calls] == [GEOCODING_URL, WEATHER_URL]
    assert calls[0].kwargs["params"] == {
        "name": "Москва, Россия",
        "language": "ru",
        "count": 100,
    }
    assert calls[1].kwargs["params"]["current"] == ("temperature_2m,weather_code,wind_speed_10m")
    assert calls[1].kwargs["params"]["wind_speed_unit"] == "ms"
    assert all(call.kwargs["allow_redirects"] is False for call in calls)
    assert all(call.kwargs["timeout"].total == 3 for call in calls)


@pytest.mark.parametrize("payload", [{}, {"results": []}])
async def test_unknown_city_returns_not_found_without_weather_request(payload):
    client, session = service(response(payload))
    result = await client.get_weather("Неизвестный город")
    assert result.model_dump() == {
        "status": "not_found",
        "weather": None,
        "candidates": [],
        "assumed_location": False,
    }
    assert session.get.call_count == 1


async def test_population_policy_returns_weather_for_known_largest_exact_match_with_marker():
    small = location(admin1="Тверская область", latitude=57, population=100)
    large = location(population=13000000)
    client, session = service(
        response({"results": [small, large]}), response(weather()), policy="population"
    )
    result = await client.get_weather("Москва, Россия")
    assert result.status == "ok" and result.assumed_location
    assert result.weather.admin1 == "Москва"
    assert session.get.call_args_list[1].kwargs["params"]["latitude"] == large["latitude"]


@pytest.mark.parametrize("first,second", [(None, None), (1000, 1000), (0, 0)])
async def test_population_policy_still_clarifies_without_unique_known_leader(first, second):
    client, session = service(
        response(
            {
                "results": [
                    location(population=first),
                    location(admin1="Другой регион", latitude=57, population=second),
                ]
            }
        ),
        policy="population",
    )
    result = await client.get_weather("Москва")
    assert result.status == "ambiguous" and not result.assumed_location
    assert session.get.call_count == 1


async def test_population_policy_uses_available_known_leader_when_other_population_is_missing():
    client, _ = service(
        response(
            {
                "results": [
                    location(population=None, latitude=57),
                    location(population=1000),
                ]
            }
        ),
        response(weather()),
        policy="population",
    )
    result = await client.get_weather("Москва")
    assert result.status == "ok" and result.assumed_location


async def test_population_policy_does_not_turn_fuzzy_prefix_into_assumed_exact_city():
    client, session = service(
        response(
            {
                "results": [
                    location(name="Ереван", population=1000000),
                    location(name="Еревани", population=100, latitude=40.1),
                ]
            }
        ),
        policy="population",
    )
    assert (await client.get_weather("Ерев")).status == "ambiguous"
    assert session.get.call_count == 1


def test_invalid_location_policy_is_rejected():
    with pytest.raises(ValueError, match="режим"):
        WeatherService(MagicMock(), location_policy="random")


def test_assumption_flag_requires_successful_weather_data():
    from pydantic import ValidationError

    from app.weather import WeatherResult

    with pytest.raises(ValidationError):
        WeatherResult(status="not_found", assumed_location=True)


def test_weather_policy_configuration_and_default(tmp_path):
    from app.config import ConfigError, Settings

    env = {
        "BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
        "POSTGRES_PASSWORD": "fake",
    }
    path = tmp_path / "absent.env"
    assert Settings.load(path, environ=env).weather_location_policy == "population"
    assert (
        Settings.load(
            path, environ={**env, "WEATHER_LOCATION_POLICY": "strict"}
        ).weather_location_policy
        == "strict"
    )
    with pytest.raises(ConfigError, match="WEATHER_LOCATION_POLICY"):
        Settings.load(path, environ={**env, "WEATHER_LOCATION_POLICY": "random"})


async def test_ambiguous_city_returns_distinguishable_candidates_without_picking_first():
    client, session = service(
        response(
            {
                "results": [
                    location(),
                    location(country="США", admin1="Айдахо", latitude=46.73, longitude=-117),
                ]
            }
        )
    )
    result = await client.get_weather("Москва")
    assert result.status == "ambiguous"
    assert result.weather is None
    assert [item.model_dump() for item in result.candidates] == [
        {"city": "Москва", "country": "Россия", "admin1": "Москва"},
        {"city": "Москва", "country": "США", "admin1": "Айдахо"},
    ]
    assert session.get.call_count == 1


async def test_duplicate_location_is_not_an_ambiguity():
    client, _ = service(response({"results": [location(), location()]}), response(weather()))
    assert (await client.get_weather("Москва")).status == "ok"


@pytest.mark.parametrize("query", ["Ереван", "Ереван, Армения", "  ЕРЕВАН, Армения  "])
async def test_exact_yerevan_excludes_prefix_matches_even_with_larger_population(query):
    main = location(
        name="Ереван",
        country="Армения",
        admin1="Ереван",
        population=1144700,
        latitude=40.17765,
        longitude=44.5126,
        timezone="Asia/Yerevan",
    )
    nearby = location(
        name="Еревани",
        country="Армения",
        admin1="Ереван",
        latitude=40.11389,
        longitude=44.50708,
        population=2000000,
    )
    other = location(name="Ереванлы", country="Азербайджан", admin1="Барда", latitude=40.33226)
    client, session = service(
        response({"results": [nearby, other, main]}),
        response(
            weather(
                timezone="Asia/Yerevan",
                utc_offset_seconds=14400,
            )
        ),
    )
    result = await client.get_weather(query)
    assert result.status == "ok" and result.weather.city == "Ереван"
    assert result.weather.country == "Армения"
    assert session.get.call_args_list[1].kwargs["params"]["latitude"] == main["latitude"]


async def test_population_orders_genuine_namesakes_without_auto_selecting_largest():
    small = location(country="США", admin1="Айдахо", latitude=46.73, population=25000)
    large = location(population=13000000)
    client, session = service(response({"results": [small, large]}))
    result = await client.get_weather("Москва")
    assert result.status == "ambiguous" and result.weather is None
    assert [item.country for item in result.candidates] == ["Россия", "США"]
    assert session.get.call_count == 1


async def test_no_exact_match_keeps_prefix_options_instead_of_guessing():
    client, session = service(
        response(
            {
                "results": [
                    location(name="Ереван", country="Армения", population=1144700),
                    location(name="Еревани", country="Армения", latitude=40.1, population=100),
                ]
            }
        )
    )
    result = await client.get_weather("Ерев")
    assert result.status == "ambiguous" and len(result.candidates) == 2
    assert session.get.call_count == 1


@pytest.mark.parametrize("population", [None, -1, True, "unknown", float("nan")])
async def test_optional_population_is_not_required_to_get_weather(population):
    client, _ = service(
        response({"results": [location(population=population)]}), response(weather())
    )
    assert (await client.get_weather("Москва")).status == "ok"


async def test_geocoding_without_country_name_keeps_both_candidates_and_asks_for_choice():
    # Реальная форма ответа на Санкт-Петербург: у Montserrat нет country, но есть MS.
    alternative = location(
        name="Санкт-Петербург",
        country=None,
        country_code="MS",
        admin1="Parish of Saint Peter",
        latitude=16.7727,
        longitude=-62.21729,
        timezone="America/Montserrat",
    )
    client, session = service(
        response(
            {
                "results": [
                    location(name="Санкт-Петербург", admin1="Санкт-Петербург"),
                    alternative,
                ]
            }
        )
    )
    result = await client.get_weather("Санкт-Петербург")
    assert result.status == "ambiguous"
    assert result.weather is None
    assert [item.country for item in result.candidates] == ["Россия", "MS"]
    assert result.candidates[1].admin1 == "Parish of Saint Peter"
    assert session.get.call_count == 1


async def test_single_location_without_country_translation_can_return_weather():
    client, _ = service(
        response({"results": [location(country=None, country_code="ru")]}), response(weather())
    )
    result = await client.get_weather("Москва, RU")
    assert result.status == "ok" and result.weather.country == "RU"


@pytest.mark.parametrize("code", [None, "", "RUS", "R1", "ЯЯ", "ß", "ſs", 42])
async def test_missing_country_requires_valid_iso_code(code):
    client, _ = service(response({"results": [location(country=None, country_code=code)]}))
    with pytest.raises(WeatherError, match="некорректные данные"):
        await client.get_weather("Москва")


async def test_weather_diagnostics_do_not_log_city_or_exception_values(caplog):
    client, _ = service(
        response(failure=aiohttp.ClientError("private-key")),
        response(failure=aiohttp.ClientError("private-key")),
    )
    with pytest.raises(WeatherError):
        await client.get_weather("private-city")
    assert "stage=geocoding" in caplog.text
    assert "kind=ClientError" in caplog.text
    assert "private-key" not in caplog.text and "private-city" not in caplog.text


@pytest.mark.parametrize("city", ["", " ", "a" * 101, None, 42])
async def test_bad_city_rejected_before_http(city):
    client, session = service()
    with pytest.raises(WeatherError, match="название города"):
        await client.get_weather(city)
    session.get.assert_not_called()


async def test_single_character_name_is_valid_input_and_can_return_not_found():
    client, _ = service(response({}))
    assert (await client.get_weather("Я")).status == "not_found"


@pytest.mark.parametrize("failure", [TimeoutError("secret"), aiohttp.ClientError("secret")])
async def test_network_failure_is_retried_once_and_error_is_safe(failure):
    client, session = service(response(failure=failure), response(failure=failure))
    with pytest.raises(WeatherError) as error:
        await client.get_weather("Москва")
    assert session.get.call_count == 2
    assert "secret" not in str(error.value)


async def test_temporary_http_failure_can_recover_on_one_retry():
    client, session = service(
        response(status=503),
        response({"results": [location()]}),
        response(weather()),
    )
    assert (await client.get_weather("Москва")).status == "ok"
    assert session.get.call_count == 3


async def test_permanent_http_error_is_not_retried_or_exposed():
    client, session = service(response({"reason": "secret"}, status=400))
    with pytest.raises(WeatherError) as error:
        await client.get_weather("Москва")
    assert session.get.call_count == 1
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"error": True, "reason": "secret"},
        {"results": None},
        {"results": [location(country=None)]},
        {"results": [location(timezone="Bad/Zone")]},
        {"results": [location(latitude=100)]},
        {"results": [location(longitude=True)]},
    ],
)
async def test_bad_geocoding_payload_is_rejected(payload):
    client, _ = service(response(payload))
    with pytest.raises(WeatherError, match="некорректные данные") as error:
        await client.get_weather("Москва")
    assert "secret" not in str(error.value)


async def test_invalid_json_is_safe():
    client, _ = service(response(body=b"secret invalid json"))
    with pytest.raises(WeatherError, match="некорректные данные") as error:
        await client.get_weather("Москва")
    assert "secret" not in str(error.value)


@pytest.mark.parametrize(
    "current",
    [
        {
            "time": "2026-11-16T12:00",
            "temperature_2m": float("nan"),
            "weather_code": 3,
            "wind_speed_10m": 4,
        },
        {"time": "2026-11-16T12:00", "temperature_2m": 1, "weather_code": 4, "wind_speed_10m": 4},
        {"time": "2026-11-16T12:00", "temperature_2m": 1, "weather_code": 3, "wind_speed_10m": -1},
        {"time": "2026-11-16", "temperature_2m": 1, "weather_code": 3, "wind_speed_10m": 4},
    ],
)
async def test_invalid_weather_values_are_rejected(current):
    client, _ = service(response({"results": [location()]}), response(weather(current=current)))
    with pytest.raises(WeatherError, match="некорректные данные"):
        await client.get_weather("Москва")


@pytest.mark.parametrize(
    "changes",
    [
        {"utc_offset_seconds": 0},
        {"timezone": "Bad/Zone"},
        {"current_units": {"temperature_2m": "F"}},
    ],
)
async def test_wrong_units_zone_or_offset_are_rejected(changes):
    client, _ = service(response({"results": [location()]}), response(weather(**changes)))
    with pytest.raises(WeatherError, match="некорректные данные"):
        await client.get_weather("Москва")


async def test_oversized_response_is_rejected_before_body_read():
    client, _ = service(response({}, content_length=600000))
    with pytest.raises(WeatherError, match="слишком большой"):
        await client.get_weather("Москва")


async def test_chunked_response_size_is_limited():
    context = response({})
    context.__aenter__.return_value.content_length = None
    context.__aenter__.return_value.content.read.side_effect = [b"x" * 300000, b"x" * 300000]
    client, _ = service(context)
    with pytest.raises(WeatherError, match="слишком большой"):
        await client.get_weather("Москва")


async def test_total_deadline_covers_all_attempts(monkeypatch):
    from app import weather as module

    async def stalled_get(url, params):
        await asyncio.Event().wait()

    client, _ = service()
    monkeypatch.setattr(module, "TOTAL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client, "_get_json", stalled_get)
    with pytest.raises(WeatherError, match="не ответил вовремя"):
        await client.get_weather("Москва")
