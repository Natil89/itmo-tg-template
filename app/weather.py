"""Текущая погода из фиксированных Open-Meteo API, без Telegram и MCP."""

import asyncio
import json
import logging
import math
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, model_validator

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
TOTAL_TIMEOUT_SECONDS = 6
MAX_RESPONSE_BYTES = 512 * 1024
logger = logging.getLogger(__name__)
WMO_CODES = {
    0,
    1,
    2,
    3,
    45,
    48,
    51,
    53,
    55,
    56,
    57,
    61,
    63,
    65,
    66,
    67,
    71,
    73,
    75,
    77,
    80,
    81,
    82,
    85,
    86,
    95,
    96,
    99,
}


class WeatherError(ValueError):
    """Безопасное сообщение об аргументах или недоступности погоды."""


class WeatherLocation(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    city: str = Field(min_length=1, max_length=200)
    country: str = Field(min_length=1, max_length=200)
    admin1: str = Field(max_length=200)


class CurrentWeather(WeatherLocation):
    temperature_c: float = Field(allow_inf_nan=False)
    weather_code: int = Field(ge=0, le=99)
    wind_speed: float = Field(ge=0, allow_inf_nan=False)
    wind_unit: Literal["m/s"] = "m/s"
    observed_at: str
    timezone: str
    source: Literal["Open-Meteo"] = "Open-Meteo"

    @model_validator(mode="after")
    def check_observation(self) -> "CurrentWeather":
        observed = datetime.fromisoformat(self.observed_at)
        if observed.tzinfo is None or self.weather_code not in WMO_CODES:
            raise ValueError("Нужны известный код погоды и момент с часовым поясом.")
        try:
            zone = ZoneInfo(self.timezone)
        except (KeyError, ValueError):
            raise ValueError("Неизвестный часовой пояс наблюдения.") from None
        if observed.utcoffset() != observed.astimezone(zone).utcoffset():
            raise ValueError("Момент наблюдения не соответствует часовому поясу.")
        return self


class GeoCity(WeatherLocation):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    latitude: float = Field(ge=-90, le=90, allow_inf_nan=False)
    longitude: float = Field(ge=-180, le=180, allow_inf_nan=False)

    @property
    def place(self) -> str:
        parts = [self.city, self.country]
        if self.admin1 and _location_name(self.admin1) != _location_name(self.city):
            parts.append(self.admin1)
        return ", ".join(parts)


class WeatherResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    status: Literal["ok", "not_found", "ambiguous"]
    weather: CurrentWeather | None = None
    candidates: list[WeatherLocation] = Field(default_factory=list, max_length=100)
    assumed_location: bool = False

    @model_validator(mode="after")
    def check_status(self) -> "WeatherResult":
        if self.status == "ok":
            valid = self.weather is not None and not self.candidates
        elif self.status == "ambiguous":
            valid = self.weather is None and len(self.candidates) >= 2
        else:
            valid = self.weather is None and not self.candidates
        if self.assumed_location and self.status != "ok":
            valid = False
        if not valid:
            raise ValueError("Поля результата не соответствуют статусу.")
        return self


@dataclass(frozen=True)
class _Location:
    public: WeatherLocation
    latitude: float
    longitude: float
    population: int = 0


def normalize_city(city: str) -> str:
    if not isinstance(city, str) or not 1 <= len(city.strip()) <= 100:
        raise WeatherError("Укажите название города длиной от 1 до 100 символов.")
    return city.strip()


def _location_name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().replace("ё", "е").split())


def _text(value: object, *, optional: bool = False) -> str:
    if optional and value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError
    value = value.strip()
    if len(value) > 200 or (not optional and not value):
        raise ValueError
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError
    number = float(value)
    if not math.isfinite(number):
        raise ValueError
    return number


def _country(row: dict) -> str:
    country = row.get("country")
    if country is not None and country != "":
        return _text(country)
    # Open-Meteo может не иметь перевода названия территории, сохраняя ISO-код.
    code = row.get("country_code")
    if not isinstance(code, str):
        raise ValueError
    code = code.strip()
    if len(code) != 2 or not code.isascii() or not code.isalpha():
        raise ValueError
    return code.upper()


class WeatherService:
    """Сессию создаёт и закрывает владелец; все GET ограничены общим бюджетом."""

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        location_policy: Literal["strict", "population"] = "strict",
    ) -> None:
        if location_policy not in {"strict", "population"}:
            raise ValueError("Неизвестный режим выбора города.")
        self._session = session
        self.location_policy = location_policy

    async def _get_json(self, url: str, params: dict) -> dict:
        stage = "geocoding" if url == GEOCODING_URL else "forecast"
        for attempt in range(2):
            try:
                async with self._session.get(
                    url,
                    params=params,
                    timeout=aiohttp.ClientTimeout(total=3),
                    allow_redirects=False,
                ) as response:
                    if response.status in {429, 500, 502, 503, 504} and attempt == 0:
                        continue
                    if response.status != 200:
                        logger.warning(
                            "Weather HTTP failure: stage=%s status=%d", stage, response.status
                        )
                        raise WeatherError("Погодный сервис сейчас недоступен. Попробуйте позже.")
                    if (
                        response.content_length is not None
                        and response.content_length > MAX_RESPONSE_BYTES
                    ):
                        raise WeatherError("Погодный сервис вернул слишком большой ответ.")
                    body = bytearray()
                    while chunk := await response.content.read(65536):
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise WeatherError("Погодный сервис вернул слишком большой ответ.")
                    data = json.loads(body)
                    if not isinstance(data, dict) or data.get("error"):
                        raise ValueError
                    return data
            except (TimeoutError, aiohttp.ClientError) as error:
                logger.warning(
                    "Weather transport failure: stage=%s attempt=%d kind=%s",
                    stage,
                    attempt + 1,
                    type(error).__name__,
                )
                if attempt == 1:
                    raise WeatherError(
                        "Не удалось получить погоду: сервис недоступен или не ответил вовремя."
                    ) from None
        raise WeatherError("Погодный сервис сейчас недоступен.")

    @staticmethod
    def _locations(data: dict, query: str | None = None) -> list[_Location]:
        rows = data.get("results", [])
        if not isinstance(rows, list) or len(rows) > 100:
            raise ValueError
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError
        # API ищет по префиксу: Ереван и Еревани не являются одноимёнными городами.
        # Отбрасываем похожие названия только при наличии точного совпадения.
        if query is not None:
            requested = _location_name(query.split(",", 1)[0].strip())
            exact = [row for row in rows if _location_name(_text(row.get("name"))) == requested]
            if exact:
                rows = exact
        locations: dict[tuple, _Location] = {}
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError
            latitude, longitude = _number(row.get("latitude")), _number(row.get("longitude"))
            if not -90 <= latitude <= 90 or not -180 <= longitude <= 180:
                raise ValueError
            ZoneInfo(_text(row.get("timezone")))
            public = WeatherLocation(
                city=_text(row.get("name")),
                country=_country(row),
                admin1=_text(row.get("admin1"), optional=True),
            )
            identity = (latitude, longitude, public.city, public.country, public.admin1)
            population = row.get("population")
            population = population if type(population) is int and population >= 0 else 0
            if identity not in locations or population > locations[identity].population:
                locations[identity] = _Location(public, latitude, longitude, population)
        # Население задаёт порядок показа, не разрешение на автоматический выбор.
        return sorted(locations.values(), key=lambda item: item.population, reverse=True)

    @staticmethod
    def _current(data: dict, location: _Location) -> CurrentWeather:
        zone_name = _text(data.get("timezone"))
        zone = ZoneInfo(zone_name)
        current, units = data.get("current"), data.get("current_units")
        if not isinstance(current, dict) or not isinstance(units, dict):
            raise ValueError
        if (
            units.get("temperature_2m") != "°C"
            or units.get("wind_speed_10m") != "m/s"
            or units.get("weather_code") != "wmo code"
            or units.get("time") != "iso8601"
        ):
            raise ValueError
        raw_time = current.get("time")
        if not isinstance(raw_time, str) or "T" not in raw_time:
            raise ValueError
        observed = datetime.fromisoformat(raw_time)
        if observed.tzinfo is None:
            offset = data.get("utc_offset_seconds")
            if isinstance(offset, bool) or not isinstance(offset, int) or abs(offset) > 50400:
                raise ValueError
            observed = observed.replace(tzinfo=timezone(timedelta(seconds=offset)))
        local = observed.astimezone(zone)
        if local.replace(tzinfo=None) != observed.replace(tzinfo=None):
            raise ValueError
        code = current.get("weather_code")
        if isinstance(code, bool) or not isinstance(code, int) or code not in WMO_CODES:
            raise ValueError
        return CurrentWeather(
            **location.public.model_dump(),
            temperature_c=_number(current.get("temperature_2m")),
            weather_code=code,
            wind_speed=_number(current.get("wind_speed_10m")),
            observed_at=local.isoformat(),
            timezone=zone_name,
        )

    async def find_cities(self, query: str) -> list[GeoCity]:
        query = normalize_city(query)
        parts = [part.strip() for part in query.split(",")]
        if any(not part for part in parts) or len(parts) > 3:
            raise WeatherError("Укажите город и при необходимости страну/регион через запятую.")
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                language = "en" if len(parts) > 1 and parts[1].isascii() else "ru"
                data = await self._get_json(
                    GEOCODING_URL, {"name": parts[0], "count": 100, "language": language}
                )
                rows = data.get("results", [])
                if (
                    not isinstance(rows, list)
                    or len(rows) > 100
                    or any(not isinstance(row, dict) for row in rows)
                ):
                    raise ValueError
                if len(parts) > 1:
                    rows = [
                        row
                        for row in rows
                        if all(
                            _location_name(part)
                            in {
                                _location_name(str(row.get(field, "")))
                                for field in ("country", "country_code", "admin1", "admin2")
                            }
                            for part in parts[1:]
                        )
                    ]
                return [
                    GeoCity(
                        **item.public.model_dump(), latitude=item.latitude, longitude=item.longitude
                    )
                    for item in self._locations({"results": rows}, parts[0])[:5]
                ]
        except (ValueError, TypeError, KeyError, TimeoutError):
            raise WeatherError(
                "Не удалось проверить город. Уточните название или повторите позже."
            ) from None

    async def weather_at(self, city: GeoCity) -> CurrentWeather:
        city = GeoCity.model_validate(city.model_dump())
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                data = await self._get_json(
                    WEATHER_URL,
                    {
                        "latitude": city.latitude,
                        "longitude": city.longitude,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "ms",
                        "timezone": "auto",
                    },
                )
                public = WeatherLocation(city=city.city, country=city.country, admin1=city.admin1)
                return self._current(data, _Location(public, city.latitude, city.longitude))
        except (ValueError, TypeError, KeyError, TimeoutError):
            raise WeatherError("Не удалось получить погоду подтверждённого города.") from None

    async def get_weather(self, city: str) -> WeatherResult:
        city = normalize_city(city)
        stage = "geocoding"
        try:
            async with asyncio.timeout(TOTAL_TIMEOUT_SECONDS):
                geo = await self._get_json(
                    GEOCODING_URL, {"name": city, "count": 100, "language": "ru"}
                )
                locations = self._locations(geo, city)
                if not locations:
                    return WeatherResult(status="not_found")
                assumed = False
                if len(locations) > 1:
                    requested = _location_name(city.split(",", 1)[0].strip())
                    can_assume = (
                        self.location_policy == "population"
                        and all(_location_name(item.public.city) == requested for item in locations)
                        and locations[0].population > locations[1].population
                    )
                    if not can_assume:
                        return WeatherResult(
                            status="ambiguous", candidates=[item.public for item in locations]
                        )
                    assumed = True
                location = locations[0]
                stage = "forecast"
                data = await self._get_json(
                    WEATHER_URL,
                    {
                        "latitude": location.latitude,
                        "longitude": location.longitude,
                        "current": "temperature_2m,weather_code,wind_speed_10m",
                        "temperature_unit": "celsius",
                        "wind_speed_unit": "ms",
                        "timezone": "auto",
                    },
                )
                return WeatherResult(
                    status="ok",
                    weather=self._current(data, location),
                    assumed_location=assumed,
                )
        except WeatherError:
            raise
        except TimeoutError:
            logger.warning("Weather deadline exceeded: stage=%s", stage)
            raise WeatherError("Погодный сервис не ответил вовремя. Попробуйте позже.") from None
        except (ValueError, TypeError, KeyError, OverflowError):
            logger.warning("Weather response invalid: stage=%s", stage)
            raise WeatherError("Погодный сервис вернул некорректные данные.") from None
