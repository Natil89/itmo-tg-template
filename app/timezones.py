"""IANA-зона по явно указанному городу, без догадки языковой модели."""

import asyncio
import json
import unicodedata
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiohttp

from app.reminders import ReminderError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
MAX_RESPONSE_BYTES = 512_000


@dataclass(frozen=True)
class TimezoneChoice:
    timezone: str
    place: str = ""


def _name(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().replace("ё", "е").split())


def _text(value: object) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= 200:
        raise ValueError
    return value.strip()


class TimezoneResolver:
    def __init__(self, session: aiohttp.ClientSession) -> None:
        self._session = session

    async def resolve(self, value: str) -> TimezoneChoice:
        query = value.strip()
        if not 1 <= len(query) <= 100:
            raise ReminderError("Укажите IANA-зону или город длиной до 100 символов.")
        try:
            return TimezoneChoice(ZoneInfo(query).key)
        except (ValueError, ZoneInfoNotFoundError):
            pass
        if "/" in query:
            raise ReminderError(
                "Неизвестная IANA-зона. Укажите город или, например, Europe/Moscow."
            )
        try:
            async with asyncio.timeout(7):
                data = await self._search(query)
            return self._choose(data, query)
        except ReminderError:
            raise
        except (aiohttp.ClientError, TimeoutError):
            raise ReminderError(
                "Поиск часового пояса сейчас недоступен. Попробуйте позже или укажите IANA-зону."
            ) from None
        except (ValueError, TypeError, KeyError, ZoneInfoNotFoundError):
            raise ReminderError(
                "Не удалось проверить часовой пояс города. Укажите IANA-зону."
            ) from None

    async def _search(self, query: str) -> dict:
        for attempt in range(2):
            try:
                async with self._session.get(
                    GEOCODING_URL,
                    params={"name": query, "count": 100, "language": "ru"},
                    timeout=aiohttp.ClientTimeout(total=3),
                    allow_redirects=False,
                ) as response:
                    if response.status in {429, 500, 502, 503, 504} and attempt == 0:
                        continue
                    if response.status != 200:
                        raise ReminderError("Сервис поиска городов сейчас недоступен.")
                    if (
                        response.content_length is not None
                        and response.content_length > MAX_RESPONSE_BYTES
                    ):
                        raise ValueError
                    body = bytearray()
                    while chunk := await response.content.read(65536):
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise ValueError
                    data = json.loads(body)
                    if not isinstance(data, dict) or data.get("error"):
                        raise ValueError
                    return data
            except (aiohttp.ClientError, TimeoutError):
                if attempt == 1:
                    raise
        raise ReminderError("Сервис поиска городов сейчас недоступен.")

    @staticmethod
    def _choose(data: dict, query: str) -> TimezoneChoice:
        rows = data.get("results", [])
        if not isinstance(rows, list) or len(rows) > 100:
            raise ValueError
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError
        requested = _name(query.split(",", 1)[0])
        exact = [row for row in rows if _name(_text(row.get("name"))) == requested]
        if exact:
            rows = exact
        candidates: dict[tuple[str, str], tuple[TimezoneChoice, int]] = {}
        for row in rows:
            zone = ZoneInfo(_text(row.get("timezone"))).key
            city = _text(row.get("name"))
            country = _text(row.get("country") or row.get("country_code"))
            admin = row.get("admin1")
            place = f"{city}, {country}"
            if admin and _name(_text(admin)) != _name(city):
                place += f", {admin}"
            population = row.get("population", 0)
            if type(population) is not int or population < 0:
                population = 0
            key = (place, zone)
            previous = candidates.get(key)
            if previous is None or population > previous[1]:
                candidates[key] = (TimezoneChoice(zone, place), population)
        ordered = sorted(candidates.values(), key=lambda item: item[1], reverse=True)
        if not ordered:
            raise ReminderError("Город не найден. Проверьте название или укажите IANA-зону.")
        # Разные места с одной зоной не мешают выбору именно часового пояса.
        if len({item[0].timezone for item in ordered}) == 1:
            return ordered[0][0]
        # Для очевидного лидера показываем выбранное место. Неизвестное население
        # остальных допускаем для большого города, но не считаем доказательством:
        # обработчик явно называет место и предлагает исправить выбор.
        largest, runner_up = ordered[0][1], ordered[1][1]
        if (runner_up > 0 and largest >= 10 * runner_up) or (runner_up == 0 and largest >= 100_000):
            return ordered[0][0]
        options = "\n".join(f"• {item.place} — {item.timezone}" for item, _ in ordered[:6])
        raise ReminderError(
            "Нашлось несколько городов с разными часовыми поясами. "
            "Уточните страну или регион, например /timezone Париж, Франция, "
            f"либо укажите IANA-зону:\n{options}"
        )
