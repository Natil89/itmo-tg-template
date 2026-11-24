"""Собственный MCP-сервер расписания. stdout зарезервирован для протокола."""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

import aiohttp
import asyncpg
import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from pydantic import ValidationError

from app.materials import MaterialsError, MaterialsService, SearchQuery, SearchResult
from app.mcp_identity import IdentityError, verify_context
from app.personal_schedule import (
    LessonDraft,
    LessonSaved,
    PersonalScheduleError,
    PersonalScheduleStore,
)
from app.reminders import ReminderDraft, ReminderError, ReminderSaved, ReminderStore
from app.schedule import DEFAULT_SCHEDULE, DaySchedule, ScheduleError, ScheduleService
from app.weather import WeatherError, WeatherResult, WeatherService


def create_server(
    service: ScheduleService,
    weather: WeatherService | None = None,
    personal_store: PersonalScheduleStore | None = None,
    context_secret: str = "",
    reminders: ReminderStore | None = None,
    materials: MaterialsService | None = None,
) -> Server:
    server = Server("study-schedule")

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        tools = [
            types.Tool(
                name="get_schedule",
                description=(
                    "Занятия на дату YYYY-MM-DD. Пустой список означает отсутствие занятий."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
                    "required": ["date"],
                    "additionalProperties": False,
                },
                outputSchema=DaySchedule.model_json_schema(),
                annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False),
            )
        ]
        if weather is not None:
            tools.append(
                types.Tool(
                    name="get_weather",
                    description=(
                        "Текущая погода в городе, не прогноз. Город можно уточнить страной "
                        "или областью через запятую. "
                        + (
                            "При нескольких точных совпадениях используется лидер по населению "
                            "с отметкой предположения; если лидера нет, требуется выбор."
                            if weather.location_policy == "population"
                            else "Несколько одноимённых вариантов требуют выбора."
                        )
                    ),
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "minLength": 1, "maxLength": 100}
                        },
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                    outputSchema=WeatherResult.model_json_schema(),
                    annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False),
                )
            )
        if personal_store is not None:
            tools.append(
                types.Tool(
                    name="add_lesson",
                    description=(
                        "Добавить занятие в личное расписание: дата, начало, конец, "
                        "название и место. Приложение обязательно запросит подтверждение."
                    ),
                    inputSchema=LessonDraft.model_json_schema(),
                    outputSchema=LessonSaved.model_json_schema(),
                    annotations=types.ToolAnnotations(
                        readOnlyHint=False, destructiveHint=False, idempotentHint=True
                    ),
                )
            )
        if reminders is not None:
            tools.append(
                types.Tool(
                    name="add_reminder",
                    description=(
                        "Подготовить личное напоминание: text и будущее remind_at ISO 8601 "
                        "с явным UTC-смещением настроенной пользователем зоны. "
                        "Предложи этот вызов сразу при известных тексте и времени: "
                        "приложение перехватит его и покажет кнопки подтверждения. "
                        "Не проси предварительного текстового согласия. "
                        "Инструмент сохраняет напоминание; доставку выполняет отдельный компонент."
                    ),
                    inputSchema=ReminderDraft.model_json_schema(),
                    outputSchema=ReminderSaved.model_json_schema(),
                    annotations=types.ToolAnnotations(
                        readOnlyHint=False, destructiveHint=False, idempotentHint=True
                    ),
                )
            )
        if materials is not None:
            tools.append(
                types.Tool(
                    name="search_materials",
                    description=(
                        "Поиск по ограниченному локальному учебному корпусу. "
                        "Возвращает фрагменты и источники."
                    ),
                    inputSchema=SearchQuery.model_json_schema(),
                    outputSchema=SearchResult.model_json_schema(),
                    annotations=types.ToolAnnotations(readOnlyHint=True, destructiveHint=False),
                )
            )
        return tools

    def identity(scope: str, arguments: dict) -> tuple[int, str | None]:
        meta = server.request_context.meta
        return verify_context(
            context_secret,
            meta.model_dump() if meta is not None else None,
            scope,
            arguments,
        )

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> types.CallToolResult:
        try:
            if name == "search_materials" and materials is not None:
                query = SearchQuery.model_validate(arguments)
                found = await asyncio.to_thread(materials.search, query)
                result = found.model_dump(mode="json")
            elif name == "get_schedule":
                if set(arguments) != {"date"} or not isinstance(arguments["date"], str):
                    raise ScheduleError("Нужен только строковый аргумент date.")
                if personal_store is not None:
                    user_id, _ = identity("tool:get_schedule", arguments)
                    result = (
                        await personal_store.for_date(user_id, arguments["date"])
                    ).model_dump()
                else:
                    result = service.for_date(arguments["date"]).model_dump()
            elif name == "get_weather" and weather is not None:
                if set(arguments) != {"city"} or not isinstance(arguments["city"], str):
                    raise WeatherError("Нужен только строковый аргумент city.")
                result = (await weather.get_weather(arguments["city"])).model_dump()
            elif name == "add_lesson" and personal_store is not None:
                user_id, action_id = identity("tool:add_lesson", arguments)
                if action_id is None:
                    raise IdentityError("Занятие требует подтверждения пользователя.")
                draft = LessonDraft.model_validate(arguments)
                result = (await personal_store.execute(user_id, action_id, draft)).model_dump()
            elif name == "add_reminder" and reminders is not None:
                user_id, action_id = identity("tool:add_reminder", arguments)
                if action_id is None:
                    raise IdentityError("Напоминание требует подтверждения пользователя.")
                draft = ReminderDraft.model_validate(arguments)
                result = (await reminders.execute(user_id, action_id, draft)).model_dump()
            else:
                raise ScheduleError("Неизвестный инструмент.")
        except ValidationError:
            return types.CallToolResult(
                isError=True,
                content=[types.TextContent(type="text", text="Аргументы не прошли проверку.")],
            )
        except (
            ScheduleError,
            WeatherError,
            PersonalScheduleError,
            ReminderError,
            IdentityError,
            MaterialsError,
        ) as error:
            return types.CallToolResult(
                isError=True, content=[types.TextContent(type="text", text=str(error))]
            )
        return types.CallToolResult(
            structuredContent=result,
            content=[types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False))],
        )

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        resources = [
            types.Resource(
                uri="schedule://current-week",
                name="Расписание текущей недели",
                description="Понедельник–воскресенье в настроенном часовом поясе сервера.",
                mimeType="application/json",
            )
        ]
        if materials is not None:
            resources.append(
                types.Resource(
                    name="Каталог учебных материалов",
                    uri="materials://catalog",
                    mimeType="application/json",
                    description="Идентификаторы, названия и источники документов.",
                )
            )
        return resources

    @server.read_resource()
    async def read_resource(uri) -> list[ReadResourceContents]:
        if str(uri) == "materials://catalog" and materials is not None:
            catalog = await asyncio.to_thread(materials.catalog)
            return [
                ReadResourceContents(
                    content=catalog.model_dump_json(), mime_type="application/json"
                )
            ]
        if str(uri) != "schedule://current-week":
            raise ValueError("Неизвестный ресурс.")
        if personal_store is not None:
            user_id, _ = identity(f"resource:{uri}", {})
            week = await personal_store.current_week(user_id, service.timezone)
        else:
            week = service.current_week()
        return [ReadResourceContents(content=week.model_dump_json(), mime_type="application/json")]

    return server


async def serve(service: ScheduleService, weather_policy: str = "strict") -> None:
    pool = None
    try:
        store = None
        reminders = None
        secret = ""
        if os.environ.get("MCP_PERSONAL_SCHEDULE") == "1":
            secret = os.environ.get("MCP_CONTEXT_SECRET", "")
            if len(secret) < 32:
                raise IdentityError("Не настроен доверенный контекст MCP.")
            pool = await asyncpg.create_pool(
                host=os.environ["MCP_DB_HOST"],
                port=int(os.environ["MCP_DB_PORT"]),
                database=os.environ["MCP_DB_NAME"],
                user=os.environ["MCP_DB_USER"],
                password=os.environ["MCP_DB_PASSWORD"],
                min_size=1,
                max_size=4,
                timeout=5,
                command_timeout=5,
            )
            store = PersonalScheduleStore(pool)
            await store.initialize()
            reminders = ReminderStore(pool)
            await reminders.initialize()
        async with aiohttp.ClientSession() as session:
            server = create_server(
                service,
                WeatherService(session, location_policy=weather_policy),
                store,
                secret,
                reminders=reminders,
                materials=MaterialsService(),
            )
            async with mcp.server.stdio.stdio_server() as (reader, writer):
                await server.run(reader, writer, server.create_initialization_options())
    finally:
        if pool is not None:
            try:
                async with asyncio.timeout(10):
                    await pool.close()
            except TimeoutError:
                pool.terminate()


def main() -> None:
    parser = argparse.ArgumentParser(description="MCP-сервер учебного расписания")
    parser.add_argument("--schedule-file", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--timezone", default="Europe/Moscow")
    parser.add_argument("--weather-policy", choices=["strict", "population"], default="strict")
    args = parser.parse_args()
    try:
        asyncio.run(serve(ScheduleService(args.schedule_file, args.timezone), args.weather_policy))
    except Exception as error:
        print(f"MCP-сервер не запущен: {type(error).__name__}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
