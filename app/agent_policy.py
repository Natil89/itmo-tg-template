"""Постоянные права компонентов и их пересечение с актуальным MCP discovery."""

import copy
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from types import MappingProxyType
from uuid import UUID

from jsonschema import Draft202012Validator
from mcp import types

from app.agent_contracts import (
    AgentRequest,
    Component,
    EmptyFacts,
    Facts,
    KnowledgeFacts,
    MaterialCatalog,
    MaterialHit,
    ReminderProposalFacts,
    ScheduleFacts,
    WeatherFacts,
)
from app.mcp_client import ScheduleMcpClient
from app.reminders import ReminderDraft, ReminderStore
from app.schedule import DaySchedule, WeekSchedule, parse_date
from app.weather import WeatherResult

TOOLS = MappingProxyType(
    {
        "router": frozenset(),
        "schedule": frozenset({"get_schedule", "add_reminder"}),
        "knowledge": frozenset({"search_materials"}),
        "lifestyle": frozenset({"get_weather"}),
        # Briefing получает чтения через соответствующие компоненты, не напрямую.
        "briefing": frozenset(),
        "lesson_compat": frozenset({"add_lesson"}),
    }
)
RESOURCES = MappingProxyType(
    {
        "schedule": frozenset({"schedule://current-week"}),
        "knowledge": frozenset({"materials://catalog"}),
    }
)
READ_TOOLS = frozenset({"get_schedule", "search_materials", "get_weather"})


class AccessDenied(ValueError):
    def __init__(self) -> None:
        super().__init__("Операция не разрешена политикой компонента.")


class ToolFailure(ValueError):
    def __init__(self) -> None:
        super().__init__("Инструмент не вернул проверенные данные.")


def allowed_tools(component: Component, discovered: list[types.Tool]) -> dict[str, types.Tool]:
    """Не кэшируем: удалённые и вновь появившиеся возможности проверяются заново."""
    names = TOOLS.get(component, frozenset())
    result = {}
    for tool in discovered:
        if tool.name in names:
            if tool.name in result:
                # Неоднозначное discovery запрещает весь запрос, а не выбирает схему.
                raise AccessDenied
            result[tool.name] = tool.model_copy(deep=True)
    return result


def require_resource(component: Component, uri: str, discovered: list[types.Resource]) -> None:
    if uri not in RESOURCES.get(component, frozenset()):
        raise AccessDenied
    if sum(str(resource.uri) == uri for resource in discovered) != 1:
        raise AccessDenied


class ScopedMcp:
    """Выдаётся одному специалисту на один запрос; не принимает owner от модели.

    Этап 1 разрешает исполнение только чтений. Запись будет подключена отдельным
    путём проверки маршрута/черновика/подтверждения; одного allowlist недостаточно.
    """

    def __init__(
        self,
        client: ScheduleMcpClient,
        component: Component,
        request: AgentRequest,
        *,
        reminders: ReminderStore | None = None,
        update_id: int | None = None,
        prepare_reminder: bool = False,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.__client = client
        self.__component = component
        self.__request = request
        self.__calls: list[str] = []
        self.__facts: Facts = EmptyFacts()
        self.__status = "needs_clarification"
        self.__resource_reads = 0
        self.__reminders = reminders
        self.__update_id = update_id
        self.__allow_prepare = prepare_reminder
        self.__clock = clock or (lambda: datetime.now(UTC))
        self.__pending = None
        self.__weather_result: WeatherResult | None = None

    @property
    def weather_result(self) -> WeatherResult | None:
        return self.__weather_result.model_copy(deep=True) if self.__weather_result else None

    @property
    def pending_reminder(self):
        return copy.deepcopy(self.__pending)

    async def prepare_reminder(self, arguments: dict) -> None:
        tool = allowed_tools(self.__component, self.__client.tools).get("add_reminder")
        if (
            tool is None
            or self.__reminders is None
            or self.__component != "schedule"
            or not self.__allow_prepare
            or self.__request.intent != "user_query"
            or type(self.__update_id) is not int
            or self.__update_id <= 0
        ):
            raise AccessDenied
        arguments = copy.deepcopy(arguments)
        Draft202012Validator(tool.inputSchema).validate(arguments)
        draft = ReminderDraft.model_validate(arguments)
        pending = await self.__reminders.prepare(
            self.__request.actor_id, self.__update_id, draft, now=self.__clock()
        )
        if (
            pending.user_id != self.__request.actor_id
            or pending.payload.text != draft.text
            or datetime.fromisoformat(pending.payload.remind_at)
            != datetime.fromisoformat(draft.remind_at)
        ):
            raise ToolFailure
        self.__pending = pending
        self.__facts = ReminderProposalFacts(
            action_id=UUID(pending.action_id),
            draft=pending.payload,
            timezone=pending.timezone,
            expires_at=pending.expires_at,
        )
        self.__status = "needs_clarification"

    @property
    def used_tools(self) -> tuple[str, ...]:
        return tuple(self.__calls)

    @property
    def facts(self) -> Facts:
        return self.__facts.model_copy(deep=True)

    @property
    def status(self) -> str:
        return self.__status

    def tools(self) -> list[types.Tool]:
        return list(allowed_tools(self.__component, self.__client.tools).values())

    async def call_tool(self, name: str, arguments: dict) -> dict:
        # Копируем до await: вызывающий не может менять проверенный dict во время вызова.
        arguments = copy.deepcopy(arguments)
        tool = allowed_tools(self.__component, self.__client.tools).get(name)
        if tool is None or name not in READ_TOOLS:
            raise AccessDenied
        if len(self.__calls) >= 2:
            raise AccessDenied
        Draft202012Validator.check_schema(tool.inputSchema)
        Draft202012Validator(tool.inputSchema).validate(arguments)
        if tool.outputSchema is None:
            raise ToolFailure
        Draft202012Validator.check_schema(tool.outputSchema)
        self.__calls.append(name)
        self.__status = "failed"
        result = await self.__client.call_tool(name, arguments, user_id=self.__request.actor_id)
        if result.isError or result.structuredContent is None:
            raise ToolFailure
        Draft202012Validator(tool.outputSchema).validate(result.structuredContent)
        # Discovery-схема не заменяет локальный типизированный контракт фактов.
        data = result.structuredContent
        if name == "get_schedule":
            day = DaySchedule.model_validate(data)
            parse_date(day.date)
            if day.date != arguments["date"]:
                raise ToolFailure
            previous = self.__facts.days if isinstance(self.__facts, ScheduleFacts) else ()
            self.__facts = ScheduleFacts(days=(*previous, day))
            self.__status = "success"
        elif name == "get_weather":
            weather = WeatherResult.model_validate(data)
            self.__weather_result = weather
            if weather.status == "ok":
                self.__facts = WeatherFacts(weather=weather.weather)
                self.__status = "success"
            else:
                self.__facts = EmptyFacts()
                self.__status = (
                    "not_found" if weather.status == "not_found" else "needs_clarification"
                )
        elif name == "search_materials":
            if set(data) != {"hits"} or not isinstance(data["hits"], list):
                raise ToolFailure
            self.__facts = KnowledgeFacts(
                hits=tuple(MaterialHit.model_validate(hit) for hit in data["hits"])
            )
            self.__status = "success" if self.__facts.hits else "not_found"
        return copy.deepcopy(result.structuredContent)

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        require_resource(self.__component, uri, self.__client.resources)
        if self.__resource_reads >= 2:
            raise AccessDenied
        self.__resource_reads += 1
        result = await self.__client.read_resource(uri, user_id=self.__request.actor_id)
        if len(result.contents) != 1:
            raise ToolFailure
        content = result.contents[0]
        if (
            not isinstance(content, types.TextResourceContents)
            or str(content.uri) != uri
            or len(content.text) > 64000
        ):
            raise ToolFailure
        if uri == "schedule://current-week":
            week = WeekSchedule.model_validate_json(content.text)
            AgentRequest.iana_zone(week.timezone)
            start = parse_date(week.start)
            if (
                start.weekday() != 0
                or week.end != (start + timedelta(days=6)).isoformat()
                or [day.date for day in week.days]
                != [(start + timedelta(days=i)).isoformat() for i in range(7)]
            ):
                raise ToolFailure
            self.__facts = ScheduleFacts(days=tuple(week.days))
            self.__status = "success"
        else:
            # Каталог — контекст поиска, сам по себе не доказательство ответа по документу.
            MaterialCatalog.model_validate_json(content.text)
        return result.model_copy(deep=True)
