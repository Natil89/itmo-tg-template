"""Постоянные права компонентов и их пересечение с актуальным MCP discovery."""

import copy
from types import MappingProxyType

from jsonschema import Draft202012Validator
from mcp import types

from app.agent_contracts import AgentRequest, Component
from app.mcp_client import ScheduleMcpClient

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
        self, client: ScheduleMcpClient, component: Component, request: AgentRequest
    ) -> None:
        self.__client = client
        self.__component = component
        self.__request = request
        self.__calls: list[str] = []

    @property
    def used_tools(self) -> tuple[str, ...]:
        return tuple(self.__calls)

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
        result = await self.__client.call_tool(name, arguments, user_id=self.__request.actor_id)
        if result.isError or result.structuredContent is None:
            raise ToolFailure
        Draft202012Validator(tool.outputSchema).validate(result.structuredContent)
        return copy.deepcopy(result.structuredContent)

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        require_resource(self.__component, uri, self.__client.resources)
        return await self.__client.read_resource(uri, user_id=self.__request.actor_id)
