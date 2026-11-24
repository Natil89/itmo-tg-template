"""Жизненный цикл stdio-соединения и обнаружение возможностей сервера."""

import asyncio
import logging
import secrets
import sys
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from datetime import timedelta
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import McpError
from mcp.types import Resource, Tool

from app.mcp_identity import IdentityError, sign_context
from app.schedule import DEFAULT_SCHEDULE
from app.tracing import span

if TYPE_CHECKING:
    from app.config import Settings

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def safe_error_summary(error: BaseException) -> str:
    """Раскрывает типы вложенных ошибок, не раскрывая их текст и данные."""
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(safe_error_summary(item) for item in error.exceptions[:8])
    if isinstance(error, McpError):
        return f"McpError(code={error.error.code})"
    return type(error).__name__


class ScheduleMcpClient:
    def __init__(self) -> None:
        self.session: ClientSession | None = None
        self.tools: list[Tool] = []
        self.resources: list[Resource] = []
        self.request_timeout_seconds = 10.0
        self.personal = False
        self._context_secret = ""

    async def call_tool(
        self,
        name: str,
        arguments: dict,
        *,
        user_id: int | None = None,
        action_id: str | None = None,
    ) -> types.CallToolResult:
        if self.session is None:
            raise ConnectionError("MCP-сервер недоступен.")
        meta = None
        if self.personal and name in {"get_schedule", "add_lesson", "add_reminder"}:
            if user_id is None:
                raise IdentityError("Отсутствует доверенный контекст пользователя.")
            meta = sign_context(
                self._context_secret,
                user_id,
                f"tool:{name}",
                arguments,
                action_id=action_id,
            )
        with span("mcp", "call_tool", tool=name) as step:
            async with asyncio.timeout(self.request_timeout_seconds):
                result = await self.session.call_tool(
                    name,
                    arguments,
                    read_timeout_seconds=timedelta(seconds=self.request_timeout_seconds),
                    meta=meta,
                )
            if result.isError is True:
                step.status, step.error_code = "failed", "tool_failed"
            return result

    async def read_current_week(self, user_id: int | None = None) -> types.ReadResourceResult:
        return await self.read_resource("schedule://current-week", user_id=user_id)

    async def read_resource(
        self, uri: str, *, user_id: int | None = None
    ) -> types.ReadResourceResult:
        """Транспорт чтения; права вызывающего проверяет ScopedMcp перед этим методом."""
        if uri not in {"schedule://current-week", "materials://catalog"}:
            raise ValueError("Неизвестный ресурс.")
        if self.session is None:
            raise ConnectionError("MCP-сервер недоступен.")
        meta = None
        if self.personal and uri == "schedule://current-week":
            if user_id is None:
                raise IdentityError("Отсутствует доверенный контекст пользователя.")
            meta = types.RequestParams.Meta(
                **sign_context(
                    self._context_secret,
                    user_id,
                    f"resource:{uri}",
                    {},
                )
            )
        with span("mcp", "read_resource"):
            async with asyncio.timeout(self.request_timeout_seconds):
                return await self.session.send_request(
                    types.ClientRequest(
                        types.ReadResourceRequest(
                            params=types.ReadResourceRequestParams(uri=uri, _meta=meta),
                        )
                    ),
                    types.ReadResourceResult,
                    request_read_timeout_seconds=timedelta(seconds=self.request_timeout_seconds),
                )

    async def refresh(self) -> None:
        if self.session is None:
            raise ConnectionError("MCP-сервер недоступен.")
        tools, resources = [], []
        cursor = None
        while True:
            async with asyncio.timeout(self.request_timeout_seconds):
                page = await self.session.list_tools(cursor=cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if cursor is None:
                break
        cursor = None
        while True:
            async with asyncio.timeout(self.request_timeout_seconds):
                page = await self.session.list_resources(cursor=cursor)
            resources.extend(page.resources)
            cursor = page.nextCursor
            if cursor is None:
                break
        self.tools, self.resources = tools, resources


@asynccontextmanager
async def connect_schedule(
    path: Path = DEFAULT_SCHEDULE,
    timezone: str = "Europe/Moscow",
    *,
    startup_timeout_seconds: float = 60,
    request_timeout_seconds: float = 10,
    database_settings: "Settings | None" = None,
    weather_policy: str | None = None,
) -> AsyncIterator[ScheduleMcpClient]:
    client = ScheduleMcpClient()
    weather_policy = weather_policy or (
        database_settings.weather_location_policy if database_settings is not None else "strict"
    )
    if weather_policy not in {"strict", "population"}:
        raise ValueError("Неизвестный режим выбора города.")
    client.request_timeout_seconds = request_timeout_seconds
    client.personal = database_settings is not None
    client._context_secret = secrets.token_hex(32) if client.personal else ""
    ready, stop = asyncio.Event(), asyncio.Event()

    async def maintain_connection() -> None:
        stage = "spawn"
        started = monotonic()
        try:
            async with AsyncExitStack() as stack:
                schedule_path = await asyncio.to_thread(path.resolve)
                process_env = {"PYTHONUTF8": "1", "PYTHONUNBUFFERED": "1"}
                if sys.dont_write_bytecode:
                    # Родительский -B должен действовать и в отдельном MCP-процессе.
                    process_env["PYTHONDONTWRITEBYTECODE"] = "1"
                if database_settings is not None:
                    process_env.update(
                        {
                            "MCP_PERSONAL_SCHEDULE": "1",
                            "MCP_CONTEXT_SECRET": client._context_secret,
                            "MCP_DB_HOST": database_settings.postgres_host,
                            "MCP_DB_PORT": str(database_settings.postgres_port),
                            "MCP_DB_NAME": database_settings.postgres_db,
                            "MCP_DB_USER": database_settings.postgres_user,
                            "MCP_DB_PASSWORD": database_settings.postgres_password,
                        }
                    )
                parameters = StdioServerParameters(
                    command=sys.executable,
                    args=[
                        "-m",
                        "app.mcp_server",
                        "--schedule-file",
                        str(schedule_path),
                        "--timezone",
                        timezone,
                        "--weather-policy",
                        weather_policy,
                    ],
                    cwd=str(PROJECT_ROOT),
                    env=process_env,
                )
                reader, writer = await stack.enter_async_context(stdio_client(parameters))
                client.session = await stack.enter_async_context(
                    ClientSession(
                        reader,
                        writer,
                        read_timeout_seconds=timedelta(seconds=startup_timeout_seconds),
                    )
                )
                stage = "initialize"
                async with asyncio.timeout(startup_timeout_seconds):
                    await client.session.initialize()
                stage = "discover"
                await client.refresh()
                stage = "connected"
                logger.info(
                    "MCP подключён: tools=%d resources=%d startup_seconds=%.2f weather_policy=%s",
                    len(client.tools),
                    len(client.resources),
                    monotonic() - started,
                    weather_policy,
                )
                ready.set()
                await stop.wait()
        except Exception as error:
            # Не пишем параметры запуска, аргументы или текст исключения с данными.
            logger.warning(
                "MCP-подключение недоступно: stage=%s error=%s elapsed_seconds=%.2f",
                stage,
                safe_error_summary(error),
                monotonic() - started,
            )
        finally:
            client.session = None
            client.tools, client.resources = [], []
            ready.set()

    # Cancel scopes SDK принадлежат отдельной задаче: отказ транспорта не отменяет polling.
    task = asyncio.create_task(maintain_connection())
    try:
        await ready.wait()
        yield client
    finally:
        stop.set()
        try:
            async with asyncio.timeout(15):
                await asyncio.shield(task)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        except asyncio.CancelledError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            raise
