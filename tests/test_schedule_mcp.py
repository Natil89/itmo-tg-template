import asyncio
import json
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from app.handlers.assistant import tools
from app.mcp_client import connect_schedule
from app.schedule import DEFAULT_SCHEDULE, ScheduleError, ScheduleService


def test_schedule_has_lessons_and_empty_day():
    service = ScheduleService(DEFAULT_SCHEDULE, "Europe/Moscow")
    assert service.for_date("2026-11-16").model_dump() == {
        "date": "2026-11-16",
        "lessons": [
            {
                "start": "09:00",
                "end": "10:30",
                "title": "Программирование",
                "location": "Учебная аудитория A",
            }
        ],
    }
    assert service.for_date("2026-11-18").lessons == []


@pytest.mark.parametrize("value", ["2026-02-30", "20261116", "16.11.2026", "2026-1-1"])
def test_invalid_date_is_rejected(value):
    with pytest.raises(ScheduleError):
        ScheduleService(DEFAULT_SCHEDULE, "Europe/Moscow").for_date(value)


@pytest.mark.parametrize(
    "instant,start,end",
    [
        (datetime(2027, 1, 1, tzinfo=UTC), "2026-12-28", "2027-01-03"),
        (datetime(2026, 12, 1, tzinfo=UTC), "2026-11-30", "2026-12-06"),
        (datetime(2026, 11, 15, 22, tzinfo=UTC), "2026-11-16", "2026-11-22"),
    ],
)
def test_week_uses_timezone_and_crosses_boundaries(instant, start, end):
    week = ScheduleService(DEFAULT_SCHEDULE, "Europe/Moscow").current_week(instant)
    assert (week.start, week.end, week.timezone) == (start, end, "Europe/Moscow")
    assert len(week.days) == 7
    assert len({day.date for day in week.days}) == 7


@pytest.mark.parametrize("contents", ["[]", '{"2026-11-16": {}}', "not json"])
def test_invalid_source_has_safe_error(tmp_path, contents):
    path = tmp_path / "private-file.json"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ScheduleError) as error:
        ScheduleService(path, "Europe/Moscow").for_date("2026-11-16")
    assert "private-file" not in str(error.value)


async def test_real_stdio_discovery_call_resource_and_rejected_arguments():
    async with connect_schedule() as client:
        assert client.session is not None
        assert {tool.name for tool in client.tools} == {"get_schedule", "get_weather"}
        tool = next(tool for tool in client.tools if tool.name == "get_schedule")
        assert tool.inputSchema["additionalProperties"] is False
        assert tool.outputSchema["properties"]["lessons"]["type"] == "array"
        assert tool.annotations.readOnlyHint is True
        assert [str(resource.uri) for resource in client.resources] == ["schedule://current-week"]
        result = await client.session.call_tool("get_schedule", {"date": "2026-11-16"})
        assert not result.isError
        assert result.structuredContent["lessons"][0]["title"] == "Программирование"
        empty = await client.session.call_tool("get_schedule", {"date": "2026-11-18"})
        assert empty.structuredContent == {"date": "2026-11-18", "lessons": []}
        for name, arguments in [
            ("get_schedule", {"date": "2026-02-30"}),
            ("get_schedule", {"date": "2026-11-16", "path": "forbidden"}),
            ("get_schedule", {"date": 123}),
            ("unknown", {}),
        ]:
            assert (await client.session.call_tool(name, arguments)).isError
        resource = await client.session.read_resource("schedule://current-week")
        assert resource.contents[0].mimeType == "application/json"
        week = json.loads(resource.contents[0].text)
        assert len(week["days"]) == 7
        assert week["timezone"] == "Europe/Moscow"
        message = AsyncMock()
        await tools(message, client)
        assert "get_schedule" in message.answer.call_args.args[0]
        assert "чтение" in message.answer.call_args.args[0]


async def test_tools_connection_failure_is_safe():
    message, client = AsyncMock(), AsyncMock()
    client.refresh.side_effect = ConnectionError("private-secret")
    await tools(message, client)
    assert "недоступны" in message.answer.call_args.args[0]
    assert "private-secret" not in message.answer.call_args.args[0]


async def test_tools_without_connection():
    message = AsyncMock()
    await tools(message)
    assert "недоступны" in message.answer.call_args.args[0]


@pytest.mark.parametrize("after_start", [False, True])
async def test_mcp_startup_and_background_failure_do_not_cancel_host(monkeypatch, after_start):
    import anyio

    from app import mcp_client

    crash = asyncio.Event()

    @asynccontextmanager
    async def broken_transport(parameters):
        if not after_start:
            raise OSError("private-secret")

        async def fail():
            await crash.wait()
            raise BrokenPipeError("private-secret")

        async with anyio.create_task_group() as group:
            group.start_soon(fail)
            yield (None, None)

    session = AsyncMock()
    session.__aenter__.return_value = session
    session.list_tools.return_value.tools = []
    session.list_tools.return_value.nextCursor = None
    session.list_resources.return_value.resources = []
    session.list_resources.return_value.nextCursor = None
    monkeypatch.setattr(mcp_client, "stdio_client", broken_transport)
    monkeypatch.setattr(mcp_client, "ClientSession", lambda *args, **kwargs: session)
    async with connect_schedule() as client:
        if after_start:
            assert client.session is session
            crash.set()
            async with asyncio.timeout(3):
                while client.session is not None:  # noqa: ASYNC110 -- наблюдаем внешний сбой
                    await asyncio.sleep(0.01)
        assert client.session is None
        message = AsyncMock()
        await tools(message, client)
        assert "недоступны" in message.answer.call_args.args[0]


async def test_application_error_keeps_its_type_when_mcp_closes():
    with pytest.raises(RuntimeError, match="polling failed"):
        async with connect_schedule() as client:
            assert client.session is not None
            raise RuntimeError("polling failed")


async def test_real_child_exiting_before_initialization_keeps_host_running(monkeypatch):
    from app import mcp_client

    original = mcp_client.stdio_client

    def exiting_process(parameters):
        parameters.args = ["-c", "pass"]
        return original(parameters)

    monkeypatch.setattr(mcp_client, "stdio_client", exiting_process)
    async with connect_schedule() as client:
        assert client.session is None
        assert client.tools == []


def test_nested_mcp_errors_are_diagnosed_without_secrets():
    from mcp.shared.exceptions import McpError
    from mcp.types import ErrorData

    from app.mcp_client import safe_error_summary

    error = ExceptionGroup(
        "private-secret",
        [
            ExceptionGroup(
                "private-secret",
                [
                    McpError(ErrorData(code=-32001, message="private-secret")),
                    BrokenPipeError("private-secret"),
                ],
            ),
        ],
    )
    assert safe_error_summary(error) == "McpError(code=-32001); BrokenPipeError"


async def test_slow_server_can_initialize_with_separate_startup_budget(monkeypatch):
    from app import mcp_client

    original = mcp_client.stdio_client

    def slow_process(parameters):
        parameters.args = [
            "-c",
            "import time, runpy; time.sleep(11); "
            'runpy.run_module("app.mcp_server", run_name="__main__")',
        ]
        return original(parameters)

    monkeypatch.setattr(mcp_client, "stdio_client", slow_process)
    async with connect_schedule(startup_timeout_seconds=60, request_timeout_seconds=2) as client:
        assert client.session is not None
        assert {tool.name for tool in client.tools} == {"get_schedule", "get_weather"}


async def test_startup_timeout_closes_transport_and_logs_safe_stage(monkeypatch, caplog):
    from app import mcp_client

    closed = asyncio.Event()

    @asynccontextmanager
    async def transport(parameters):
        try:
            yield (None, None)
        finally:
            closed.set()

    async def stalled_initialize():
        await asyncio.Event().wait()

    session = AsyncMock()
    session.__aenter__.return_value = session
    session.initialize.side_effect = stalled_initialize
    monkeypatch.setattr(mcp_client, "stdio_client", transport)
    monkeypatch.setattr(mcp_client, "ClientSession", lambda *args, **kwargs: session)
    async with connect_schedule(startup_timeout_seconds=0.02) as client:
        assert client.session is None
        assert closed.is_set()
    assert "stage=initialize error=TimeoutError" in caplog.text


async def test_cancellation_during_shutdown_is_propagated(monkeypatch):
    from app import mcp_client

    closing = asyncio.Event()

    @asynccontextmanager
    async def slow_transport(parameters):
        try:
            yield (None, None)
        finally:
            closing.set()
            await asyncio.Event().wait()

    session = AsyncMock()
    session.__aenter__.return_value = session
    for method in (session.list_tools, session.list_resources):
        method.return_value.nextCursor = None
    session.list_tools.return_value.tools = []
    session.list_resources.return_value.resources = []
    monkeypatch.setattr(mcp_client, "stdio_client", slow_transport)
    monkeypatch.setattr(mcp_client, "ClientSession", lambda *args, **kwargs: session)

    async def host():
        async with connect_schedule():
            pass

    task = asyncio.create_task(host())
    await asyncio.wait_for(closing.wait(), timeout=3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


def test_schedule_configuration_validates_timezone_and_source(tmp_path):
    from app.config import ConfigError, Settings

    values = {
        "BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
        "POSTGRES_PASSWORD": "fake-db-password",
        "SCHEDULE_FILE": "app/data/schedule.json",
        "SCHEDULE_TIMEZONE": "Europe/Moscow",
    }
    settings = Settings.load(tmp_path / "absent.env", environ=values)
    assert settings.schedule_file.as_posix() == "app/data/schedule.json"
    assert settings.schedule_timezone == "Europe/Moscow"
    with pytest.raises(ConfigError, match="SCHEDULE_TIMEZONE"):
        Settings.load(
            tmp_path / "absent.env",
            environ={
                **values,
                "SCHEDULE_TIMEZONE": "Unknown/Zone",
            },
        )
