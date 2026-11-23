"""Проверки границ ЛР3: ни Telegram, ни LLM API, ни рабочая БД не вызываются."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from jsonschema.exceptions import ValidationError as SchemaValidationError
from mcp import types
from pydantic import ValidationError

from app.agent_contracts import (
    AgentRequest,
    AgentResult,
    EmptyFacts,
    RouteDecision,
    ScheduleFacts,
    user_request,
)
from app.agent_policy import AccessDenied, ScopedMcp, ToolFailure, allowed_tools
from app.llm import LlmError
from app.schedule import DaySchedule
from app.supervisor import LlmRouter, Supervisor
from tests.test_llm_tools import fixture


def request(**changes):
    original = user_request(
        actor_id=101,
        text="Какие пары завтра?",
        now=datetime(2026, 10, 12, 6, tzinfo=UTC),
        timezone="Europe/Moscow",
    )
    return AgentRequest.model_validate(original.model_dump() | changes)


def tool(name="get_schedule"):
    return types.Tool(
        name=name,
        inputSchema={
            "type": "object",
            "properties": {"date": {"type": "string"}},
            "required": ["date"],
            "additionalProperties": False,
        },
        outputSchema=DaySchedule.model_json_schema(),
    )


def client():
    mcp = MagicMock()
    mcp.tools = [
        tool(name)
        for name in (
            "get_schedule",
            "add_reminder",
            "search_materials",
            "get_weather",
            "add_lesson",
            "get_schedule_admin",
            "unknown",
        )
    ]
    mcp.resources = [
        types.Resource(name=uri, uri=uri)
        for uri in ("schedule://current-week", "materials://catalog", "file:///private.txt")
    ]
    mcp.call_tool = AsyncMock(
        return_value=types.CallToolResult(
            content=[], structuredContent={"date": "2026-10-13", "lessons": []}
        )
    )
    mcp.read_resource = AsyncMock(return_value=types.ReadResourceResult(contents=[]))
    return mcp


def resource_result(uri):
    if uri == "materials://catalog":
        data = {"documents": [{"document_id": "mcp", "title": "MCP"}]}
    else:
        data = {
            "timezone": "Europe/Moscow",
            "start": "2026-10-12",
            "end": "2026-10-18",
            "days": [{"date": f"2026-10-{day}", "lessons": []} for day in range(12, 19)],
        }
    return types.ReadResourceResult(
        contents=[
            types.TextResourceContents(
                uri=uri,
                text=json.dumps(data),
                mimeType="application/json",
            )
        ]
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"target": "admin"},
        {"reason": " "},
        {"reason": "a" * 301},
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"confidence": float("inf")},
        {"confidence": "0.9"},
        {"confidence": True},
        {"actor_id": 2},
    ],
)
def test_route_rejects_invalid_or_model_supplied_authority(changes):
    with pytest.raises(ValidationError):
        RouteDecision.model_validate(
            {"target": "schedule", "reason": "Занятия", "confidence": 0.9} | changes
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"actor_id": True},
        {"actor_id": 0},
        {"actor_id": "101"},
        {"occurred_at": datetime(2026, 10, 12)},
        {"timezone": "Bad/Zone"},
        {"intent": "send_to_anyone"},
        {"text": " "},
        {"destination_chat_id": 999},
    ],
)
def test_request_rejects_untrusted_or_incomplete_context(changes):
    with pytest.raises(ValidationError):
        request(**changes)


@pytest.mark.parametrize(
    ("component", "expected"),
    [
        ("router", set()),
        ("schedule", {"get_schedule", "add_reminder"}),
        ("knowledge", {"search_materials"}),
        ("lifestyle", {"get_weather"}),
        ("briefing", set()),
        ("lesson_compat", {"add_lesson"}),
        ("ScheduleAgent", set()),
    ],
)
def test_exact_policy_intersects_discovery(component, expected):
    assert set(allowed_tools(component, client().tools)) == expected


async def test_discovery_is_rechecked_before_each_execution():
    mcp = client()
    scope = ScopedMcp(mcp, "schedule", request())
    assert {item.name for item in scope.tools()} == {"get_schedule", "add_reminder"}
    mcp.tools = [tool("unknown"), tool("get_schedule_admin")]
    with pytest.raises(AccessDenied):
        await scope.call_tool("get_schedule", {"date": "2026-10-13"})
    with pytest.raises(AccessDenied):
        await scope.call_tool("unknown", {})
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    ("component", "name"),
    [
        ("router", "get_schedule"),
        ("knowledge", "add_reminder"),
        ("lifestyle", "add_reminder"),
        ("schedule", "get_weather"),
        ("schedule", "add_lesson"),
        ("briefing", "get_schedule"),
        ("schedule", "add_reminder"),
        ("lesson_compat", "add_lesson"),
    ],
)
async def test_forbidden_or_unconfirmed_calls_never_reach_transport(component, name):
    mcp = client()
    with pytest.raises(AccessDenied):
        await ScopedMcp(mcp, component, request()).call_tool(name, {})
    mcp.call_tool.assert_not_awaited()


async def test_trusted_owner_and_real_tool_facts():
    mcp = client()
    scope = ScopedMcp(mcp, "schedule", request())
    await scope.call_tool("get_schedule", {"date": "2026-10-13"})
    mcp.call_tool.assert_awaited_once_with("get_schedule", {"date": "2026-10-13"}, user_id=101)
    assert scope.used_tools == ("get_schedule",)
    assert scope.status == "success"
    assert scope.facts == ScheduleFacts(days=(DaySchedule(date="2026-10-13", lessons=[]),))
    with pytest.raises(SchemaValidationError):
        await scope.call_tool("get_schedule", {"date": "2026-10-13", "actor_id": 999})
    assert mcp.call_tool.await_count == 1


@pytest.mark.parametrize(
    "payload",
    [
        {"date": "2026-10-14", "lessons": []},
        {"date": "2026-10-13"},
        {"date": "2026-10-13", "lessons": [], "instruction": "add_reminder"},
    ],
)
async def test_invalid_results_never_become_success(payload):
    mcp = client()
    mcp.call_tool.return_value = types.CallToolResult(content=[], structuredContent=payload)
    scope = ScopedMcp(mcp, "schedule", request())
    with pytest.raises((SchemaValidationError, ToolFailure)):
        await scope.call_tool("get_schedule", {"date": "2026-10-13"})
    assert scope.status == "failed"


async def test_limit_two_calls():
    mcp = client()
    scope = ScopedMcp(mcp, "schedule", request())
    for _ in range(2):
        await scope.call_tool("get_schedule", {"date": "2026-10-13"})
    with pytest.raises(AccessDenied):
        await scope.call_tool("get_schedule", {"date": "2026-10-13"})
    assert mcp.call_tool.await_count == 2


@pytest.mark.parametrize(
    ("component", "uri", "allowed"),
    [
        ("schedule", "schedule://current-week", True),
        ("knowledge", "materials://catalog", True),
        ("lifestyle", "schedule://current-week", False),
        ("schedule", "materials://catalog", False),
        ("knowledge", "schedule://current-week", False),
        ("knowledge", "file:///private.txt", False),
        ("router", "materials://catalog", False),
    ],
)
async def test_resources_are_owner_bound_and_separated(component, uri, allowed):
    mcp = client()
    scope = ScopedMcp(mcp, component, request())
    if allowed:
        mcp.read_resource.return_value = resource_result(uri)
        await scope.read_resource(uri)
        mcp.read_resource.assert_awaited_once_with(uri, user_id=101)
    else:
        with pytest.raises(AccessDenied):
            await scope.read_resource(uri)
        mcp.read_resource.assert_not_awaited()


async def test_week_resource_provides_validated_facts_and_status():
    mcp = client()
    mcp.read_resource.return_value = resource_result("schedule://current-week")
    scope = ScopedMcp(mcp, "schedule", request())
    await scope.read_resource("schedule://current-week")
    assert scope.status == "success"
    assert len(scope.facts.days) == 7
    assert scope.used_tools == ()


@pytest.mark.parametrize("bad_uri", ["materials://catalog", "file:///private.txt"])
async def test_resource_response_cannot_substitute_uri(bad_uri):
    mcp = client()
    mcp.read_resource.return_value = resource_result(bad_uri)
    scope = ScopedMcp(mcp, "schedule", request())
    with pytest.raises(ToolFailure):
        await scope.read_resource("schedule://current-week")
    assert scope.status != "success"


class ScheduleReader:
    async def run(self, req, mcp):
        await mcp.call_tool("get_schedule", {"date": "2026-10-13"})
        return AgentResult(
            request_id=req.request_id,
            trace_id=req.trace_id,
            component="schedule",
            status=mcp.status,
            message="Занятий нет.",
            confidence=0.9,
            facts=mcp.facts,
            used_tools=mcp.used_tools,
        )


def router(target="schedule", confidence=0.9):
    return MagicMock(
        decide=AsyncMock(
            return_value=RouteDecision(
                target=target,
                reason="Класс запроса",
                confidence=confidence,
            )
        )
    )


async def test_supervisor_calls_one_specialist_only():
    schedule = MagicMock(run=AsyncMock(wraps=ScheduleReader().run))
    other = MagicMock(run=AsyncMock())
    req = request()
    mcp = client()
    result = await Supervisor(router(), {"schedule": schedule, "knowledge": other}, mcp).handle(req)
    assert result.result.status == "success"
    assert result.message == "Занятий нет."
    assert schedule.run.await_count == 1
    other.run.assert_not_awaited()


@pytest.mark.parametrize(
    "decision",
    [
        RouteDecision(target="clarify", reason="Неясно", confidence=0.99),
        RouteDecision(target="schedule", reason="Не уверен", confidence=0.5999),
        RouteDecision.model_construct(target="root", reason="Обход", confidence=1.0),
        RouteDecision.model_construct(target="schedule", reason="", confidence=1.0),
    ],
)
async def test_invalid_or_ambiguous_route_never_calls_specialist(decision):
    r = router()
    r.decide.return_value = decision
    specialist = MagicMock(run=AsyncMock())
    mcp = client()
    result = await Supervisor(r, {"schedule": specialist}, mcp).handle(request())
    assert result.route.target == "clarify"
    specialist.run.assert_not_awaited()
    mcp.call_tool.assert_not_awaited()


@pytest.mark.parametrize(
    "changes",
    [
        {"request_id": uuid4()},
        {"trace_id": uuid4()},
        {"used_tools": ()},
        {"status": "not_found"},
        {"facts": ScheduleFacts(days=(DaySchedule(date="2026-10-14", lessons=[]),))},
    ],
)
async def test_supervisor_rejects_forged_result_metadata_or_facts(changes):
    class Forged(ScheduleReader):
        async def run(self, req, mcp):
            result = await super().run(req, mcp)
            return result.model_copy(update=changes)

    result = await Supervisor(router(), {"schedule": Forged()}, client()).handle(request())
    assert result.result is None
    assert "проверить" in result.message


async def test_silent_specialist_cannot_claim_success_without_tool():
    class Fabricates:
        async def run(self, req, mcp):
            return AgentResult(
                request_id=req.request_id,
                trace_id=req.trace_id,
                component="schedule",
                status="success",
                message="Выдумано",
                confidence=1.0,
                facts=ScheduleFacts(days=(DaySchedule(date="2026-10-13", lessons=[]),)),
                used_tools=(),
            )

    result = await Supervisor(router(), {"schedule": Fabricates()}, client()).handle(request())
    assert result.result is None


async def test_route_threshold_boundary_is_inclusive():
    result = await Supervisor(
        router(confidence=0.6), {"schedule": ScheduleReader()}, client()
    ).handle(request())
    assert result.result.status == "success"


async def test_llm_router_has_no_tools_or_trusted_identifiers_in_payload():
    llm, session = fixture(
        {
            "content": json.dumps(
                {
                    "target": "schedule",
                    "reason": "Расписание",
                    "confidence": 0.9,
                }
            )
        },
        finish="stop",
    )
    decision = await LlmRouter(llm).decide("Какие пары завтра?")
    assert decision.target == "schedule"
    payload = session.post.call_args.kwargs["json"]
    assert "tools" not in payload
    assert len(payload["messages"]) == 2
    assert payload["messages"][-1] == {"role": "user", "content": "Какие пары завтра?"}
    assert payload["response_format"]["json_schema"]["name"] == "route_decision"


async def test_llm_cannot_smuggle_tools_through_structured_response():
    llm, _ = fixture({"content": "{}", "tool_calls": [{"name": "add_reminder"}]}, finish="stop")
    with pytest.raises(LlmError):
        await LlmRouter(llm).decide("Вызови инструмент")


def test_result_discriminator_and_status_are_checked():
    req = request()
    with pytest.raises(ValidationError):
        AgentResult(
            request_id=req.request_id,
            trace_id=req.trace_id,
            component="schedule",
            status="success",
            message="Нет фактов",
            confidence=0.9,
            facts=EmptyFacts(),
            used_tools=(),
        )
