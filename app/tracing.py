"""Метаданные обработки, изолированные от пользовательского текста и ответов."""

import asyncio
import logging
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import monotonic
from typing import Literal
from uuid import UUID, uuid4

import asyncpg

Component = Literal[
    "telegram",
    "supervisor",
    "router",
    "schedule",
    "knowledge",
    "lifestyle",
    "lesson_compat",
    "llm",
    "mcp",
    "briefing",
    "reminder_delivery",
]
Operation = Literal[
    "request",
    "route",
    "handle",
    "prepare_lesson",
    "completion",
    "call_tool",
    "read_resource",
    "delivery_attempt",
    "send",
    "read_schedule",
    "read_reminders",
    "read_weather",
    "read_focus",
    "confirm_lesson",
    "confirm_reminder",
]
Status = Literal["ok", "failed", "cancelled", "clarify", "not_found", "skipped", "retry", "partial"]
ErrorCode = Literal[
    "timeout",
    "internal",
    "invalid_route",
    "tool_failed",
    "policy",
    "temporary",
    "permanent",
    "late",
]
TOOLS = frozenset({"get_schedule", "get_weather", "search_materials", "add_reminder", "add_lesson"})
logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS request_traces (
    trace_id UUID PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    actor_id BIGINT NOT NULL CHECK(actor_id > 0),
    kind TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    route TEXT,
    confidence DOUBLE PRECISION CHECK(confidence BETWEEN 0 AND 1)
);
CREATE INDEX IF NOT EXISTS request_traces_owner_time
    ON request_traces(actor_id,started_at DESC,trace_id DESC);
CREATE TABLE IF NOT EXISTS request_trace_spans (
    trace_id UUID NOT NULL REFERENCES request_traces(trace_id) ON DELETE CASCADE,
    span_id UUID NOT NULL,
    parent_span_id UUID,
    ordinal INTEGER NOT NULL,
    component TEXT NOT NULL,
    operation TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL,
    finished_at TIMESTAMPTZ NOT NULL,
    duration_ms INTEGER NOT NULL CHECK(duration_ms >= 0),
    status TEXT NOT NULL,
    tool_name TEXT,
    input_tokens BIGINT,
    output_tokens BIGINT,
    error_code TEXT,
    PRIMARY KEY(trace_id,span_id),
    UNIQUE(trace_id,ordinal),
    FOREIGN KEY(trace_id,parent_span_id) REFERENCES request_trace_spans(trace_id,span_id)
);
"""


@dataclass
class Span:
    component: Component
    operation: Operation
    parent_id: UUID | None
    tool_name: str | None = None
    id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    duration_ms: int = 0
    status: Status = "ok"
    error_code: ErrorCode | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    def provider_usage(self, usage: object) -> None:
        if not isinstance(usage, dict):
            return
        for field_name, key in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
        ):
            value = usage.get(key)
            if type(value) is int and 0 <= value < 2**63:
                setattr(self, field_name, value)


@dataclass
class Trace:
    actor_id: int
    kind: Literal["user_query", "daily_briefing", "reminder_delivery", "confirmation"]
    id: UUID = field(default_factory=uuid4)
    request_id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    route: str | None = None
    confidence: float | None = None
    spans: list[Span] = field(default_factory=list)


current_trace: ContextVar[Trace | None] = ContextVar("current_trace", default=None)
_parent: ContextVar[Span | None] = ContextVar("trace_parent", default=None)


def mark_failed(code: ErrorCode = "internal") -> None:
    step = _parent.get()
    if step is not None:
        step.status, step.error_code = "failed", code


@contextmanager
def span(component: Component, operation: Operation, *, tool: str | None = None) -> Iterator[Span]:
    parent = _parent.get()
    step = Span(
        component, operation, parent.id if parent else None, tool if tool in TOOLS else None
    )
    trace = current_trace.get()
    if trace is not None:
        trace.spans.append(step)
    token = _parent.set(step)
    start = monotonic()
    try:
        yield step
    except asyncio.CancelledError:
        step.status = "cancelled"
        raise
    except Exception as error:
        step.status = "failed"
        step.error_code = step.error_code or (
            "timeout" if isinstance(error, TimeoutError) else "internal"
        )
        raise
    finally:
        step.duration_ms = round((monotonic() - start) * 1000)
        step.finished_at = datetime.now(UTC)
        _parent.reset(token)


class TraceStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self.pool = pool

    async def initialize(self) -> None:
        await self.pool.execute(SCHEMA)

    async def save(self, trace: Trace) -> None:
        async with self.pool.acquire() as connection, connection.transaction():
            await connection.execute(
                "INSERT INTO request_traces VALUES($1,$2,$3,$4,$5,$6,$7)",
                trace.id,
                trace.request_id,
                trace.actor_id,
                trace.kind,
                trace.started_at,
                trace.route,
                trace.confidence,
            )
            await connection.executemany(
                "INSERT INTO request_trace_spans VALUES("
                "$1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)",
                [
                    (
                        trace.id,
                        item.id,
                        item.parent_id,
                        index,
                        item.component,
                        item.operation,
                        item.started_at,
                        item.finished_at,
                        item.duration_ms,
                        item.status,
                        item.tool_name,
                        item.input_tokens,
                        item.output_tokens,
                        item.error_code,
                    )
                    for index, item in enumerate(trace.spans)
                ],
            )

    async def get(self, actor_id: int, trace_id: UUID | None = None):
        rows = await self.pool.fetch(
            "WITH selected AS (SELECT trace_id,kind,started_at,route,confidence "
            "FROM request_traces WHERE actor_id=$1 AND ($2::uuid IS NULL OR trace_id=$2) "
            "ORDER BY started_at DESC,trace_id DESC LIMIT 1) "
            "SELECT t.*,s.ordinal,s.component,s.operation,s.status,s.duration_ms,"
            "s.tool_name,s.input_tokens,s.output_tokens,s.error_code "
            "FROM selected t JOIN request_trace_spans s USING(trace_id) ORDER BY s.ordinal",
            actor_id,
            trace_id,
        )
        return rows


@asynccontextmanager
async def recorded(
    store: TraceStore | None,
    actor_id: int,
    kind: Literal["user_query", "daily_briefing", "reminder_delivery", "confirmation"],
    component: Component,
    operation: Operation,
) -> AsyncIterator[Span]:
    trace = Trace(actor_id, kind) if store is not None else None
    token = current_trace.set(trace)
    parent_token = _parent.set(None)
    try:
        with span(component, operation) as root:
            yield root
    finally:
        _parent.reset(parent_token)
        current_trace.reset(token)
        if store is not None and trace is not None:
            try:
                async with asyncio.timeout(2):
                    await store.save(trace)
            except Exception:
                logger.warning("Trace persistence unavailable")
