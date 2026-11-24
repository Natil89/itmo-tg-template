import os

import pytest
from test_integration import database as database

from app.db import create_pool
from app.tracing import TraceStore, current_trace, recorded, span

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("RUN_INTEGRATION") != "1", reason="Нужен RUN_INTEGRATION=1"),
]


async def test_postgres_persists_tree_and_isolates_trace_reads(database):
    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        store = TraceStore(pool)
        await store.initialize()
        async with recorded(store, 101, "user_query", "telegram", "request") as root:
            trace = current_trace.get()
            trace.route, trace.confidence = "schedule", 0.9
            with span("mcp", "call_tool", tool="get_schedule") as child:
                pass
        async with recorded(store, 202, "daily_briefing", "briefing", "delivery_attempt"):
            pass
        assert await store.get(202, trace.id) == []
        own = await store.get(101)
        assert [row["component"] for row in own] == ["telegram", "mcp"]
        assert own[0]["trace_id"] == trace.id
        assert own[1]["tool_name"] == "get_schedule"
        assert (
            await pool.fetchval(
                "SELECT parent_span_id FROM request_trace_spans WHERE trace_id=$1 AND span_id=$2",
                trace.id,
                child.id,
            )
            == root.id
        )
    finally:
        await pool.close()
    pool = await create_pool(settings)
    try:
        assert len(await TraceStore(pool).get(101, trace.id)) == 2
    finally:
        await pool.close()
