"""Замер маршрутов через настроенную модель. Данные и подтверждения изолированы."""

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from aiohttp import ClientSession
from mcp import types

from app.agent_contracts import MaterialCatalog, MaterialEntry, MaterialHit, user_request
from app.config import Settings
from app.llm import LlmClient
from app.materials import SearchQuery, SearchResult
from app.mcp_server import create_server
from app.schedule import DEFAULT_SCHEDULE, ScheduleService
from app.specialists import KnowledgeAgent, LifestyleAgent, ScheduleSpecialist
from app.supervisor import LlmRouter, Supervisor
from scripts.evaluate_lab2 import FixtureMcp, MemoryReminders, RecordingSession

DATASET = Path("docs/evaluation/lab3-routing.json")
DATASET_SHA256 = "3a1e058d8f5c467eb84ac775f886b148b14a74c326015ebda9d175e26fb39236"


class SyntheticSchedule(ScheduleService):
    def _load(self):
        return {}


class SyntheticMaterials:
    """Тексты стенда заданы здесь и не читаются из пользовательских файлов."""

    hits = (
        MaterialHit(document_id="mcp", title="MCP", excerpt="MCP даёт доступ к инструментам."),
        MaterialHit(
            document_id="context",
            title="Контекстное окно",
            excerpt="Контекстное окно ограничивает объём данных, доступных модели.",
        ),
        MaterialHit(
            document_id="structured",
            title="Структурированный вывод",
            excerpt="Структурированный вывод проверяют по схеме.",
        ),
        MaterialHit(
            document_id="idempotency",
            title="Идемпотентность",
            excerpt="Повтор операции сохраняет результат первого применения.",
        ),
    )

    def catalog(self):
        return MaterialCatalog(
            documents=tuple(
                MaterialEntry(document_id=hit.document_id, title=hit.title) for hit in self.hits
            )
        )

    def search(self, query: SearchQuery):
        return SearchResult(hits=self.hits[: query.limit])


class ObservedRouter(LlmRouter):
    decision = None

    async def decide(self, *args, **kwargs):
        self.decision = await super().decide(*args, **kwargs)
        return self.decision


class ReadFixtures(FixtureMcp):
    def __init__(self, tools, resources, now, schedule, materials):
        super().__init__(tools, now)
        self.resources = resources
        self.schedule = schedule
        self.materials = materials
        self.reads = []

    async def call_tool(self, name, arguments, **kwargs):
        if name == "search_materials":
            self.calls.append({"name": name})
            data = self.materials.search(SearchQuery.model_validate(arguments))
            return types.CallToolResult(content=[], structuredContent=data.model_dump(mode="json"))
        return await super().call_tool(name, arguments, **kwargs)

    async def read_resource(self, uri, *, user_id):
        self.reads.append(uri)
        if uri == "schedule://current-week":
            result = self.schedule.current_week(self.now)
        elif uri == "materials://catalog":
            result = self.materials.catalog()
        else:
            raise ValueError("Неизвестный ресурс набора.")
        return types.ReadResourceResult(
            contents=[types.TextResourceContents(uri=uri, text=result.model_dump_json())]
        )


async def run(args):
    raw = (await asyncio.to_thread(DATASET.read_text, encoding="utf-8")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != DATASET_SHA256:
        raise ValueError("Контрольная сумма набора ЛР3 не совпала.")
    dataset = json.loads(raw)
    assert len(dataset["cases"]) == 15
    context = dataset["evaluation_context"]
    now = datetime.fromisoformat(context["now"])
    settings = Settings.load(require_llm=True)
    if settings.llm_base_url.rstrip("/") != "https://openrouter.ai/api/v1":
        raise ValueError("Проверка рассчитана на официальный endpoint OpenRouter.")
    schedule = SyntheticSchedule(DEFAULT_SCHEDULE, context["timezone"])
    materials = SyntheticMaterials()
    server = create_server(
        schedule,
        weather=SimpleNamespace(location_policy="strict"),
        reminders=MemoryReminders(context["timezone"]),
        materials=materials,
    )
    tools = (
        await server.request_handlers[types.ListToolsRequest](
            types.ListToolsRequest(method="tools/list")
        )
    ).root.tools
    resources = (
        await server.request_handlers[types.ListResourcesRequest](
            types.ListResourcesRequest(method="resources/list")
        )
    ).root.resources
    cases = dataset["cases"]
    model = settings.llm_model
    if args.retry_from:
        previous = json.loads(args.retry_from.read_text(encoding="utf-8"))
        if previous["model"] != model or previous["dataset_sha256"] != digest or not args.change:
            raise ValueError("Для повтора нужны тот же набор, та же модель и описание изменения.")
        failed = {row["id"] for row in previous["cases"] if not row["correct"]}
        cases = [case for case in cases if case["id"] in failed]
    results = {
        "started_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "model": model,
        "temperature": 0.3,
        "supervisor_timeout_seconds": args.timeout,
        "dataset_sha256": digest,
        "evaluation_context": context,
        "retry_from": str(args.retry_from) if args.retry_from else None,
        "change": args.change,
        "cases": [],
    }
    async with ClientSession() as session:
        for index, case in enumerate(cases, 1):
            recording = RecordingSession(session)
            llm = LlmClient(
                recording,
                api_key=settings.llm_api_key,
                model=model,
                base_url=settings.llm_base_url,
                history_limits=settings.history_limits,
            )
            reminders = MemoryReminders(context["timezone"])
            mcp = ReadFixtures(tools, resources, now, schedule, materials)
            router = ObservedRouter(llm)
            supervisor = Supervisor(
                router,
                {
                    "schedule": ScheduleSpecialist(llm),
                    "knowledge": KnowledgeAgent(llm),
                    "lifestyle": LifestyleAgent(llm),
                },
                mcp,
                reminders=reminders,
                timeout_seconds=args.timeout,
                clock=lambda: now,
            )
            request = user_request(
                actor_id=101, text=case["query"], now=now, timezone=context["timezone"]
            )
            started = perf_counter()
            response = await supervisor.handle(request, update_id=index)
            actual = response.route.target if router.decision is not None else None
            row = {
                **case,
                "actual_target": actual,
                "correct": actual == case["expected_target"],
                "confidence": response.route.confidence,
                "duration_ms": round((perf_counter() - started) * 1000),
                "tools": [call["name"] for call in mcp.calls],
                "resources": mcp.reads,
                "reminder_prepared_in_memory": bool(response.pending_reminder),
                "specialist_status": response.result.status if response.result else None,
                "provider_responses": recording.responses,
            }
            results["cases"].append(row)
            print(f"{case['id']}: {actual} ({row['duration_ms']} ms)", file=sys.stderr, flush=True)
    results["correct"] = sum(row["correct"] for row in results["cases"])
    results["total"] = len(cases)
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            json.dump(results, output, ensure_ascii=False, indent=2)
            output.write("\n")
    else:
        print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument("--change")
    parser.add_argument("--timeout", type=float, default=30)
    asyncio.run(run(parser.parse_args()))
