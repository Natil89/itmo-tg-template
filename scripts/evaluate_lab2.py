"""Воспроизводимый routing experiment: реальная LLM, изолированные тестовые данные."""

import argparse
import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from aiohttp import ClientSession
from mcp import types

from app.audit import AuditTrace
from app.config import ConfigError, Settings
from app.llm import LlmClient, LlmError
from app.mcp_server import create_server
from app.modes import MODES
from app.personal_schedule import PendingLesson, PersonalScheduleError
from app.reminders import PendingReminder, ReminderDraft, ReminderError, _normalize
from app.schedule import DEFAULT_SCHEDULE, DaySchedule, ScheduleService
from app.schedule_agent import ScheduleAgent
from app.weather import WeatherResult

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "docs/evaluation/lab2-routing.json"
SOURCES = (
    "app/schedule_agent.py",
    "app/llm.py",
    "app/mcp_server.py",
    "app/modes.py",
    "app/reminders.py",
    "app/audit.py",
    "app/weather.py",
    "app/schedule.py",
    "scripts/evaluate_lab2.py",
)


def canonical(value: str) -> str:
    return " ".join(value.casefold().replace("ё", "е").strip().rstrip(".!?").split())


def arguments_correct(case: dict, action: str, arguments: dict) -> bool | None:
    expected = case["expected_arguments"]
    if case["expected_action"] in {"respond", "clarify"}:
        return None
    if action != case["expected_action"] or set(arguments) != set(expected):
        return False
    if action == "add_reminder":
        try:
            ReminderDraft.model_validate(arguments)
        except ValueError:
            return False
    for key, value in expected.items():
        actual = arguments[key]
        if not isinstance(actual, str):
            return False
        if key == "remind_at":
            try:
                wanted, received = datetime.fromisoformat(value), datetime.fromisoformat(actual)
                if received.utcoffset() is None or received != wanted:
                    return False
                if received.utcoffset() != wanted.utcoffset():
                    return False
            except ValueError:
                return False
        elif key == "city":
            if not 1 <= len(actual.strip()) <= 100:
                return False
            # Запятая является частью контракта country/admin1 фильтра геокодера.
            actual_city = ",".join(
                " ".join(part.casefold().replace("ё", "е").split()) for part in actual.split(",")
            )
            expected_city = ",".join(
                " ".join(part.casefold().replace("ё", "е").split()) for part in value.split(",")
            )
            if actual_city != expected_city:
                return False
        elif key == "date":
            if actual != value:
                return False
        elif canonical(actual) != canonical(value):
            return False
    return True


def metrics(rows: list[dict]) -> dict:
    required = [row for row in rows if row["arguments_correct"] is not None]
    return {
        "routing_correct": sum(row["routing_correct"] for row in rows),
        "routing_total": len(rows),
        "arguments_correct": sum(row["arguments_correct"] for row in required),
        "arguments_total": len(required),
    }


class RecordingResponse:
    def __init__(self, response, metadata: dict):
        self._response = response
        self._metadata = metadata

    @property
    def status(self):
        return self._response.status

    async def json(self):
        data = await self._response.json()
        if isinstance(data, dict):
            for key in ("model", "provider"):
                if isinstance(data.get(key), str):
                    self._metadata[key] = data[key]
            usage = data.get("usage", {})
            if isinstance(usage, dict):
                self._metadata["usage"] = {
                    key: usage[key]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if type(usage.get(key)) is int
                }
        return data


class RecordingRequest:
    def __init__(self, request, responses: list[dict]):
        self._request = request
        self._responses = responses

    async def __aenter__(self):
        response = await self._request.__aenter__()
        metadata = {"status": response.status}
        self._responses.append(metadata)
        return RecordingResponse(response, metadata)

    async def __aexit__(self, *args):
        return await self._request.__aexit__(*args)


class RecordingSession:
    def __init__(self, session):
        self._session = session
        self.responses: list[dict] = []

    def post(self, *args, **kwargs):
        return RecordingRequest(self._session.post(*args, **kwargs), self.responses)


class RecordingLlm(LlmClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.first_decision: dict | None = None

    async def complete_with_tools(self, *args, **kwargs):
        decision = await super().complete_with_tools(*args, **kwargs)
        if self.first_decision is None:
            self.first_decision = decision
        return decision


class MemoryReminders:
    def __init__(self, zone: str):
        self.zone = zone
        self.prepared = []

    async def get_timezone(self, user_id):
        return self.zone

    async def prepare(self, user_id, update_id, draft, *, now):
        normalized = _normalize(draft, self.zone)
        if datetime.fromisoformat(normalized.remind_at) <= now:
            raise ReminderError("Время не находится в будущем тестового контекста.")
        result = PendingReminder(
            "00000000-0000-0000-0000-000000000001",
            user_id,
            normalized,
            self.zone,
            now + timedelta(minutes=5),
            "prepared",
        )
        self.prepared.append(result)
        return result


class MemoryLessons:
    async def prepare(self, user_id, update_id, draft, *, now):
        return PendingLesson(
            "00000000-0000-0000-0000-000000000002",
            user_id,
            draft,
            now + timedelta(minutes=5),
            "prepared",
        )


class FixtureMcp:
    def __init__(self, tools, now: datetime):
        self.tools = tools
        self.session = object()
        self.request_timeout_seconds = 10
        self.now = now
        self.calls: list[dict] = []

    async def call_tool(self, name, arguments, **kwargs):
        self.calls.append({"name": name, "arguments": arguments})
        if name == "get_schedule":
            data = DaySchedule(date=arguments["date"], lessons=[]).model_dump()
        elif name == "get_weather":
            query = arguments["city"]
            data = WeatherResult.model_validate(
                {
                    "status": "ok",
                    "weather": {
                        "city": query.split(",", 1)[0],
                        "country": "Грузия" if "Тбилиси" in query else "Россия",
                        "admin1": "",
                        "temperature_c": 10.0,
                        "wind_speed": 2.0,
                        "weather_code": 3,
                        "observed_at": self.now.isoformat(),
                        "timezone": "Europe/Moscow",
                        "source": "Open-Meteo",
                    },
                }
            ).model_dump()
        else:
            raise RuntimeError("Запись через MCP в эксперименте запрещена.")
        return types.CallToolResult(content=[], structuredContent=data)


async def evaluate_case(case, index, tools, settings, session, context):
    recording = RecordingSession(session)
    llm = RecordingLlm(
        recording,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        history_limits=settings.history_limits,
    )
    now = datetime.fromisoformat(context["now"])
    reminders = MemoryReminders(context["timezone"])
    mcp = FixtureMcp(tools, now)
    agent = ScheduleAgent(
        llm, mcp, context["timezone"], personal_store=MemoryLessons(), reminders=reminders
    )
    trace = AuditTrace()
    answer, error = None, None
    try:
        result = await agent.answer(
            case["query"],
            system_prompt=MODES["study"].system_prompt,
            temperature=0.3,
            now=now,
            user_id=1,
            update_id=index,
            audit=trace,
        )
        answer = result if isinstance(result, str) else "Подготовлено предложение с подтверждением."
    except (LlmError, ReminderError, PersonalScheduleError) as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    except Exception as exc:
        error = {"type": type(exc).__name__}
    first = llm.first_decision or {}
    calls = first.get("tool_calls", [])
    actual, arguments = "undetermined", {}
    if calls:
        actual = " + ".join(item["function"]["name"] for item in calls)
        try:
            arguments = json.loads(calls[0]["function"]["arguments"])
            if not isinstance(arguments, dict):
                arguments = {}
        except ValueError:
            arguments = {}
    elif first and trace.steps:
        actual = trace.steps[-1].action
    correct = arguments_correct(case, actual, arguments)
    return {
        **case,
        "actual_action": actual,
        "actual_arguments": arguments,
        "routing_correct": actual == case["expected_action"],
        "arguments_correct": correct,
        "answer": answer,
        "error": error,
        "duration_ms": trace.duration_ms,
        "first_decision": first,
        "audit": [step.model_dump() for step in trace.steps],
        "mcp_calls": mcp.calls,
        "prepared_reminders": [item.payload.model_dump() for item in reminders.prepared],
        "api_responses": recording.responses,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Лабораторная 2: эксперимент маршрутизации",
        "",
        f"Дата запуска: {report['started_at']}",
        f"Модель: `{report['requested_model']}`; temperature=0.3; history=[]; weather=strict.",
        f"Фиксированный контекст: `{report['evaluation_context']}`.",
        "",
        "Реальная LLM и схемы из list_tools приложения. Результаты чтений — фикстуры;",
        "подготовка изменений — в памяти. Рабочая БД, погодный API и Telegram не вызываются.",
        "Проверяется первое native решение и его аргументы. Финальный текст не влияет на метрики.",
        "Для текста напоминания регистр, ё/е и конечная пунктуация игнорируются; "
        "для города запятая значима; дата строгая; момент и offset сравниваются.",
        "",
        "| № | Запрос | Ожидается | Фактически | Аргументы | Результат |",
        "|---:|---|---|---|---|---|",
    ]
    for index, row in enumerate(report["cases"], 1):
        args = (
            "—" if row["arguments_correct"] is None else "да" if row["arguments_correct"] else "нет"
        )
        passed = row["routing_correct"] and row["arguments_correct"] is not False
        query = row["query"].replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {index} | {query} | {row['expected_action']} | {row['actual_action']} | "
            f"{args} | {'PASS' if passed else 'FAIL'} |"
        )
    score = report["metrics"]
    lines += [
        "",
        f"Маршрутизация: **{score['routing_correct']}/{score['routing_total']}**.",
        f"Аргументы: **{score['arguments_correct']}/{score['arguments_total']}**.",
    ]
    if len(report["cases"]) == 12:
        passed = score["routing_correct"] >= 10 and score["arguments_correct"] >= 8
        lines.append(f"Порог методички (10/12 и 8/9): **{'PASS' if passed else 'FAIL'}**.")
    if report.get("retry_from"):
        lines += ["", f"Повтор только ошибок из `{report['retry_from']}`.", report["change_note"]]
    lines += ["", "## Фактические аргументы и ошибки", ""]
    for row in report["cases"]:
        lines += [
            f"### {row['id']}",
            "",
            "```json",
            json.dumps(row["actual_arguments"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
        if row["error"]:
            lines.append(f"Ошибка обработки: {row['error']}\n")
    return "\n".join(lines) + "\n"


async def run(args):
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if len(cases) != 12 or sum(bool(c["expected_arguments"]) for c in cases) != 9:
        raise ValueError("Базовый набор должен содержать 12 запросов, 9 с аргументами.")
    settings = Settings.load(require_llm=True)
    if args.retry_from:
        baseline = json.loads(args.retry_from.read_text(encoding="utf-8"))
        if baseline["requested_model"] != settings.llm_model or not args.change_note:
            raise ValueError("Для повтора нужны прежняя модель и описание одного изменения.")
        failed = {
            row["id"]
            for row in baseline["cases"]
            if not row["routing_correct"] or row["arguments_correct"] is False or row["error"]
        }
        cases = [case for case in cases if case["id"] in failed]
    output = args.output
    md = output.with_suffix(".md")
    if output.exists() or md.exists():
        raise ValueError("Выходной файл уже существует. Выберите новое имя.")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "in_progress",
        "started_at": datetime.now(ZoneInfo("Europe/Moscow")).isoformat(),
        "dataset_sha256": hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES
        },
        "requested_model": settings.llm_model,
        "temperature": 0.3,
        "max_tokens": 1000,
        "context_budget": settings.history_limits.context_budget,
        "history_max_messages": settings.history_limits.max_messages,
        "evaluation_context": dataset["evaluation_context"],
        "history": [],
        "weather_policy": "strict",
        "retry_from": str(args.retry_from) if args.retry_from else None,
        "change_note": args.change_note,
        "cases": [],
    }
    with output.open("x", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    server = create_server(
        ScheduleService(DEFAULT_SCHEDULE, dataset["evaluation_context"]["timezone"]),
        weather=SimpleNamespace(location_policy="strict"),
        personal_store=MemoryLessons(),
        reminders=MemoryReminders(dataset["evaluation_context"]["timezone"]),
    )
    discovered = await server.request_handlers[types.ListToolsRequest](
        types.ListToolsRequest(method="tools/list")
    )
    schema = [tool.model_dump(mode="json") for tool in discovered.root.tools]
    report["tool_names"] = [tool.name for tool in discovered.root.tools]
    report["tool_schemas_sha256"] = hashlib.sha256(
        json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    async with ClientSession() as session:
        for index, case in enumerate(cases, 1):
            row = await evaluate_case(
                case, index, discovered.root.tools, settings, session, dataset["evaluation_context"]
            )
            report["cases"].append(row)
            output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "actual": row["actual_action"],
                        "route": row["routing_correct"],
                        "arguments": row["arguments_correct"],
                    }
                ),
                flush=True,
            )
    report["status"] = "complete"
    report["metrics"] = metrics(report["cases"])
    report["reported_models"] = sorted(
        {
            response["model"]
            for row in report["cases"]
            for response in row["api_responses"]
            if "model" in response
        }
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with md.open("x", encoding="utf-8") as file:
        file.write(markdown(report))
    print(json.dumps(report["metrics"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--retry-from", type=Path)
    parser.add_argument("--change-note", default="")
    args = parser.parse_args()
    try:
        asyncio.run(run(args))
    except (ConfigError, ValueError, OSError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
