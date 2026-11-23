"""Ограниченный аудит решений без истории диалогов и доверенных идентификаторов."""

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from time import monotonic
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import asyncpg
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


class AuditError(ValueError):
    """Безопасная ошибка чтения или записи аудита."""


Action = Literal["respond", "clarify", "tool", "undetermined"]
Validation = Literal["not_required", "pending", "passed", "rejected"]
Execution = Literal[
    "not_run", "success", "error", "awaiting_confirmation", "cancelled", "expired", "limit_exceeded"
]
Reason = Literal[
    "no_tool_needed",
    "missing_details",
    "live_weather",
    "personal_schedule",
    "requires_confirmation",
    "invalid_arguments",
    "tool_limit",
    "source_ambiguous",
    "city_not_found",
    "request_failed",
    "confirmation_accepted",
    "cancelled_by_user",
    "unavailable_confirmation",
    "classification_unavailable",
]
_TOOLS = frozenset({"get_schedule", "get_weather", "add_lesson", "add_reminder"})
_TIME = re.compile(r"^(?:[01]\d|2[0-3]):[0-5]\d$")
_INSTANT = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2}(?:\.\d{1,6})?)?(?:Z|[+-]\d{2}:\d{2})$"
)


def _safe_date(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = date.fromisoformat(value)
        return value if parsed.isoformat() == value else None
    except ValueError:
        return None


def _safe_city(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    city = value.strip()
    if not 1 <= len(city) <= 100:
        return None
    if not all(char.isalpha() or char in " ,'-’." for char in city):
        return None
    words = re.split(r"[ ,'.’\-]+", city)
    if not any(words) or any(len(word) > 30 for word in words):
        return None
    return city


def _length(arguments: dict, name: str, maximum: int) -> int | None:
    raw = arguments.get(name)
    if isinstance(raw, str):
        return min(len(raw.strip()), maximum)
    # Повторная проверка уже сокращённой записи сохраняет безопасные длины.
    value = arguments.get(f"{name}_length")
    if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum:
        return value
    return None


def safe_arguments(name: str | None, args: object) -> dict:
    """Возвращает только безопасное резюме заранее известных аргументов."""
    if not isinstance(name, str) or name not in _TOOLS or not isinstance(args, dict):
        return {}
    result: dict = {}
    if name in {"get_schedule", "add_lesson"}:
        day = _safe_date(args.get("date"))
        if day is not None:
            result["date"] = day
    if name == "get_weather":
        city = _safe_city(args.get("city"))
        if city is not None:
            result["city"] = city
    if name == "add_lesson":
        for key in ("start", "end"):
            value = args.get(key)
            if isinstance(value, str) and _TIME.fullmatch(value):
                result[key] = value
        for key in ("title", "location"):
            length = _length(args, key, 200)
            if length is not None:
                result[f"{key}_length"] = length
    if name == "add_reminder":
        value = args.get("remind_at")
        if isinstance(value, str) and _INSTANT.fullmatch(value):
            try:
                instant = datetime.fromisoformat(value)
                if instant.utcoffset() is not None:
                    result["remind_at"] = instant.isoformat()
            except ValueError:
                pass
        length = _length(args, "text", 500)
        if length is not None:
            result["text_length"] = length
    return result


class AuditStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, validate_assignment=True)

    action: Action
    tool: str | None = None
    arguments: dict = Field(default_factory=dict)
    validation: Validation = "not_required"
    execution: Execution = "not_run"
    reason: Reason = "request_failed"

    @field_validator("tool", mode="before")
    @classmethod
    def known_tool(cls, value: object) -> str | None:
        return value if isinstance(value, str) and value in _TOOLS else None

    @model_validator(mode="after")
    def sanitize_arguments(self) -> "AuditStep":
        object.__setattr__(self, "arguments", safe_arguments(self.tool, self.arguments))
        return self


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise AuditError("Время аудита должно содержать часовой пояс.")
    return value.astimezone(UTC)


@dataclass
class AuditTrace:
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    steps: list[AuditStep] = field(default_factory=list)
    _monotonic_start: float = field(default_factory=monotonic, init=False, repr=False)

    def __post_init__(self) -> None:
        self.started_at = _utc(self.started_at)
        self.steps = [AuditStep.model_validate(step.model_dump()) for step in self.steps[:10]]

    def add(
        self,
        action: Action,
        *,
        tool: str | None = None,
        arguments: dict | None = None,
        validation: Validation = "not_required",
        execution: Execution = "not_run",
        reason: Reason = "request_failed",
    ) -> AuditStep:
        try:
            step = AuditStep(
                action=action,
                tool=tool if tool in _TOOLS else None,
                arguments=arguments if arguments is not None else {},
                validation=validation,
                execution=execution,
                reason=reason,
            )
        except (ValidationError, TypeError):
            raise AuditError("Некорректное событие аудита.") from None
        if len(self.steps) < 10:
            self.steps.append(step)
        return step

    def update(self, **kwargs) -> AuditStep:
        if not self.steps:
            raise AuditError("Нет события аудита для обновления.")
        try:
            step = AuditStep.model_validate({**self.steps[-1].model_dump(), **kwargs})
        except ValidationError:
            raise AuditError("Некорректное обновление аудита.") from None
        self.steps[-1] = step
        return step

    @property
    def duration_ms(self) -> int:
        return max(0, int((monotonic() - self._monotonic_start) * 1000))


@dataclass(frozen=True)
class AuditRecord:
    started_at: datetime
    duration_ms: int
    steps: list[AuditStep]

    def __post_init__(self) -> None:
        object.__setattr__(self, "started_at", _utc(self.started_at))
        if isinstance(self.duration_ms, bool) or not isinstance(self.duration_ms, int):
            raise AuditError("Некорректная длительность аудита.")
        if self.duration_ms < 0 or len(self.steps) > 10:
            raise AuditError("Некорректный размер аудита.")
        object.__setattr__(
            self, "steps", [AuditStep.model_validate(step.model_dump()) for step in self.steps]
        )


_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_audit_events (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL CHECK (user_id > 0),
    request_key TEXT NOT NULL CHECK (length(request_key) BETWEEN 1 AND 200),
    started_at TIMESTAMPTZ NOT NULL,
    duration_ms BIGINT NOT NULL CHECK (duration_ms >= 0),
    steps JSONB NOT NULL CHECK (jsonb_typeof(steps) = 'array'),
    UNIQUE (user_id,request_key)
);
CREATE INDEX IF NOT EXISTS assistant_audit_events_owner_time
    ON assistant_audit_events (user_id,started_at DESC,id DESC);
"""
_DB_ERRORS = (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError)


def _owner(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuditError("Не удалось определить владельца аудита.")


class AuditStore:
    def __init__(self, pool: asyncpg.Pool, *, secrets: tuple[str, ...] = ()) -> None:
        self._pool = pool
        self._secrets = tuple(secret for secret in secrets if isinstance(secret, str) and secret)

    def _safe_step(self, step: AuditStep) -> AuditStep:
        data = AuditStep.model_validate(step.model_dump()).model_dump()
        # Убираем только значения аргументов. Ключи и структурные коды не редактируются.
        data["arguments"] = {
            key: value
            for key, value in data["arguments"].items()
            if not isinstance(value, str) or not any(secret in value for secret in self._secrets)
        }
        return AuditStep.model_validate(data)

    async def initialize(self) -> None:
        try:
            await self._pool.execute(_SCHEMA)
        except _DB_ERRORS:
            raise AuditError("Не удалось подготовить хранилище аудита.") from None

    async def save(self, user_id: int, request_key: str, trace: AuditTrace) -> None:
        _owner(user_id)
        if not isinstance(request_key, str) or not 1 <= len(request_key) <= 200:
            raise AuditError("Не удалось определить запрос для аудита.")
        try:
            record = AuditRecord(
                started_at=trace.started_at,
                duration_ms=trace.duration_ms,
                steps=[self._safe_step(step) for step in trace.steps[:10]],
            )
            payload = json.dumps([step.model_dump() for step in record.steps], ensure_ascii=False)
            await self._pool.execute(
                "INSERT INTO assistant_audit_events "
                "(user_id,request_key,started_at,duration_ms,steps) VALUES ($1,$2,$3,$4,$5) "
                "ON CONFLICT (user_id,request_key) DO NOTHING",
                user_id,
                request_key,
                record.started_at,
                record.duration_ms,
                payload,
            )
        except _DB_ERRORS:
            raise AuditError("Не удалось сохранить аудит запроса.") from None
        except (ValueError, TypeError, AttributeError):
            raise AuditError("Некорректные данные аудита.") from None

    async def latest(self, user_id: int) -> AuditRecord | None:
        _owner(user_id)
        try:
            row = await self._pool.fetchrow(
                "SELECT started_at,duration_ms,steps FROM assistant_audit_events "
                "WHERE user_id=$1 ORDER BY started_at DESC,id DESC LIMIT 1",
                user_id,
            )
            if row is None:
                return None
            steps = json.loads(row["steps"]) if isinstance(row["steps"], str) else row["steps"]
            if not isinstance(steps, list) or len(steps) > 10:
                raise ValueError
            return AuditRecord(
                started_at=row["started_at"],
                duration_ms=row["duration_ms"],
                steps=[self._safe_step(AuditStep.model_validate(step)) for step in steps],
            )
        except _DB_ERRORS:
            raise AuditError("Не удалось прочитать аудит запроса.") from None
        except (ValueError, TypeError, KeyError, AttributeError):
            raise AuditError("Запись аудита имеет неверный формат.") from None


_ACTIONS = {
    "respond": "ответ без инструмента",
    "clarify": "уточнение",
    "tool": "вызов инструмента",
    "undetermined": "решение не определено",
}
_VALIDATIONS = {
    "not_required": "не требуется",
    "pending": "ожидается",
    "passed": "пройдена",
    "rejected": "отклонена",
}
_EXECUTIONS = {
    "not_run": "не выполнялось",
    "success": "успешно",
    "error": "ошибка",
    "awaiting_confirmation": "ожидает подтверждения",
    "cancelled": "отменено",
    "expired": "срок подтверждения истёк",
    "limit_exceeded": "превышен лимит",
}
_REASONS = {
    "no_tool_needed": "выбран ответ без вызова инструмента",
    "missing_details": "нужно уточнить недостающие данные",
    "live_weather": "нужны актуальные погодные данные",
    "personal_schedule": "нужны данные личного расписания",
    "requires_confirmation": "изменение требует подтверждения",
    "invalid_arguments": "аргументы не прошли проверку",
    "tool_limit": "достигнут лимит вызовов инструментов",
    "source_ambiguous": "источник данных допускает несколько вариантов",
    "city_not_found": "город не найден",
    "request_failed": "обработка запроса завершилась ошибкой",
    "confirmation_accepted": "получено подтверждение пользователя",
    "cancelled_by_user": "пользователь отменил действие",
    "unavailable_confirmation": "подтверждение недоступно или просрочено",
    "classification_unavailable": "классификация решения недоступна",
}
_ARGUMENTS = {
    "date": "дата",
    "city": "город",
    "start": "начало",
    "end": "конец",
    "title_length": "символов в названии",
    "location_length": "символов в месте",
    "text_length": "символов в тексте",
    "remind_at": "время напоминания",
}


def render_audit(record: AuditRecord, timezone: str = "UTC") -> str:
    try:
        zone = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError, TypeError):
        raise AuditError("Неизвестный часовой пояс для аудита.") from None
    lines = [
        f"Последний запрос: {record.started_at.astimezone(zone):%d.%m.%Y %H:%M:%S} ({zone.key})",
        f"Длительность: {record.duration_ms} мс",
    ]
    for index, original in enumerate(record.steps, start=1):
        step = AuditStep.model_validate(original.model_dump())
        action = _ACTIONS[step.action] + (f" {step.tool}" if step.tool else "")
        arguments = ", ".join(
            f"{_ARGUMENTS[key]}: {value}" for key, value in step.arguments.items()
        )
        lines.extend(
            [
                f"{index}. Действие: {action}",
                f"Аргументы: {arguments or 'нет'}",
                f"Проверка: {_VALIDATIONS[step.validation]}",
                f"Выполнение: {_EXECUTIONS[step.execution]}",
                f"Причина: {_REASONS[step.reason]}",
            ]
        )
    if not record.steps:
        lines.append("Этапы обработки не записаны.")
    return "\n".join(lines)
