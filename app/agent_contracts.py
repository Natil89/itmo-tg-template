"""Внутренние контракты лабораторной 3. Не являются сетевым A2A-протоколом."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.reminders import ReminderDraft
from app.schedule import DaySchedule
from app.weather import CurrentWeather

Target = Literal["schedule", "knowledge", "lifestyle", "clarify"]
Component = Literal["router", "schedule", "knowledge", "lifestyle", "briefing", "lesson_compat"]
Intent = Literal["user_query", "daily_briefing"]
Confidence = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid", strict=True, frozen=True, revalidate_instances="always"
    )


class RouteDecision(Contract):
    target: Target
    # Короткое публичное обоснование, не chain-of-thought и не диагностический лог.
    reason: str = Field(min_length=1, max_length=300)
    confidence: Confidence
    dialogue: bool = False
    operation: Literal["read", "prepare_reminder", "prepare_lesson"] = "read"

    @model_validator(mode="after")
    def consistent_operation(self) -> "RouteDecision":
        if self.operation != "read" and self.target != "schedule":
            raise ValueError("Изменения относятся только к расписанию.")
        if self.dialogue and (self.target != "clarify" or self.operation != "read"):
            raise ValueError("Обычный диалог не получает предметных действий.")
        return self

    @field_validator("reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Нужно краткое обоснование маршрута.")
        return value.strip()


class DialogueTurn(Contract):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AgentRequest(Contract):
    request_id: UUID
    trace_id: UUID
    actor_id: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=16000)
    intent: Intent
    occurred_at: datetime
    timezone: str
    history: tuple[DialogueTurn, ...] = Field(default=(), max_length=4)

    @model_validator(mode="after")
    def background_has_no_history(self) -> "AgentRequest":
        if self.intent != "user_query" and self.history:
            raise ValueError("Фоновый запрос не наследует историю диалога.")
        return self

    @field_validator("occurred_at")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Нужно время приложения с часовым поясом.")
        return value

    @field_validator("timezone")
    @classmethod
    def iana_zone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ValueError, ZoneInfoNotFoundError):
            raise ValueError("Нужна известная IANA-зона.") from None
        return value

    @field_validator("text")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Нужен непустой запрос.")
        return value


def user_request(*, actor_id: int, text: str, now: datetime, timezone: str) -> AgentRequest:
    """Вызывает приложение с владельцем из update и зоной из хранилища, не LLM."""
    return AgentRequest(
        request_id=uuid4(),
        trace_id=uuid4(),
        actor_id=actor_id,
        text=text,
        intent="user_query",
        occurred_at=now,
        timezone=timezone,
    )


class ScheduleFacts(Contract):
    kind: Literal["schedule"] = "schedule"
    days: tuple[DaySchedule, ...] = Field(min_length=1, max_length=7)


class MaterialHit(Contract):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=200)
    excerpt: str = Field(min_length=1, max_length=4000)
    source_url: str | None = Field(
        default=None,
        pattern=r"^https://ai-in-sd-y29\.github\.io/\.github/profile/lectures/[^\s]*$",
    )

    @property
    def source(self) -> str:
        return self.source_url or f"materials://{self.document_id}"


class MaterialEntry(Contract):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=200)
    source_url: str | None = Field(
        default=None,
        pattern=r"^https://ai-in-sd-y29\.github\.io/\.github/profile/lectures/[^\s]*$",
    )


class MaterialCatalog(Contract):
    documents: tuple[MaterialEntry, ...] = Field(max_length=100)

    @model_validator(mode="after")
    def unique_documents(self) -> "MaterialCatalog":
        ids = [entry.document_id for entry in self.documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Идентификаторы документов должны быть уникальны.")
        return self


class KnowledgeFacts(Contract):
    kind: Literal["knowledge"] = "knowledge"
    hits: tuple[MaterialHit, ...] = Field(max_length=10)


class WeatherFacts(Contract):
    kind: Literal["weather"] = "weather"
    weather: CurrentWeather


class EmptyFacts(Contract):
    kind: Literal["empty"] = "empty"


class ReminderProposalFacts(Contract):
    kind: Literal["reminder_proposal"] = "reminder_proposal"
    action_id: UUID
    draft: ReminderDraft
    timezone: str
    expires_at: datetime


Facts = Annotated[
    ScheduleFacts | KnowledgeFacts | WeatherFacts | EmptyFacts | ReminderProposalFacts,
    Field(discriminator="kind"),
]


class AgentResult(Contract):
    request_id: UUID
    trace_id: UUID
    component: Literal["schedule", "knowledge", "lifestyle"]
    status: Literal["success", "needs_clarification", "not_found", "failed"]
    message: str = Field(min_length=1, max_length=12000)
    confidence: Confidence
    facts: Facts
    used_tools: tuple[str, ...] = Field(max_length=2)
    error_code: (
        Literal["access_denied", "invalid_contract", "unavailable", "timeout", "tool_failed"] | None
    ) = None

    @model_validator(mode="after")
    def consistent_result(self) -> "AgentResult":
        expected = {"schedule": "schedule", "knowledge": "knowledge", "lifestyle": "weather"}
        kinds = {expected[self.component], "empty"}
        if self.component == "schedule":
            kinds.add("reminder_proposal")
        if self.facts.kind not in kinds:
            raise ValueError("Факты не соответствуют специалисту.")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("Код ошибки должен соответствовать статусу failed.")
        if self.status == "success" and self.facts.kind == "empty":
            raise ValueError("Успех требует проверенных фактов.")
        return self
