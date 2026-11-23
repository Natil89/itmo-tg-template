"""Внутренние контракты лабораторной 3. Не являются сетевым A2A-протоколом."""

from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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

    @field_validator("reason")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Нужно краткое обоснование маршрута.")
        return value.strip()


class AgentRequest(Contract):
    request_id: UUID
    trace_id: UUID
    actor_id: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=16000)
    intent: Intent
    occurred_at: datetime
    timezone: str

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
    days: tuple[DaySchedule, ...]


class MaterialHit(Contract):
    document_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,63}$")
    title: str = Field(min_length=1, max_length=200)
    excerpt: str = Field(min_length=1, max_length=4000)

    @property
    def source(self) -> str:
        return f"materials://{self.document_id}"


class KnowledgeFacts(Contract):
    kind: Literal["knowledge"] = "knowledge"
    hits: tuple[MaterialHit, ...] = Field(max_length=10)


class WeatherFacts(Contract):
    kind: Literal["weather"] = "weather"
    weather: CurrentWeather


class EmptyFacts(Contract):
    kind: Literal["empty"] = "empty"


Facts = Annotated[
    ScheduleFacts | KnowledgeFacts | WeatherFacts | EmptyFacts, Field(discriminator="kind")
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
        if self.facts.kind not in {expected[self.component], "empty"}:
            raise ValueError("Факты не соответствуют специалисту.")
        if (self.status == "failed") != (self.error_code is not None):
            raise ValueError("Код ошибки должен соответствовать статусу failed.")
        if self.status == "success" and self.facts.kind == "empty":
            raise ValueError("Успех требует проверенных фактов.")
        return self
