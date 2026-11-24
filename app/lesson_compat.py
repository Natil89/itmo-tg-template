"""Отдельный путь add_lesson: новая Schedule-специализация не получает это право."""

import json
import re
from typing import Literal

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from app.agent_contracts import AgentRequest
from app.agent_policy import AccessDenied, allowed_tools
from app.llm import LlmClient
from app.mcp_client import ScheduleMcpClient
from app.personal_schedule import (
    LessonDraft,
    PendingLesson,
    PersonalScheduleError,
    PersonalScheduleStore,
)


def requests_lesson(text: str) -> bool:
    lowered = text.casefold().replace("ё", "е")
    return bool(
        re.search(r"\b(?:добав|внес|запиш|запис|созда|сохрани)[а-я]*\b", lowered)
        and re.search(r"\b(?:занят|пар[ауы]|урок|лекци|семинар|расписан)[а-я]*\b", lowered)
    )


def requests_reminder(text: str) -> bool:
    lowered = text.casefold().replace("ё", "е")
    return bool(
        re.search(r"\bнапомни(?:ть)?\b", lowered)
        or (
            re.search(r"\b(?:созда|добав|постав|сохрани)[а-я]*\b", lowered)
            and "напоминан" in lowered
        )
    )


def action_requested(request: AgentRequest, kind: Literal["lesson", "reminder"]) -> bool:
    texts = [turn.content for turn in request.history if turn.role == "user"] + [request.text]
    for text in reversed(texts):
        lesson, reminder = requests_lesson(text), requests_reminder(text)
        if lesson or reminder:
            return lesson if kind == "lesson" else reminder
    return False


class LessonCompatibility:
    def __init__(
        self, llm: LlmClient, mcp: ScheduleMcpClient, store: PersonalScheduleStore
    ) -> None:
        self._llm, self._mcp, self._store = llm, mcp, store

    async def prepare(self, request: AgentRequest, update_id: int | None) -> PendingLesson:
        if (
            request.intent != "user_query"
            or not action_requested(request, "lesson")
            or type(update_id) is not int
            or update_id <= 0
        ):
            raise AccessDenied
        tool = allowed_tools("lesson_compat", self._mcp.tools).get("add_lesson")
        if tool is None:
            raise AccessDenied
        raw = await self._llm.structured(
            json.dumps(
                {
                    "recent_dialogue_data": [turn.model_dump() for turn in request.history],
                    "current_query": request.text,
                },
                ensure_ascii=False,
            ),
            system_prompt=(
                "Извлеки параметры добавления занятия. Верни только JSON по схеме. "
                "Не придумывай дату, начало, конец, название или место. "
                "Если поля нет, укажи пустую строку: приложение попросит уточнение. "
                "Не обещай сохранение и не проси подтверждения текстом: "
                "карточку создаёт приложение. "
                f"Текущее время: {request.occurred_at.isoformat()}; timezone: {request.timezone}."
            ),
            schema=tool.inputSchema,
            schema_name="lesson_proposal",
        )
        try:
            draft = LessonDraft.model_validate_json(raw)
        except ValidationError:
            raise PersonalScheduleError(
                "Для занятия укажите дату, время начала и конца, название и место."
            ) from None
        # Discovery мог измениться за время ожидания LLM.
        tool = allowed_tools("lesson_compat", self._mcp.tools).get("add_lesson")
        if tool is None:
            raise AccessDenied
        Draft202012Validator(tool.inputSchema).validate(draft.model_dump())
        pending = await self._store.prepare(request.actor_id, update_id, draft)
        if pending.user_id != request.actor_id or pending.payload != draft:
            raise AccessDenied
        return pending
