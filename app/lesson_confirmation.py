"""Подтверждение занятия — действие приложения, без обращения к языковой модели."""

import re
import unicodedata

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from app.agent_policy import allowed_tools
from app.audit import AuditTrace, safe_arguments
from app.mcp_client import ScheduleMcpClient
from app.personal_schedule import LessonSaved, PersonalScheduleError, PersonalScheduleStore

AFFIRMATIONS = frozenset(
    {
        "да",
        "верно",
        "все верно",
        "все правильно",
        "подтверждаю",
        "подтвердить",
        "да все верно",
        "да верно",
        "да подтверждаю",
        "да все правильно",
    }
)


def is_affirmation(text: str) -> bool:
    # Точное распознавание: «да, но поменяй время» не является согласием.
    normalized = unicodedata.normalize("NFKC", text).casefold().replace("ё", "е")
    normalized = re.sub(r"[,!.]+", " ", normalized)
    return " ".join(normalized.split()) in AFFIRMATIONS


async def confirm_lesson(
    user_id: int,
    action_id: str,
    store: PersonalScheduleStore,
    mcp: ScheduleMcpClient,
    trace: AuditTrace | None = None,
) -> str:
    if "add_lesson" not in allowed_tools("lesson_compat", mcp.tools):
        raise PersonalScheduleError("Добавление занятий сейчас недоступно.")
    pending = await store.confirm(user_id, action_id)
    if (
        pending.user_id != user_id
        or pending.action_id != action_id
        or pending.status not in {"confirmed", "executed"}
    ):
        raise PersonalScheduleError("Подтверждение недоступно.")
    if trace is not None:
        trace.add(
            "tool",
            tool="add_lesson",
            arguments=safe_arguments("add_lesson", pending.payload.model_dump()),
            validation="passed",
            reason="confirmation_accepted",
        )
    tool = allowed_tools("lesson_compat", mcp.tools).get("add_lesson")
    if tool is None:
        raise PersonalScheduleError("Добавление занятий сейчас недоступно.")
    Draft202012Validator(tool.inputSchema).validate(pending.payload.model_dump())
    result = await mcp.call_tool(
        "add_lesson",
        pending.payload.model_dump(),
        user_id=user_id,
        action_id=action_id,
    )
    if result.isError:
        raise PersonalScheduleError("Не удалось сохранить занятие. Попробуйте подтвердить снова.")
    try:
        saved = LessonSaved.model_validate(result.structuredContent)
    except ValidationError:
        raise PersonalScheduleError("Сервер вернул неожиданный результат сохранения.") from None
    if any(getattr(saved, key) != value for key, value in pending.payload.model_dump().items()):
        raise PersonalScheduleError("Сервер вернул неожиданный результат сохранения.")
    if trace is not None:
        trace.update(execution="success")
    return (
        "Занятие добавлено" if saved.created else "Занятие уже было добавлено"
    ) + f": {saved.date}, {saved.start}–{saved.end}\n{saved.title}\nМесто: {saved.location}"
