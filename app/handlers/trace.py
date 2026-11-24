from uuid import UUID

from aiogram import BaseMiddleware
from aiogram.filters import CommandObject
from aiogram.types import CallbackQuery, Message

from app.formatting import split_text
from app.tracing import TraceStore, recorded


class TraceMiddleware(BaseMiddleware):
    async def __call__(self, handler, event, data):
        store = data.get("trace_store")
        if store is None:
            return await handler(event, data)
        if isinstance(event, CallbackQuery):
            message, user = event.message, event.from_user
            kind = "confirmation"
            operation = (
                "confirm_lesson" if (event.data or "").startswith("lesson:") else "confirm_reminder"
            )
        else:
            message, user = event, event.from_user
            kind, operation = "user_query", "request"
        if (
            not isinstance(message, Message)
            or user is None
            or message.chat.type != "private"
            or message.chat.id != user.id
        ):
            return await handler(event, data)
        async with recorded(store, user.id, kind, "telegram", operation):
            return await handler(event, data)


async def show_trace(
    message: Message,
    command: CommandObject,
    trace_store: TraceStore | None = None,
) -> None:
    if (
        message.from_user is None
        or message.chat.type != "private"
        or message.chat.id != message.from_user.id
    ):
        return
    arg = (command.args or "last").strip()
    try:
        identity = None if arg == "last" else UUID(arg)
    except ValueError:
        await message.answer("Укажите /trace last или /trace <id>.", parse_mode=None)
        return
    try:
        rows = await trace_store.get(message.from_user.id, identity) if trace_store else None
    except Exception:
        rows = None
    if rows is None:
        await message.answer("Трассировка временно недоступна.", parse_mode=None)
        return
    if not rows:
        await message.answer("Трасса не найдена.", parse_mode=None)
        return
    first = rows[0]
    lines = [f"Трасса {first['trace_id']}", f"Сценарий: {first['kind']}"]
    if first["route"]:
        lines.append(f"Маршрут: {first['route']} (уверенность {first['confidence']:.2f})")
    for row in rows:
        details = [row["tool_name"], row["error_code"]]
        if row["input_tokens"] is not None or row["output_tokens"] is not None:
            input_tokens = row["input_tokens"] if row["input_tokens"] is not None else "—"
            output_tokens = row["output_tokens"] if row["output_tokens"] is not None else "—"
            details.append(f"токены вход/выход {input_tokens}/{output_tokens}")
        suffix = ", ".join(item for item in details if item)
        lines.append(
            f"{row['ordinal'] + 1}. {row['component']}/{row['operation']} "
            f"— {row['status']}, {row['duration_ms']} мс" + (f" ({suffix})" if suffix else "")
        )
    for part in split_text("\n".join(lines)):
        await message.answer(part, parse_mode=None)
