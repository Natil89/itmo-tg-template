"""Текстовые сообщения личного чата передаются клиенту языковой модели."""

import logging
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from aiogram.utils.chat_action import ChatActionSender

from app.assistant_service import AssistantService
from app.audit import AuditStore, AuditTrace, render_audit, safe_arguments
from app.delivery_telegram import DeliveryDestinationMiddleware
from app.formatting import format_answer
from app.generation import TEMPERATURES
from app.handlers.notifications import register_notifications
from app.handlers.trace import TraceMiddleware, show_trace
from app.history import DEFAULT_HISTORY_LIMITS, HistoryLimits, HistoryMessage, recent_history
from app.lesson_confirmation import confirm_lesson, is_affirmation
from app.llm import LlmClient, LlmError
from app.mcp_client import ScheduleMcpClient
from app.modes import MODES
from app.personal_schedule import (
    PendingLesson,
    PersonalScheduleError,
    PersonalScheduleStore,
)
from app.reminders import PendingReminder, ReminderError, ReminderSaved, ReminderStore
from app.schedule_agent import ScheduleAgent
from app.timezones import TimezoneResolver
from app.tracing import mark_failed, span
from app.user_state import UserStateError, UserStateStore

logger = logging.getLogger(__name__)
AUDIT_WARNING = "Диагностика запроса временно недоступна; /why может показывать предыдущее решение."

START_TEXT = (
    "Привет! ≽^•⩊•^≼ Я учебный AI-помощник. Отправьте текстовый вопрос — я обращусь к модели.\n"
    "Бот учитывает недавние сообщения в этом режиме. Выбор режима очищает историю.\n"
    "Режимы:\n/study — объяснение программирования (по умолчанию);\n"
    "/translate — перевод;\n/quiz — мини-тест по указанной теме.\n"
    "/settings — настройки, /settings 0.7 — изменить температуру.\n"
    "/reset — очистить историю и сохранить настройки.\n"
    "/tools — инструменты MCP-сервера.\n"
    "/why — объяснить последнее решение по вашему запросу.\n"
    "/trace last — показать этапы обработки последнего запроса.\n"
    "/timezone Москва — установить часовой пояс; /timezone — показать его.\n"
    "/city — выбрать город; /briefing_on и /briefing_off — сводка в 08:00;\n"
    "/notifications — настройки уведомлений.\n"
    "Можно спросить текущую погоду или своё расписание. Чтобы добавить занятие, "
    "напишите дату, время начала и конца, название и место — я попрошу подтверждение.\n"
    "Можно попросить создать напоминание: сначала покажу время и попрошу подтверждение. "
    "Напоминания пока только сохраняются; отправка уведомлений по времени ещё не подключена.\n"
    "Переключите режим командой, затем отправьте текст. /start показывает эту справку."
)
UNKNOWN_COMMAND_TEXT = (
    "Доступные команды: /start, /study, /translate, /quiz, /settings, /reset, /tools, "
    "/timezone, /why, /trace ≽(-˕ - ≼マ. "
    "Для вопроса отправьте обычный текст."
)
TEMPERATURE_BY_TEXT = {f"{value:.1f}": value for value in TEMPERATURES}


async def start(message: Message, delivery_enabled: bool = False) -> None:
    text = START_TEXT
    if delivery_enabled:
        text = text.replace(
            "Напоминания пока только сохраняются; "
            "отправка уведомлений по времени ещё не подключена.",
            "Подтверждённые напоминания доставляются автоматически.",
        )
    await message.answer(text, parse_mode=None)


async def _save_audit(
    audit_store: AuditStore | None,
    user_id: int | None,
    request_key: str,
    trace: AuditTrace | None,
    *,
    failed: bool = False,
) -> bool:
    """Сбой диагностики не изменяет результат и не повторяет бизнес-операцию."""
    if audit_store is None or trace is None:
        return True
    if user_id is None:
        return False
    try:
        if not trace.steps:
            trace.add(
                "undetermined",
                execution="error" if failed else "success",
                reason="request_failed" if failed else "classification_unavailable",
            )
        elif failed and trace.steps[-1].execution in {"success", "awaiting_confirmation"}:
            # Чтение/подготовка могли пройти, а последующий этап запроса — нет.
            trace.add("undetermined", execution="error", reason="request_failed")
        await audit_store.save(user_id, request_key, trace)
    except Exception as error:
        # Никаких аргументов, Telegram ID, callback data и текста исключения в логе.
        logger.warning("Audit save failed: %s", type(error).__name__)
        return False
    return True


async def why(
    message: Message,
    audit_store: AuditStore,
    reminders: ReminderStore | None = None,
) -> None:
    if message.from_user is None:
        await message.answer("Не удалось определить пользователя.", parse_mode=None)
        return
    user_id = message.from_user.id
    try:
        record = await audit_store.latest(user_id)
        if record is None:
            text = "У вас пока нет сохранённых решений. Отправьте обычный запрос боту."
        else:
            zone = "UTC"
            if reminders is not None:
                try:
                    zone = await reminders.get_timezone(user_id) or "UTC"
                except Exception as error:
                    logger.warning("Audit timezone unavailable: %s", type(error).__name__)
            text = render_audit(record, timezone=zone)
    except Exception as error:
        logger.warning("Audit read failed: %s", type(error).__name__)
        text = "Диагностика сейчас недоступна. Попробуйте /why позже."
    for part in format_answer(text):
        await message.answer(part.text, entities=part.entities, parse_mode=None)


def _callback_audit_failure(trace: AuditTrace | None, tool: str) -> None:
    if trace is None:
        return
    if trace.steps:
        trace.update(execution="error", reason="request_failed")
    else:
        trace.add(
            "tool",
            tool=tool,
            validation="rejected",
            execution="not_run",
            reason="unavailable_confirmation",
        )


async def timezone(
    message: Message,
    command: CommandObject,
    reminders: ReminderStore,
    timezone_resolver: TimezoneResolver | None = None,
) -> None:
    if message.from_user is None:
        await message.answer("Не удалось определить пользователя.", parse_mode=None)
        return
    user_id = message.from_user.id
    argument = (command.args or "").strip()
    try:
        if argument:
            choice = await timezone_resolver.resolve(argument) if timezone_resolver else None
            zone = choice.timezone if choice else argument
            await reminders.set_timezone(user_id, zone)
            local = datetime.now(ZoneInfo(zone))
            place = f"Город: {choice.place}.\n" if choice and choice.place else ""
            text = (
                f"{place}Часовой пояс сохранён: {zone}.\nСейчас в нём: {local:%d.%m.%Y %H:%M:%S}."
            )
            if place:
                text += "\nЕсли имели в виду другой город, укажите страну или регион в /timezone."
        else:
            saved_zone = await reminders.get_timezone(user_id)
            text = (
                f"Ваш часовой пояс: {saved_zone}."
                if saved_zone is not None
                else "Часовой пояс не задан. Укажите, например: /timezone Europe/Moscow."
            )
    except ReminderError as error:
        text = str(error)
    except (ZoneInfoNotFoundError, ValueError):
        text = "Неизвестный часовой пояс. Пример: /timezone Europe/Moscow."
    await message.answer(text, parse_mode=None)


async def tools(message: Message, schedule_mcp: ScheduleMcpClient | None = None) -> None:
    if schedule_mcp is None:
        await message.answer("Инструменты сейчас недоступны: нет подключения к MCP-серверу.")
        return
    try:
        await schedule_mcp.refresh()
    except Exception:
        await message.answer("Инструменты сейчас недоступны: нет подключения к MCP-серверу.")
        return
    lines = ["Доступные инструменты:"]
    for tool in schedule_mcp.tools:
        read_only = tool.annotations is not None and tool.annotations.readOnlyHint is True
        kind = "чтение" if read_only else "изменение, нужно подтверждение"
        lines.append(f"• {tool.name} — {tool.description or 'без описания'} ({kind})")
    if not schedule_mcp.tools:
        lines.append("Сервер не предоставил инструментов.")
    for part in format_answer("\n".join(lines)):
        await message.answer(part.text, entities=part.entities, parse_mode=None)


async def unknown_command(message: Message) -> None:
    await message.answer(UNKNOWN_COMMAND_TEXT, parse_mode=None)


async def switch_mode(message: Message, command: CommandObject, user_state: UserStateStore) -> None:
    mode = MODES[command.command]
    try:
        await user_state.set_mode(message.chat.id, command.command)
    except UserStateError as error:
        await message.answer(str(error), parse_mode=None)
        return
    await message.answer(f"Режим «{mode.title}» включён.\n{mode.hint}", parse_mode=None)


async def settings(
    message: Message, command: CommandObject, user_state: UserStateStore, llm: LlmClient
) -> None:
    chat_id = message.chat.id
    argument = (command.args or "").strip()
    try:
        if argument:
            temperature = TEMPERATURE_BY_TEXT.get(argument)
            if temperature is None:
                values = ", ".join(TEMPERATURE_BY_TEXT)
                await message.answer(f"Выберите температуру: {values}.", parse_mode=None)
                return
            await user_state.set_temperature(chat_id, temperature)
            text = f"Температура сохранена: {temperature:.1f}."
        else:
            mode = MODES[await user_state.get_mode(chat_id)]
            temperature = await user_state.get_temperature(chat_id)
            text = f"Режим: {mode.title}\nМодель: {llm.model}\nТемпература: {temperature:.1f}"
    except UserStateError as error:
        text = str(error)
    await message.answer(text, parse_mode=None)


async def reset(message: Message, user_state: UserStateStore) -> None:
    try:
        await user_state.save_history(message.chat.id, [])
    except UserStateError as error:
        await message.answer(str(error), parse_mode=None)
        return
    await message.answer("История очищена. Режим и температура сохранены.", parse_mode=None)


async def answer_question(
    message: Message,
    llm: LlmClient,
    user_state: UserStateStore,
    history_limits: HistoryLimits = DEFAULT_HISTORY_LIMITS,
    schedule_agent: ScheduleAgent | None = None,
    audit_store: AuditStore | None = None,
    personal_schedule: PersonalScheduleStore | None = None,
    schedule_mcp: ScheduleMcpClient | None = None,
    assistant_service: AssistantService | None = None,
    delivery_enabled: bool = False,
) -> None:
    if message.text is None:
        return
    chat_id = message.chat.id
    user_id = message.from_user.id if message.from_user is not None else None
    trace = AuditTrace() if audit_store is not None else None
    try:
        mode_name = await user_state.get_mode(chat_id)
        mode = MODES[mode_name]
        if personal_schedule is not None and schedule_mcp is not None:
            if await text_lesson_confirmation(
                message,
                personal_schedule,
                schedule_mcp,
                audit_store,
                allow_unreplied=mode_name == "study",
            ):
                return
        temperature = await user_state.get_temperature(chat_id)
        history = await user_state.get_history(chat_id)
        async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
            arguments = dict(
                system_prompt=mode.system_prompt,
                examples=mode.examples,
                history=history,
                temperature=temperature,
            )
            if assistant_service is not None:
                answer = await assistant_service.answer(
                    message.text,
                    **arguments,
                    mode_name=mode_name,
                    user_id=user_id,
                    update_id=message.message_id,
                    audit=trace,
                )
            elif schedule_agent is not None:
                if trace is not None:
                    arguments["audit"] = trace
                answer = await schedule_agent.answer(
                    message.text,
                    **arguments,
                    user_id=user_id,
                    update_id=message.message_id,
                )
            else:
                answer = await llm.generate(message.text, **arguments)
        markup = None
        lesson_action_id = None
        if isinstance(answer, PendingLesson):
            pending = answer
            draft = pending.payload
            answer = (
                f"Добавить занятие в ваше расписание?\n{draft.date}, {draft.start}–{draft.end}\n"
                f"{draft.title}\nМесто: {draft.location}\n"
                f"Часовой пояс: {(assistant_service or schedule_agent).timezone}\n"
                "Подтверждение действует пять минут. "
                "Нажмите кнопку или ответьте «да» / «всё верно»."
            )
            if pending.status in {"prepared", "confirmed"}:
                lesson_action_id = pending.action_id
                markup = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Подтвердить",
                                callback_data=f"lesson:confirm:{pending.action_id}",
                            ),
                            InlineKeyboardButton(
                                text="Отменить", callback_data=f"lesson:cancel:{pending.action_id}"
                            ),
                        ]
                    ]
                )
            else:
                answer = "Это предложение уже обработано или просрочено. Отправьте новый запрос."
        elif isinstance(answer, PendingReminder):
            pending = answer
            try:
                local = datetime.fromisoformat(pending.payload.remind_at).astimezone(
                    ZoneInfo(pending.timezone)
                )
            except (ValueError, ZoneInfoNotFoundError):
                raise ReminderError("Не удалось проверить время напоминания.") from None
            answer = (
                f"Сохранить напоминание?\n{pending.payload.text}\n"
                f"Когда: {local:%d.%m.%Y %H:%M:%S}\nЧасовой пояс: {pending.timezone}\n"
                "Подтверждение действует не более пяти минут. "
                + (
                    "Автоматическая доставка включена."
                    if delivery_enabled
                    else "Автоматическая доставка отключена."
                )
            )
            if pending.status in {"prepared", "confirmed"}:
                markup = InlineKeyboardMarkup(
                    inline_keyboard=[
                        [
                            InlineKeyboardButton(
                                text="Подтвердить",
                                callback_data=f"reminder:confirm:{pending.action_id}",
                            ),
                            InlineKeyboardButton(
                                text="Отменить",
                                callback_data=f"reminder:cancel:{pending.action_id}",
                            ),
                        ]
                    ]
                )
            else:
                answer = "Это напоминание уже обработано или просрочено. Отправьте новый запрос."
        turn: list[HistoryMessage] = [
            {"role": "user", "content": message.text},
            {"role": "assistant", "content": answer},
        ]
        await user_state.save_history(chat_id, recent_history(history + turn, history_limits))
    except (LlmError, UserStateError, PersonalScheduleError, ReminderError) as error:
        mark_failed()
        saved = await _save_audit(
            audit_store,
            user_id,
            f"message:{message.message_id}",
            trace,
            failed=True,
        )
        text = str(error) + (f"\n\n{AUDIT_WARNING}" if not saved else "")
        await message.answer(text, parse_mode=None)
        return
    except Exception:
        await _save_audit(
            audit_store,
            user_id,
            f"message:{message.message_id}",
            trace,
            failed=True,
        )
        raise

    saved = await _save_audit(audit_store, user_id, f"message:{message.message_id}", trace)
    if not saved:
        answer += f"\n\n{AUDIT_WARNING}"
    parts = format_answer(answer)
    for index, part in enumerate(parts):
        kwargs = {"reply_markup": markup} if markup is not None and index == len(parts) - 1 else {}
        with span("telegram", "send"):
            sent = await message.answer(
                part.text, entities=part.entities, parse_mode=None, **kwargs
            )
        if (
            lesson_action_id is not None
            and index == len(parts) - 1
            and personal_schedule is not None
            and isinstance(sent, Message)
        ):
            try:
                await personal_schedule.bind_prompt(
                    user_id, lesson_action_id, chat_id, sent.message_id
                )
            except PersonalScheduleError:
                await message.answer(
                    "Текстовое подтверждение временно недоступно. Используйте кнопку.",
                    parse_mode=None,
                )


async def text_lesson_confirmation(
    message: Message,
    personal_schedule: PersonalScheduleStore,
    schedule_mcp: ScheduleMcpClient,
    audit_store: AuditStore | None = None,
    *,
    allow_unreplied: bool = True,
) -> bool:
    if (
        not message.text
        or not is_affirmation(message.text)
        or message.from_user is None
        or message.chat.type != "private"
        or message.chat.id != message.from_user.id
        or (not allow_unreplied and message.reply_to_message is None)
    ):
        return False
    user_id = message.from_user.id
    trace = AuditTrace() if audit_store is not None else None
    processed = False
    try:
        pending = await personal_schedule.text_confirmation_candidate(
            user_id,
            message.chat.id,
            message.message_id,
            reply_to=message.reply_to_message.message_id if message.reply_to_message else None,
        )
        if pending is None:
            return False
        text = await confirm_lesson(
            user_id, pending.action_id, personal_schedule, schedule_mcp, trace
        )
        processed = True
    except PersonalScheduleError as error:
        mark_failed()
        _callback_audit_failure(trace, "add_lesson")
        text = str(error)
    except Exception:
        mark_failed()
        _callback_audit_failure(trace, "add_lesson")
        text = "Не удалось обработать подтверждение. Попробуйте кнопку или повторите позже."
    await _save_audit(
        audit_store, user_id, f"message:{message.message_id}", trace, failed=not processed
    )
    await message.answer(text, parse_mode=None)
    return True


async def lesson_confirmation(
    callback: CallbackQuery,
    personal_schedule: PersonalScheduleStore,
    schedule_mcp: ScheduleMcpClient,
    audit_store: AuditStore | None = None,
) -> None:
    trace = AuditTrace() if audit_store is not None else None
    user_id = callback.from_user.id
    if (
        not isinstance(callback.message, Message)
        or callback.message.chat.type != "private"
        or callback.message.chat.id != user_id
    ):
        _callback_audit_failure(trace, "add_lesson")
        await _save_audit(audit_store, user_id, f"callback:{callback.id}", trace, failed=True)
        await callback.answer("Действие доступно только в личном чате.", show_alert=True)
        return
    await callback.answer()
    processed = False
    try:
        _, action, action_id = (callback.data or "").split(":", 2)
        user_id = callback.from_user.id
        if action == "cancel":
            cancelled = await personal_schedule.cancel(user_id, action_id)
            text = (
                "Добавление занятия отменено."
                if cancelled
                else "Действие уже завершено или недоступно."
            )
            if trace is not None:
                trace.add(
                    "tool",
                    tool="add_lesson",
                    validation="passed" if cancelled else "rejected",
                    execution="cancelled" if cancelled else "not_run",
                    reason="cancelled_by_user" if cancelled else "unavailable_confirmation",
                )
        elif action == "confirm":
            text = await confirm_lesson(user_id, action_id, personal_schedule, schedule_mcp, trace)
        else:
            raise PersonalScheduleError("Неизвестное действие.")
        processed = True
    except PersonalScheduleError as error:
        mark_failed()
        _callback_audit_failure(trace, "add_lesson")
        text = str(error)
    except Exception:
        mark_failed()
        _callback_audit_failure(trace, "add_lesson")
        text = "Не удалось обработать подтверждение. Попробуйте ещё раз."
    finally:
        await _save_audit(
            audit_store,
            user_id,
            f"callback:{callback.id}",
            trace,
            failed=not processed,
        )
    await callback.message.answer(text, parse_mode=None)
    if not processed:
        return
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        # Повторный callback может прийти после удаления кнопок; БД уже защитила запись.
        pass


async def reminder_confirmation(
    callback: CallbackQuery,
    reminders: ReminderStore,
    schedule_mcp: ScheduleMcpClient,
    audit_store: AuditStore | None = None,
    delivery_enabled: bool = False,
) -> None:
    trace = AuditTrace() if audit_store is not None else None
    user_id = callback.from_user.id
    if (
        not isinstance(callback.message, Message)
        or callback.message.chat.type != "private"
        or callback.message.chat.id != callback.from_user.id
    ):
        _callback_audit_failure(trace, "add_reminder")
        await _save_audit(audit_store, user_id, f"callback:{callback.id}", trace, failed=True)
        await callback.answer("Действие доступно только в вашем личном чате.", show_alert=True)
        return
    await callback.answer()
    processed = False
    try:
        _, action, action_id = (callback.data or "").split(":", 2)
        user_id = callback.from_user.id
        if action == "cancel":
            cancelled = await reminders.cancel(user_id, action_id)
            text = (
                "Создание напоминания отменено."
                if cancelled
                else "Действие уже завершено или недоступно."
            )
            if trace is not None:
                trace.add(
                    "tool",
                    tool="add_reminder",
                    validation="passed" if cancelled else "rejected",
                    execution="cancelled" if cancelled else "not_run",
                    reason="cancelled_by_user" if cancelled else "unavailable_confirmation",
                )
        elif action == "confirm":
            pending = await reminders.confirm(user_id, action_id)
            if (
                pending.user_id != user_id
                or pending.action_id != action_id
                or pending.status not in {"confirmed", "executed"}
            ):
                raise ReminderError("Подтверждение недоступно.")
            if trace is not None:
                trace.add(
                    "tool",
                    tool="add_reminder",
                    arguments=safe_arguments("add_reminder", pending.payload.model_dump()),
                    validation="passed",
                    reason="confirmation_accepted",
                )
            result = await schedule_mcp.call_tool(
                "add_reminder",
                pending.payload.model_dump(),
                user_id=user_id,
                action_id=action_id,
            )
            if result.isError:
                raise ReminderError(
                    "Не удалось сохранить напоминание. Попробуйте подтвердить снова."
                )
            saved = ReminderSaved.model_validate(result.structuredContent)
            if (
                saved.reminder_id <= 0
                or saved.text != pending.payload.text
                or datetime.fromisoformat(saved.remind_at)
                != datetime.fromisoformat(pending.payload.remind_at)
                or saved.timezone != pending.timezone
            ):
                raise ReminderError("Сервер вернул неожиданный результат сохранения.")
            local = datetime.fromisoformat(saved.remind_at).astimezone(ZoneInfo(saved.timezone))
            text = (
                "Напоминание сохранено" if saved.created else "Напоминание уже было сохранено"
            ) + (
                f" №{saved.reminder_id}: {saved.text}\n"
                f"Когда: {local:%d.%m.%Y %H:%M:%S} ({saved.timezone}).\n"
                + (
                    "Автоматическая доставка включена."
                    if delivery_enabled
                    else "Автоматическая доставка отключена."
                )
            )
            if trace is not None:
                trace.update(execution="success")
        else:
            raise ReminderError("Неизвестное действие.")
        processed = True
    except ReminderError as error:
        mark_failed()
        _callback_audit_failure(trace, "add_reminder")
        text = str(error)
    except Exception:
        mark_failed()
        _callback_audit_failure(trace, "add_reminder")
        text = "Не удалось обработать подтверждение напоминания. Попробуйте ещё раз."
    finally:
        await _save_audit(
            audit_store,
            user_id,
            f"callback:{callback.id}",
            trace,
            failed=not processed,
        )
    await callback.message.answer(text, parse_mode=None)
    if not processed:
        return
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:
        # Повторный callback допустим: уникальность подтверждённого действия проверяет БД.
        pass


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.outer_middleware(DeliveryDestinationMiddleware())
    router.callback_query.outer_middleware(DeliveryDestinationMiddleware())
    router.message.filter(F.chat.type == "private", F.text)
    router.message.register(start, CommandStart())
    router.message.register(switch_mode, Command(commands=list(MODES)))
    router.message.register(settings, Command("settings"))
    router.message.register(reset, Command("reset"))
    router.message.register(tools, Command("tools"))
    router.message.register(timezone, Command("timezone"))
    router.message.register(why, Command("why"))
    router.message.register(show_trace, Command("trace"))
    register_notifications(router)
    router.message.register(unknown_command, F.text.startswith("/"))
    traced = Router(name="traced_requests")
    traced.message.middleware(TraceMiddleware())
    traced.callback_query.middleware(TraceMiddleware())
    traced.message.register(answer_question)
    traced.callback_query.register(lesson_confirmation, F.data.startswith("lesson:"))
    traced.callback_query.register(reminder_confirmation, F.data.startswith("reminder:"))
    router.include_router(traced)
    return router
