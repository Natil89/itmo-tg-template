"""Текстовые сообщения личного чата передаются клиенту языковой модели."""

from aiogram import F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from app.formatting import format_answer
from app.generation import TEMPERATURES
from app.history import DEFAULT_HISTORY_LIMITS, HistoryLimits, HistoryMessage, recent_history
from app.llm import LlmClient, LlmError
from app.modes import MODES
from app.user_state import UserStateError, UserStateStore

START_TEXT = (
    "Привет! ≽^•⩊•^≼ Я учебный AI-помощник. Отправьте текстовый вопрос — я обращусь к модели.\n"
    "Бот учитывает недавние сообщения в этом режиме. Выбор режима очищает историю.\n"
    "Режимы:\n/study — объяснение программирования (по умолчанию);\n"
    "/translate — перевод;\n/quiz — мини-тест по указанной теме.\n"
    "/settings — настройки, /settings 0.7 — изменить температуру.\n"
    "/reset — очистить историю и сохранить настройки.\n"
    "Переключите режим командой, затем отправьте текст. /start показывает эту справку."
)
UNKNOWN_COMMAND_TEXT = (
    "Доступные команды: /start, /study, /translate, /quiz, /settings, /reset ≽(-˕ - ≼マ. "
    "Для вопроса отправьте обычный текст."
)
TEMPERATURE_BY_TEXT = {f"{value:.1f}": value for value in TEMPERATURES}


async def start(message: Message) -> None:
    await message.answer(START_TEXT, parse_mode=None)


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
) -> None:
    if message.text is None:
        return
    chat_id = message.chat.id
    try:
        mode = MODES[await user_state.get_mode(chat_id)]
        temperature = await user_state.get_temperature(chat_id)
        history = await user_state.get_history(chat_id)
        async with ChatActionSender.typing(bot=message.bot, chat_id=chat_id):
            answer = await llm.generate(
                message.text,
                system_prompt=mode.system_prompt,
                examples=mode.examples,
                history=history,
                temperature=temperature,
            )
        turn: list[HistoryMessage] = [
            {"role": "user", "content": message.text},
            {"role": "assistant", "content": answer},
        ]
        await user_state.save_history(chat_id, recent_history(history + turn, history_limits))
    except (LlmError, UserStateError) as error:
        await message.answer(str(error), parse_mode=None)
        return

    for part in format_answer(answer):
        await message.answer(part.text, entities=part.entities, parse_mode=None)


def create_router() -> Router:
    router = Router(name="assistant")
    router.message.filter(F.chat.type == "private", F.text)
    router.message.register(start, CommandStart())
    router.message.register(switch_mode, Command(commands=list(MODES)))
    router.message.register(settings, Command("settings"))
    router.message.register(reset, Command("reset"))
    router.message.register(unknown_command, F.text.startswith("/"))
    router.message.register(answer_question)
    return router
