from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram import Bot, Dispatcher
from aiogram.methods import SendMessage
from aiogram.types import Chat, Message, PhotoSize, Update, User

from app.formatting import split_text
from app.handlers.assistant import START_TEXT, UNKNOWN_COMMAND_TEXT, create_router
from app.llm import LlmError
from app.modes import MODES
from app.user_state import UserStateError

TOKEN = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk"


async def dispatch(
    text="Вопрос",
    *,
    chat_type="private",
    error=None,
    answer="Ответ",
    photo=False,
    mode="study",
    user_state=None,
    chat_id=42,
):
    bot = Bot(TOKEN)
    bot.session = AsyncMock()
    llm = AsyncMock()
    llm.model = "test/model"
    llm.generate.return_value = answer
    llm.generate.side_effect = error
    if user_state is None:
        user_state = AsyncMock()
        user_state.get_mode.return_value = mode
        user_state.get_history.return_value = []
        user_state.get_temperature.return_value = 0.3
    dispatcher = Dispatcher()
    dispatcher.include_router(create_router())
    message = Message(
        message_id=1,
        date=datetime.now(UTC),
        chat=Chat(id=chat_id, type=chat_type),
        from_user=User(id=42, is_bot=False, first_name="Студент"),
        text=None if photo else text,
        photo=[PhotoSize(file_id="a", file_unique_id="b", width=1, height=1)] if photo else None,
    )
    await dispatcher.feed_update(
        bot, Update(update_id=1, message=message), llm=llm, user_state=user_state
    )
    sent = [
        call.args[1] for call in bot.session.call_args_list if isinstance(call.args[1], SendMessage)
    ]
    return llm, sent


async def test_question_is_sent_to_llm_and_answer_is_plain_text():
    # Arrange / Act
    llm, sent = await dispatch("Что такое Python?", answer="<b>Ответ</b> & текст")
    # Assert
    llm.generate.assert_awaited_once_with(
        "Что такое Python?",
        system_prompt=MODES["study"].system_prompt,
        examples=MODES["study"].examples,
        history=[],
        temperature=0.3,
    )
    assert [message.text for message in sent] == ["<b>Ответ</b> & текст"]
    assert sent[0].parse_mode is None
    assert sent[0].chat_id == 42


@pytest.mark.parametrize(
    "text,expected", [("/start", START_TEXT), ("/unknown", UNKNOWN_COMMAND_TEXT)]
)
async def test_commands_are_handled_without_llm(text, expected):
    # Arrange / Act
    llm, sent = await dispatch(text)
    # Assert
    llm.generate.assert_not_awaited()
    assert [message.text for message in sent] == [expected]


@pytest.mark.parametrize(
    "chat_type,photo", [("group", False), ("supergroup", False), ("private", True)]
)
async def test_groups_and_photos_are_ignored(chat_type, photo):
    # Arrange / Act
    llm, sent = await dispatch(chat_type=chat_type, photo=photo)
    # Assert
    llm.generate.assert_not_awaited()
    assert sent == []


async def test_llm_error_is_shown_without_traceback():
    # Arrange
    error = LlmError("Модель не ответила вовремя. Попробуйте позже.")
    store = AsyncMock()
    store.get_mode.return_value = "study"
    store.get_history.return_value = []
    # Act
    _, sent = await dispatch(error=error, user_state=store)
    # Assert
    assert [message.text for message in sent] == [str(error)]
    assert sent[0].parse_mode is None
    store.save_history.assert_not_awaited()


async def test_markdown_is_sent_as_telegram_entities():
    # Arrange / Act
    _, sent = await dispatch(answer="## Python\n\n**Список** и `tuple`")
    # Assert
    assert len(sent) == 1
    assert sent[0].text == "Python\n\nСписок и tuple"
    assert sent[0].parse_mode is None
    assert {entity.type for entity in sent[0].entities} == {"bold", "code"}


@pytest.mark.parametrize("answer", ["а" * 4096, "a" * 8193, "🙂" * 4097, "<b>\n" * 2000])
async def test_long_answer_is_sent_in_order_without_loss(answer):
    # Arrange / Act
    _, sent = await dispatch(answer=answer)
    # Assert
    assert "".join(message.text for message in sent) == answer
    assert all(0 < len(message.text.encode("utf-16-le")) // 2 <= 4096 for message in sent)
    assert all(message.parse_mode is None for message in sent)


def test_empty_text_does_not_create_empty_chunks():
    # Arrange / Act / Assert
    assert split_text("") == []


@pytest.mark.parametrize("mode", ["study", "translate", "quiz"])
async def test_mode_command_is_saved_without_calling_llm(mode):
    # Arrange
    store = AsyncMock()
    # Act
    llm, sent = await dispatch("/" + mode, user_state=store)
    # Assert
    store.set_mode.assert_awaited_once_with(42, mode)
    store.get_mode.assert_not_awaited()
    llm.generate.assert_not_awaited()
    assert [message.text for message in sent] == [
        f"Режим «{MODES[mode].title}» включён.\n{MODES[mode].hint}"
    ]


@pytest.mark.parametrize("mode", ["study", "translate", "quiz"])
async def test_saved_mode_controls_prompt_and_examples(mode):
    # Arrange / Act
    llm, _ = await dispatch("Текст вопроса", mode=mode)
    # Assert
    llm.generate.assert_awaited_once_with(
        "Текст вопроса",
        system_prompt=MODES[mode].system_prompt,
        examples=MODES[mode].examples,
        history=[],
        temperature=0.3,
    )


async def test_failed_mode_save_is_not_confirmed():
    # Arrange
    store = AsyncMock()
    store.set_mode.side_effect = UserStateError("Не удалось сохранить режим. Попробуйте позже.")
    # Act
    llm, sent = await dispatch("/translate", user_state=store)
    # Assert
    llm.generate.assert_not_awaited()
    assert [message.text for message in sent] == ["Не удалось сохранить режим. Попробуйте позже."]


async def test_unavailable_storage_does_not_call_model_with_wrong_mode():
    # Arrange
    store = AsyncMock()
    store.get_mode.side_effect = UserStateError("Не удалось загрузить режим. Попробуйте позже.")
    # Act
    llm, sent = await dispatch(user_state=store)
    # Assert
    llm.generate.assert_not_awaited()
    assert [message.text for message in sent] == ["Не удалось загрузить режим. Попробуйте позже."]


async def test_private_chat_identifier_is_used_for_settings():
    # Arrange
    store = AsyncMock()
    store.get_mode.return_value = "quiz"
    store.get_history.return_value = []
    # Act
    await dispatch(user_state=store, chat_id=777)
    # Assert
    store.get_mode.assert_awaited_once_with(777)
    store.get_history.assert_awaited_once_with(777)


async def test_follow_up_uses_saved_turn_and_appends_answer():
    history = []
    store = AsyncMock()
    store.get_mode.return_value = "study"
    store.get_history.side_effect = lambda _: history.copy()

    def save(chat_id, messages):
        assert chat_id == 777
        history[:] = messages

    store.save_history.side_effect = save
    await dispatch(
        "Что такое список?", answer="Изменяемая последовательность.", user_state=store, chat_id=777
    )
    llm, _ = await dispatch(
        "А как добавить элемент?", answer="Через append().", user_state=store, chat_id=777
    )
    first_turn = [
        {"role": "user", "content": "Что такое список?"},
        {"role": "assistant", "content": "Изменяемая последовательность."},
    ]
    assert llm.generate.call_args.kwargs["history"] == first_turn
    assert history == first_turn + [
        {"role": "user", "content": "А как добавить элемент?"},
        {"role": "assistant", "content": "Через append()."},
    ]


async def test_start_does_not_reset_selected_mode():
    # Arrange
    store = AsyncMock()
    # Act
    llm, _ = await dispatch("/start", user_state=store)
    # Assert
    store.set_mode.assert_not_awaited()
    llm.generate.assert_not_awaited()


async def test_settings_shows_mode_model_and_temperature():
    _, sent = await dispatch("/settings", mode="quiz")
    assert [message.text for message in sent] == [
        "Режим: Мини-тест\nМодель: test/model\nТемпература: 0.3"
    ]


@pytest.mark.parametrize("value", ["0.0", "0.3", "0.7", "1.0"])
async def test_settings_saves_temperature_used_by_next_question(value):
    store = AsyncMock()
    store.get_mode.return_value = "study"
    store.get_history.return_value = []

    def save(chat_id, temperature):
        assert chat_id == 777
        store.get_temperature.return_value = temperature

    store.set_temperature.side_effect = save
    llm, sent = await dispatch(f"/settings {value}", user_state=store, chat_id=777)
    llm.generate.assert_not_awaited()
    assert [message.text for message in sent] == [f"Температура сохранена: {value}."]
    llm, _ = await dispatch(user_state=store, chat_id=777)
    assert llm.generate.call_args.kwargs["temperature"] == float(value)


@pytest.mark.parametrize("value", ["0.5", "NaN", "0.3 0.7"])
async def test_settings_rejects_invalid_temperature_without_saving(value):
    store = AsyncMock()
    llm, sent = await dispatch(f"/settings {value}", user_state=store)
    assert [message.text for message in sent] == ["Выберите температуру: 0.0, 0.3, 0.7, 1.0."]
    store.set_temperature.assert_not_awaited()
    llm.generate.assert_not_awaited()


async def test_reset_clears_only_current_chat_history():
    store = AsyncMock()
    llm, sent = await dispatch("/reset", user_state=store, chat_id=777)
    store.save_history.assert_awaited_once_with(777, [])
    store.set_mode.assert_not_awaited()
    store.set_temperature.assert_not_awaited()
    llm.generate.assert_not_awaited()
    assert [message.text for message in sent] == ["История очищена. Режим и температура сохранены."]
