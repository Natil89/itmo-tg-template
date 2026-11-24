import json
from unittest.mock import AsyncMock, Mock

import asyncpg
import pytest

from app.modes import MODES
from app.user_state import UserStateError, UserStateStore


async def test_new_user_defaults_to_study():
    # Arrange
    pool = Mock(fetchval=AsyncMock(return_value=None))
    # Act
    mode = await UserStateStore(pool).get_mode(42)
    # Assert
    assert mode == "study"
    assert pool.fetchval.call_args.args[1:] == (42,)


@pytest.mark.parametrize("mode", ["study", "translate", "quiz"])
async def test_saved_mode_is_returned(mode):
    # Arrange
    pool = Mock(fetchval=AsyncMock(return_value=mode))
    # Act / Assert
    assert await UserStateStore(pool).get_mode(42) == mode


async def test_unknown_mode_is_rejected_before_database_write():
    # Arrange
    pool = Mock(execute=AsyncMock())
    # Act / Assert
    with pytest.raises(ValueError, match="Неизвестный режим"):
        await UserStateStore(pool).set_mode(42, "not-a-mode")
    pool.execute.assert_not_awaited()


async def test_unknown_saved_mode_is_not_silently_replaced():
    # Arrange
    pool = Mock(fetchval=AsyncMock(return_value="unknown"))
    # Act / Assert
    with pytest.raises(UserStateError, match="Сохранённый режим недоступен"):
        await UserStateStore(pool).get_mode(42)


@pytest.mark.parametrize("operation", ["initialize", "get_mode", "set_mode"])
@pytest.mark.parametrize(
    "failure", [asyncpg.PostgresError("fake-secret"), TimeoutError("fake-secret")]
)
async def test_database_errors_do_not_expose_exception_details(operation, failure):
    # Arrange
    pool = Mock(execute=AsyncMock(side_effect=failure), fetchval=AsyncMock(side_effect=failure))
    store = UserStateStore(pool)
    args = () if operation == "initialize" else ((42,) if operation == "get_mode" else (42, "quiz"))
    # Act / Assert
    with pytest.raises(UserStateError) as error:
        await getattr(store, operation)(*args)
    assert "fake-secret" not in str(error.value)


def test_modes_are_distinct_and_translation_has_examples():
    # Arrange / Act / Assert
    assert set(MODES) == {"study", "translate", "quiz"}
    assert len({mode.system_prompt for mode in MODES.values()}) == 3
    assert len(MODES["translate"].examples) >= 2


async def test_new_chat_has_empty_history():
    pool = Mock(fetchval=AsyncMock(return_value=None))
    assert await UserStateStore(pool).get_history(42) == []
    assert pool.fetchval.call_args.args[1:] == (42,)


async def test_history_preserves_roles_order_and_unicode():
    history = [
        {"role": "user", "content": "Что такое список? 👋"},
        {"role": "assistant", "content": "Изменяемая последовательность."},
    ]
    pool = Mock(fetchval=AsyncMock(return_value=json.dumps(history)))
    assert await UserStateStore(pool).get_history(42) == history


@pytest.mark.parametrize("history", [[], [{"role": "user", "content": "Вопрос"}]])
async def test_save_history_passes_chat_and_json_to_database(history):
    pool = Mock(execute=AsyncMock())
    await UserStateStore(pool).save_history(42, history)
    _, chat_id, encoded = pool.execute.call_args.args
    assert chat_id == 42
    assert json.loads(encoded) == history


@pytest.mark.parametrize(
    "history",
    [
        {},
        ["text"],
        [{"role": "system", "content": "Инструкция"}],
        [{"role": "user", "content": "  "}],
        [{"role": "user", "content": 123}],
        [{"role": "user"}],
        [{"role": "user", "content": "Вопрос", "extra": True}],
    ],
)
async def test_invalid_history_is_rejected_before_write(history):
    pool = Mock(execute=AsyncMock())
    with pytest.raises(ValueError):
        await UserStateStore(pool).save_history(42, history)
    pool.execute.assert_not_awaited()


@pytest.mark.parametrize("raw", ["not-json", "{}", '[{"role":"system","content":"secret"}]'])
async def test_corrupt_history_returns_safe_error(raw):
    pool = Mock(fetchval=AsyncMock(return_value=raw))
    with pytest.raises(UserStateError, match="^Сохранённая история повреждена\\.$"):
        await UserStateStore(pool).get_history(42)


@pytest.mark.parametrize("operation", ["get_history", "save_history"])
@pytest.mark.parametrize("failure", [asyncpg.PostgresError("secret"), TimeoutError("secret")])
async def test_history_database_errors_are_safe(operation, failure):
    pool = Mock(fetchval=AsyncMock(side_effect=failure), execute=AsyncMock(side_effect=failure))
    args = (42,) if operation == "get_history" else (42, [])
    with pytest.raises(UserStateError) as error:
        await getattr(UserStateStore(pool), operation)(*args)
    expected = (
        "Не удалось загрузить историю. Попробуйте позже."
        if operation == "get_history"
        else "Не удалось сохранить историю. Попробуйте позже."
    )
    assert str(error.value) == expected
