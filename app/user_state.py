"""Режим и история пользователя в PostgreSQL; пул принадлежит приложению."""

import json

import asyncpg

from app.generation import DEFAULT_TEMPERATURE, TEMPERATURES
from app.history import HistoryMessage
from app.modes import DEFAULT_MODE, MODES


def _validate_history(history: object) -> list[HistoryMessage]:
    if not isinstance(history, list):
        raise ValueError("История должна быть списком сообщений.")
    for message in history:
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or message["role"] not in ("user", "assistant")
            or not isinstance(message["content"], str)
            or not message["content"].strip()
        ):
            raise ValueError("Сообщение истории должно содержать роль и непустой текст.")
    return history


_SCHEMA = """
CREATE TABLE IF NOT EXISTS assistant_user_state (
    chat_id BIGINT PRIMARY KEY,
    mode TEXT NOT NULL DEFAULT 'study' CHECK (mode IN ('study', 'translate', 'quiz')),
    history JSONB NOT NULL DEFAULT '[]'::jsonb CHECK (jsonb_typeof(history) = 'array')
);
ALTER TABLE assistant_user_state ADD COLUMN IF NOT EXISTS
    temperature DOUBLE PRECISION NOT NULL DEFAULT 0.3
    CHECK (temperature IN (0.0, 0.3, 0.7, 1.0));
"""


class UserStateError(Exception):
    """Безопасное сообщение об ошибке хранения пользовательских настроек."""


class UserStateStore:
    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool

    async def initialize(self) -> None:
        """Повторный запуск не удаляет таблицу и существующие данные."""
        try:
            await self._pool.execute(_SCHEMA)
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError(
                "Не удалось подготовить хранилище настроек. Проверьте БД."
            ) from None

    async def get_mode(self, chat_id: int) -> str:
        try:
            mode = await self._pool.fetchval(
                "SELECT mode FROM assistant_user_state WHERE chat_id = $1", chat_id
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError("Не удалось загрузить режим. Попробуйте позже.") from None
        if mode is None:
            return DEFAULT_MODE
        if mode not in MODES:
            raise UserStateError(
                "Сохранённый режим недоступен. Выберите /study, /translate или /quiz."
            )
        return mode

    async def get_temperature(self, chat_id: int) -> float:
        try:
            temperature = await self._pool.fetchval(
                "SELECT temperature FROM assistant_user_state WHERE chat_id = $1", chat_id
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError("Не удалось загрузить температуру. Попробуйте позже.") from None
        return DEFAULT_TEMPERATURE if temperature is None else temperature

    async def set_temperature(self, chat_id: int, temperature: float) -> None:
        if isinstance(temperature, bool) or temperature not in TEMPERATURES:
            raise ValueError("Недопустимая температура.")
        try:
            await self._pool.execute(
                "INSERT INTO assistant_user_state (chat_id, temperature) VALUES ($1, $2) "
                "ON CONFLICT (chat_id) DO UPDATE SET temperature = EXCLUDED.temperature",
                chat_id,
                temperature,
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError("Не удалось сохранить температуру. Попробуйте позже.") from None

    async def get_history(self, chat_id: int) -> list[HistoryMessage]:
        """Возвращает сообщения по порядку; для нового чата — пустой список."""
        try:
            raw = await self._pool.fetchval(
                "SELECT history::text FROM assistant_user_state WHERE chat_id = $1", chat_id
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError("Не удалось загрузить историю. Попробуйте позже.") from None
        if raw is None:
            return []
        try:
            return _validate_history(json.loads(raw))
        except (ValueError, TypeError):
            raise UserStateError("Сохранённая история повреждена.") from None

    async def save_history(self, chat_id: int, history: list[HistoryMessage]) -> None:
        """Заменяет историю чата, сохраняя режим и температуру.

        Чтение, изменение и сохранение одного чата вызывающий код должен
        выполнять последовательно, чтобы параллельные запросы не теряли сообщения.
        """
        _validate_history(history)
        encoded = json.dumps(history, ensure_ascii=False)
        try:
            await self._pool.execute(
                "INSERT INTO assistant_user_state (chat_id, history) VALUES ($1, $2::jsonb) "
                "ON CONFLICT (chat_id) DO UPDATE SET history = EXCLUDED.history",
                chat_id,
                encoded,
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError("Не удалось сохранить историю. Попробуйте позже.") from None

    async def set_mode(self, chat_id: int, mode: str) -> None:
        if mode not in MODES:
            raise ValueError("Неизвестный режим.")
        try:
            # Смена режима и сброс истории атомарны.
            await self._pool.execute(
                "INSERT INTO assistant_user_state (chat_id, mode) VALUES ($1, $2) "
                "ON CONFLICT (chat_id) DO UPDATE SET mode = EXCLUDED.mode, history = '[]'::jsonb",
                chat_id,
                mode,
            )
        except (asyncpg.PostgresError, asyncpg.InterfaceError, OSError, TimeoutError):
            raise UserStateError("Не удалось сохранить режим. Попробуйте позже.") from None
