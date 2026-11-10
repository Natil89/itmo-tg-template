"""Асинхронный клиент OpenRouter без чтения .env и зависимостей от Telegram."""

import logging
import math
from collections.abc import Sequence
from time import perf_counter
from urllib.parse import urlsplit
from uuid import uuid4

import aiohttp

from app.generation import DEFAULT_TEMPERATURE, TEMPERATURES
from app.history import (
    DEFAULT_HISTORY_LIMITS,
    HistoryLimits,
    HistoryMessage,
    estimate_message,
    recent_history,
)

logger = logging.getLogger(__name__)


class LlmError(Exception):
    """Безопасное сообщение об ошибке, которое можно показать пользователю."""


class LlmClient:
    """Переиспользует сессию приложения; закрывает её создавший сессию код.

    Ключ, адрес API и модель передаются из конфигурации при запуске бота.
    generate() возвращает текст, а при сбое поднимает LlmError.
    """

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60,
        max_tokens: int = 1000,
        history_limits: HistoryLimits = DEFAULT_HISTORY_LIMITS,
    ) -> None:
        if not api_key.strip():
            raise ValueError("LLM_API_KEY: ключ не задан.")
        if not model.strip():
            raise ValueError("LLM_MODEL: имя модели не задано.")
        try:
            parsed = urlsplit(base_url)
            valid_url = (
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and (parsed.port is None or 1 <= parsed.port <= 65535)
            )
        except ValueError:
            valid_url = False
        if not valid_url:
            raise ValueError("LLM_BASE_URL: нужен HTTPS-адрес API без учётных данных.")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Таймаут LLM должен быть положительным конечным числом.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("Лимит ответа LLM должен быть положительным целым числом.")

        self._session = session
        self._api_key = api_key.strip()
        self.model = model.strip()
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        self._max_tokens = max_tokens
        self._history_limits = history_limits

    async def generate(
        self,
        text: str,
        *,
        system_prompt: str = "Ты учебный помощник. Отвечай понятно и на русском языке.",
        temperature: float = DEFAULT_TEMPERATURE,
        examples: tuple[tuple[str, str], ...] = (),
        history: Sequence[HistoryMessage] = (),
    ) -> str:
        """Отправляет новый вопрос с примерами и подходящей частью истории."""
        if not text.strip():
            raise LlmError("Отправьте непустое текстовое сообщение.")
        if not system_prompt.strip():
            raise ValueError("Системная инструкция не должна быть пустой.")
        if isinstance(temperature, bool) or temperature not in TEMPERATURES:
            raise LlmError("Выберите temperature: 0.0, 0.3, 0.7 или 1.0.")

        messages = [{"role": "system", "content": system_prompt}]
        for question, answer in examples:
            messages.extend(
                [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
        reserved = (
            self._max_tokens
            + estimate_message(text)
            + sum(estimate_message(message["content"]) for message in messages)
        )
        if reserved > self._history_limits.context_budget:
            raise LlmError("Вопрос слишком длинный для текущего режима. Сократите его.")
        for message in recent_history(history, self._history_limits, reserved=reserved):
            messages.append({"role": message["role"], "content": message["content"]})
        messages.append({"role": "user", "content": text})
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "reasoning": {"enabled": False},
            "max_tokens": self._max_tokens,
        }
        request_id = uuid4().hex
        started = perf_counter()
        outcome = "error"
        logger.info("LLM start request_id=%s model=%s", request_id, self.model)
        try:
            async with self._session.post(
                self._url,
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self._timeout,
                allow_redirects=False,
            ) as response:
                if response.status in (401, 403):
                    raise LlmError("Нет доступа к модели. Проверьте настройки LLM.")
                if response.status == 402:
                    raise LlmError("Недостаточно средств или превышен лимит расходов LLM.")
                if response.status == 429:
                    raise LlmError("Слишком много запросов к модели. Попробуйте позже.")
                if response.status != 200:
                    raise LlmError("Сервис модели недоступен. Попробуйте позже.")
                result = await response.json()

            if not isinstance(result, dict) or result.get("error"):
                raise LlmError("Сервис модели вернул некорректный ответ. Попробуйте позже.")
            choices = result.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise LlmError("Модель не вернула текст ответа. Попробуйте ещё раз.")
            choice = choices[0]
            if choice.get("finish_reason") == "length":
                raise LlmError("Ответ модели превысил лимит. Попробуйте уточнить вопрос.")
            if choice.get("finish_reason") != "stop":
                raise LlmError("Модель не завершила ответ. Попробуйте переформулировать вопрос.")
            message = choice.get("message")
            answer = message.get("content") if isinstance(message, dict) else None
            if not isinstance(answer, str) or not answer.strip():
                raise LlmError("Модель не вернула текст ответа. Попробуйте ещё раз.")
            outcome = "ok"
            return answer
        except TimeoutError:
            raise LlmError("Модель не ответила вовремя. Попробуйте позже.") from None
        except aiohttp.ClientError:
            raise LlmError("Не удалось связаться с моделью. Попробуйте позже.") from None
        except ValueError:
            raise LlmError("Сервис модели вернул некорректный ответ. Попробуйте позже.") from None
        finally:
            logger.info(
                "LLM finish request_id=%s model=%s outcome=%s duration_ms=%.0f",
                request_id,
                self.model,
                outcome,
                (perf_counter() - started) * 1000,
            )
