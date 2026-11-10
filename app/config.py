"""Единственное место чтения и проверки настроек приложения."""

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from aiogram.utils.token import TokenValidationError, validate_token
from dotenv import dotenv_values

from app.history import DEFAULT_HISTORY_LIMITS, HistoryLimits


class ConfigError(ValueError):
    """Ошибка настройки без секретных значений в сообщении."""


@dataclass(frozen=True)
class Settings:
    bot_token: str = field(repr=False)
    postgres_password: str = field(repr=False)
    telegram_proxy_url: str = field(default="", repr=False)
    postgres_host: str = "127.0.0.1"
    postgres_port: int = 5432
    postgres_db: str = "bot"
    postgres_user: str = "bot"
    log_level: str = "INFO"
    health_port: int = 8080
    llm_api_key: str = field(default="", repr=False)
    llm_model: str = ""
    llm_base_url: str = field(default="https://openrouter.ai/api/v1", repr=False)
    history_limits: HistoryLimits = DEFAULT_HISTORY_LIMITS

    def validate_llm(self) -> None:
        """LLM обязателен для бота, но не для подготовки БД и healthcheck."""
        if not self.llm_api_key.strip():
            raise ConfigError("LLM_API_KEY: укажите ключ доступа к модели.")
        if not self.llm_model.strip():
            raise ConfigError("LLM_MODEL: укажите идентификатор модели.")
        try:
            parsed = urlsplit(self.llm_base_url)
            if (
                parsed.scheme != "https"
                or not parsed.hostname
                or parsed.username is not None
                or parsed.password is not None
                or parsed.query
                or parsed.fragment
                or (parsed.port is not None and not 1 <= parsed.port <= 65535)
            ):
                raise ValueError
        except ValueError:
            raise ConfigError("LLM_BASE_URL: нужен HTTPS-адрес API без учётных данных.") from None

    @classmethod
    def load(
        cls,
        env_file: Path | str = ".env",
        *,
        environ: Mapping[str, str] | None = None,
        require_llm: bool = False,
    ) -> "Settings":
        values = {
            **dotenv_values(env_file, interpolate=False),
            **(os.environ if environ is None else environ),
        }

        def value(key: str, default: str = "") -> str:
            return values.get(key) or default

        def port(key: str, default: str) -> int:
            try:
                result = int(value(key, default))
                if not 1 <= result <= 65535:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужен номер порта от 1 до 65535.") from None

        def history_limit(key: str, default: int, minimum: int) -> int:
            try:
                result = int(value(key, str(default)))
                if result < minimum:
                    raise ValueError
                return result
            except ValueError:
                raise ConfigError(f"{key}: нужно целое число от {minimum}.") from None

        token = value("BOT_TOKEN")
        try:
            validate_token(token)
        except TokenValidationError:
            raise ConfigError("BOT_TOKEN: укажите токен, полученный у BotFather.") from None
        password = value("POSTGRES_PASSWORD")
        if not password:
            raise ConfigError("POSTGRES_PASSWORD: пароль базы данных не задан.")
        proxy = value("TELEGRAM_PROXY_URL")
        if proxy:
            try:
                parsed = urlsplit(proxy)
                if (
                    parsed.scheme not in {"http", "socks5"}
                    or not parsed.hostname
                    or not parsed.port
                    or parsed.path not in {"", "/"}
                    or parsed.query
                    or parsed.fragment
                ):
                    raise ValueError
            except ValueError:
                raise ConfigError(
                    "TELEGRAM_PROXY_URL: нужен http://host:port или socks5://host:port; "
                    "при необходимости добавьте user:password@."
                ) from None
        level = value("LOG_LEVEL", "INFO").upper()
        if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigError("LOG_LEVEL: используйте DEBUG, INFO, WARNING, ERROR или CRITICAL.")
        settings = cls(
            bot_token=token,
            postgres_password=password,
            telegram_proxy_url=proxy,
            postgres_host=value("POSTGRES_HOST", "127.0.0.1"),
            postgres_port=port("POSTGRES_PORT", "5432"),
            postgres_db=value("POSTGRES_DB", "bot"),
            postgres_user=value("POSTGRES_USER", "bot"),
            log_level=level,
            health_port=port("HEALTH_PORT", "8080"),
            llm_api_key=value("LLM_API_KEY").strip(),
            llm_model=value("LLM_MODEL").strip(),
            llm_base_url=value("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip(),
            history_limits=HistoryLimits(
                max_messages=history_limit(
                    "HISTORY_MAX_MESSAGES", DEFAULT_HISTORY_LIMITS.max_messages, 2
                ),
                context_budget=history_limit(
                    "LLM_CONTEXT_BUDGET", DEFAULT_HISTORY_LIMITS.context_budget, 1001
                ),
            ),
        )
        if require_llm:
            settings.validate_llm()
        return settings
