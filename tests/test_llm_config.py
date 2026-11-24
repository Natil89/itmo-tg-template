import logging
from unittest.mock import Mock

import pytest

from app.config import ConfigError, Settings
from app.logging_setup import configure_logging

BASE = {
    "BOT_TOKEN": "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
    "POSTGRES_PASSWORD": "fake-db-password",
    "LLM_API_KEY": "fake-llm-key",
    "LLM_MODEL": "test/model",
}


def test_llm_settings_environment_overrides_file_without_exposing_key(tmp_path):
    # Arrange
    path = tmp_path / ".env"
    path.write_text("LLM_MODEL=old/model\nLLM_API_KEY=old-key\n", encoding="utf-8")
    # Act
    settings = Settings.load(path, environ=BASE, require_llm=True)
    # Assert
    assert settings.llm_api_key == "fake-llm-key"
    assert settings.llm_model == "test/model"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
    assert "fake-llm-key" not in repr(settings)


@pytest.mark.parametrize(
    "key,value",
    [
        ("LLM_API_KEY", ""),
        ("LLM_API_KEY", " "),
        ("LLM_MODEL", ""),
        ("LLM_BASE_URL", "http://example.org/api"),
        ("LLM_BASE_URL", "https://user:fake-secret@example.org/api"),
        ("LLM_BASE_URL", "https://example.org:99999/api"),
        ("LLM_BASE_URL", "https://example.org/api?key=fake-secret"),
    ],
)
def test_invalid_llm_configuration_has_safe_error(tmp_path, key, value):
    # Arrange / Act / Assert
    with pytest.raises(ConfigError, match=key) as error:
        Settings.load(tmp_path / ".env", environ={**BASE, key: value}, require_llm=True)
    assert "fake-secret" not in str(error.value)


def test_database_setup_does_not_require_llm_key(tmp_path):
    # Arrange
    values = {key: value for key, value in BASE.items() if not key.startswith("LLM_")}
    # Act
    settings = Settings.load(tmp_path / ".env", environ=values)
    # Assert
    assert settings.llm_api_key == ""


def test_logging_registers_and_redacts_llm_key(monkeypatch):
    # Arrange
    basic_config = Mock()
    monkeypatch.setattr(logging, "basicConfig", basic_config)
    settings = Settings(bot_token="fake-token", postgres_password="fake-db", llm_api_key="fake-key")
    # Act
    configure_logging(settings)
    handler = basic_config.call_args.kwargs["handlers"][0]
    record = logging.LogRecord("test", logging.ERROR, "", 1, "Error fake-key", (), None)
    handler.filter(record)
    # Assert
    assert record.getMessage() == "Error [скрыто]"
