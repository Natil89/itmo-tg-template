"""RUN_INTEGRATION=1 python -m pytest -m integration -v.

Тест создаёт собственные контейнер и volume; пользовательскую БД не использует.
"""

import asyncio
import json
import os
import secrets
import socket
import time
from dataclasses import replace

import asyncpg
import pytest

from app.config import Settings
from app.db import create_pool
from app.health import HealthState, health_result
from scripts.common import CommandError, run_command

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("RUN_INTEGRATION") != "1",
        reason="Включите RUN_INTEGRATION=1 для Docker-тестов",
    ),
]


@pytest.fixture
def database():
    name = "itmo-test-" + secrets.token_hex(6)
    volume = name + "-data"
    password = secrets.token_urlsafe(24)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

    def docker(*args):
        return run_command(["docker", *args], label="Интеграционный Docker-тест", timeout=180)

    def wait_until_ready():
        deadline = time.monotonic() + 60
        while True:
            try:
                docker("exec", name, "pg_isready", "-h", "127.0.0.1", "-U", "bot", "-d", "bot")
                return
            except CommandError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)

    docker("info")
    docker("volume", "create", volume)
    try:
        docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{port}:5432",
            "-e",
            "POSTGRES_DB=bot",
            "-e",
            "POSTGRES_USER=bot",
            "-e",
            f"POSTGRES_PASSWORD={password}",
            "--mount",
            f"type=volume,source={volume},target=/var/lib/postgresql/data",
            "postgres:16-bookworm",
        )
        info = json.loads(docker("inspect", name))[0]
        assert int(info["NetworkSettings"]["Ports"]["5432/tcp"][0]["HostPort"]) == port
        wait_until_ready()
        yield (
            Settings(
                bot_token="123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijk",
                postgres_password=password,
                postgres_port=port,
            ),
            docker,
            wait_until_ready,
            name,
        )
    finally:
        try:
            docker("rm", "-f", name)
        finally:
            docker("volume", "rm", volume)


async def test_real_db_password_restart_health_and_persistence(database):
    # Arrange
    settings, docker, wait_until_ready, name = database
    pool = await create_pool(settings)
    task = asyncio.create_task(asyncio.Event().wait())
    state = HealthState(pool=pool, initialized=True, polling_task=task)
    try:
        await pool.execute("CREATE TABLE integration_probe (value text NOT NULL)")
        await pool.execute("INSERT INTO integration_probe VALUES ($1)", "Сохранилось 👋")
        # Act / Assert: реальное подключение и ошибка пароля.
        assert (await health_result(state))[0] == 200
        with pytest.raises(asyncpg.InvalidPasswordError):
            await create_pool(replace(settings, postgres_password="wrong-password"))
        # Act / Assert: потеря соединения и сохранность данных после запуска.
        await asyncio.to_thread(docker, "stop", name)
        assert (await health_result(state))[0] == 503
        await asyncio.to_thread(docker, "start", name)
        await asyncio.to_thread(wait_until_ready)
        for _ in range(60):
            if (await health_result(state))[0] == 200:
                break
            await asyncio.sleep(1)
        assert (await health_result(state))[0] == 200
        assert await pool.fetchval("SELECT value FROM integration_probe") == "Сохранилось 👋"
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await pool.close()


async def test_user_modes_are_isolated_clear_history_and_survive_reconnection(database):
    # Arrange
    from app.user_state import UserStateStore

    settings, _, _, _ = database
    pool = await create_pool(settings)
    try:
        store = UserStateStore(pool)
        # Таблица из предыдущей версии приложения, без колонки temperature.
        await pool.execute(
            "CREATE TABLE assistant_user_state ("
            "chat_id BIGINT PRIMARY KEY, "
            "mode TEXT NOT NULL DEFAULT 'study' CHECK (mode IN ('study', 'translate', 'quiz')), "
            "history JSONB NOT NULL DEFAULT '[]'::jsonb)"
        )
        await pool.execute(
            "INSERT INTO assistant_user_state (chat_id, mode, history) VALUES (404, 'quiz', $1)",
            json.dumps([{"role": "user", "content": "Старый вопрос"}]),
        )
        await store.initialize()
        assert await store.get_mode(404) == "quiz"
        assert await store.get_history(404) == [{"role": "user", "content": "Старый вопрос"}]
        assert await store.get_temperature(404) == 0.3
        assert await store.get_mode(101) == "study"
        assert await store.get_temperature(101) == 0.3
        assert await store.get_history(101) == []
        history = [
            {"role": "user", "content": "Учебный пример 👋"},
            {"role": "assistant", "content": "Ответ модели"},
        ]
        await store.save_history(101, history)
        assert await store.get_mode(101) == "study"
        assert await store.get_history(101) == history
        await store.set_mode(101, "translate")
        assert await store.get_history(101) == []
        await store.set_mode(202, "quiz")
        await store.save_history(101, history)
        await store.save_history(202, history)
        await store.set_temperature(101, 0.7)
        await store.set_temperature(202, 1.0)
        assert await store.get_history(101) == history
        assert await store.get_mode(101) == "translate"
        assert await store.get_mode(202) == "quiz"
        replacement = [{"role": "user", "content": "Другой вопрос"}]
        await store.save_history(101, replacement)
        assert await store.get_history(101) == replacement
        assert await store.get_history(202) == history
        # Сброс истории сохраняет режим и температуру и не трогает второй чат.
        await store.save_history(101, [])
        assert await store.get_mode(101) == "translate"
        assert await store.get_temperature(101) == 0.7
        assert await store.get_history(101) == []
        assert await store.get_history(202) == history
        await store.save_history(101, replacement)
        # Act: переключаем только одного пользователя.
        await store.set_mode(101, "study")
        # Assert
        assert await store.get_mode(101) == "study"
        assert await store.get_mode(202) == "quiz"
        assert await store.get_history(101) == []
        assert await store.get_history(202) == history
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                "UPDATE assistant_user_state SET temperature = 0.5 WHERE chat_id = 101"
            )
        # Ограничение схемы также отклоняет неизвестный режим.
        with pytest.raises(asyncpg.CheckViolationError):
            await pool.execute(
                "UPDATE assistant_user_state SET mode = $1 WHERE chat_id = $2", "bad", 101
            )
    finally:
        await pool.close()

    # Act: новый пул и новый экземпляр хранилища, как после перезапуска приложения.
    pool = await create_pool(settings)
    try:
        store = UserStateStore(pool)
        await store.initialize()
        # Assert: повторная подготовка схемы не сбрасывает настройки.
        assert await store.get_mode(101) == "study"
        assert await store.get_mode(202) == "quiz"
        assert await store.get_mode(303) == "study"
        assert await store.get_temperature(101) == 0.7
        assert await store.get_temperature(202) == 1.0
        assert await store.get_history(101) == []
        assert await store.get_history(202) == history
        assert await store.get_history(303) == []
    finally:
        await pool.close()
