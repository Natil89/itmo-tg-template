import argparse
import asyncio
import logging

from aiogram import Dispatcher
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiohttp import ClientSession

from app.config import ConfigError, Settings
from app.db import create_pool
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.llm import LlmClient
from app.logging_setup import configure_logging
from app.telegram import create_bot
from app.user_state import UserStateStore

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    settings.validate_llm()
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    try:
        state.pool = await create_pool(settings)
        user_state = UserStateStore(state.pool)
        await user_state.initialize()
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me = await bot.get_me()
            webhook = await bot.get_webhook_info()
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        async with ClientSession() as llm_session:
            llm = LlmClient(
                llm_session,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                history_limits=settings.history_limits,
            )
            # В личном чате вопрос и команда режима меняют одну строку истории.
            dispatcher = Dispatcher(events_isolation=SimpleEventIsolation())
            dispatcher.include_router(create_router())
            runner = await start_health_server(state, settings.health_port)
            state.polling_task = asyncio.create_task(
                dispatcher.start_polling(
                    bot,
                    db=state.pool,
                    llm=llm,
                    user_state=user_state,
                    history_limits=settings.history_limits,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    close_bot_session=False,
                )
            )
            state.initialized = True
            try:
                await state.polling_task
            finally:
                state.initialized = False
                if not state.polling_task.done():
                    state.polling_task.cancel()
                    await asyncio.gather(state.polling_task, return_exceptions=True)
    finally:
        state.initialized = False
        if state.polling_task and not state.polling_task.done():
            state.polling_task.cancel()
            await asyncio.gather(state.polling_task, return_exceptions=True)
        if runner:
            await runner.cleanup()
        await bot.session.close()
        if state.pool:
            try:
                async with asyncio.timeout(10):
                    await state.pool.close()
            except TimeoutError:
                state.pool.terminate()


def main() -> int:
    parser = argparse.ArgumentParser(description="Учебный Telegram-бот с языковой моделью")
    parser.add_argument("--env-file", default=".env")
    args = parser.parse_args()
    try:
        settings = Settings.load(args.env_file, require_llm=True)
    except ConfigError as exc:
        print(str(exc))
        return 1
    configure_logging(settings)
    try:
        asyncio.run(run(settings))
    except KeyboardInterrupt:
        logger.info("Бот остановлен.")
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен, прокси и настройки LLM.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
