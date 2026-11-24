import argparse
import asyncio
import logging
from urllib.parse import unquote, urlsplit

from aiogram import Dispatcher
from aiogram.fsm.storage.memory import SimpleEventIsolation
from aiohttp import ClientSession

from app.assistant_service import AssistantService
from app.audit import AuditStore
from app.briefing import BriefingService, BriefingStore
from app.briefing_sources import BriefingSources
from app.config import ConfigError, Settings
from app.db import create_pool
from app.delivery import DeliveryStore, ReminderDeliveryService
from app.delivery_telegram import TelegramReminderSender
from app.handlers.assistant import create_router
from app.health import HealthState, start_health_server
from app.lesson_compat import LessonCompatibility
from app.llm import LlmClient
from app.logging_setup import configure_logging
from app.materials import MaterialsService
from app.mcp_client import connect_schedule
from app.notifications import CitySelection, NotificationStore
from app.personal_schedule import PersonalScheduleStore
from app.reminders import ReminderStore
from app.specialists import KnowledgeAgent, LifestyleAgent, ScheduleSpecialist
from app.supervisor import LlmRouter, Supervisor
from app.telegram import TelegramStartupError, check_telegram_startup, create_bot
from app.timezones import TimezoneResolver
from app.tracing import TraceStore
from app.user_state import UserStateStore
from app.weather import WeatherService

logger = logging.getLogger("app")


async def run(settings: Settings) -> None:
    settings.validate_llm()
    state = HealthState()
    bot = create_bot(settings)
    runner = None
    delivery_task = None
    briefing_task = None
    city_cleanup_task = None
    try:
        state.pool = await create_pool(settings)
        user_state = UserStateStore(state.pool)
        await user_state.initialize()
        personal_schedule = PersonalScheduleStore(state.pool)
        await personal_schedule.initialize()
        reminders = ReminderStore(state.pool)
        await reminders.initialize()
        notification_store = NotificationStore(state.pool)
        await notification_store.initialize()
        briefing_store = BriefingStore(notification_store)
        await briefing_store.initialize()
        delivery_store = DeliveryStore(state.pool) if settings.reminder_delivery_enabled else None
        if delivery_store is not None:
            await delivery_store.initialize()
        proxy_password = urlsplit(settings.telegram_proxy_url).password or ""
        audit_store = AuditStore(
            state.pool,
            secrets=(
                settings.bot_token,
                settings.llm_api_key,
                settings.postgres_password,
                settings.telegram_proxy_url,
                proxy_password,
                unquote(proxy_password),
            ),
        )
        await audit_store.initialize()
        trace_store = TraceStore(state.pool)
        await trace_store.initialize()
        logger.info("PostgreSQL подключён: SELECT 1 выполнен.")
        # Начальная проверка токена и маршрута через прокси ограничена по времени.
        async with asyncio.timeout(30):
            me, webhook = await check_telegram_startup(bot)
        if webhook.url:
            raise ConfigError(
                "У бота установлен webhook. Удалите его перед запуском polling "
                "или используйте отдельного учебного бота."
            )
        logger.info("Telegram доступен. Бот @%s запускает polling.", me.username)
        async with (
            ClientSession() as llm_session,
            connect_schedule(
                settings.schedule_file,
                settings.schedule_timezone,
                database_settings=settings,
            ) as schedule_mcp,
        ):
            llm = LlmClient(
                llm_session,
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                history_limits=settings.history_limits,
            )
            weather = WeatherService(llm_session, location_policy=settings.weather_location_policy)
            city_selection = CitySelection(weather, notification_store)
            supervisor = Supervisor(
                LlmRouter(llm),
                {
                    "schedule": ScheduleSpecialist(llm),
                    "knowledge": KnowledgeAgent(llm),
                    "lifestyle": LifestyleAgent(llm),
                },
                schedule_mcp,
                reminders=reminders,
                lesson_compat=LessonCompatibility(llm, schedule_mcp, personal_schedule),
            )
            assistant_service = AssistantService(
                llm, supervisor, reminders, settings.schedule_timezone
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
                    schedule_mcp=schedule_mcp,
                    assistant_service=assistant_service,
                    personal_schedule=personal_schedule,
                    reminders=reminders,
                    audit_store=audit_store,
                    trace_store=trace_store,
                    delivery_store=delivery_store,
                    delivery_enabled=settings.reminder_delivery_enabled,
                    notification_store=notification_store,
                    city_selection=city_selection,
                    timezone_resolver=TimezoneResolver(llm_session),
                    user_state=user_state,
                    history_limits=settings.history_limits,
                    allowed_updates=dispatcher.resolve_used_update_types(),
                    close_bot_session=False,
                )
            )
            state.initialized = True
            city_cleanup_task = asyncio.create_task(city_selection.run_cleanup())
            briefing_task = asyncio.create_task(
                BriefingService(
                    briefing_store,
                    BriefingSources(state.pool, personal_schedule, weather, MaterialsService()),
                    TelegramReminderSender(bot),
                    trace_store=trace_store,
                ).run()
            )
            if settings.reminder_delivery_enabled:
                delivery_task = asyncio.create_task(
                    ReminderDeliveryService(
                        delivery_store, TelegramReminderSender(bot), trace_store=trace_store
                    ).run()
                )
            try:
                await state.polling_task
            finally:
                state.initialized = False
                for task in (briefing_task, delivery_task, city_cleanup_task):
                    if task is not None:
                        task.cancel()
                        await asyncio.gather(task, return_exceptions=True)
                if not state.polling_task.done():
                    state.polling_task.cancel()
                    await asyncio.gather(state.polling_task, return_exceptions=True)
    finally:
        state.initialized = False
        if city_cleanup_task is not None:
            city_cleanup_task.cancel()
            await asyncio.gather(city_cleanup_task, return_exceptions=True)
        if briefing_task is not None:
            briefing_task.cancel()
            await asyncio.gather(briefing_task, return_exceptions=True)
        if delivery_task is not None:
            delivery_task.cancel()
            await asyncio.gather(delivery_task, return_exceptions=True)
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
    except TelegramStartupError as error:
        logger.error("%s", error)
        return 1
    except Exception:
        logger.exception("Не удалось запустить бот. Проверьте БД, токен, прокси и настройки LLM.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
