"""Команды проактивных настроек: без SQL, LLM и выбора адреса моделью."""

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.formatting import split_text
from app.notifications import CitySelection, NotificationError, NotificationStore
from app.reminders import ReminderStore
from app.weather import WeatherError


def owner(message: Message) -> int | None:
    if (
        message.from_user is None
        or message.chat.type != "private"
        or message.chat.id != message.from_user.id
    ):
        return None
    return message.from_user.id


async def city(
    message: Message,
    command: CommandObject,
    notification_store: NotificationStore | None = None,
    city_selection: CitySelection | None = None,
) -> None:
    actor = owner(message)
    if actor is None:
        return
    if notification_store is None or city_selection is None:
        await message.answer("Настройки города сейчас недоступны.", parse_mode=None)
        return
    try:
        query = (command.args or "").strip()
        if not query:
            profile = await notification_store.get(actor)
            text = f"Подтверждённый город: {profile.city.place}.\n" if profile else ""
            await message.answer(
                text + "Выбор: /city город, страна. Изменение требует подтверждения.",
                parse_mode=None,
            )
            return
        proposals = await city_selection.propose(actor, message.chat.id, query)
        rows = ["Подтвердите город для ежедневной сводки:"]
        buttons = []
        for number, proposal in enumerate(proposals, 1):
            rows.append(f"{number}. {proposal.city.place}")
            buttons.append(
                [
                    InlineKeyboardButton(
                        text=f"Подтвердить город №{number}",
                        callback_data=f"city:{proposal.token}",
                    )
                ]
            )
        rows.append(
            "Выбор действует пять минут. До подтверждения координаты в БД не сохраняются. "
            "После выбора сводку нужно отдельно включить командой /briefing_on."
        )
        parts = split_text("\n".join(rows))
        for index, part in enumerate(parts):
            await message.answer(
                part,
                parse_mode=None,
                reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons)
                if index == len(parts) - 1
                else None,
            )
    except (NotificationError, WeatherError) as error:
        await message.answer(str(error), parse_mode=None)


async def city_confirmation(
    callback: CallbackQuery, city_selection: CitySelection | None = None
) -> None:
    message = callback.message
    if (
        not isinstance(message, Message)
        or message.chat.type != "private"
        or message.chat.id != callback.from_user.id
    ):
        await callback.answer("Выбор доступен только в вашем личном чате.", show_alert=True)
        return
    await callback.answer()
    try:
        if city_selection is None:
            raise NotificationError("Выбор города сейчас недоступен.")
        profile = await city_selection.confirm(
            (callback.data or "").removeprefix("city:"),
            callback.from_user.id,
            message.chat.id,
        )
        await message.answer(
            f"Город подтверждён: {profile.city.place}.\n"
            "Сводка выключена. Для включения используйте /briefing_on после настройки /timezone.",
            parse_mode=None,
        )
        try:
            await message.edit_reply_markup(reply_markup=None)
        except Exception:
            pass  # Повторный callback не меняет уже подтверждённые настройки.
    except NotificationError as error:
        await message.answer(str(error), parse_mode=None)


async def briefing_toggle(
    message: Message,
    command: CommandObject,
    notification_store: NotificationStore | None = None,
) -> None:
    actor = owner(message)
    if actor is None:
        return
    try:
        if notification_store is None:
            raise NotificationError("Настройки сводки сейчас недоступны.")
        enabled = command.command == "briefing_on"
        if enabled:
            await notification_store.touch_destination(actor, message.chat.id)
        profile = await notification_store.set_enabled(actor, enabled)
        text = (
            f"Ежедневная сводка включена: 08:00 ({profile.timezone}), {profile.city.place}. "
            "Отключить: /briefing_off."
            if enabled
            else "Ежедневная сводка выключена."
        )
        await message.answer(text, parse_mode=None)
    except NotificationError as error:
        await message.answer(str(error), parse_mode=None)


async def notifications(
    message: Message,
    reminders: ReminderStore,
    notification_store: NotificationStore | None = None,
    delivery_enabled: bool = False,
) -> None:
    actor = owner(message)
    if actor is None:
        return
    try:
        if notification_store is None:
            raise NotificationError("Настройки уведомлений сейчас недоступны.")
        profile = await notification_store.get(actor)
        timezone = await reminders.get_timezone(actor)
        daily = "включена, в 08:00" if profile and profile.enabled else "выключена"
        await message.answer(
            f"Город: {profile.city.place if profile else 'не подтверждён'}.\n"
            f"Часовой пояс: {timezone or 'не настроен'}.\n"
            f"Ежедневная сводка: {daily}.\n"
            f"Автодоставка напоминаний: {'включена' if delivery_enabled else 'выключена'}.\n"
            "Город: /city; сводка: /briefing_on или /briefing_off.",
            parse_mode=None,
        )
    except NotificationError as error:
        await message.answer(str(error), parse_mode=None)


def register_notifications(router: Router) -> None:
    router.message.register(city, Command("city"))
    router.message.register(briefing_toggle, Command(commands=["briefing_on", "briefing_off"]))
    router.message.register(notifications, Command("notifications"))
    router.callback_query.register(city_confirmation, F.data.startswith("city:"))
