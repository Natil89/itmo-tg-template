"""Telegram-адаптер и сохранение адреса из доверенных private updates."""

from collections.abc import Awaitable, Callable
from typing import Any

from aiogram import BaseMiddleware, Bot
from aiogram.exceptions import (
    TelegramBadRequest,
    TelegramForbiddenError,
    TelegramNetworkError,
    TelegramRetryAfter,
    TelegramServerError,
)
from aiogram.types import CallbackQuery, Message, TelegramObject

from app.delivery import DeliveryError, PermanentDeliveryError, TemporaryDeliveryError
from app.notifications import NotificationError


class TelegramReminderSender:
    def __init__(self, bot: Bot) -> None:
        self._bot = bot

    async def send(self, destination_chat_id: int, text: str) -> int:
        try:
            message = await self._bot.send_message(destination_chat_id, text, parse_mode=None)
        except TelegramRetryAfter as error:
            raise TemporaryDeliveryError(error.retry_after) from None
        except (TelegramNetworkError, TelegramServerError):
            raise TemporaryDeliveryError from None
        except (TelegramForbiddenError, TelegramBadRequest):
            raise PermanentDeliveryError from None
        return message.message_id


class DeliveryDestinationMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        store = data.get("delivery_store")
        notifications = data.get("notification_store")
        message = (
            event
            if isinstance(event, Message)
            else event.message
            if isinstance(event, CallbackQuery)
            else None
        )
        user = event.from_user if isinstance(event, (Message, CallbackQuery)) else None
        if (
            (store is not None or notifications is not None)
            and isinstance(message, Message)
            and user is not None
            and message.chat.type == "private"
            and message.chat.id == user.id
        ):
            try:
                if store is not None:
                    await store.bind_destination(user.id, message.chat.id)
                if notifications is not None:
                    await notifications.touch_destination(user.id, message.chat.id)
            except (DeliveryError, NotificationError):
                await message.answer(
                    "Не удалось проверить адрес доставки. Повторите позже.", parse_mode=None
                )
                return None
        return await handler(event, data)
