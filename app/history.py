"""Ограничения истории и приблизительная оценка контекста."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypedDict


class HistoryMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class HistoryLimits:
    max_messages: int = 20
    context_budget: int = 32_000


DEFAULT_HISTORY_LIMITS = HistoryLimits()


def estimate_message(content: str) -> int:
    """Байты UTF-8 и запас на роль и границы сообщения, без токенизатора."""
    return len(content.encode("utf-8")) + 16


def recent_history(
    history: Sequence[HistoryMessage], limits: HistoryLimits, *, reserved: int = 0
) -> list[HistoryMessage]:
    """Возвращает самый свежий непрерывный отрезок из полных пар."""
    end = len(history)
    start = end
    remaining = limits.context_budget - reserved
    while start >= 2 and end - start + 2 <= limits.max_messages:
        question, answer = history[start - 2 : start]
        if question["role"] != "user" or answer["role"] != "assistant":
            break
        cost = estimate_message(question["content"]) + estimate_message(answer["content"])
        if cost > remaining:
            break
        remaining -= cost
        start -= 2
    return list(history[start:end])
