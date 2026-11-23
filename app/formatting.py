"""Markdown модели → текст и Telegram entities, без разбора HTML со стороны Telegram."""

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from aiogram.types import MessageEntity
from markdown_it import MarkdownIt
from markdown_it.token import Token


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def split_text(text: str) -> list[str]:
    parts = []
    start = length = 0
    for index, character in enumerate(text):
        size = 2 if ord(character) > 0xFFFF else 1
        if length + size > 4096:
            parts.append(text[start:index])
            start, length = index, 0
        length += size
    if start < len(text):
        parts.append(text[start:])
    return parts


@dataclass
class FormattedMessage:
    text: str
    entities: list[MessageEntity] = field(default_factory=list)


class _Renderer:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.entities: list[MessageEntity] = []
        self.length = 0
        self.characters = 0
        self.content_end = 0
        self.tail = ""
        self.open_entities: list[tuple[str, str | None, int, dict]] = []
        self.lists: list[int | None] = []
        self.formatted = False

    def append(self, text: str, *, layout: bool = False) -> None:
        self.parts.append(text)
        self.length += utf16_length(text)
        self.characters += len(text)
        self.tail = (self.tail + text)[-2:]
        if text and not layout:
            self.content_end = self.characters

    def newline(self, count: int = 2) -> None:
        if self.length:
            existing = len(self.tail) - len(self.tail.rstrip("\n"))
            self.append("\n" * max(0, count - existing), layout=True)

    def open(self, tag: str, kind: str | None, **attributes: str) -> None:
        # Вложенные одинаковые выделения избыточны; code/pre остаются отдельными entities.
        if kind and any(active == kind for _, active, _, _ in self.open_entities):
            kind = None
        self.open_entities.append((tag, kind, self.length, attributes))

    def close(self) -> None:
        _, kind, start, attributes = self.open_entities.pop()
        if kind and self.length > start:
            self.entities.append(
                MessageEntity(type=kind, offset=start, length=self.length - start, **attributes)
            )

    def span(self, text: str, kind: str, **attributes: str) -> None:
        self.open(kind, kind, **attributes)
        self.append(text)
        self.close()

    def render(self, tokens: list[Token]) -> None:
        for token in tokens:
            kind = token.type
            if kind in {
                "strong_open",
                "em_open",
                "s_open",
                "link_open",
                "code_inline",
                "fence",
                "code_block",
                "heading_open",
                "bullet_list_open",
                "ordered_list_open",
                "hr",
                "image",
                "blockquote_open",
                "hardbreak",
            }:
                self.formatted = True
            if kind == "inline":
                self.render(token.children or [])
            elif kind == "text":
                self.append(token.content)
            elif kind in {"softbreak", "hardbreak"}:
                self.append("\n")
            elif kind in {"strong_open", "em_open", "s_open"}:
                self.open(
                    token.tag, {"strong": "bold", "em": "italic", "s": "strikethrough"}[token.tag]
                )
            elif kind in {"strong_close", "em_close", "s_close", "link_close"}:
                self.close()
            elif kind == "link_open":
                url = token.attrGet("href") or ""
                try:
                    allowed = urlsplit(url).scheme in {"http", "https", "mailto"}
                except ValueError:
                    allowed = False
                self.open("a", "text_link" if allowed else None, url=url)
            elif kind == "code_inline":
                self.span(token.content, "code")
            elif kind in {"fence", "code_block"}:
                self.newline()
                language = token.info.split()[0] if token.info.strip() else ""
                language = language if re.fullmatch(r"[\w+.-]{1,32}", language) else ""
                self.span(token.content, "pre", language=language)
                self.newline()
            elif kind == "heading_open":
                self.newline()
                self.open(token.tag, "bold")
            elif kind == "heading_close":
                self.close()
                self.newline()
            elif kind == "paragraph_open":
                if not self.lists:
                    self.newline()
            elif kind == "paragraph_close":
                self.newline(1 if self.lists else 2)
            elif kind in {"bullet_list_open", "ordered_list_open"}:
                self.newline(1 if self.lists else 2)
                self.lists.append(
                    int(token.attrGet("start") or 1) if kind == "ordered_list_open" else None
                )
            elif kind in {"bullet_list_close", "ordered_list_close"}:
                self.lists.pop()
                self.newline(1 if self.lists else 2)
            elif kind == "list_item_open":
                self.newline(1)
                number = self.lists[-1]
                self.append(
                    "  " * (len(self.lists) - 1) + ("• " if number is None else f"{number}. ")
                )
                if number is not None:
                    self.lists[-1] = number + 1
            elif kind == "list_item_close":
                self.newline(1)
            elif kind == "hr":
                self.newline()
                self.append("────────")
                self.newline()
            elif kind == "image":
                self.append(token.content + " (" + (token.attrGet("src") or "") + ")")
            elif kind == "blockquote_open":
                self.newline()
            elif kind == "blockquote_close":
                self.newline()


def _without_code_overlap(entities: list[MessageEntity]) -> list[MessageEntity]:
    """Telegram не допускает другие entities внутри code/pre или вокруг них."""
    code = [entity for entity in entities if entity.type in {"code", "pre"}]
    result = list(code)
    for entity in entities:
        if entity.type in {"code", "pre"}:
            continue
        ranges = [(entity.offset, entity.offset + entity.length)]
        for protected in code:
            remaining = []
            left, right = protected.offset, protected.offset + protected.length
            for start, stop in ranges:
                if stop <= left or start >= right:
                    remaining.append((start, stop))
                else:
                    if start < left:
                        remaining.append((start, left))
                    if stop > right:
                        remaining.append((right, stop))
            ranges = remaining
        result.extend(
            entity.model_copy(update={"offset": start, "length": stop - start})
            for start, stop in ranges
        )
    return result


def format_answer(markdown: str) -> list[FormattedMessage]:
    """Поддерживает заголовки, выделение, ссылки, списки и код; HTML остаётся текстом.

    Entities обрезаются по границе каждой части и получают локальные UTF-16 offsets.
    Таблицы и прочие расширения Markdown не включены.
    """
    renderer = _Renderer()
    renderer.render(
        MarkdownIt("commonmark", {"html": False}).enable("strikethrough").parse(markdown)
    )
    renderer.entities = _without_code_overlap(renderer.entities)
    text = "".join(renderer.parts)[: renderer.content_end]
    # Одни разделители Markdown не должны приводить к отправке пустого сообщения.
    if not renderer.formatted or not text.strip():
        return [FormattedMessage(part) for part in split_text(markdown)]
    messages = []
    offset = 0
    for part in split_text(text):
        end = offset + utf16_length(part)
        entities = []
        for entity in renderer.entities:
            start = max(offset, entity.offset)
            stop = min(end, entity.offset + entity.length)
            if stop > start:
                entities.append(
                    entity.model_copy(update={"offset": start - offset, "length": stop - start})
                )
        entities.sort(key=lambda entity: (entity.offset, -entity.length))
        messages.append(FormattedMessage(part, entities))
        offset = end
    return messages
