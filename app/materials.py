"""Ограниченный обезличенный корпус. Пути не входят в аргументы поиска."""

import re
from pathlib import Path
from types import MappingProxyType

from pydantic import Field, field_validator

from app.agent_contracts import Contract, MaterialCatalog, MaterialEntry, MaterialHit

CORPUS_ROOT = Path(__file__).resolve().parent.parent / "docs" / "materials"
DOCUMENTS = MappingProxyType(
    {
        "mcp": ("mcp.md", "MCP: инструменты и ресурсы"),
        "context-output": ("context-output.md", "Контекстное окно и структурированный вывод"),
        "idempotency": ("idempotency.md", "Идемпотентность и доставка уведомлений"),
        "lecture-01": ("lecture-01.md", "Лекция 1: введение в AI-инжиниринг"),
        "lecture-02": ("lecture-02.md", "Лекция 2: модели и параметры инференса"),
        "lecture-03": ("lecture-03.md", "Лекция 3: контекстное окно, KV-cache и RAG"),
        "lecture-04": ("lecture-04.md", "Лекция 4: промпты и структурированный вывод"),
        "lecture-05": ("lecture-05.md", "Лекция 5: агентный цикл и ограничения"),
        "lecture-06": ("lecture-06.md", "Лекция 6: spec-driven development"),
    }
)
SOURCE_URLS = MappingProxyType(
    {
        f"lecture-{i:02}": "https://ai-in-sd-y29.github.io/.github/profile/lectures/" + path
        for i, path in enumerate(
            ("1/", "2/", "3/", "4/", "5/", "lecture-06-spec-driven-development.html"), start=1
        )
    }
)


class MaterialsError(ValueError):
    def __init__(self) -> None:
        super().__init__("Учебные материалы недоступны.")


class SearchQuery(Contract):
    query: str = Field(min_length=1, max_length=300)
    limit: int = Field(default=5, ge=1, le=5)

    @field_validator("query")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Нужен непустой поисковый запрос.")
        return value.strip()


class SearchResult(Contract):
    hits: tuple[MaterialHit, ...] = Field(max_length=5)


def _terms(text: str) -> set[str]:
    # Простой объяснимый поиск: нормализация и обрезка окончаний длинных слов.
    words = re.findall(r"[а-яa-z0-9]+", text.casefold().replace("ё", "е"))
    return {word[:-2] if len(word) > 5 else word for word in words if len(word) > 2}


class MaterialsService:
    def __init__(self, root: Path = CORPUS_ROOT) -> None:
        # Конфигурация приложения/фикстуры, никогда не аргумент модели.
        self._root = root.resolve()

    def _documents(self) -> list[tuple[str, str, str]]:
        documents = []
        try:
            for document_id, (filename, title) in DOCUMENTS.items():
                path = (self._root / filename).resolve(strict=True)
                if not path.is_relative_to(self._root) or not path.is_file():
                    raise MaterialsError
                with path.open("rb") as stream:
                    raw = stream.read(128001)
                if len(raw) > 128000:
                    raise MaterialsError
                text = raw.decode("utf-8").strip()
                if not text:
                    raise MaterialsError
                documents.append((document_id, title, text))
        except (OSError, UnicodeError, RuntimeError):
            raise MaterialsError from None
        return documents

    def catalog(self) -> MaterialCatalog:
        return MaterialCatalog(
            documents=tuple(
                MaterialEntry(
                    document_id=document_id, title=title, source_url=SOURCE_URLS.get(document_id)
                )
                for document_id, title, _ in self._documents()
            )
        )

    def search(self, query: SearchQuery) -> SearchResult:
        query = SearchQuery.model_validate(query.model_dump())
        wanted = _terms(query.query)
        ranked = []
        for document_id, title, text in self._documents():
            # В выдачу идёт лучший ограниченный фрагмент, а не вся лекция.
            windows = [text[i : i + 1800] for i in range(0, len(text), 1400)]
            excerpt = max(windows, key=lambda window: len(wanted & _terms(window)))
            score = len(wanted & _terms(excerpt)) + len(wanted & _terms(title))
            if score:
                ranked.append((score, document_id, title, excerpt))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return SearchResult(
            hits=tuple(
                MaterialHit(
                    document_id=document_id,
                    title=title,
                    excerpt=text,
                    source_url=SOURCE_URLS.get(document_id),
                )
                for _, document_id, title, text in ranked[: query.limit]
            )
        )
