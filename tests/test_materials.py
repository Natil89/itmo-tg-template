import json

import pytest
from pydantic import ValidationError

from app.agent_contracts import MaterialCatalog
from app.agent_policy import ScopedMcp
from app.materials import DOCUMENTS, MaterialsError, MaterialsService, SearchQuery
from app.mcp_client import connect_schedule
from tests.test_agent_boundaries import request


def corpus(root):
    for filename, title in DOCUMENTS.values():
        (root / filename).write_text(f"# {title}\nУчебный материал MCP", encoding="utf-8")
    return MaterialsService(root)


def test_real_corpus_has_stable_ids_and_course_links():
    catalog = MaterialsService().catalog()
    assert {doc.document_id for doc in catalog.documents} == set(DOCUMENTS)
    assert len([doc for doc in catalog.documents if doc.source_url]) == 6


@pytest.mark.parametrize("query", ["назначение MCP", "контекстное окно", "идемпотентность"])
def test_search_has_bounded_citable_excerpts(query):
    result = MaterialsService().search(SearchQuery(query=query, limit=3))
    assert 1 <= len(result.hits) <= 3
    assert all(0 < len(hit.excerpt) <= 1800 and hit.source for hit in result.hits)


def test_absent_topic_has_no_invented_result():
    assert MaterialsService().search(SearchQuery(query="несуществующеесловоxyz")).hits == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"query": " "},
        {"limit": True},
        {"limit": 0},
        {"limit": 6},
        {"path": "../.env"},
        {"actor_id": 999},
    ],
)
def test_search_rejects_arbitrary_path_owner_and_invalid_arguments(changes):
    with pytest.raises(ValidationError):
        SearchQuery.model_validate({"query": "MCP"} | changes)


def test_missing_or_oversized_file_is_error_not_no_results(tmp_path):
    service = corpus(tmp_path)
    (tmp_path / "mcp.md").write_text("x" * 128001, encoding="utf-8")
    with pytest.raises(MaterialsError):
        service.search(SearchQuery(query="MCP"))
    (tmp_path / "mcp.md").unlink()
    with pytest.raises(MaterialsError):
        service.catalog()


def test_symlink_cannot_escape_configured_corpus(tmp_path):
    root = tmp_path / "corpus"
    root.mkdir()
    service = corpus(root)
    outside = tmp_path / "outside.md"
    outside.write_text("private marker", encoding="utf-8")
    (root / "mcp.md").unlink()
    (root / "mcp.md").symlink_to(outside)
    with pytest.raises(MaterialsError) as error:
        service.catalog()
    assert "private" not in str(error.value)


async def test_real_stdio_catalog_search_and_rejected_injection():
    async with connect_schedule() as mcp:
        assert mcp.session is not None
        scope = ScopedMcp(mcp, "knowledge", request())
        response = await scope.read_resource("materials://catalog")
        catalog = MaterialCatalog.model_validate_json(response.contents[0].text)
        assert len(catalog.documents) == 9
        result = await scope.call_tool(
            "search_materials", {"query": "контекстное окно", "limit": 3}
        )
        assert result["hits"]
        assert scope.facts.hits
        assert scope.used_tools == ("search_materials",)
        assert scope.status == "success"
        invalid = await mcp.session.call_tool(
            "search_materials", {"query": "MCP", "path": "../.env"}
        )
        assert invalid.isError
        assert "POSTGRES_PASSWORD" not in json.dumps(invalid.model_dump())
