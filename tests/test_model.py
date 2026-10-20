"""End-to-end checks with the real default embedding model (downloads once)."""

from pathlib import Path

import pytest

from mcp_contrast.analyze import analyze
from mcp_contrast.config import Settings
from mcp_contrast.embed import Model2VecEmbedder
from mcp_contrast.sources import Connect, from_file

pytestmark = pytest.mark.model
EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module")
def embedder() -> Model2VecEmbedder:
    return Model2VecEmbedder()


def test_vectors_are_normalized(embedder) -> None:
    vectors = embedder.embed(["Update a person record.", "Update a company record."])
    assert vectors.shape[0] == 2
    assert abs(float(vectors[0] @ vectors[0]) - 1.0) < 1e-5


def test_crm_example(embedder) -> None:
    servers = from_file(EXAMPLES / "crm.json", Connect())
    report = analyze(servers, Settings(), embedder)
    found = {(f.rule, f.a.name, f.b.name) for f in report.findings}
    assert found == {
        ("synonym-contrast", "get_person", "fetch_person"),
        ("unmarked-contrast", "list_deals", "list_all_deals"),
        ("hidden-overlap", "search_notes", "find_notes_by_text"),
    }
    clean = {(c.a.name, c.b.name) for c in report.clean}
    assert ("update_person", "update_company") in clean
    assert ("delete_person", "archive_person") in clean
