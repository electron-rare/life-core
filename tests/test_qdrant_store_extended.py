"""Extended tests for QdrantVectorStore — iter_chunks & search_multi."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from life_core.rag.pipeline import Chunk, SearchHit


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.get_collections.return_value.collections = []
    return client


@pytest.fixture
def store(mock_client):
    with patch("life_core.rag.qdrant_store.QdrantClient", return_value=mock_client):
        from life_core.rag.qdrant_store import QdrantVectorStore
        return QdrantVectorStore(url="http://localhost:6333", collection_name="test_chunks")


# ---------------------------------------------------------------------------
# iter_chunks
# ---------------------------------------------------------------------------


def test_iter_chunks_empty(store, mock_client):
    mock_client.scroll.return_value = ([], None)
    chunks = store.iter_chunks()
    assert chunks == []
    mock_client.scroll.assert_called_once()


def test_iter_chunks_single_page(store, mock_client):
    point = MagicMock()
    point.payload = {
        "content": "hello",
        "document_id": "doc1",
        "chunk_index": 0,
        "metadata": {"source": "test"},
    }
    mock_client.scroll.return_value = ([point], None)

    chunks = store.iter_chunks()
    assert len(chunks) == 1
    assert chunks[0].content == "hello"
    assert chunks[0].document_id == "doc1"
    assert chunks[0].metadata["collection"] == "test_chunks"


def test_iter_chunks_multiple_pages(store, mock_client):
    point1 = MagicMock()
    point1.payload = {"content": "page1", "document_id": "d1", "chunk_index": 0, "metadata": {}}
    point2 = MagicMock()
    point2.payload = {"content": "page2", "document_id": "d2", "chunk_index": 1, "metadata": {}}

    mock_client.scroll.side_effect = [
        ([point1], "offset-abc"),
        ([point2], None),
    ]

    chunks = store.iter_chunks()
    assert len(chunks) == 2
    assert chunks[0].content == "page1"
    assert chunks[1].content == "page2"
    assert mock_client.scroll.call_count == 2


def test_iter_chunks_missing_payload_fields(store, mock_client):
    point = MagicMock()
    point.payload = {}
    mock_client.scroll.return_value = ([point], None)

    chunks = store.iter_chunks()
    assert len(chunks) == 1
    assert chunks[0].content == ""
    assert chunks[0].document_id == "unknown"
    assert chunks[0].chunk_index == 0


def test_iter_chunks_none_payload(store, mock_client):
    point = MagicMock()
    point.payload = None
    mock_client.scroll.return_value = ([point], None)

    chunks = store.iter_chunks()
    assert len(chunks) == 1
    assert chunks[0].content == ""


# ---------------------------------------------------------------------------
# search_multi
# ---------------------------------------------------------------------------


def _make_collection(name: str) -> MagicMock:
    c = MagicMock()
    c.name = name
    return c


def _make_point(content: str, doc_id: str, idx: int, score: float) -> MagicMock:
    p = MagicMock()
    p.payload = {"content": content, "document_id": doc_id, "chunk_index": idx, "metadata": {}}
    p.score = score
    return p


def test_search_multi_single_collection(store, mock_client):
    mock_client.get_collections.return_value.collections = [_make_collection("col_a")]
    mock_client.query_points.return_value.points = [
        _make_point("hit1", "d1", 0, 0.9),
    ]

    hits = store.search_multi([0.1, 0.2], collections=["col_a"], top_k=5)
    assert len(hits) == 1
    assert hits[0].chunk.content == "hit1"
    assert hits[0].score == 0.9
    assert hits[0].chunk.metadata["collection"] == "col_a"


def test_search_multi_merges_and_sorts(store, mock_client):
    mock_client.get_collections.return_value.collections = [
        _make_collection("col_a"),
        _make_collection("col_b"),
    ]
    mock_client.query_points.side_effect = [
        MagicMock(points=[_make_point("low", "d1", 0, 0.3)]),
        MagicMock(points=[_make_point("high", "d2", 0, 0.95)]),
    ]

    hits = store.search_multi([0.1], collections=["col_a", "col_b"], top_k=5)
    assert len(hits) == 2
    assert hits[0].chunk.content == "high"
    assert hits[1].chunk.content == "low"


def test_search_multi_truncates_to_top_k(store, mock_client):
    mock_client.get_collections.return_value.collections = [_make_collection("col_a")]
    mock_client.query_points.return_value.points = [
        _make_point(f"hit{i}", "d1", i, 1.0 - i * 0.1)
        for i in range(5)
    ]

    hits = store.search_multi([0.1], collections=["col_a"], top_k=2)
    assert len(hits) == 2


def test_search_multi_skips_unknown_collection(store, mock_client):
    mock_client.get_collections.return_value.collections = [_make_collection("col_a")]
    mock_client.query_points.return_value.points = [
        _make_point("hit1", "d1", 0, 0.9),
    ]

    hits = store.search_multi([0.1], collections=["col_a", "nonexistent"], top_k=5)
    assert len(hits) == 1
    # query_points called once (only for col_a)
    mock_client.query_points.assert_called_once()


def test_search_multi_empty_collections(store, mock_client):
    mock_client.get_collections.return_value.collections = []
    hits = store.search_multi([0.1], collections=["col_a"], top_k=5)
    assert hits == []
    mock_client.query_points.assert_not_called()
