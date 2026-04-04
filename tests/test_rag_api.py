"""Tests for RAG API endpoints."""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from fastapi.testclient import TestClient

from life_core.rag.api import rag_router, set_rag_pipeline, _get_rag
from life_core.rag.pipeline import RAGPipeline, Chunk


@pytest.fixture
def mock_rag():
    rag = MagicMock(spec=RAGPipeline)
    rag.get_stats.return_value = {"documents": 2, "chunks": 10, "vectors": 10}
    rag.query = AsyncMock(return_value=[
        Chunk(content="test result", document_id="doc1", chunk_index=0),
    ])
    rag.index_document = AsyncMock()
    rag.list_documents = MagicMock(return_value=[
        {"id": "doc1", "name": "file1.txt", "chunks": 5, "metadata": {}},
        {"id": "doc2", "name": "file2.txt", "chunks": 5, "metadata": {}},
    ])
    rag.delete_document = AsyncMock(return_value=True)
    set_rag_pipeline(rag)
    yield rag
    set_rag_pipeline(None)


def test_rag_stats(mock_rag):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/stats")
    assert response.status_code == 200
    data = response.json()
    assert data["documents"] == 2
    assert data["chunks"] == 10


def test_rag_search(mock_rag):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/search?q=test+query&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test query"
    assert len(data["results"]) == 1
    assert data["results"][0]["content"] == "test result"


def test_rag_stats_no_pipeline():
    from fastapi import FastAPI
    set_rag_pipeline(None)
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/stats")
    assert response.status_code == 503


def test_rag_list_documents(mock_rag):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.get("/rag/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert len(data["documents"]) == 2
    assert data["documents"][0]["id"] == "doc1"
    assert data["documents"][1]["name"] == "file2.txt"
    mock_rag.list_documents.assert_called_once()


def test_rag_delete_document(mock_rag):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.delete("/rag/documents/abc123")
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] is True
    assert data["id"] == "abc123"
    mock_rag.delete_document.assert_called_once_with("abc123")


def test_rag_delete_document_not_found(mock_rag):
    from fastapi import FastAPI
    mock_rag.delete_document = AsyncMock(return_value=False)
    app = FastAPI()
    app.include_router(rag_router)
    client = TestClient(app)

    response = client.delete("/rag/documents/nonexistent")
    assert response.status_code == 404
