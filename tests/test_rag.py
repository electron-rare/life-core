"""Tests pour le pipeline RAG."""

import sys
import types

import pytest

from life_core.rag import Chunk, Document, DocumentChunker, EmbeddingModel, RAGPipeline


def test_document_creation():
    """Test la création d'un document."""
    doc = Document(content="Test content", metadata={"id": "doc1"})
    assert doc.content == "Test content"
    assert doc.metadata["id"] == "doc1"


def test_document_chunker_creation():
    """Test la création du chunker."""
    chunker = DocumentChunker(chunk_size=256, overlap=32)
    assert chunker.chunk_size == 256
    assert chunker.overlap == 32


def test_document_chunker_chunk():
    """Test le découpage d'un document."""
    chunker = DocumentChunker(chunk_size=100, overlap=10)
    content = "This is a test document. " * 10
    doc = Document(content=content, metadata={"id": "test"})

    chunks = chunker.chunk(doc)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)


def test_chunk_get_id():
    """Test la génération d'ID unique pour un chunk."""
    chunk1 = Chunk(content="test", document_id="doc1", chunk_index=0)
    chunk2 = Chunk(content="test", document_id="doc1", chunk_index=0)

    # Même contenu doit donner même ID
    assert chunk1.get_id() == chunk2.get_id()

    # Contenu différent doit donner ID différent
    chunk3 = Chunk(content="different", document_id="doc1", chunk_index=0)
    assert chunk1.get_id() != chunk3.get_id()


def test_rag_pipeline_creation():
    """Test la création du pipeline RAG."""
    pipeline = RAGPipeline()
    assert pipeline.chunker is not None
    assert pipeline.embeddings is not None
    assert pipeline.vector_store is not None


@pytest.mark.asyncio
async def test_rag_pipeline_index_document():
    """Test l'indexation d'un document."""
    pipeline = RAGPipeline(chunk_size=100)
    doc = Document(
        content="This is a test document. " * 5,
        metadata={"id": "test_doc"}
    )

    await pipeline.index_document(doc)

    stats = pipeline.get_stats()
    assert stats["documents"] == 1
    assert stats["chunks"] > 0


@pytest.mark.asyncio
async def test_rag_pipeline_query():
    """Test l'interrogation du RAG."""
    pipeline = RAGPipeline(chunk_size=100)
    doc = Document(
        content="The quick brown fox jumps. " * 5,
        metadata={"id": "test"}
    )

    await pipeline.index_document(doc)
    results = await pipeline.query("fox jumps", top_k=3)

    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_rag_pipeline_hybrid_prefers_lexical_overlap():
    """Hybrid retrieval should surface chunks matching lexical intent."""
    pipeline = RAGPipeline(chunk_size=160, retrieval_mode="hybrid", hybrid_dense_weight=0.4)
    doc = Document(
        content=(
            "Factory 4 Life uses Qdrant and hybrid retrieval for industrial search. "
            "The rerank stage promotes exact lexical matches for critical queries. "
        ) * 3,
        metadata={"id": "hybrid_doc"},
    )

    await pipeline.index_document(doc)
    hits = await pipeline.query_with_scores("hybrid retrieval rerank", top_k=2)

    assert hits
    assert hits[0].sparse_score >= 0
    assert "hybrid retrieval" in hits[0].chunk.content.lower()


@pytest.mark.asyncio
async def test_rag_pipeline_augment_context():
    """Test l'augmentation de contexte."""
    pipeline = RAGPipeline(chunk_size=100)
    doc = Document(
        content="Machine learning is important. " * 3,
        metadata={"id": "ml_doc"}
    )

    await pipeline.index_document(doc)
    context = await pipeline.augment_context("machine learning", top_k=2)

    assert isinstance(context, str)


@pytest.mark.asyncio
async def test_rag_pipeline_empty_query():
    """Test une requête sur un RAG vide."""
    pipeline = RAGPipeline()
    results = await pipeline.query("test", top_k=5)
    assert results == []


@pytest.mark.asyncio
async def test_embedding_model_uses_tei(monkeypatch):
    """EMBED_URL should be used for TEI embedding requests."""
    seen: dict[str, object] = {}

    class FakeResponse:
        status_code = 200

        def json(self):
            return [[0.1, 0.2, 0.3]]

    class FakeAsyncClient:
        def __init__(self, timeout: float):
            seen["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, json: dict):
            seen["url"] = url
            seen["json"] = json
            return FakeResponse()

    monkeypatch.setenv("EMBED_URL", "http://embed.local:11437")
    monkeypatch.setitem(sys.modules, "httpx", types.SimpleNamespace(AsyncClient=FakeAsyncClient))

    model = EmbeddingModel()
    embeddings = await model._embed_via_tei(["Factory 4 Life hybrid retrieval"])

    assert embeddings == [[0.1, 0.2, 0.3]]
    assert seen["url"] == "http://embed.local:11437/embed"
    assert seen["json"] == {
        "inputs": ["Factory 4 Life hybrid retrieval"],
    }


def test_vector_store_add_retrieve():
    """Test l'ajout et la récupération dans le stockage de vecteurs."""
    from life_core.rag import VectorStore

    store = VectorStore()
    chunk = Chunk(content="test", document_id="doc1", chunk_index=0)
    embedding = [0.1, 0.2, 0.3]

    store.add("chunk1", embedding, chunk)
    assert "chunk1" in store.vectors
    assert store.vectors["chunk1"]["chunk"] == chunk


def test_cosine_similarity():
    """Test le calcul de similarité cosinus."""
    from life_core.rag import VectorStore

    v1 = [1, 0, 0]
    v2 = [1, 0, 0]
    v3 = [0, 1, 0]

    sim_same = VectorStore._cosine_similarity(v1, v2)
    sim_diff = VectorStore._cosine_similarity(v1, v3)

    assert sim_same > sim_diff  # Même vecteur plus similaire
    assert sim_same == 1.0  # Vecteurs identiques = similarité 1
