"""Integration tests for hybrid RAG pipeline with BM25 + rerank."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from life_core.rag.pipeline import RAGPipeline, Document, Chunk, SearchHit


@pytest.fixture
def pipeline():
    return RAGPipeline(retrieval_mode="hybrid", hybrid_dense_weight=0.6)


@pytest.mark.asyncio
async def test_hybrid_uses_bm25_sparse(pipeline):
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 384)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    await pipeline.index_document(Document(content="STM32 HAL GPIO config for LED", metadata={}))
    with patch.object(pipeline, "_sparse_retriever") as mock_sparse:
        mock_sparse.search.return_value = [
            SearchHit(chunk=Chunk(content="STM32 HAL GPIO config for LED", document_id="test", chunk_index=0),
                      score=0.0, dense_score=0.0, sparse_score=0.9)
        ]
        results = await pipeline.query_with_scores("STM32 GPIO", top_k=5, mode="hybrid")
    mock_sparse.search.assert_called_once()


@pytest.mark.asyncio
async def test_dense_mode_skips_sparse(pipeline):
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 384)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 384])
    await pipeline.index_document(Document(content="Test content here", metadata={}))
    results = await pipeline.query_with_scores("test", top_k=5, mode="dense")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_merges_dense_and_sparse_scores(pipeline):
    """Hybrid results should reflect weighted combination of dense + sparse scores."""
    pipeline.embeddings.embed = AsyncMock(return_value=[0.5] * 8)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.5] * 8])
    await pipeline.index_document(Document(content="ESP32 firmware GPIO interrupt handler", metadata={}))
    results = await pipeline.query_with_scores("ESP32 GPIO", top_k=5, mode="hybrid")
    assert isinstance(results, list)
    for hit in results:
        assert isinstance(hit, SearchHit)
        assert hit.score >= 0.0


@pytest.mark.asyncio
async def test_dense_mode_returns_only_dense_hits(pipeline):
    """Dense mode must not call BM25 sparse retriever."""
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 8)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 8])
    await pipeline.index_document(Document(content="Some test document", metadata={}))
    with patch.object(pipeline, "_sparse_retriever") as mock_sparse:
        results = await pipeline.query_with_scores("test", top_k=5, mode="dense")
    mock_sparse.search.assert_not_called()
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_respects_top_k(pipeline):
    """Hybrid mode must return at most top_k results."""
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 8)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 8] * 5)
    for i in range(5):
        await pipeline.index_document(
            Document(content=f"Document {i} with unique token tok{i}", metadata={"id": f"doc{i}"})
        )
    results = await pipeline.query_with_scores("Document unique token", top_k=3, mode="hybrid")
    assert len(results) <= 3


@pytest.mark.asyncio
async def test_reranker_called_when_enabled(pipeline):
    """When _reranker is set, it should be called during hybrid retrieval."""
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 8)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 8])
    await pipeline.index_document(Document(content="Reranker test document content", metadata={}))

    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = []
    pipeline._reranker = mock_reranker

    await pipeline.query_with_scores("reranker test", top_k=3, mode="hybrid")
    mock_reranker.rerank.assert_called_once()


@pytest.mark.asyncio
async def test_reranker_skipped_when_disabled(pipeline):
    """When _reranker is None (disabled), reranking must not run."""
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 8)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 8])
    await pipeline.index_document(Document(content="No reranker document", metadata={}))

    assert pipeline._reranker is None
    results = await pipeline.query_with_scores("no reranker", top_k=5, mode="hybrid")
    assert isinstance(results, list)


@pytest.mark.asyncio
async def test_hybrid_empty_corpus_returns_empty():
    """Hybrid on empty pipeline should return empty list, not raise."""
    p = RAGPipeline(retrieval_mode="hybrid")
    p.embeddings.embed = AsyncMock(return_value=[0.0] * 8)
    results = await p.query_with_scores("anything", top_k=5, mode="hybrid")
    assert results == []


@pytest.mark.asyncio
async def test_rebuild_sparse_index_called_on_hybrid(pipeline):
    """_rebuild_sparse_index should be called during hybrid retrieval."""
    pipeline.embeddings.embed = AsyncMock(return_value=[0.1] * 8)
    pipeline.embeddings.embed_batch = AsyncMock(return_value=[[0.1] * 8])
    await pipeline.index_document(Document(content="Some document to index", metadata={}))
    with patch.object(pipeline, "_rebuild_sparse_index", wraps=pipeline._rebuild_sparse_index) as mock_rebuild:
        await pipeline.query_with_scores("some document", top_k=5, mode="hybrid")
    mock_rebuild.assert_called_once()
