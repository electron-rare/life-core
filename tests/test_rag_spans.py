"""Tests for OTEL spans on RAG pipeline operations."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from life_core.rag.pipeline import RAGPipeline, EmbeddingModel, VectorStore, Chunk


def _make_test_tracer():
    exporter = InMemorySpanExporter()
    tp = SdkTracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("test")
    return tracer, exporter


@pytest.mark.asyncio
async def test_augment_context_emits_embed_and_search_spans():
    tracer, exporter = _make_test_tracer()

    mock_embedding = MagicMock(spec=EmbeddingModel)
    mock_embedding.embed = AsyncMock(return_value=[0.1] * 768)

    mock_chunk = Chunk(content="relevant doc", document_id="d1", chunk_index=0, metadata={})
    mock_store = MagicMock(spec=VectorStore)
    mock_store.search = MagicMock(return_value=[mock_chunk])

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embeddings = mock_embedding
    pipeline.vector_store = mock_store
    pipeline.stats = {"queries": 0, "documents": 0}

    with patch("life_core.telemetry.get_tracer", return_value=tracer):
        result = await pipeline.augment_context("What is X?", top_k=3)

    assert "relevant doc" in result

    spans = exporter.get_finished_spans()
    span_names = [s.name for s in spans]
    assert "rag.embed" in span_names
    assert "rag.search" in span_names

    search_span = next(s for s in spans if s.name == "rag.search")
    assert search_span.attributes["rag.results_count"] == 1

    embed_span = next(s for s in spans if s.name == "rag.embed")
    assert embed_span.attributes["rag.embedding_dim"] == 768


@pytest.mark.asyncio
async def test_augment_context_empty_results():
    tracer, exporter = _make_test_tracer()

    mock_embedding = MagicMock(spec=EmbeddingModel)
    mock_embedding.embed = AsyncMock(return_value=[0.0] * 768)

    mock_store = MagicMock(spec=VectorStore)
    mock_store.search = MagicMock(return_value=[])

    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline.embeddings = mock_embedding
    pipeline.vector_store = mock_store
    pipeline.stats = {"queries": 0, "documents": 0}

    with patch("life_core.telemetry.get_tracer", return_value=tracer):
        result = await pipeline.augment_context("Unknown query")

    assert result == ""

    spans = exporter.get_finished_spans()
    search_span = next(s for s in spans if s.name == "rag.search")
    assert search_span.attributes["rag.results_count"] == 0
