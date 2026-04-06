"""Tests for BM25 sparse retrieval."""
import pytest
from life_core.rag.pipeline import Chunk
from life_core.rag.sparse import BM25SparseRetriever

@pytest.fixture
def chunks():
    return [
        Chunk(content="STM32 HAL GPIO configuration for LED blinking", document_id="d1", chunk_index=0),
        Chunk(content="KiCad schematic symbol library management", document_id="d2", chunk_index=0),
        Chunk(content="ESP32 WiFi connection with WPA2 enterprise", document_id="d3", chunk_index=0),
        Chunk(content="STM32 timer interrupt configuration for PWM", document_id="d4", chunk_index=0),
    ]

def test_build_index(chunks):
    retriever = BM25SparseRetriever()
    retriever.build_index(chunks)
    assert retriever.corpus_size == 4

def test_search_returns_relevant(chunks):
    retriever = BM25SparseRetriever()
    retriever.build_index(chunks)
    results = retriever.search("STM32 GPIO", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.document_id == "d1"

def test_search_with_scores(chunks):
    retriever = BM25SparseRetriever()
    retriever.build_index(chunks)
    results = retriever.search("STM32 GPIO", top_k=2)
    assert all(hit.sparse_score > 0 for hit in results)
    assert results[0].sparse_score >= results[1].sparse_score

def test_search_empty_index():
    retriever = BM25SparseRetriever()
    results = retriever.search("anything", top_k=5)
    assert results == []

def test_search_no_match(chunks):
    retriever = BM25SparseRetriever()
    retriever.build_index(chunks)
    results = retriever.search("quantum computing blockchain", top_k=2)
    assert all(hit.sparse_score < 0.1 for hit in results)

def test_tokenize():
    retriever = BM25SparseRetriever()
    tokens = retriever._tokenize("STM32 HAL GPIO configuration")
    assert "stm32" in tokens
    assert "hal" in tokens
    assert "gpio" in tokens
