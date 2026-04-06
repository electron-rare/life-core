"""Tests for cross-encoder reranker."""
import pytest
from unittest.mock import patch, MagicMock
from life_core.rag.pipeline import Chunk, SearchHit
from life_core.rag.reranker import Reranker

@pytest.fixture
def hits():
    return [
        SearchHit(chunk=Chunk(content="STM32 GPIO config", document_id="d1", chunk_index=0), score=0.8, dense_score=0.8, sparse_score=0.0),
        SearchHit(chunk=Chunk(content="KiCad symbols", document_id="d2", chunk_index=0), score=0.7, dense_score=0.7, sparse_score=0.0),
        SearchHit(chunk=Chunk(content="STM32 timer PWM", document_id="d3", chunk_index=0), score=0.6, dense_score=0.6, sparse_score=0.0),
    ]

def test_rerank_reorders_by_cross_encoder_score(hits):
    reranker = Reranker(model_name="mock")
    with patch.object(reranker, "_score_pairs", return_value=[0.1, 0.9, 0.5]):
        result = reranker.rerank("STM32 GPIO", hits, top_k=3)
    assert result[0].chunk.document_id == "d2"
    assert result[1].chunk.document_id == "d3"
    assert result[2].chunk.document_id == "d1"

def test_rerank_truncates_to_top_k(hits):
    reranker = Reranker(model_name="mock")
    with patch.object(reranker, "_score_pairs", return_value=[0.9, 0.8, 0.7]):
        result = reranker.rerank("query", hits, top_k=2)
    assert len(result) == 2

def test_rerank_empty_hits():
    reranker = Reranker(model_name="mock")
    result = reranker.rerank("query", [], top_k=5)
    assert result == []

def test_rerank_updates_score(hits):
    reranker = Reranker(model_name="mock")
    with patch.object(reranker, "_score_pairs", return_value=[0.3, 0.9, 0.6]):
        result = reranker.rerank("query", hits, top_k=3)
    assert result[0].score == 0.9
