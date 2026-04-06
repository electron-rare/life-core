"""Tests for RAG metrics instrumentation."""
import pytest
from unittest.mock import patch, MagicMock
from life_core.rag.metrics import RAGMetrics


def test_rag_metrics_init():
    metrics = RAGMetrics()
    assert metrics is not None


def test_record_retrieval():
    metrics = RAGMetrics()
    metrics.record_retrieval(query="test query", mode="hybrid", n_results=5, latency_ms=42.3, top_score=0.87)


def test_record_retrieval_to_langfuse():
    mock_score = MagicMock()
    with patch("life_core.rag.metrics.score_trace", mock_score):
        metrics = RAGMetrics()
        metrics.record_retrieval(query="test", mode="dense", n_results=3, latency_ms=50.0, top_score=0.92, trace_id="trace-123")
    mock_score.assert_called_once()
