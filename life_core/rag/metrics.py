"""RAG pipeline metrics for OTEL and Langfuse."""
from __future__ import annotations

import logging

from life_core.langfuse_tracing import score_trace

logger = logging.getLogger("life_core.rag.metrics")


class RAGMetrics:
    """Record RAG retrieval metrics to OTEL and Langfuse."""

    def __init__(self) -> None:
        from life_core.telemetry import create_rag_instruments

        self._instruments = create_rag_instruments()

    def record_retrieval(
        self,
        query: str,
        mode: str,
        n_results: int,
        latency_ms: float,
        top_score: float,
        trace_id: str | None = None,
    ) -> None:
        """Record retrieval metrics and optionally score a Langfuse trace."""
        attrs = {"rag.mode": mode}
        self._instruments["retrieval_latency"].record(latency_ms, attrs)
        self._instruments["retrieval_count"].add(1, attrs)
        self._instruments["retrieval_top_score"].record(top_score, attrs)
        self._instruments["retrieval_results"].record(n_results, attrs)

        if trace_id:
            try:
                score_trace(trace_id, "rag_relevance", top_score, f"mode={mode}, n={n_results}")
            except Exception as e:
                logger.debug("Langfuse scoring error: %s", e)
