"""Cross-encoder reranker for RAG pipeline."""
from __future__ import annotations
import logging
import os
from .pipeline import SearchHit

logger = logging.getLogger("life_core.rag.reranker")
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    """Rerank search hits using a cross-encoder model."""
    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("RERANK_MODEL", DEFAULT_MODEL)
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
            logger.info(f"Loaded reranker: {self.model_name}")

    def _score_pairs(self, query: str, texts: list[str]) -> list[float]:
        self._ensure_model()
        pairs = [(query, text) for text in texts]
        scores = self._model.predict(pairs)
        return [float(s) for s in scores]

    def rerank(self, query: str, hits: list[SearchHit], top_k: int = 5) -> list[SearchHit]:
        if not hits:
            return []
        texts = [hit.chunk.content for hit in hits]
        scores = self._score_pairs(query, texts)
        scored_hits = []
        for hit, score in zip(hits, scores):
            scored_hits.append(SearchHit(chunk=hit.chunk, score=score, dense_score=hit.dense_score, sparse_score=hit.sparse_score))
        scored_hits.sort(key=lambda h: h.score, reverse=True)
        return scored_hits[:top_k]
