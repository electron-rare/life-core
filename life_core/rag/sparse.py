"""BM25 sparse retrieval for hybrid RAG pipeline."""
from __future__ import annotations
import re
from rank_bm25 import BM25Plus
from .pipeline import Chunk, SearchHit


class BM25SparseRetriever:
    """BM25-based sparse retriever over a chunk corpus.

    Uses BM25+ (BM25Plus) variant which guarantees non-zero scores for
    documents sharing at least one query token, avoiding the score-floor
    issue of standard BM25Okapi.
    """

    def __init__(self) -> None:
        self._bm25: BM25Plus | None = None
        self._chunks: list[Chunk] = []

    @property
    def corpus_size(self) -> int:
        return len(self._chunks)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 2]

    def build_index(self, chunks: list[Chunk]) -> None:
        self._chunks = list(chunks)
        if not self._chunks:
            self._bm25 = None
            return
        corpus = [self._tokenize(c.content) for c in self._chunks]
        self._bm25 = BM25Plus(corpus)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        if not self._bm25 or not self._chunks:
            return []
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        max_score = max(scores) if max(scores) > 0 else 1.0
        indexed = sorted(
            ((i, s) for i, s in enumerate(scores) if s > 0),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]
        return [
            SearchHit(
                chunk=self._chunks[idx],
                score=0.0,
                dense_score=0.0,
                sparse_score=float(score / max_score),
            )
            for idx, score in indexed
        ]
