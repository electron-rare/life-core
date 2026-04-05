"""VectorStore backed par Qdrant."""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient, models

from .pipeline import Chunk, SearchHit

logger = logging.getLogger("life_core.rag.qdrant")

# nomic-embed-text output dimension (768).
# MiniLM-L6-v2 uses 384 — update if you switch embedding models.
EMBEDDING_DIM = 768


class QdrantVectorStore:
    """Stockage de vecteurs via Qdrant."""

    def __init__(self, url: str = "http://localhost:6333", collection_name: str = "life_chunks"):
        self.url = url
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        """Creer la collection si elle n'existe pas."""
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection_name}")

    @staticmethod
    def _to_qdrant_id(chunk_id: str) -> str:
        """Convert an arbitrary string chunk ID to a valid Qdrant UUID.

        Qdrant only accepts unsigned integers or UUIDs as point IDs.
        We derive a deterministic UUID-v5 from the chunk_id so that
        re-indexing the same chunk always produces the same point ID
        (enabling idempotent upserts).
        """
        return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))

    def add(self, chunk_id: str, embedding: list[float], chunk: Chunk) -> None:
        """Ajouter un vecteur."""
        point_id = self._to_qdrant_id(chunk_id)
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "content": chunk.content,
                        "document_id": chunk.document_id,
                        "chunk_index": chunk.chunk_index,
                        "metadata": chunk.metadata,
                        "chunk_id": chunk_id,
                    },
                )
            ],
        )

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Backward-compatible search API."""
        return [hit.chunk for hit in self.search_with_scores(query_embedding, top_k=top_k)]

    def search_with_scores(self, query_embedding: list[float], top_k: int = 5) -> list[SearchHit]:
        """Rechercher les chunks les plus similaires avec leurs scores."""
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
        )
        hits: list[SearchHit] = []
        for point in results.points:
            payload = point.payload
            metadata = dict(payload.get("metadata", {}))
            metadata.setdefault("collection", self.collection_name)
            chunk = Chunk(
                content=payload["content"],
                document_id=payload["document_id"],
                chunk_index=payload["chunk_index"],
                metadata=metadata,
            )
            hits.append(
                SearchHit(
                    chunk=chunk,
                    score=float(point.score or 0.0),
                    dense_score=float(point.score or 0.0),
                )
            )
        return hits

    def iter_chunks(self) -> list[Chunk]:
        """Scroll all chunks for lightweight lexical retrieval."""
        chunks: list[Chunk] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in records:
                payload = point.payload or {}
                metadata = dict(payload.get("metadata", {}))
                metadata.setdefault("collection", self.collection_name)
                chunks.append(
                    Chunk(
                        content=payload.get("content", ""),
                        document_id=payload.get("document_id", "unknown"),
                        chunk_index=payload.get("chunk_index", 0),
                        metadata=metadata,
                    )
                )
            if offset is None:
                break
        return chunks

    def search_multi(
        self,
        query_embedding: list[float],
        collections: list[str],
        top_k: int = 5,
    ) -> list[SearchHit]:
        """Search across multiple Qdrant collections and merge top hits."""
        available = {collection.name for collection in self.client.get_collections().collections}
        merged: list[SearchHit] = []
        for collection_name in collections:
            if collection_name not in available:
                logger.warning("Skipping unknown Qdrant collection: %s", collection_name)
                continue
            results = self.client.query_points(
                collection_name=collection_name,
                query=query_embedding,
                limit=top_k,
            )
            for point in results.points:
                payload = point.payload
                metadata = dict(payload.get("metadata", {}))
                metadata.setdefault("collection", collection_name)
                merged.append(
                    SearchHit(
                        chunk=Chunk(
                            content=payload["content"],
                            document_id=payload["document_id"],
                            chunk_index=payload["chunk_index"],
                            metadata=metadata,
                        ),
                        score=float(point.score or 0.0),
                        dense_score=float(point.score or 0.0),
                    )
                )
        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[:top_k]
