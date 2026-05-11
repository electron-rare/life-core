"""RAG API endpoints for document management and search."""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

from .pipeline import Document, RAGPipeline

logger = logging.getLogger("life_core.rag.api")

rag_router = APIRouter(prefix="/rag", tags=["RAG"])

# Will be set during app startup
_rag: RAGPipeline | None = None


def set_rag_pipeline(rag: RAGPipeline | None) -> None:
    global _rag
    _rag = rag


def _get_rag() -> RAGPipeline:
    if _rag is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    return _rag


# Models
class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    metadata: dict[str, Any] = {}


class SearchResult(BaseModel):
    content: str
    document_id: str
    chunk_index: int
    score: float = 0.0
    dense_score: float = 0.0
    sparse_score: float = 0.0


class RagStats(BaseModel):
    documents: int
    chunks: int
    vectors: int
    collections: dict[str, int] = {}


# Endpoints
@rag_router.get("/stats", response_model=RagStats)
async def rag_stats():
    """Return aggregated RAG stats across all Qdrant collections.

    Combines:
    - In-memory pipeline.stats (docs/chunks indexed via the API this
      session)
    - Qdrant points_count for every collection the vector store can
      reach (covers documents indexed by nc-rag-indexer running
      out-of-band against the same Qdrant instance).
    """
    rag = _get_rag()
    pipeline_stats = rag.get_stats()
    collections: dict[str, int] = {}
    qdrant_vectors_total = 0
    try:
        client = getattr(getattr(rag, "vector_store", None), "client", None)
        if client is not None:
            for c in client.get_collections().collections:
                try:
                    info = client.get_collection(c.name)
                    pts = int(getattr(info, "points_count", 0) or 0)
                except Exception:
                    pts = 0
                collections[c.name] = pts
                qdrant_vectors_total += pts
    except Exception as exc:
        logger.warning("rag_stats: Qdrant collection enumeration failed: %s", exc)

    return RagStats(
        documents=pipeline_stats.get("documents", 0),
        chunks=pipeline_stats.get("chunks", 0),
        vectors=qdrant_vectors_total or pipeline_stats.get("vectors", 0),
        collections=collections,
    )


@rag_router.post("/documents", response_model=DocumentInfo)
async def index_document(file: UploadFile = File(...)):
    rag = _get_rag()

    content = await file.read()
    text = content.decode("utf-8", errors="replace")
    doc_id = hashlib.md5(text[:1000].encode()).hexdigest()[:12]

    doc = Document(
        content=text,
        metadata={"id": doc_id, "name": file.filename or "unnamed", "source": "upload"},
    )
    await rag.index_document(doc)

    stats = rag.get_stats()
    return DocumentInfo(
        id=doc_id,
        name=file.filename or "unnamed",
        chunks=stats.get("chunks", 0),
        metadata=doc.metadata,
    )


@rag_router.get("/documents")
async def list_documents():
    rag = _get_rag()
    documents = rag.list_documents()
    return {"documents": documents}


@rag_router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    rag = _get_rag()
    deleted = await rag.delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found")
    return {"deleted": True, "id": doc_id}


@rag_router.get("/search")
async def search_documents(
    q: str,
    top_k: int = 5,
    mode: str | None = None,
    collections: str | None = None,
):
    rag = _get_rag()

    collection_list = [item.strip() for item in collections.split(",") if item.strip()] if collections else None

    if collection_list and getattr(rag, "vector_store", None) and hasattr(rag.vector_store, "search_multi"):
        query_embedding = await rag.embeddings.embed(q)
        hits = rag.vector_store.search_multi(
            query_embedding=query_embedding,
            collections=collection_list,
            top_k=top_k,
        )
    else:
        try:
            hits = await rag.query_with_scores(q, top_k=top_k, mode=mode)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    effective_mode = mode.lower() if mode else getattr(rag, "retrieval_mode", "dense")
    return {
        "query": q,
        "mode": effective_mode,
        "collections": collection_list or ["life_chunks"],
        "results": [
            {
                "content": hit.chunk.content,
                "document_id": hit.chunk.document_id,
                "chunk_index": hit.chunk.chunk_index,
                "metadata": hit.chunk.metadata,
                "score": hit.score,
                "dense_score": hit.dense_score,
                "sparse_score": hit.sparse_score,
            }
            for hit in hits
        ],
    }
