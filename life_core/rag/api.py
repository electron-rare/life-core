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


class RagStats(BaseModel):
    documents: int
    chunks: int
    vectors: int


# Endpoints
@rag_router.get("/stats", response_model=RagStats)
async def rag_stats():
    rag = _get_rag()
    stats = rag.get_stats()
    return RagStats(
        documents=stats.get("documents", 0),
        chunks=stats.get("chunks", 0),
        vectors=stats.get("vectors", 0),
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
async def search_documents(q: str, top_k: int = 5):
    rag = _get_rag()
    chunks = await rag.query(q, top_k=top_k)
    return {
        "query": q,
        "results": [
            {
                "content": chunk.content,
                "document_id": chunk.document_id,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ],
    }
