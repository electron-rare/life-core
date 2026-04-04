"""RAG Pipeline pour life-core."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("life_core.rag")


def _tokenize(text: str) -> set[str]:
    """Tokenize text for lightweight lexical retrieval."""
    return {token for token in re.findall(r"[a-z0-9]+", text.lower()) if len(token) > 1}


@dataclass
class Document:
    """Document pour le RAG."""

    content: str
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Chunk:
    """Chunk de document."""

    content: str
    document_id: str
    chunk_index: int
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_id(self) -> str:
        """Obtenir un ID unique pour le chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()
        return f"{self.document_id}_{self.chunk_index}_{content_hash}"


@dataclass
class SearchHit:
    """Ranked search result with dense and sparse scores."""

    chunk: Chunk
    score: float
    dense_score: float = 0.0
    sparse_score: float = 0.0


class DocumentChunker:
    """Découpe les documents en chunks."""

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Créer un chunker.
        
        Args:
            chunk_size: Taille des chunks en caractères
            overlap: Chevauchement entre chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Découper un document en chunks.
        
        Args:
            document: Document à découper
            
        Returns:
            Liste de chunks
        """
        content = document.content
        chunks = []
        
        step_size = self.chunk_size - self.overlap
        for i in range(0, len(content), step_size):
            chunk_content = content[i : i + self.chunk_size]
            if len(chunk_content) > 50:  # Ignorer les chunks trop petits
                chunk = Chunk(
                    content=chunk_content,
                    document_id=document.metadata.get("id", "unknown"),
                    chunk_index=len(chunks),
                    metadata=document.metadata,
                )
                chunks.append(chunk)
        
        return chunks


class EmbeddingModel:
    """Modèle d'embeddings simple.

    Priorité de résolution :
    1. Ollama (OLLAMA_URL env var, modèle nomic-embed-text) — léger, pas de dépendance lourde.
    2. sentence-transformers (fallback si Ollama indisponible).
    """

    OLLAMA_EMBED_MODEL = "nomic-embed-text"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Créer un modèle d'embeddings.

        Args:
            model_name: Nom du modèle HuggingFace (utilisé si Ollama indisponible)
        """
        self.model_name = model_name
        self._model = None

    async def _embed_via_ollama(self, texts: list[str]) -> list[list[float]] | None:
        """
        Générer des embeddings via l'API Ollama.

        Args:
            texts: Liste de textes à embedder.

        Returns:
            Liste de vecteurs, ou None si Ollama est indisponible.
        """
        ollama_url = os.environ.get("OLLAMA_URL")
        if not ollama_url:
            return None
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                embeddings: list[list[float]] = []
                for text in texts:
                    resp = await client.post(
                        f"{ollama_url}/api/embed",
                        json={"model": self.OLLAMA_EMBED_MODEL, "input": text},
                    )
                    if resp.status_code != 200:
                        logger.warning(
                            "Ollama embed returned HTTP %s, falling back to sentence-transformers",
                            resp.status_code,
                        )
                        return None
                    data = resp.json()
                    embeddings.append(data["embeddings"][0])
                return embeddings
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ollama embed failed (%s), falling back to sentence-transformers", exc)
            return None

    async def _get_model(self):
        """Charger le modèle sentence-transformers de manière lazy."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package requis. "
                    "Installez avec: pip install sentence-transformers  "
                    "(ou définissez OLLAMA_URL pour utiliser Ollama)"
                )
        return self._model

    async def embed(self, text: str) -> list[float]:
        """
        Générer un embedding pour un texte.

        Args:
            text: Texte à embedder

        Returns:
            Vecteur d'embedding
        """
        results = await self._embed_via_ollama([text])
        if results is not None:
            return results[0]
        model = await self._get_model()
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Générer des embeddings pour plusieurs textes.

        Args:
            texts: Liste de textes

        Returns:
            Liste de vecteurs
        """
        results = await self._embed_via_ollama(texts)
        if results is not None:
            return results
        model = await self._get_model()
        embeddings = model.encode(texts, convert_to_tensor=False)
        return embeddings.tolist()


class VectorStore:
    """Stockage simple des vecteurs (en-mémoire)."""

    def __init__(self):
        """Créer un stockage de vecteurs."""
        self.vectors: dict[str, dict[str, Any]] = {}

    def add(
        self,
        chunk_id: str,
        embedding: list[float],
        chunk: Chunk
    ) -> None:
        """
        Ajouter un vecteur au stockage.
        
        Args:
            chunk_id: ID du chunk
            embedding: Vecteur d'embedding
            chunk: Chunk associé
        """
        self.vectors[chunk_id] = {
            "embedding": embedding,
            "chunk": chunk,
        }

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[Chunk]:
        """Backward-compatible search API."""
        return [hit.chunk for hit in self.search_with_scores(query_embedding, top_k=top_k)]

    def search_with_scores(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[SearchHit]:
        """
        Rechercher les chunks les plus similaires.

        Args:
            query_embedding: Embedding de la requête
            top_k: Nombre de résultats

        Returns:
            Liste de résultats triés par similarité
        """
        results: list[SearchHit] = []
        for data in self.vectors.values():
            embedding = data["embedding"]
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append(
                SearchHit(
                    chunk=data["chunk"],
                    score=similarity,
                    dense_score=similarity,
                )
            )

        results.sort(key=lambda hit: hit.score, reverse=True)

        return results[:top_k]

    def iter_chunks(self) -> list[Chunk]:
        """Iterate over indexed chunks for lexical retrieval."""
        return [data["chunk"] for data in self.vectors.values()]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calcul la similarité cosinus entre deux vecteurs."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class RAGPipeline:
    """Pipeline RAG complet."""

    def __init__(
        self,
        chunk_size: int = 512,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_url: str | None = None,
        retrieval_mode: str | None = None,
        hybrid_dense_weight: float | None = None,
    ):
        """
        Créer un pipeline RAG.

        Args:
            chunk_size: Taille des chunks
            embedding_model: Modèle d'embeddings
            qdrant_url: URL Qdrant optionnelle (si None, utilise le store en mémoire)
        """
        self.chunker = DocumentChunker(chunk_size=chunk_size)
        self.embeddings = EmbeddingModel(model_name=embedding_model)
        if qdrant_url:
            from .qdrant_store import QdrantVectorStore
            self.vector_store = QdrantVectorStore(url=qdrant_url)
        else:
            self.vector_store = VectorStore()
        self.retrieval_mode = (retrieval_mode or os.environ.get("RAG_RETRIEVAL_MODE", "dense")).lower()
        self.hybrid_dense_weight = (
            hybrid_dense_weight
            if hybrid_dense_weight is not None
            else float(os.environ.get("RAG_HYBRID_DENSE_WEIGHT", "0.7"))
        )
        self.hybrid_dense_weight = min(max(self.hybrid_dense_weight, 0.0), 1.0)
        self.stats = {"documents": 0, "chunks": 0}
        self._documents: dict[str, dict] = {}

    async def index_document(self, document: Document) -> None:
        """
        Indexer un document.
        
        Args:
            document: Document à indexer
        """
        # Découper le document
        chunks = self.chunker.chunk(document)
        self.stats["chunks"] += len(chunks)

        # Générer les embeddings
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embeddings.embed_batch(chunk_texts)

        # Ajouter au stockage
        chunk_ids = []
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get_id()
            self.vector_store.add(chunk_id, embedding, chunk)
            chunk_ids.append(chunk_id)

        doc_id = document.metadata.get("id", "unknown")
        self._documents[doc_id] = {
            "id": doc_id,
            "name": document.metadata.get("name", "unnamed"),
            "chunks": len(chunks),
            "chunk_ids": chunk_ids,
            "metadata": document.metadata,
        }

        self.stats["documents"] += 1
        logger.info(f"Indexed document with {len(chunks)} chunks")

    def _dense_candidate_count(self, top_k: int) -> int:
        return max(top_k * 3, 10)

    @staticmethod
    def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        max_score = max(scores.values()) or 1.0
        return {key: value / max_score for key, value in scores.items()}

    def _lexical_hits(self, query_text: str, top_k: int) -> list[SearchHit]:
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        hits: list[SearchHit] = []
        for chunk in self.vector_store.iter_chunks():
            chunk_tokens = _tokenize(chunk.content)
            if not chunk_tokens:
                continue
            overlap = len(query_tokens & chunk_tokens)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(query_tokens) * len(chunk_tokens))
            hits.append(SearchHit(chunk=chunk, score=score, sparse_score=score))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def _merge_hybrid_hits(
        self,
        *,
        dense_hits: list[SearchHit],
        sparse_hits: list[SearchHit],
        top_k: int,
    ) -> list[SearchHit]:
        dense_scores = self._normalize_scores({hit.chunk.get_id(): hit.score for hit in dense_hits})
        sparse_scores = self._normalize_scores({hit.chunk.get_id(): hit.score for hit in sparse_hits})
        dense_weight = self.hybrid_dense_weight
        sparse_weight = 1.0 - dense_weight

        chunk_by_id: dict[str, Chunk] = {}
        for hit in dense_hits + sparse_hits:
            chunk_by_id.setdefault(hit.chunk.get_id(), hit.chunk)

        merged: list[SearchHit] = []
        for chunk_id, chunk in chunk_by_id.items():
            dense_score = dense_scores.get(chunk_id, 0.0)
            sparse_score = sparse_scores.get(chunk_id, 0.0)
            merged.append(
                SearchHit(
                    chunk=chunk,
                    score=(dense_score * dense_weight) + (sparse_score * sparse_weight),
                    dense_score=dense_score,
                    sparse_score=sparse_score,
                )
            )

        merged.sort(key=lambda hit: hit.score, reverse=True)
        return merged[:top_k]

    async def query_with_scores(self, query_text: str, top_k: int = 5) -> list[SearchHit]:
        """Interroger le RAG avec scores détaillés."""
        query_embedding = await self.embeddings.embed(query_text)
        dense_hits = self.vector_store.search_with_scores(
            query_embedding,
            top_k=self._dense_candidate_count(top_k),
        )
        if self.retrieval_mode != "hybrid":
            return dense_hits[:top_k]

        sparse_hits = self._lexical_hits(query_text, top_k=self._dense_candidate_count(top_k))
        return self._merge_hybrid_hits(dense_hits=dense_hits, sparse_hits=sparse_hits, top_k=top_k)

    async def query(self, query_text: str, top_k: int = 5) -> list[Chunk]:
        """
        Interroger le RAG.
        
        Args:
            query_text: Texte de la requête
            top_k: Nombre de résultats
            
        Returns:
            Chunks les plus pertinents
        """
        return [hit.chunk for hit in await self.query_with_scores(query_text, top_k=top_k)]

    async def augment_context(self, query_text: str, top_k: int = 5) -> str:
        """
        Augmenter le contexte pour une requête.

        Args:
            query_text: Texte de la requête
            top_k: Nombre de chunks

        Returns:
            Contexte augmenté
        """
        from life_core.telemetry import get_tracer
        tracer = get_tracer()

        with tracer.start_as_current_span("rag.embed") as span:
            query_embedding = await self.embeddings.embed(query_text)
            span.set_attribute("rag.embedding_dim", len(query_embedding))

        with tracer.start_as_current_span("rag.search") as span:
            dense_hits = self.vector_store.search_with_scores(
                query_embedding,
                top_k=self._dense_candidate_count(top_k),
            )
            if self.retrieval_mode == "hybrid":
                sparse_hits = self._lexical_hits(query_text, top_k=self._dense_candidate_count(top_k))
                hits = self._merge_hybrid_hits(
                    dense_hits=dense_hits,
                    sparse_hits=sparse_hits,
                    top_k=top_k,
                )
                span.set_attribute("rag.sparse_results_count", len(sparse_hits))
            else:
                hits = dense_hits[:top_k]

            span.set_attribute("rag.search_mode", self.retrieval_mode)
            span.set_attribute("rag.results_count", len(hits))
            span.set_attribute("rag.dense_results_count", len(dense_hits))

        if not hits:
            return ""

        context = "\n\n".join([hit.chunk.content for hit in hits])
        return context

    def list_documents(self) -> list[dict]:
        """Lister les documents indexés.

        Returns:
            Liste des métadonnées de documents
        """
        return list(self._documents.values())

    async def delete_document(self, doc_id: str) -> bool:
        """Supprimer un document et ses chunks du vector store.

        Args:
            doc_id: Identifiant du document

        Returns:
            True si supprimé, False si introuvable
        """
        if doc_id not in self._documents:
            return False

        doc = self._documents.pop(doc_id)
        chunk_ids = doc.get("chunk_ids", [])

        # Remove chunks from the in-memory vector store
        if hasattr(self.vector_store, "vectors"):
            for chunk_id in chunk_ids:
                self.vector_store.vectors.pop(chunk_id, None)

        removed_chunks = len(chunk_ids)
        self.stats["documents"] = max(0, self.stats["documents"] - 1)
        self.stats["chunks"] = max(0, self.stats["chunks"] - removed_chunks)
        logger.info("Deleted document %s (%d chunks removed)", doc_id, removed_chunks)
        return True

    def get_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques du RAG."""
        return {
            **self.stats,
            "vectors": len(getattr(self.vector_store, "vectors", {})),
            "retrieval_mode": self.retrieval_mode,
        }
