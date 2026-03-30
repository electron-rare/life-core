"""RAG Pipeline pour life-core."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("life_core.rag")


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
    """Modèle d'embeddings simple."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Créer un modèle d'embeddings.
        
        Args:
            model_name: Nom du modèle HuggingFace
        """
        self.model_name = model_name
        self._model = None

    async def _get_model(self):
        """Charger le modèle de manière lazy."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers package requis. "
                    "Installez avec: pip install sentence-transformers"
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
        """
        Rechercher les chunks les plus similaires.
        
        Args:
            query_embedding: Embedding de la requête
            top_k: Nombre de résultats
            
        Returns:
            Liste de chunks triés par similarité
        """
        import math

        # Calcul de la similarité cosinus
        results = []
        for chunk_id, data in self.vectors.items():
            embedding = data["embedding"]
            similarity = self._cosine_similarity(query_embedding, embedding)
            results.append((chunk_id, similarity, data["chunk"]))

        # Trier par similarité décroissante
        results.sort(key=lambda x: x[1], reverse=True)

        return [chunk for _, _, chunk in results[:top_k]]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calcul la similarité cosinus entre deux vecteurs."""
        import math

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
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Créer un pipeline RAG.
        
        Args:
            chunk_size: Taille des chunks
            embedding_model: Modèle d'embeddings
        """
        self.chunker = DocumentChunker(chunk_size=chunk_size)
        self.embeddings = EmbeddingModel(model_name=embedding_model)
        self.vector_store = VectorStore()
        self.stats = {"documents": 0, "chunks": 0}

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
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = chunk.get_id()
            self.vector_store.add(chunk_id, embedding, chunk)

        self.stats["documents"] += 1
        logger.info(f"Indexed document with {len(chunks)} chunks")

    async def query(self, query_text: str, top_k: int = 5) -> list[Chunk]:
        """
        Interroger le RAG.
        
        Args:
            query_text: Texte de la requête
            top_k: Nombre de résultats
            
        Returns:
            Chunks les plus pertinents
        """
        # Générer l'embedding de la requête
        query_embedding = await self.embeddings.embed(query_text)

        # Rechercher les chunks similaires
        results = self.vector_store.search(query_embedding, top_k=top_k)

        return results

    async def augment_context(self, query_text: str, top_k: int = 5) -> str:
        """
        Augmenter le contexte pour une requête.
        
        Args:
            query_text: Texte de la requête
            top_k: Nombre de chunks
            
        Returns:
            Contexte augmenté
        """
        chunks = await self.query(query_text, top_k=top_k)

        if not chunks:
            return ""

        context = "\n\n".join([chunk.content for chunk in chunks])
        return context

    def get_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques du RAG."""
        return {
            **self.stats,
            "vectors": len(self.vector_store.vectors),
        }
