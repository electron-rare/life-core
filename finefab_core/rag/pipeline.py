"""Minimal RAG pipeline placeholder for phase 1 bootstrap."""

from __future__ import annotations


class RagPipeline:
    def chunk_text(self, text: str, chunk_size: int = 200) -> list[str]:
        normalized = text.strip()
        if not normalized:
            return []
        return [normalized[index:index + chunk_size] for index in range(0, len(normalized), chunk_size)]

    def prepare_context(self, documents: list[str]) -> str:
        return "\n\n".join(document.strip() for document in documents if document.strip())