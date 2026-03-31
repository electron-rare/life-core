"""Client for the Cils Document Store."""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger("life_core.docstore")

DOCSTORE_URL = os.environ.get("DOCSTORE_URL", "")


async def search_docstore(query: str, top_k: int = 3) -> list[dict]:
    """Search the Cils document store for relevant context."""
    if not DOCSTORE_URL:
        return []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{DOCSTORE_URL}/search",
                params={"q": query, "top_k": top_k},
            )
            if resp.status_code == 200:
                data = resp.json()
                return data.get("results", [])
    except Exception as e:
        logger.warning(f"Docstore search failed: {e}")

    return []


async def augment_with_docstore(query: str, top_k: int = 3) -> str:
    """Get augmented context from Cils docstore."""
    results = await search_docstore(query, top_k)
    if not results:
        return ""

    context_parts = []
    for r in results:
        doc_name = r.get("document_name", "?")
        content = r.get("content", "")
        score = r.get("score", 0)
        if score > 0.4:  # Only include relevant results
            context_parts.append(f"[{doc_name}] {content}")

    return "\n\n".join(context_parts)
