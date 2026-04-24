"""Langfuse LLM tracing integration."""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger("life_core.langfuse")

_langfuse = None


def init_langfuse() -> None:
    """Initialize Langfuse if credentials are set."""
    global _langfuse

    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    host = os.environ.get("LANGFUSE_HOST")

    if not all([public_key, secret_key, host]):
        logger.info("Langfuse disabled (LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST not set)")
        return

    try:
        from langfuse import Langfuse
        _langfuse = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info(f"Langfuse initialized, host={host}")
    except ImportError:
        logger.warning("langfuse package not installed")
    except Exception as e:
        logger.warning(f"Langfuse init failed: {e}")


def trace_llm_call(
    provider: str,
    model: str,
    messages: list[dict[str, str]],
    response: dict[str, Any],
    duration_ms: float,
) -> None:
    """Record an LLM call in Langfuse."""
    if not _langfuse:
        return

    try:
        trace = _langfuse.trace(
            name=f"chat/{provider}",
            metadata={"provider": provider, "model": model},
        )
        trace.generation(
            name=f"{provider}/{model}",
            model=model,
            input=messages,
            output=response.get("content", ""),
            usage={
                "input": response.get("usage", {}).get("prompt_tokens", 0),
                "output": response.get("usage", {}).get("completion_tokens", 0),
            },
            metadata={
                "provider": provider,
                "duration_ms": duration_ms,
            },
        )
    except Exception as e:
        logger.warning(f"Langfuse trace failed: {e}")


def get_langfuse_prompt(name: str, version: int | None = None):
    """Fetch a prompt from Langfuse. Returns None if Langfuse unavailable."""
    if _langfuse is None:
        return None
    try:
        return _langfuse.get_prompt(name, version=version)
    except Exception as e:
        logger.warning("Failed to fetch Langfuse prompt '%s': %s", name, e)
        return None


def score_trace(trace_id: str, name: str, value: float, comment: str | None = None):
    """Send a score to Langfuse for a given trace."""
    if _langfuse is None:
        logger.warning("Langfuse not initialized — score discarded")
        return
    try:
        _langfuse.score(trace_id=trace_id, name=name, value=value, comment=comment)
    except Exception as e:
        logger.error("Failed to score trace %s: %s", trace_id, e)


def trace_rag_query(query: str, mode: str, n_results: int, latency_ms: float, top_score: float) -> None:
    """Trace a RAG query in Langfuse."""
    if _langfuse is None:
        return
    try:
        trace = _langfuse.trace(name="rag-query", input={"query": query, "mode": mode})
        trace.span(
            name="retrieval",
            input={"query": query, "mode": mode},
            output={"n_results": n_results, "top_score": top_score},
            metadata={"latency_ms": latency_ms},
        )
    except Exception as e:
        logger.debug("Langfuse RAG trace error: %s", e)


def flush_langfuse() -> None:
    """Flush pending Langfuse events."""
    if _langfuse:
        try:
            _langfuse.flush()
        except Exception:
            pass


def forward_generation_run(
    generation_run_id: str,
    agent_run_id: str,
    deliverable_slug: str,
    llm_model: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    user_id: str | None = None,
) -> None:
    """Forward a generation_run row into Langfuse as a trace+generation pair."""
    if _langfuse is None:
        return
    try:
        trace = _langfuse.trace(
            id=agent_run_id,
            name=f"inner_trace/{deliverable_slug}",
            user_id=user_id,
            metadata={
                "deliverable_slug": deliverable_slug,
                "generation_run_id": generation_run_id,
            },
        )
        trace.generation(
            id=generation_run_id,
            name=llm_model,
            model=llm_model,
            usage={"input": tokens_in, "output": tokens_out},
            metadata={
                "cost_usd": cost_usd,
                "user_id": user_id,
            },
        )
    except Exception as exc:
        logger.warning("forward_generation_run failed: %s", exc)
