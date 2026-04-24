"""LLM Providers."""

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk
from life_core.router.providers.compat_providers import (
    AnthropicProvider,
    GoogleProvider,
    GroqProvider,
    MistralProvider,
    OpenAIProvider,
)
from life_core.router.providers.litellm_provider import LiteLLMProvider

__all__ = [
    "LLMProvider",
    "LLMResponse",
    "LLMStreamChunk",
    "LiteLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "MistralProvider",
    "GroqProvider",
    "litellm_cost_callback",
]


def litellm_cost_callback(
    kwargs: dict,
    completion_response: dict,
    start_time: float,
    end_time: float,
    writer=None,
) -> None:
    """LiteLLM success callback that records a cost row.

    writer: injectable sink; defaults to a Postgres upsert into
    inner_trace.cost_ledger when None.
    """
    usage = completion_response.get("usage", {}) or {}
    hidden = completion_response.get("_hidden_params", {}) or {}
    row = {
        "model": kwargs.get("model"),
        "tokens_in": int(usage.get("prompt_tokens", 0) or 0),
        "tokens_out": int(usage.get("completion_tokens", 0) or 0),
        "cost_usd": float(hidden.get("response_cost", 0.0) or 0.0),
        "latency_ms": int((end_time - start_time) * 1000),
    }
    if writer is not None:
        writer(row)
        return
    try:
        from sqlalchemy import create_engine, text
        import os
        dsn = os.environ.get("DATABASE_URL")
        if not dsn:
            return
        engine = create_engine(dsn, pool_pre_ping=True)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO inner_trace.cost_ledger "
                    "(model, tokens_in, tokens_out, cost_usd, latency_ms) "
                    "VALUES (:model,:tokens_in,:tokens_out,:cost_usd,:latency_ms)"
                ),
                row,
            )
    except Exception:
        pass
