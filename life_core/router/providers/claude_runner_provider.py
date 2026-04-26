"""ClaudeRunnerProvider — HTTP adapter for the claude-runner sidecar.

Routes LLM calls to the local claude-runner microservice (port 9300 by
default) which proxies to the real Claude API and adds latency tracking.

Timeout rationale (J1 validation, commit 28fa737):
  Claude latency is multimodal — 12-20 s warm, up to 60-90 s on ~30% of
  calls. A 30 s default balances warm-call coverage against letting
  LiteLLM fall back to qwen-14b on slow calls (the antifragile P2
  behaviour the demo requires). Override via CLAUDE_RUNNER_TIMEOUT_S env
  var or by passing ``timeout=`` explicitly.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx
from opentelemetry import trace

from life_core.router.providers.base import (
    LLMProvider,
    LLMResponse,
    LLMStreamChunk,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 30
_TIMEOUT_ENV = "CLAUDE_RUNNER_TIMEOUT_S"

tracer = trace.get_tracer("life_core.providers.claude_runner")


class ClaudeRunnerProvider(LLMProvider):
    """HTTP adapter for the claude-runner sidecar (:9300).

    Implements the standard LLMProvider interface so the Router can use
    it alongside LiteLLMProvider and fall back transparently.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:9300",
        timeout: int | None = None,
        max_retries: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(provider_id="claude-runner", **kwargs)
        self.base_url = base_url.rstrip("/")
        self.timeout = (
            timeout
            if timeout is not None
            else int(os.getenv(_TIMEOUT_ENV, str(_DEFAULT_TIMEOUT)))
        )
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # LLMProvider abstract methods
    # ------------------------------------------------------------------

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-7",
        **kwargs: Any,
    ) -> LLMResponse:
        """POST /v1/chat/completions and return a normalised LLMResponse.

        Retries on 503/504 with exponential backoff (0.2s, 0.4s, 0.8s …).
        Emits an OTEL span named ``claude_runner.send`` with attributes
        ``model``, ``attempts``, and ``latency_ms``.
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **kwargs,
        }

        with tracer.start_as_current_span("claude_runner.send") as span:
            span.set_attribute("model", model)
            attempts = 0

            for attempt in range(self.max_retries):
                attempts += 1
                try:
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        resp = await client.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=payload,
                        )
                    resp.raise_for_status()
                    break  # success — exit retry loop
                except httpx.HTTPStatusError as exc:
                    if (
                        exc.response.status_code in (503, 504)
                        and attempt < self.max_retries - 1
                    ):
                        logger.warning(
                            "ClaudeRunnerProvider: %s on attempt %d, retrying",
                            exc.response.status_code,
                            attempt + 1,
                        )
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue
                    raise

            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            latency_ms = data.get("latency_ms", 0)

            span.set_attribute("attempts", attempts)
            span.set_attribute("latency_ms", latency_ms)

            return LLMResponse(
                content=choice["message"]["content"],
                model=model,
                provider="claude-runner",
                usage={
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
                metadata={"latency_ms": latency_ms, "raw": data},
            )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-7",
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream SSE chunks from claude-runner's streaming endpoint.

        Falls back to a single synthetic chunk when the server returns a
        non-streaming response (e.g. during cold-start or when
        stream=False is forced).
        """
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[len("data: "):]
                    if raw.strip() == "[DONE]":
                        break
                    import json as _json  # noqa: PLC0415 — deferred import
                    chunk_data = _json.loads(raw)
                    for choice in chunk_data.get("choices", []):
                        delta = choice.get("delta", {})
                        content = delta.get("content") or ""
                        if content:
                            yield LLMStreamChunk(
                                content=content,
                                model=model,
                                finish_reason=choice.get("finish_reason"),
                            )

    async def health_check(self) -> bool:
        """GET /health and check the returned status field."""
        try:
            async with httpx.AsyncClient(timeout=2) as client:
                resp = await client.get(f"{self.base_url}/health")
            return resp.status_code == 200 and resp.json().get("status") == "ok"
        except (httpx.HTTPError, KeyError, ValueError):
            return False

    async def list_models(self) -> list[str]:
        """Return models advertised by the sidecar via /v1/models."""
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.base_url}/v1/models")
            resp.raise_for_status()
            data = resp.json()
            return [m["id"] for m in data.get("data", [])]
        except Exception:  # noqa: BLE001
            logger.warning("ClaudeRunnerProvider: failed to list models")
            return []
