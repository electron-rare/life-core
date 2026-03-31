"""Provider Ollama pour inference LLM locale."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import httpx

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.providers.ollama")


class OllamaProvider(LLMProvider):
    """Provider pour Ollama (local ou remote via Tailscale)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        name: str = "ollama",
        config: dict[str, Any] | None = None,
    ):
        super().__init__(name, config)
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    def _get_client(self, timeout: float = 120.0) -> httpx.AsyncClient:
        """Retourner le client partagé, en le créant si nécessaire."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=timeout)
        return self._client

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2",
        **kwargs: Any,
    ) -> LLMResponse:
        """Envoyer un message a Ollama et retourner une reponse complete."""
        client = self._get_client(timeout=120.0)
        response = await client.post(
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
            },
        )
        response.raise_for_status()
        data = response.json()

        return LLMResponse(
            content=data["message"]["content"],
            model=data.get("model", model),
            provider=self.provider_id,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
        )

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "llama3.2",
        **kwargs: Any,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streamer une reponse depuis Ollama."""
        client = self._get_client(timeout=120.0)
        async with client.stream(
            "POST",
            f"{self.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                msg = data.get("message", {})
                content = msg.get("content", "")
                done = data.get("done", False)
                if content:
                    yield LLMStreamChunk(
                        content=content,
                        model=model,
                        finish_reason="stop" if done else None,
                    )

    async def list_models(self) -> list[str]:
        """Lister les modeles disponibles sur Ollama."""
        client = self._get_client(timeout=10.0)
        response = await client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return [m["name"] for m in data.get("models", [])]

    async def health_check(self) -> bool:
        """Verifier si Ollama est accessible."""
        try:
            client = self._get_client(timeout=5.0)
            response = await client.get(f"{self.base_url}/api/tags")
            healthy = response.status_code == 200
            self.is_available = healthy
            return healthy
        except Exception:
            logger.warning(f"Ollama at {self.base_url} is not reachable")
            self.is_available = False
            return False
