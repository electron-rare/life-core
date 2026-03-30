"""Provider Mistral AI."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.router.providers.mistral")


class MistralProvider(LLMProvider):
    """Provider pour l'API Mistral."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        super().__init__("mistral", config)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                from mistralai import Mistral

                self._client = Mistral(api_key=self.api_key)
            except ImportError:
                raise ImportError("mistralai package requis. Installez avec: pip install mistralai")
        return self._client

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "mistral-large-latest",
        **kwargs,
    ) -> LLMResponse:
        client = await self._get_client()

        try:
            response = await client.chat.complete_async(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
            )

            choice = response.choices[0]
            usage = response.usage

            return LLMResponse(
                content=choice.message.content,
                model=model,
                provider="mistral",
                usage={
                    "input_tokens": getattr(usage, "prompt_tokens", 0),
                    "output_tokens": getattr(usage, "completion_tokens", 0),
                },
                metadata={"finish_reason": getattr(choice, "finish_reason", None)},
            )
        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            self.is_available = False
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "mistral-large-latest",
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        client = await self._get_client()

        try:
            stream = await client.chat.stream_async(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
            )

            async for event in stream:
                delta = getattr(event.data.choices[0], "delta", None)
                content = getattr(delta, "content", None) if delta else None
                if content:
                    yield LLMStreamChunk(content=content, model=model, finish_reason=None)
        except Exception as e:
            logger.error(f"Mistral streaming error: {e}")
            self.is_available = False
            raise

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await client.chat.complete_async(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            self.is_available = True
            return True
        except Exception as e:
            logger.warning(f"Mistral health check failed: {e}")
            self.is_available = False
            return False

    async def list_models(self) -> list[str]:
        return [
            "mistral-large-latest",
            "mistral-small-latest",
            "ministral-8b-latest",
        ]
