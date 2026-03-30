"""Provider Groq."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.router.providers.groq")


class GroqProvider(LLMProvider):
    """Provider pour l'API Groq (OpenAI-compatible)."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        super().__init__("groq", config)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        if self._client is None:
            try:
                from groq import AsyncGroq

                self._client = AsyncGroq(api_key=self.api_key)
            except ImportError:
                raise ImportError("groq package requis. Installez avec: pip install groq")
        return self._client

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "llama-3.3-70b-versatile",
        **kwargs,
    ) -> LLMResponse:
        client = await self._get_client()

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
            )

            return LLMResponse(
                content=response.choices[0].message.content,
                model=model,
                provider="groq",
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                metadata={"finish_reason": response.choices[0].finish_reason},
            )
        except Exception as e:
            logger.error(f"Groq API error: {e}")
            self.is_available = False
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "llama-3.3-70b-versatile",
        **kwargs,
    ) -> AsyncIterator[LLMStreamChunk]:
        client = await self._get_client()

        try:
            async with await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4096),
                stream=True,
            ) as stream:
                async for event in stream:
                    delta = event.choices[0].delta
                    if delta.content:
                        yield LLMStreamChunk(
                            content=delta.content,
                            model=model,
                            finish_reason=event.choices[0].finish_reason,
                        )
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            self.is_available = False
            raise

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            await client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            self.is_available = True
            return True
        except Exception as e:
            logger.warning(f"Groq health check failed: {e}")
            self.is_available = False
            return False

    async def list_models(self) -> list[str]:
        return [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
        ]
