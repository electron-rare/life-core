"""Provider OpenAI GPT."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.router.providers.openai")


class OpenAIProvider(LLMProvider):
    """Provider pour l'API OpenAI."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        """
        Initialiser le provider OpenAI.
        
        Args:
            api_key: Clé API OpenAI (si None, lire depuis env OPENAI_API_KEY)
            config: Configuration additionnelle
        """
        super().__init__("openai", config)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Initialiser le client OpenAI de manière lazy."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package requis. Installez avec: pip install openai")
        return self._client

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> LLMResponse:
        """Envoyer un message à OpenAI et retourner une réponse complète."""
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
                provider="openai",
                usage={
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                },
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                }
            )
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            self.is_available = False
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "gpt-4o-mini",
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streamer une réponse de OpenAI."""
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
                    if event.choices[0].delta.content:
                        yield LLMStreamChunk(
                            content=event.choices[0].delta.content,
                            model=model,
                            finish_reason=event.choices[0].finish_reason
                        )
        except Exception as e:
            logger.error(f"OpenAI streaming error: {e}")
            self.is_available = False
            raise

    async def health_check(self) -> bool:
        """Vérifier si OpenAI est disponible."""
        try:
            client = await self._get_client()
            # Test avec un appel minimal
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            self.is_available = True
            return True
        except Exception as e:
            logger.warning(f"OpenAI health check failed: {e}")
            self.is_available = False
            return False

    async def list_models(self) -> list[str]:
        """Lister les modèles OpenAI disponibles."""
        return [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]
