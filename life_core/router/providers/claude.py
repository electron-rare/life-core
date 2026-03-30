"""Provider Claude Anthropic."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.router.providers.claude")


class ClaudeProvider(LLMProvider):
    """Provider pour l'API Claude d'Anthropic."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        """
        Initialiser le provider Claude.
        
        Args:
            api_key: Clé API Anthropic (si None, lire depuis env ANTHROPIC_API_KEY)
            config: Configuration additionnelle
        """
        super().__init__("claude", config)
        self.api_key = api_key
        # lazy import pour éviter les dépendances forcées
        self._client = None

    async def _get_client(self):
        """Initialiser le client Anthropic de manière lazy."""
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package requis. Installez avec: pip install anthropic")
        return self._client

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs
    ) -> LLMResponse:
        """Envoyer un message à Claude et retourner une réponse complète."""
        client = await self._get_client()
        
        try:
            response = client.messages.create(
                model=model,
                max_tokens=kwargs.get("max_tokens", 4096),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                system=kwargs.get("system", None),
            )
            
            return LLMResponse(
                content=response.content[0].text,
                model=model,
                provider="claude",
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
                metadata={
                    "stop_reason": response.stop_reason,
                }
            )
        except Exception as e:
            logger.error(f"Claude API error: {e}")
            self.is_available = False
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streamer une réponse de Claude."""
        client = await self._get_client()
        
        try:
            with client.messages.stream(
                model=model,
                max_tokens=kwargs.get("max_tokens", 4096),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                system=kwargs.get("system", None),
            ) as stream:
                for text in stream.text_stream:
                    yield LLMStreamChunk(
                        content=text,
                        model=model,
                        finish_reason=None
                    )
        except Exception as e:
            logger.error(f"Claude streaming error: {e}")
            self.is_available = False
            raise

    async def health_check(self) -> bool:
        """Vérifier si Claude est disponible."""
        try:
            client = await self._get_client()
            # Test avec un appel minimal
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}],
            )
            self.is_available = True
            return True
        except Exception as e:
            logger.warning(f"Claude health check failed: {e}")
            self.is_available = False
            return False

    async def list_models(self) -> list[str]:
        """Lister les modèles Claude disponibles."""
        return [
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20250219",
        ]
