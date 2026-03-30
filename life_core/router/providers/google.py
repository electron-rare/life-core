"""Provider Google Gemini."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk

logger = logging.getLogger("life_core.router.providers.google")


class GoogleProvider(LLMProvider):
    """Provider pour l'API Google Gemini."""

    def __init__(self, api_key: str | None = None, config: dict[str, Any] | None = None):
        """
        Initialiser le provider Google.
        
        Args:
            api_key: Clé API Google (si None, lire depuis env GOOGLE_API_KEY)
            config: Configuration additionnelle
        """
        super().__init__("google", config)
        self.api_key = api_key
        self._client = None

    async def _get_client(self):
        """Initialiser le client Google de manière lazy."""
        if self._client is None:
            try:
                import google.generativeai as genai
                if self.api_key:
                    genai.configure(api_key=self.api_key)
                self._client = genai
            except ImportError:
                raise ImportError("google-generativeai package requis. Installez avec: pip install google-generativeai")
        return self._client

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str = "gemini-2.0-flash",
        **kwargs
    ) -> LLMResponse:
        """Envoyer un message à Google et retourner une réponse complète."""
        genai = await self._get_client()
        
        try:
            # Convertir le format des messages
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "parts": [msg.get("content", "")]
                })
            
            model_obj = genai.GenerativeModel(model)
            response = await model_obj.generate_content_async(
                formatted_messages,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_tokens", 4096),
                )
            )
            
            return LLMResponse(
                content=response.text,
                model=model,
                provider="google",
                usage={
                    "input_tokens": response.usage_metadata.prompt_token_count,
                    "output_tokens": response.usage_metadata.candidates_token_count,
                },
                metadata={
                    "finish_reason": response.candidates[0].finish_reason.name if response.candidates else None,
                }
            )
        except Exception as e:
            logger.error(f"Google API error: {e}")
            self.is_available = False
            raise

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str = "gemini-2.0-flash",
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streamer une réponse de Google."""
        genai = await self._get_client()
        
        try:
            # Convertir le format des messages
            formatted_messages = []
            for msg in messages:
                formatted_messages.append({
                    "role": msg.get("role", "user"),
                    "parts": [msg.get("content", "")]
                })
            
            model_obj = genai.GenerativeModel(model)
            async for chunk in await model_obj.generate_content_async(
                formatted_messages,
                stream=True,
                generation_config=genai.types.GenerationConfig(
                    temperature=kwargs.get("temperature", 0.7),
                    max_output_tokens=kwargs.get("max_tokens", 4096),
                )
            ):
                if chunk.text:
                    yield LLMStreamChunk(
                        content=chunk.text,
                        model=model,
                        finish_reason=None
                    )
        except Exception as e:
            logger.error(f"Google streaming error: {e}")
            self.is_available = False
            raise

    async def health_check(self) -> bool:
        """Vérifier si Google est disponible."""
        try:
            genai = await self._get_client()
            model = genai.GenerativeModel("gemini-2.0-flash")
            response = await model.generate_content_async("ping", generation_config={"max_output_tokens": 5})
            self.is_available = True
            return True
        except Exception as e:
            logger.warning(f"Google health check failed: {e}")
            self.is_available = False
            return False

    async def list_models(self) -> list[str]:
        """Lister les modèles Google disponibles."""
        return [
            "gemini-2.0-flash",
            "gemini-1.5-pro",
            "gemini-1.5-flash",
        ]
