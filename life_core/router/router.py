"""Routeur LLM multi-provider avec fallback et circuit breaker."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from opentelemetry import trace

from life_core.router.providers.base import LLMProvider, LLMResponse, LLMStreamChunk
from life_core.telemetry import get_tracer

logger = logging.getLogger("life_core.router")
tracer = get_tracer()


class Router:
    """
    Routeur LLM intelligent.
    
    Gère:
    - Dispatch entre multiples providers
    - Fallback automatique en cas d'erreur
    - Circuit breaker pour prévenir les cascades de panne
    - Health checks périodiques
    """

    def __init__(self):
        """Initialiser le routeur."""
        self.providers: dict[str, LLMProvider] = {}
        self.primary_provider: str | None = None
        self._health_status: dict[str, bool] = {}

    def register_provider(
        self,
        provider: LLMProvider,
        is_primary: bool = False
    ) -> None:
        """
        Enregistrer un provider.
        
        Args:
            provider: Instance du provider
            is_primary: Définir comme provider primaire
        """
        self.providers[provider.provider_id] = provider
        self._health_status[provider.provider_id] = True
        
        if is_primary:
            self.primary_provider = provider.provider_id
        
        logger.info(f"Registered provider: {provider.provider_id}")

    async def send(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str | None = None,
        **kwargs
    ) -> LLMResponse:
        """
        Envoyer un message via le routeur.
        
        Args:
            messages: Liste de messages (role/content)
            model: Modèle à utiliser
            provider: Provider spécifique (None = auto-select)
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse normalisée
            
        Raises:
            ValueError: Si aucun provider n'est disponible
        """
        # Si un provider est spécifié, l'utiliser
        if provider:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not registered")
            return await self._call_with_fallback(
                messages=messages,
                model=model,
                primary_provider=provider,
                **kwargs
            )
        
        # Sinon, utiliser le provider primaire
        if self.primary_provider:
            return await self._call_with_fallback(
                messages=messages,
                model=model,
                primary_provider=self.primary_provider,
                **kwargs
            )
        
        raise ValueError("No primary provider configured")

    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        provider: str | None = None,
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Streamer une réponse via le routeur.
        
        Args:
            messages: Liste de messages (role/content)
            model: Modèle à utiliser
            provider: Provider spécifique (None = auto-select)
            **kwargs: Paramètres additionnels
            
        Yields:
            Chunks de la réponse
        """
        # Si un provider est spécifié, l'utiliser
        if provider:
            if provider not in self.providers:
                raise ValueError(f"Provider {provider} not registered")
            async for chunk in self._stream_with_fallback(
                messages=messages,
                model=model,
                primary_provider=provider,
                **kwargs
            ):
                yield chunk
            return
        
        # Sinon, utiliser le provider primaire
        if self.primary_provider:
            async for chunk in self._stream_with_fallback(
                messages=messages,
                model=model,
                primary_provider=self.primary_provider,
                **kwargs
            ):
                yield chunk
            return
        
        raise ValueError("No primary provider configured")

    async def _call_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: str,
        primary_provider: str,
        **kwargs
    ) -> LLMResponse:
        """Appeler un provider avec fallback automatique."""
        from life_core.router.fallback_config import KIKI_TO_VLLM_FALLBACKS

        # Early path: for kiki-* aliases, inject the static fallback
        # chain as the LiteLLM fallbacks= kwarg so the SDK handles
        # model swaps within a single acompletion call (Studio → KXKM
        # → cloud). The router's cross-provider retry loop below then
        # catches any remaining failure at the provider level.
        if isinstance(model, str) and model in KIKI_TO_VLLM_FALLBACKS:
            provider = self.providers[primary_provider]
            call_kwargs = dict(kwargs)
            call_kwargs.setdefault(
                "fallbacks", list(KIKI_TO_VLLM_FALLBACKS[model])
            )
            return await provider.send(
                messages=messages, model=model, **call_kwargs
            )

        providers_to_try = [primary_provider]
        
        # Ajouter les autres providers comme fallback
        for pid in self.providers:
            if pid != primary_provider and self._health_status.get(pid, True):
                providers_to_try.append(pid)
        
        with tracer.start_as_current_span("llm.call") as span:
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.provider.hint", primary_provider)
            
            for fallback_index, provider_id in enumerate(providers_to_try):
                try:
                    provider = self.providers[provider_id]
                    if not self._health_status.get(provider_id, True):
                        logger.debug(f"Skipping unhealthy provider: {provider_id}")
                        continue
                    
                    span.set_attribute("llm.provider", provider_id)
                    span.set_attribute("llm.fallback_index", fallback_index)
                    
                    response = await provider.send(messages=messages, model=model, **kwargs)
                    self._health_status[provider_id] = True
                    
                    # Ajouter les tokens à la span
                    if response.usage:
                        span.set_attribute("llm.tokens.input", response.usage.get("input_tokens", 0))
                        span.set_attribute("llm.tokens.output", response.usage.get("output_tokens", 0))
                    
                    logger.info(f"Response from {provider_id}")
                    return response
                except Exception as e:
                    logger.warning(f"Provider {provider_id} failed: {e}")
                    self._health_status[provider_id] = False
                    if fallback_index == len(providers_to_try) - 1:
                        # Dernière tentative
                        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                    continue
        
        raise RuntimeError(f"All providers failed for model {model}")

    async def _stream_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: str,
        primary_provider: str,
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """Streamer avec fallback automatique."""
        providers_to_try = [primary_provider]
        
        # Ajouter les autres providers comme fallback
        for pid in self.providers:
            if pid != primary_provider and self._health_status.get(pid, True):
                providers_to_try.append(pid)
        
        for provider_id in providers_to_try:
            try:
                provider = self.providers[provider_id]
                if not self._health_status.get(provider_id, True):
                    logger.debug(f"Skipping unhealthy provider: {provider_id}")
                    continue
                
                async for chunk in provider.stream(
                    messages=messages,
                    model=model,
                    **kwargs
                ):
                    yield chunk
                
                self._health_status[provider_id] = True
                logger.info(f"Stream completed from {provider_id}")
                return
            except Exception as e:
                logger.warning(f"Provider {provider_id} stream failed: {e}")
                self._health_status[provider_id] = False
                continue
        
        raise RuntimeError(f"All providers failed for streaming {model}")

    async def health_check_all(self) -> dict[str, bool]:
        """
        Vérifier la santé de tous les providers.
        
        Returns:
            Dictionnaire {provider_id: is_healthy}
        """
        results = {}
        tasks = [
            self._check_provider_health(pid, provider)
            for pid, provider in self.providers.items()
        ]
        await asyncio.gather(*tasks)
        return self._health_status.copy()

    async def _check_provider_health(self, provider_id: str, provider: LLMProvider) -> None:
        """Vérifier la santé d'un provider."""
        try:
            is_healthy = await provider.health_check()
            self._health_status[provider_id] = is_healthy
            logger.debug(f"Health check {provider_id}: {is_healthy}")
        except Exception as e:
            logger.warning(f"Health check failed for {provider_id}: {e}")
            self._health_status[provider_id] = False

    def list_available_providers(self) -> list[str]:
        """Lister les providers disponibles."""
        return list(self.providers.keys())

    def get_provider_status(self) -> dict[str, bool]:
        """Obtenir le statut de tous les providers."""
        return self._health_status.copy()
