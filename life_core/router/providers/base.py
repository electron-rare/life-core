"""Interface abstraite pour les providers LLM."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("life_core.router.providers")


@dataclass
class LLMResponse:
    """Réponse normalisée d'un provider LLM."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMStreamChunk:
    """Chunk de streaming pour les réponses LLM."""

    content: str
    model: str
    finish_reason: str | None = None


class LLMProvider(ABC):
    """
    Interface commune pour tous les providers LLM.
    
    Chaque provider doit implémenter:
    - send(): Envoyer un prompt et retourner une réponse
    - stream(): Envoyer un prompt et streamer la réponse
    - health_check(): Vérifier la disponibilité du provider
    """

    def __init__(self, provider_id: str, config: dict[str, Any] | None = None):
        """
        Initialiser le provider.
        
        Args:
            provider_id: ID unique du provider (e.g., 'claude', 'openai', 'mistral')
            config: Configuration spécifique au provider
        """
        self.provider_id = provider_id
        self.config = config or {}
        self.is_available = True

    @abstractmethod
    async def send(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs
    ) -> LLMResponse:
        """
        Envoyer un prompt et retourner une réponse complète.
        
        Args:
            messages: Liste de messages (role/content)
            model: Modèle à utiliser
            **kwargs: Paramètres additionnels (temperature, max_tokens, etc.)
            
        Returns:
            LLMResponse normalisée
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        **kwargs
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Envoyer un prompt et streamer la réponse.
        
        Args:
            messages: Liste de messages (role/content)
            model: Modèle à utiliser
            **kwargs: Paramètres additionnels
            
        Yields:
            LLMStreamChunk avec chunks de contenu
        """
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Vérifier la disponibilité du provider.
        
        Returns:
            True si le provider est disponible, False sinon
        """
        pass

    async def list_models(self) -> list[str]:
        """
        Lister les modèles disponibles du provider.
        
        Returns:
            Liste des noms de modèles
        """
        return []
