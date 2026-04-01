"""Cache multi-tier pour life-core."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Any

logger = logging.getLogger("life_core.cache")


class CacheEntry:
    """Entrée de cache avec métadonnées."""

    def __init__(self, key: str, value: Any, ttl: int | None = None):
        """
        Créer une entrée de cache.
        
        Args:
            key: Clé de cache
            value: Valeur à cacher
            ttl: Temps de vie en secondes (None = pas d'expiration)
        """
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.hits = 0

    def is_expired(self) -> bool:
        """Vérifier si l'entrée a expiré."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def hit(self) -> None:
        """Enregistrer un accès."""
        self.hits += 1


class L1Cache:
    """Cache mémoire de niveau 1 (en-process)."""

    def __init__(self, max_size: int = 1000):
        """
        Créer un cache L1.
        
        Args:
            max_size: Taille maximale du cache
        """
        self.max_size = max_size
        self.entries: dict[str, CacheEntry] = {}
        self.stats = {"hits": 0, "misses": 0}

    def get(self, key: str) -> Any:
        """Récupérer une valeur du cache L1."""
        if key in self.entries:
            entry = self.entries[key]
            if not entry.is_expired():
                entry.hit()
                self.stats["hits"] += 1
                logger.debug(f"L1 cache hit: {key}")
                return entry.value
            else:
                del self.entries[key]
                logger.debug(f"L1 cache expired: {key}")
        
        self.stats["misses"] += 1
        return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Stocker une valeur dans le cache L1."""
        if len(self.entries) >= self.max_size:
            # Supprimer l'entrée la moins utilisée (LRU simplifié)
            lru_key = min(self.entries, key=lambda k: self.entries[k].hits)
            del self.entries[lru_key]
            logger.debug(f"L1 cache evicted: {lru_key}")
        
        self.entries[key] = CacheEntry(key, value, ttl)
        logger.debug(f"L1 cache set: {key}")

    def delete(self, key: str) -> None:
        """Supprimer une entrée du cache L1."""
        if key in self.entries:
            del self.entries[key]
            logger.debug(f"L1 cache deleted: {key}")

    def clear(self) -> None:
        """Vider le cache L1."""
        self.entries.clear()
        logger.debug("L1 cache cleared")

    def get_stats(self) -> dict[str, int]:
        """Obtenir les statistiques du cache."""
        return {
            **self.stats,
            "size": len(self.entries),
            "max_size": self.max_size,
        }


class L2Cache:
    """Cache Redis de niveau 2 (optionnel, partagé)."""

    def __init__(self, redis_url: str | None = None):
        """
        Créer un cache L2.
        
        Args:
            redis_url: URL de connexion Redis (ex: redis://localhost:6379/0)
        """
        self.redis_url = redis_url
        self._client = None
        self.stats = {"hits": 0, "misses": 0}
        self.available = redis_url is not None

    async def _get_client(self):
        """Initialiser le client Redis de manière lazy."""
        if self._client is None and self.redis_url:
            try:
                import redis.asyncio as redis
                self._client = await redis.from_url(self.redis_url)
                logger.info(f"L2 cache connected to Redis: {self.redis_url}")
            except Exception as e:
                logger.warning(f"L2 cache Redis connection failed: {e}")
                self.available = False
        return self._client

    async def get(self, key: str) -> Any:
        """Récupérer une valeur du cache L2."""
        if not self.available:
            self.stats["misses"] += 1
            return None
        
        try:
            client = await self._get_client()
            if client is None:
                self.stats["misses"] += 1
                return None
            
            value = await client.get(key)
            if value:
                self.stats["hits"] += 1
                logger.debug(f"L2 cache hit: {key}")
                try:
                    return json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    return value
            else:
                self.stats["misses"] += 1
                return None
        except Exception as e:
            logger.warning(f"L2 cache get error: {e}")
            self.stats["misses"] += 1
            return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Stocker une valeur dans le cache L2."""
        if not self.available:
            return
        
        try:
            client = await self._get_client()
            if client is None:
                return
            
            # Sérialiser en JSON si possible
            try:
                serialized = json.dumps(value)
            except TypeError:
                serialized = str(value)
            
            if ttl:
                await client.setex(key, ttl, serialized)
            else:
                await client.set(key, serialized)
            
            logger.debug(f"L2 cache set: {key}")
        except Exception as e:
            logger.warning(f"L2 cache set error: {e}")

    async def delete(self, key: str) -> None:
        """Supprimer une entrée du cache L2."""
        if not self.available:
            return
        
        try:
            client = await self._get_client()
            if client:
                await client.delete(key)
                logger.debug(f"L2 cache deleted: {key}")
        except Exception as e:
            logger.warning(f"L2 cache delete error: {e}")

    async def clear(self) -> None:
        """Vider le cache L2."""
        if not self.available:
            return
        
        try:
            client = await self._get_client()
            if client:
                await client.flushdb()
                logger.debug("L2 cache cleared")
        except Exception as e:
            logger.warning(f"L2 cache clear error: {e}")

    def get_stats(self) -> dict[str, int]:
        """Obtenir les statistiques du cache."""
        return {
            **self.stats,
            "available": self.available,
        }


class MultiTierCache:
    """Cache multi-tier avec L1 (mémoire) et L2 (Redis optionnel)."""

    def __init__(self, redis_url: str | None = None, l1_max_size: int = 1000):
        """
        Créer un cache multi-tier.
        
        Args:
            redis_url: URL Redis pour L2 (optionnel)
            l1_max_size: Taille maximale du cache L1
        """
        self.l1 = L1Cache(max_size=l1_max_size)
        self.l2 = L2Cache(redis_url=redis_url)

    async def get(self, key: str, default: Any = None) -> Any:
        """
        Récupérer une valeur du cache.

        Tente d'abord L1, puis L2.
        """
        from life_core.telemetry import get_tracer
        tracer = get_tracer()

        # Essayer L1
        with tracer.start_as_current_span("cache.l1.get", attributes={"cache.tier": "l1"}) as span:
            value = self.l1.get(key)
            span.set_attribute("cache.hit", value is not None)
            if value is not None:
                return value

        # Essayer L2
        with tracer.start_as_current_span("cache.l2.get", attributes={"cache.tier": "l2"}) as span:
            value = await self.l2.get(key)
            span.set_attribute("cache.hit", value is not None)
            if value is not None:
                # Repeupler L1
                self.l1.set(key, value)
                return value

        return default

    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None
    ) -> None:
        """
        Stocker une valeur dans le cache.

        Stocke dans les deux niveaux disponibles.
        """
        from life_core.telemetry import get_tracer
        tracer = get_tracer()

        with tracer.start_as_current_span("cache.store", attributes={"cache.ttl": ttl or 0}):
            self.l1.set(key, value, ttl)
            await self.l2.set(key, value, ttl)

    async def delete(self, key: str) -> None:
        """Supprimer une clé du cache."""
        self.l1.delete(key)
        await self.l2.delete(key)

    async def clear(self) -> None:
        """Vider tous les caches."""
        self.l1.clear()
        await self.l2.clear()

    def get_stats(self) -> dict[str, Any]:
        """Obtenir les statistiques combinées."""
        return {
            "l1": self.l1.get_stats(),
            "l2": self.l2.get_stats(),
        }

    async def health_check(self) -> dict[str, bool]:
        """Vérifier la santé des caches."""
        return {
            "l1": True,  # L1 est toujours disponible
            "l2": self.l2.available,
        }
