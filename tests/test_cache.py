"""Tests pour le cache multi-tier."""

import pytest

from life_core.cache import L1Cache, MultiTierCache


def test_l1_cache_creation():
    """Test la création du cache L1."""
    cache = L1Cache(max_size=100)
    assert cache.max_size == 100
    assert len(cache.entries) == 0


def test_l1_cache_set_get():
    """Test le stockage et récupération L1."""
    cache = L1Cache()
    
    cache.set("key1", "value1")
    assert cache.get("key1") == "value1"
    assert cache.stats["misses"] == 0  # Pas de miss
    assert cache.stats["hits"] == 1


def test_l1_cache_miss():
    """Test un miss de cache L1."""
    cache = L1Cache()
    
    result = cache.get("nonexistent")
    assert result is None
    assert cache.stats["misses"] == 1
    assert cache.stats["hits"] == 0


def test_l1_cache_expiration():
    """Test l'expiration d'une clé L1."""
    cache = L1Cache()
    
    cache.set("expire_me", "value", ttl=1)
    assert cache.get("expire_me") == "value"
    
    import time
    time.sleep(1.1)
    
    result = cache.get("expire_me")
    assert result is None  # Doit avoir expiré


def test_l1_cache_delete():
    """Test la suppression d'une clé L1."""
    cache = L1Cache()
    
    cache.set("key", "value")
    assert cache.get("key") == "value"
    
    cache.delete("key")
    assert cache.get("key") is None


def test_l1_cache_clear():
    """Test le vidage du cache L1."""
    cache = L1Cache()
    
    cache.set("key1", "value1")
    cache.set("key2", "value2")
    assert len(cache.entries) == 2
    
    cache.clear()
    assert len(cache.entries) == 0


def test_l1_cache_lru_eviction():
    """Test l'éviction LRU quand la taille max est atteinte."""
    cache = L1Cache(max_size=2)
    
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)  # Cela doit évincer une clé
    
    assert len(cache.entries) == 2
    # La clé moins utilisée doit avoir été supprimée


def test_l1_cache_stats():
    """Test les statistiques du cache L1."""
    cache = L1Cache(max_size=50)
    
    cache.set("key", "value")
    cache.get("key")
    cache.get("key")
    cache.get("missing")
    
    stats = cache.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["size"] == 1
    assert stats["max_size"] == 50


@pytest.mark.asyncio
async def test_multi_tier_cache_creation():
    """Test la création du cache multi-tier."""
    cache = MultiTierCache()
    assert cache.l1 is not None
    assert cache.l2 is not None


@pytest.mark.asyncio
async def test_multi_tier_cache_set_get():
    """Test le stockage et récupération multi-tier."""
    cache = MultiTierCache()
    
    await cache.set("key1", "value1")
    value = await cache.get("key1")
    assert value == "value1"


@pytest.mark.asyncio
async def test_multi_tier_cache_l1_fallback():
    """Test le fallback L1 quand L2 n'est pas disponible."""
    cache = MultiTierCache()
    
    await cache.set("test_key", {"data": "value"})
    value = await cache.get("test_key")
    assert value == {"data": "value"}


@pytest.mark.asyncio
async def test_multi_tier_cache_delete():
    """Test la suppression dans le cache multi-tier."""
    cache = MultiTierCache()
    
    await cache.set("key", "value")
    await cache.delete("key")
    
    value = await cache.get("key")
    assert value is None


@pytest.mark.asyncio
async def test_multi_tier_cache_clear():
    """Test le vidage du cache multi-tier."""
    cache = MultiTierCache()
    
    await cache.set("a", 1)
    await cache.set("b", 2)
    await cache.clear()
    
    a = await cache.get("a")
    b = await cache.get("b")
    assert a is None
    assert b is None


@pytest.mark.asyncio
async def test_multi_tier_cache_stats():
    """Test les statistiques du cache multi-tier."""
    cache = MultiTierCache()
    
    await cache.set("key", "value")
    await cache.get("key")
    
    stats = cache.get_stats()
    assert "l1" in stats
    assert "l2" in stats
    assert stats["l1"]["hits"] > 0


@pytest.mark.asyncio
async def test_multi_tier_cache_health_check():
    """Test le health check du cache multi-tier."""
    cache = MultiTierCache()
    
    health = await cache.health_check()
    assert health["l1"] is True  # L1 toujours healthy
    assert "l2" in health
