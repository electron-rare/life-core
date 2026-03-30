"""Very small in-memory cache used as a bootstrap for phase 1."""

from __future__ import annotations


class MultiTierCache:
    def __init__(self) -> None:
        self._store: dict[str, object] = {}
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> object | None:
        if key in self._store:
            self._hits += 1
            return self._store[key]
        self._misses += 1
        return None

    def set(self, key: str, value: object) -> None:
        self._store[key] = value

    def get_stats(self) -> dict[str, int]:
        return {
            "hits": self._hits,
            "misses": self._misses,
            "entries": len(self._store),
        }