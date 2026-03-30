"""Minimal intelligent router for phase 1 bootstrap."""

from __future__ import annotations

from enum import StrEnum

from finefab_core.router.circuit_breaker import CircuitBreaker
from finefab_core.router.fallback import FallbackState
from finefab_core.router.providers.base import LLMProvider, LLMResponse


class Strategy(StrEnum):
    BEST = "best"
    CHEAPEST = "cheapest"
    FASTEST = "fastest"
    SPECIFIC = "specific"


class Router:
    def __init__(self) -> None:
        self._providers: dict[str, LLMProvider] = {}
        self.circuit_breaker = CircuitBreaker()
        self.fallback = FallbackState()

    @property
    def available_providers(self) -> list[str]:
        return sorted(self._providers)

    def register(self, provider: LLMProvider) -> None:
        self._providers[provider.name] = provider

    def _select_provider(self, strategy: Strategy, provider_name: str | None = None) -> LLMProvider:
        if not self._providers:
            raise ValueError("No provider registered")

        if strategy == Strategy.SPECIFIC:
            if not provider_name or provider_name not in self._providers:
                raise ValueError(f"Unknown provider: {provider_name}")
            return self._providers[provider_name]

        providers = list(self._providers.values())
        if strategy == Strategy.CHEAPEST:
            return min(providers, key=lambda provider: provider.cost_per_million)
        if strategy == Strategy.FASTEST:
            return min(providers, key=lambda provider: provider.speed_rank)
        return max(providers, key=lambda provider: provider.quality_rank)

    def _provider_order(self, strategy: Strategy, provider_name: str | None = None) -> list[LLMProvider]:
        if strategy == Strategy.SPECIFIC:
            return [self._select_provider(strategy, provider_name)]

        primary = self._select_provider(strategy)
        others = [provider for provider in self._providers.values() if provider.name != primary.name]
        return [primary, *others]

    async def send(
        self,
        messages: list[dict],
        *,
        strategy: Strategy | str = Strategy.BEST,
        provider_name: str | None = None,
    ) -> LLMResponse:
        chosen_strategy = Strategy(strategy)
        last_error: Exception | None = None

        for provider in self._provider_order(chosen_strategy, provider_name):
            if not self.circuit_breaker.allow_request():
                raise RuntimeError("Circuit breaker is open")
            try:
                response = await provider.send(messages)
                self.circuit_breaker.record_success()
                return response
            except Exception as exc:  # pragma: no cover - intentionally generic fallback path
                self.circuit_breaker.record_failure()
                self.fallback.record_failure()
                last_error = exc

        if last_error is not None:
            raise last_error
        raise RuntimeError("No provider available")