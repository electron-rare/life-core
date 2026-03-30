"""Service wrapper around the phase 1 router scaffold."""

from __future__ import annotations

from finefab_core.router.router import Router, Strategy
from finefab_core.router.providers.base import LLMResponse


class RouterService:
    def __init__(self, router: Router | None = None) -> None:
        self.router = router or Router()

    async def chat(
        self,
        messages: list[dict],
        *,
        strategy: Strategy | str = Strategy.BEST,
        provider_name: str | None = None,
    ) -> LLMResponse:
        return await self.router.send(messages, strategy=strategy, provider_name=provider_name)

    def list_providers(self) -> list[str]:
        return self.router.available_providers