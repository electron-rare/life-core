"""Shared fixtures for life-core bootstrap tests."""

from __future__ import annotations

import asyncio

import pytest

from life_core.router.providers.base import LLMProvider, LLMResponse
from life_core.router.router import Router


class MockProvider(LLMProvider):
    def __init__(self, name: str, cost: tuple[float, float], speed: int, quality: int, fail: bool = False):
        self.name = name
        self.default_model = f"{name}-model"
        self.cost_per_million = cost
        self.speed_rank = speed
        self.quality_rank = quality
        self.fail = fail

    async def send(self, messages: list[dict], **kwargs: object) -> LLMResponse:
        if self.fail:
            raise ConnectionError(f"{self.name} failed")
        return LLMResponse(content=f"response from {self.name}", model=self.default_model, provider=self.name)

    async def stream(self, messages: list[dict], **kwargs: object):
        yield f"token from {self.name}"

    def available_models(self) -> list[str]:
        return [self.default_model]


@pytest.fixture
def router() -> Router:
    current = Router()
    current.register(MockProvider("cheap", (1.0, 2.0), speed=3, quality=1))
    current.register(MockProvider("fast", (3.0, 3.0), speed=1, quality=2))
    current.register(MockProvider("best", (6.0, 6.0), speed=2, quality=5))
    return current


@pytest.fixture
def asyncio_run():
    return asyncio.run