"""Provider interface shared by life-core router tests and services."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass(slots=True)
class LLMResponse:
    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)


class LLMProvider(ABC):
    name: str
    default_model: str
    cost_per_million: tuple[float, float] = (0.0, 0.0)
    speed_rank: int = 0
    quality_rank: int = 0

    @abstractmethod
    async def send(self, messages: list[dict], **kwargs: object) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    async def stream(self, messages: list[dict], **kwargs: object) -> AsyncIterator[str]:
        raise NotImplementedError

    @abstractmethod
    def available_models(self) -> list[str]:
        raise NotImplementedError