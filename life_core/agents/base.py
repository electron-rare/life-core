"""Base classes for F4L workflow LLM agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentResult:
    ok: bool
    output: str
    reasons: list[str] = field(default_factory=list)


class AgentBase:
    role: str = ""

    def __init__(self) -> None:
        if not self.role:
            raise NotImplementedError(
                f"{self.__class__.__name__} must set a non-empty `role`"
            )

    async def run(self, payload: dict[str, Any]) -> AgentResult:
        raise NotImplementedError
