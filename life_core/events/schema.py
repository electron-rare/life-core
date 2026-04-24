"""V1.7 Track II — SSE event schema.

One EventType value per spec Section 5.2 row. Event is a frozen
dataclass that serialises to the `{event, data}` shape that
sse-starlette's EventSourceResponse expects.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any


class EventType(str, Enum):
    ROUTER_STATUS = "router.status"
    ROUTER_STATS = "router.stats"
    INFRA_NETWORK_HOST = "infra.network.host"
    INFRA_CONTAINER = "infra.container"
    LANGFUSE_TRACE = "langfuse.trace"
    GOOSE_STATS = "goose.stats"


@dataclass(frozen=True)
class Event:
    type: EventType
    data: dict[str, Any]
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                "Event.timestamp must be timezone-aware"
            )

    def to_sse(self) -> dict[str, str]:
        payload = {
            **self.data,
            "timestamp": self.timestamp.isoformat(),
        }
        return {
            "event": self.type.value,
            "data": json.dumps(payload, separators=(",", ":")),
        }
