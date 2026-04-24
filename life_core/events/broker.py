"""V1.7 Track II — in-memory pub/sub broker for SSE /events.

Each subscriber gets its own bounded asyncio.Queue. Publish copies
the Event reference to every live subscriber. Bounded queues drop
the oldest event on overflow so a slow consumer cannot stall the
producers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List

from life_core.events.schema import Event

logger = logging.getLogger(__name__)


DEFAULT_QUEUE_SIZE = 256


class EventBroker:
    def __init__(self, queue_size: int = DEFAULT_QUEUE_SIZE) -> None:
        self._queue_size = queue_size
        self._subscribers: List[asyncio.Queue[Event]] = []
        self._lock = asyncio.Lock()

    def subscribe(self) -> asyncio.Queue[Event]:
        q: asyncio.Queue[Event] = asyncio.Queue(maxsize=self._queue_size)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue[Event]) -> None:
        try:
            self._subscribers.remove(q)
        except ValueError:
            pass

    async def publish(self, event: Event) -> None:
        dead: List[asyncio.Queue[Event]] = []
        for q in list(self._subscribers):
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest to keep cockpit fresh.
                try:
                    q.get_nowait()
                    q.put_nowait(event)
                except Exception:
                    dead.append(q)
            except Exception as exc:
                logger.warning("broker: drop sub on %s: %s", event.type, exc)
                dead.append(q)
        for q in dead:
            self.unsubscribe(q)


_broker: EventBroker | None = None


def get_broker() -> EventBroker:
    """Process-wide singleton accessor."""
    global _broker
    if _broker is None:
        _broker = EventBroker()
    return _broker
