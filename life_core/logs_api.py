"""Logs API — recent log entries for the cockpit."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("life_core.logs_api")

logs_router = APIRouter(prefix="/logs", tags=["Logs"])

# In-memory ring buffer for recent logs
_log_buffer: deque[dict[str, Any]] = deque(maxlen=200)


def add_log(level: str, message: str, source: str = "life-core") -> None:
    """Add a log entry to the buffer."""
    _log_buffer.append({
        "timestamp": time.strftime("%H:%M:%S"),
        "level": level,
        "message": message,
        "source": source,
    })


@logs_router.get("/recent")
async def recent_logs(limit: int = 50):
    """Return recent log entries."""
    logs = list(_log_buffer)
    return {"logs": logs[-limit:], "total": len(logs)}


# Hook into Python logging to capture life-core logs
class BufferHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("life_core"):
            add_log(
                level=record.levelname,
                message=record.getMessage(),
                source=record.name,
            )


# Install handler on root life_core logger
_handler = BufferHandler()
_handler.setLevel(logging.INFO)
logging.getLogger("life_core").addHandler(_handler)
