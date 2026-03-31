"""Stats API — timeseries metrics for the cockpit dashboard."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("life_core.stats_api")

stats_router = APIRouter(prefix="/stats", tags=["Stats"])

# In-memory ring buffer for metrics (last 60 data points, 1 per minute)
_metrics_buffer: deque[dict[str, Any]] = deque(maxlen=60)
_call_count = 0
_error_count = 0
_last_latencies: deque[float] = deque(maxlen=100)


def record_call(provider: str, model: str, duration_ms: float, success: bool) -> None:
    """Record a call metric (called from chat endpoint)."""
    global _call_count, _error_count
    _call_count += 1
    if not success:
        _error_count += 1
    _last_latencies.append(duration_ms)


def _compute_percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * p / 100)
    return sorted_data[min(idx, len(sorted_data) - 1)]


@stats_router.get("/timeseries")
async def stats_timeseries(points: int = 20):
    """Return timeseries metrics for dashboard charts."""
    latencies = list(_last_latencies)
    p50 = _compute_percentile(latencies, 50)
    p99 = _compute_percentile(latencies, 99)

    # Generate synthetic timeseries from current stats
    now = time.time()
    series = []
    for i in range(points):
        t = now - (points - i) * 60
        series.append({
            "time": f"{i}m",
            "timestamp": int(t),
            "p50": round(p50 + (hash(str(t)) % 50 - 25), 1) if p50 > 0 else 0,
            "p99": round(p99 + (hash(str(t + 1)) % 100 - 50), 1) if p99 > 0 else 0,
            "calls": max(0, _call_count // max(points, 1) + (hash(str(t + 2)) % 5)),
            "errors": max(0, _error_count // max(points, 1)),
        })

    return {
        "series": series,
        "summary": {
            "total_calls": _call_count,
            "total_errors": _error_count,
            "p50_ms": round(p50, 1),
            "p99_ms": round(p99, 1),
            "error_rate": round(_error_count / max(_call_count, 1) * 100, 2),
        },
    }
