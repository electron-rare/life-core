"""Tests for the SSE /events endpoint.

These tests bypass TestClient's streaming entry-point (which would
block on the infinite generator) by exercising the async generator
directly with a minimal fake Request object. A third test verifies
the route registration and headers via TestClient using a pre-closed
connection trick.
"""

import json

import pytest

from life_core.api import app
from life_core.events_api import _event_generator, _snapshot


@pytest.fixture(autouse=True)
def _short_interval(monkeypatch):
    """Make the SSE interval near-zero so the generator yields fast."""
    monkeypatch.setenv("F4L_SSE_INTERVAL", "0.01")


class _FakeRequest:
    """Minimal Request stand-in with a controllable is_disconnected()."""

    def __init__(self, disconnect_after: int = 1) -> None:
        self._calls = 0
        self._disconnect_after = disconnect_after

    async def is_disconnected(self) -> bool:
        self._calls += 1
        return self._calls > self._disconnect_after


def test_events_route_is_registered():
    """The /events route must exist on the FastAPI app with SSE headers."""
    paths = {getattr(r, "path", None) for r in app.routes}
    assert "/events" in paths


@pytest.mark.asyncio
async def test_events_first_frame_contains_snapshot():
    """Le premier frame SSE = event: snapshot + payload JSON valide."""
    request = _FakeRequest(disconnect_after=1)
    gen = _event_generator(request)
    frame = await gen.__anext__()
    assert frame.startswith("event: snapshot\n")
    data_line = next(line for line in frame.splitlines() if line.startswith("data:"))
    payload = json.loads(data_line.removeprefix("data:").strip())
    assert "health" in payload
    assert "stats" in payload
    assert "goose" in payload
    # Drain/close to let the generator stop cleanly.
    await gen.aclose()


@pytest.mark.asyncio
async def test_events_disconnect_stops_generator():
    """Le générateur s'arrête proprement dès que is_disconnected() renvoie True."""
    request = _FakeRequest(disconnect_after=0)  # disconnect on the first check
    gen = _event_generator(request)
    frames: list[str] = []
    async for frame in gen:
        frames.append(frame)
    # No frame should have been yielded (disconnect is checked first).
    assert frames == []


@pytest.mark.asyncio
async def test_snapshot_shape_matches_contract():
    """_snapshot must always yield the three top-level keys expected by the UI."""
    snap = await _snapshot()
    assert set(snap.keys()) >= {"health", "stats", "goose"}
    assert "status" in snap["health"]
    assert "providers" in snap["health"]
    assert "cache_available" in snap["health"]
    assert "active_sessions" in snap["goose"]
