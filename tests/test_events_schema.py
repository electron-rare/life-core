"""V1.7 Track II Task 1 — SSE event schema unit tests."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest


def test_event_type_enum_has_six_members():
    from life_core.events.schema import EventType

    names = {e.value for e in EventType}
    assert names == {
        "router.status",
        "router.stats",
        "infra.network.host",
        "infra.container",
        "langfuse.trace",
        "goose.stats",
    }


def test_event_dataclass_fields():
    from life_core.events.schema import Event, EventType

    ts = datetime(2026, 4, 23, 21, 0, tzinfo=timezone.utc)
    ev = Event(
        type=EventType.ROUTER_STATUS,
        data={"active_model": "qwen-14b-awq-kxkm"},
        timestamp=ts,
    )
    assert ev.type is EventType.ROUTER_STATUS
    assert ev.data == {"active_model": "qwen-14b-awq-kxkm"}
    assert ev.timestamp == ts


def test_event_to_sse_dict():
    from life_core.events.schema import Event, EventType

    ev = Event(
        type=EventType.ROUTER_STATS,
        data={"rpm": 42, "tpm": 128000, "errors_1h": 0},
        timestamp=datetime(2026, 4, 23, 21, 0, tzinfo=timezone.utc),
    )
    out = ev.to_sse()
    assert out["event"] == "router.stats"
    assert '"rpm":42' in out["data"] or '"rpm": 42' in out["data"]
    assert '"timestamp"' in out["data"]


def test_event_requires_timezone_aware_timestamp():
    from life_core.events.schema import Event, EventType

    naive = datetime(2026, 4, 23, 21, 0)
    with pytest.raises(ValueError):
        Event(type=EventType.ROUTER_STATUS, data={}, timestamp=naive)
