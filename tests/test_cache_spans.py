"""Tests for OTEL spans on MultiTierCache operations."""
import pytest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider as SdkTracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from life_core.cache.multi_tier_cache import MultiTierCache


def _make_test_tracer():
    """Create a test tracer with in-memory exporter."""
    exporter = InMemorySpanExporter()
    tp = SdkTracerProvider()
    tp.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = tp.get_tracer("test")
    return tracer, exporter


@pytest.mark.asyncio
async def test_cache_get_emits_l1_span_on_hit():
    tracer, exporter = _make_test_tracer()
    cache = MultiTierCache()

    with patch("life_core.telemetry.get_tracer", return_value=tracer):
        await cache.set("key1", "value1", ttl=60)
        await cache.get("key1")

    spans = exporter.get_finished_spans()
    l1_get_spans = [s for s in spans if s.name == "cache.l1.get"]
    assert len(l1_get_spans) >= 1
    assert l1_get_spans[-1].attributes["cache.hit"] is True


@pytest.mark.asyncio
async def test_cache_get_emits_l1_span_on_miss():
    tracer, exporter = _make_test_tracer()
    cache = MultiTierCache()

    with patch("life_core.telemetry.get_tracer", return_value=tracer):
        await cache.get("nonexistent")

    spans = exporter.get_finished_spans()
    l1_get_spans = [s for s in spans if s.name == "cache.l1.get"]
    assert len(l1_get_spans) >= 1
    assert l1_get_spans[-1].attributes["cache.hit"] is False


@pytest.mark.asyncio
async def test_cache_set_emits_store_span():
    tracer, exporter = _make_test_tracer()
    cache = MultiTierCache()

    with patch("life_core.telemetry.get_tracer", return_value=tracer):
        await cache.set("key1", "value1", ttl=120)

    spans = exporter.get_finished_spans()
    store_spans = [s for s in spans if s.name == "cache.store"]
    assert len(store_spans) >= 1
    assert store_spans[-1].attributes["cache.ttl"] == 120
