"""OpenTelemetry instrumentation for life-core."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger("life_core.telemetry")

_tracer = None
_meter = None


def init_telemetry() -> None:
    """Initialize OpenTelemetry tracing and metrics if OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    global _tracer, _meter

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        logger.info("OpenTelemetry disabled (OTEL_EXPORTER_OTLP_ENDPOINT not set)")
        return

    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({
            "service.name": os.environ.get("OTEL_SERVICE_NAME", "life-core"),
            "service.version": "1.0.0",
        })

        # Tracing
        tracer_provider = TracerProvider(resource=resource)
        tracer_provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint, insecure=True))
        )
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer("life-core")

        # Metrics
        metric_reader = PeriodicExportingMetricReader(
            OTLPMetricExporter(endpoint=endpoint, insecure=True),
            export_interval_millis=30000,
        )
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter("life-core")

        logger.info(f"OpenTelemetry initialized, exporting to {endpoint}")

    except ImportError:
        logger.warning("OpenTelemetry packages not installed, telemetry disabled")
    except Exception as e:
        logger.warning(f"OpenTelemetry init failed: {e}")


def get_tracer():
    """Get the OpenTelemetry tracer (or a no-op if not initialized)."""
    if _tracer:
        return _tracer
    from opentelemetry import trace
    return trace.get_tracer("life-core")


def get_meter():
    """Get the OpenTelemetry meter (or a no-op if not initialized)."""
    if _meter:
        return _meter
    from opentelemetry import metrics
    return metrics.get_meter("life-core")


def create_llm_instruments() -> dict:
    """Create OTEL instruments for LLM call tracking."""
    meter = get_meter()
    return {
        "llm_calls": meter.create_counter(
            "finefab.llm.calls.total",
            description="Total LLM calls",
            unit="1",
        ),
        "llm_errors": meter.create_counter(
            "finefab.llm.errors.total",
            description="Total LLM errors",
            unit="1",
        ),
        "llm_duration": meter.create_histogram(
            "finefab.llm.duration.ms",
            description="LLM call duration",
            unit="ms",
        ),
    }


def create_rag_instruments():
    """Create OTEL instruments for RAG metrics."""
    meter = get_meter()
    return {
        "retrieval_latency": meter.create_histogram(
            "rag.retrieval.latency_ms",
            description="RAG retrieval latency in milliseconds",
            unit="ms",
        ),
        "retrieval_count": meter.create_counter(
            "rag.retrieval.count",
            description="Total RAG retrieval queries",
        ),
        "retrieval_top_score": meter.create_histogram(
            "rag.retrieval.top_score",
            description="Top retrieval score per query",
        ),
        "retrieval_results": meter.create_histogram(
            "rag.retrieval.results_count",
            description="Number of results returned per query",
        ),
    }
