"""Tests for life_core/tracing.py — OTel span wrapping and metrics."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, AsyncMock, patch, call
from contextlib import contextmanager

import life_core.tracing as tracing_module
from life_core.tracing import traced_llm_call, _ensure_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_span():
    """Return a MagicMock that behaves as an OTel span context manager."""
    span = MagicMock()
    return span


def _make_tracer(span):
    """Return a MagicMock tracer whose start_as_current_span yields *span*."""
    tracer = MagicMock()

    @contextmanager
    def _start(name, attributes=None):
        yield span

    tracer.start_as_current_span.side_effect = _start
    return tracer


def _make_meter():
    """Return a MagicMock meter with counter and histogram."""
    meter = MagicMock()
    counter = MagicMock()
    histogram = MagicMock()
    meter.create_counter.return_value = counter
    meter.create_histogram.return_value = histogram
    return meter, counter, histogram


def _reset_metrics():
    """Reset module-level metric globals between tests."""
    tracing_module._llm_calls = None
    tracing_module._llm_errors = None
    tracing_module._llm_duration = None


# ---------------------------------------------------------------------------
# Tests for _ensure_metrics()
# ---------------------------------------------------------------------------

class TestEnsureMetrics:
    def setup_method(self):
        _reset_metrics()

    def test_initialises_metrics_on_first_call(self):
        meter, counter, histogram = _make_meter()
        with patch("life_core.tracing.get_meter", return_value=meter):
            _ensure_metrics()

        assert tracing_module._llm_calls is not None
        assert tracing_module._llm_errors is not None
        assert tracing_module._llm_duration is not None

    def test_creates_correct_counter_names(self):
        meter, counter, histogram = _make_meter()
        with patch("life_core.tracing.get_meter", return_value=meter):
            _ensure_metrics()

        calls = [c[0][0] for c in meter.create_counter.call_args_list]
        assert "llm.calls" in calls
        assert "llm.errors" in calls

    def test_creates_histogram(self):
        meter, counter, histogram = _make_meter()
        with patch("life_core.tracing.get_meter", return_value=meter):
            _ensure_metrics()

        meter.create_histogram.assert_called_once()
        assert meter.create_histogram.call_args[0][0] == "llm.duration_ms"

    def test_idempotent_second_call(self):
        meter, counter, histogram = _make_meter()
        with patch("life_core.tracing.get_meter", return_value=meter):
            _ensure_metrics()
            _ensure_metrics()

        # get_meter called only once — second call is no-op
        assert meter.create_counter.call_count == 2  # llm.calls + llm.errors
        assert meter.create_histogram.call_count == 1


# ---------------------------------------------------------------------------
# Tests for traced_llm_call()
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTracedLlmCallSuccess:
    def setup_method(self):
        _reset_metrics()

    async def test_returns_call_fn_result(self):
        expected = {"content": "Hello", "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        call_fn = AsyncMock(return_value=expected)
        span = _make_span()
        tracer = _make_tracer(span)
        meter, counter, histogram = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            result = await traced_llm_call("openai", "gpt-4", [{"role": "user", "content": "Hi"}], call_fn)

        assert result == expected

    async def test_passes_messages_and_model_to_call_fn(self):
        messages = [{"role": "user", "content": "Hello"}]
        call_fn = AsyncMock(return_value={"content": "ok", "usage": {}})
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            await traced_llm_call("anthropic", "claude-3", messages, call_fn, temperature=0.7)

        call_fn.assert_awaited_once_with(messages=messages, model="claude-3", temperature=0.7)

    async def test_span_attributes_set_on_success(self):
        response = {"content": "Hello world", "usage": {"prompt_tokens": 8, "completion_tokens": 3}}
        call_fn = AsyncMock(return_value=response)
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            await traced_llm_call("openai", "gpt-4o", [{"role": "user", "content": "Hi"}], call_fn)

        set_attr_calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "llm.duration_ms" in set_attr_calls
        assert set_attr_calls["llm.tokens.prompt"] == 8
        assert set_attr_calls["llm.tokens.completion"] == 3
        assert set_attr_calls["llm.response_length"] == len("Hello world")

    async def test_span_name_and_initial_attributes(self):
        call_fn = AsyncMock(return_value={"content": "", "usage": {}})
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()
        messages = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            await traced_llm_call("mistral", "mistral-large", messages, call_fn)

        tracer.start_as_current_span.assert_called_once_with(
            "llm.call",
            attributes={
                "llm.provider": "mistral",
                "llm.model": "mistral-large",
                "llm.message_count": 2,
            },
        )

    async def test_calls_counter_and_histogram_on_success(self):
        call_fn = AsyncMock(return_value={"content": "ok", "usage": {}})
        span = _make_span()
        tracer = _make_tracer(span)
        meter, counter, histogram = _make_meter()

        # Pre-inject so we can track the specific objects
        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            await traced_llm_call("groq", "llama3", [{"role": "user", "content": "x"}], call_fn)

        # _llm_calls.add(1, ...) was called
        tracing_module._llm_calls.add.assert_called_once_with(1, {"provider": "groq", "model": "llama3"})
        # _llm_duration.record(ms, ...) was called
        tracing_module._llm_duration.record.assert_called_once()
        record_args = tracing_module._llm_duration.record.call_args
        assert record_args[0][1] == {"provider": "groq", "model": "llama3"}
        assert record_args[0][0] >= 0  # duration_ms is non-negative


# ---------------------------------------------------------------------------
# Tests for traced_llm_call() — failure path
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
class TestTracedLlmCallFailure:
    def setup_method(self):
        _reset_metrics()

    async def test_re_raises_exception(self):
        call_fn = AsyncMock(side_effect=RuntimeError("timeout"))
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            with pytest.raises(RuntimeError, match="timeout"):
                await traced_llm_call("openai", "gpt-4", [{"role": "user", "content": "hi"}], call_fn)

    async def test_records_error_on_span(self):
        call_fn = AsyncMock(side_effect=ValueError("bad request"))
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            with pytest.raises(ValueError):
                await traced_llm_call("anthropic", "claude-3", [], call_fn)

        set_attr_calls = {c[0][0]: c[0][1] for c in span.set_attribute.call_args_list}
        assert "llm.error" in set_attr_calls
        assert "bad request" in set_attr_calls["llm.error"]

    async def test_sets_span_status_error(self):
        from opentelemetry.trace import StatusCode
        call_fn = AsyncMock(side_effect=ConnectionError("network down"))
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            with pytest.raises(ConnectionError):
                await traced_llm_call("openai", "gpt-4", [], call_fn)

        span.set_status.assert_called_once()
        status_args = span.set_status.call_args[0]
        assert status_args[0] == StatusCode.ERROR

    async def test_increments_error_counter(self):
        call_fn = AsyncMock(side_effect=TimeoutError("too slow"))
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            with pytest.raises(TimeoutError):
                await traced_llm_call("groq", "llama3", [], call_fn)

        tracing_module._llm_errors.add.assert_called_once_with(
            1,
            {"provider": "groq", "model": "llama3", "error": "TimeoutError"},
        )

    async def test_does_not_increment_calls_counter_on_error(self):
        """llm.calls counter must NOT be incremented; only llm.errors is."""
        call_fn = AsyncMock(side_effect=RuntimeError("boom"))
        span = _make_span()
        tracer = _make_tracer(span)
        meter, _, _ = _make_meter()

        with patch("life_core.tracing.get_tracer", return_value=tracer), \
             patch("life_core.tracing.get_meter", return_value=meter):
            with pytest.raises(RuntimeError):
                await traced_llm_call("openai", "gpt-4", [], call_fn)

        # _llm_errors was called; _llm_calls was not.
        # Both are MagicMocks but they are distinct objects (create_counter returns
        # the same counter mock for both calls by default — so we inspect call args).
        add_calls = tracing_module._llm_calls.add.call_args_list
        # If _llm_calls is the same object as _llm_errors (both returned from the
        # same mock), we verify that none of the add() calls used the "success" signature
        # (i.e. no call with only provider+model and without "error" key).
        success_calls = [
            c for c in add_calls
            if "error" not in c[0][1]  # second positional arg is the attributes dict
        ]
        assert success_calls == [], f"llm.calls counter was incremented on error path: {success_calls}"
