"""Tests for life_core.tools.cad_mcp_client — JSON-RPC MCP client."""
from __future__ import annotations

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from life_core.tools import cad_mcp_client
from life_core.tools.cad_mcp_client import (
    DEFAULT_BASE_URL,
    _extract_tool_payload,
    _normalise_url,
    _resolve_base_url,
    format_partial_read_for_prompt,
    read_partial_sch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mk_call_tool_result(
    *,
    structured: dict | None = None,
    text: str | None = None,
    is_error: bool = False,
):
    """Build a duck-typed CallToolResult-shaped object."""
    content = []
    if text is not None:
        content.append(SimpleNamespace(text=text))
    return SimpleNamespace(
        structuredContent=structured,
        content=content,
        isError=is_error,
    )


def _patch_mcp_call(call_tool_return=None, call_tool_side_effect=None,
                    initialize_side_effect=None, transport_side_effect=None):
    """Patch streamablehttp_client + ClientSession in cad_mcp_client.

    Returns a tuple ``(transport_patch, session_patch, call_tool_mock)``
    suitable for use as context managers via ``contextlib.ExitStack`` if
    needed; tests below typically call this once per test.
    """
    call_tool = AsyncMock(
        return_value=call_tool_return, side_effect=call_tool_side_effect
    )
    session = MagicMock()
    session.initialize = AsyncMock(side_effect=initialize_side_effect)
    session.call_tool = call_tool

    @asynccontextmanager
    async def fake_session_cm(read, write):
        yield session

    @asynccontextmanager
    async def fake_transport_cm(url, **kwargs):
        if transport_side_effect:
            raise transport_side_effect
        yield (MagicMock(), MagicMock(), lambda: "session-id")

    transport_patch = patch.object(
        cad_mcp_client, "streamablehttp_client", fake_transport_cm
    )
    session_patch = patch.object(cad_mcp_client, "ClientSession", fake_session_cm)
    return transport_patch, session_patch, call_tool


# ---------------------------------------------------------------------------
# URL resolution
# ---------------------------------------------------------------------------


def test_normalise_url_appends_mcp_path() -> None:
    assert _normalise_url("http://host:8022") == "http://host:8022/mcp"
    assert _normalise_url("http://host:8022/") == "http://host:8022/mcp"
    assert _normalise_url("http://host:8022/mcp") == "http://host:8022/mcp"
    assert _normalise_url("http://host:8022/mcp/") == "http://host:8022/mcp"


def test_resolve_base_url_default(monkeypatch) -> None:
    monkeypatch.delenv("CAD_MCP_HTTP_URL", raising=False)
    assert _resolve_base_url() == DEFAULT_BASE_URL


def test_resolve_base_url_env_override(monkeypatch) -> None:
    monkeypatch.setenv("CAD_MCP_HTTP_URL", "http://override:9999")
    assert _resolve_base_url() == "http://override:9999/mcp"


def test_resolve_base_url_catalog_lookup(monkeypatch) -> None:
    monkeypatch.delenv("CAD_MCP_HTTP_URL", raising=False)
    fake_module = SimpleNamespace(
        get_mcp_catalog=lambda: [
            {"name": "other", "url": "http://nope:1"},
            {"name": "cad", "http_url": "http://cad-from-catalog:8022"},
        ]
    )
    with patch.dict(
        "sys.modules", {"life_core.providers.registry": fake_module}
    ):
        assert _resolve_base_url() == "http://cad-from-catalog:8022/mcp"


def test_resolve_base_url_catalog_url_field_fallback(monkeypatch) -> None:
    monkeypatch.delenv("CAD_MCP_HTTP_URL", raising=False)
    fake_module = SimpleNamespace(
        get_mcp_catalog=lambda: [{"name": "cad", "url": "http://only-url:8022"}]
    )
    with patch.dict(
        "sys.modules", {"life_core.providers.registry": fake_module}
    ):
        assert _resolve_base_url() == "http://only-url:8022/mcp"


def test_resolve_base_url_catalog_failure_falls_back(monkeypatch) -> None:
    monkeypatch.delenv("CAD_MCP_HTTP_URL", raising=False)

    def boom():
        raise RuntimeError("registry exploded")

    fake_module = SimpleNamespace(get_mcp_catalog=boom)
    with patch.dict(
        "sys.modules", {"life_core.providers.registry": fake_module}
    ):
        assert _resolve_base_url() == DEFAULT_BASE_URL


def test_resolve_base_url_env_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("CAD_MCP_HTTP_URL", "http://env-wins:1")
    fake_module = SimpleNamespace(
        get_mcp_catalog=lambda: [
            {"name": "cad", "http_url": "http://catalog-loses:2"}
        ]
    )
    with patch.dict(
        "sys.modules", {"life_core.providers.registry": fake_module}
    ):
        assert _resolve_base_url() == "http://env-wins:1/mcp"


# ---------------------------------------------------------------------------
# _extract_tool_payload
# ---------------------------------------------------------------------------


def test_extract_payload_structured_dict() -> None:
    payload = {"sch_text": "x", "bom": [], "net_count": 0}
    result = _mk_call_tool_result(structured=payload)
    assert _extract_tool_payload(result) == payload


def test_extract_payload_unwraps_result_wrapper() -> None:
    inner = {"sch_text": "x", "bom": [], "net_count": 1}
    result = _mk_call_tool_result(structured={"result": inner})
    assert _extract_tool_payload(result) == inner


def test_extract_payload_keeps_structured_when_keys_present_at_top() -> None:
    # Already-flat structured payload (no spurious unwrap)
    payload = {
        "sch_text": "x",
        "bom": [],
        "net_count": 0,
        "result": "ignored",
    }
    result = _mk_call_tool_result(structured=payload)
    assert _extract_tool_payload(result) == payload


def test_extract_payload_text_fallback() -> None:
    payload = {"sch_text": "y", "bom": [], "net_count": 2}
    result = _mk_call_tool_result(structured=None, text=json.dumps(payload))
    assert _extract_tool_payload(result) == payload


def test_extract_payload_invalid_text_returns_none() -> None:
    result = _mk_call_tool_result(structured=None, text="not-json")
    assert _extract_tool_payload(result) is None


def test_extract_payload_skips_empty_text_blocks() -> None:
    # First block has empty text, second carries the real JSON.
    payload = {"sch_text": "x", "bom": [], "net_count": 0}
    result = SimpleNamespace(
        structuredContent=None,
        content=[
            SimpleNamespace(text=""),
            SimpleNamespace(text=json.dumps(payload)),
        ],
        isError=False,
    )
    assert _extract_tool_payload(result) == payload


def test_extract_payload_text_with_non_dict_json_returns_none() -> None:
    # Top-level JSON is a list, not a dict — not a tool payload.
    result = _mk_call_tool_result(structured=None, text="[1, 2, 3]")
    assert _extract_tool_payload(result) is None


def test_extract_payload_is_error_returns_none() -> None:
    result = _mk_call_tool_result(
        structured={"sch_text": "x", "bom": [], "net_count": 0}, is_error=True
    )
    assert _extract_tool_payload(result) is None


# ---------------------------------------------------------------------------
# read_partial_sch — full async paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_partial_sch_happy_path() -> None:
    payload = {
        "sch_text": "(kicad_sch ...)",
        "bom": [{"reference": "R1", "value": "10k", "footprint": "0603"}],
        "net_count": 4,
    }
    result = _mk_call_tool_result(structured=payload)
    transport_p, session_p, call_tool = _patch_mcp_call(call_tool_return=result)
    with transport_p, session_p:
        out = await read_partial_sch(
            "demo-slug", 1, base_url="http://stub:8022"
        )
    assert out == payload
    call_tool.assert_awaited_once_with(
        "read_partial_sch",
        arguments={"deliverable_slug": "demo-slug", "version": 1},
    )


@pytest.mark.asyncio
async def test_read_partial_sch_partial_failure_returns_payload() -> None:
    payload = {
        "sch_text": "raw",
        "bom": [],
        "net_count": 0,
        "error": "kiutils parse failed: bad token",
    }
    result = _mk_call_tool_result(structured=payload)
    transport_p, session_p, _ = _patch_mcp_call(call_tool_return=result)
    with transport_p, session_p:
        out = await read_partial_sch("d", 2, base_url="http://stub")
    assert out == payload  # partial failure still bubbles up


@pytest.mark.asyncio
async def test_read_partial_sch_transport_error_returns_none() -> None:
    transport_p, session_p, _ = _patch_mcp_call(
        transport_side_effect=ConnectionRefusedError("no server")
    )
    with transport_p, session_p:
        out = await read_partial_sch("d", 1, base_url="http://stub")
    assert out is None


@pytest.mark.asyncio
async def test_read_partial_sch_initialize_error_returns_none() -> None:
    transport_p, session_p, _ = _patch_mcp_call(
        initialize_side_effect=RuntimeError("handshake failed")
    )
    with transport_p, session_p:
        out = await read_partial_sch("d", 1, base_url="http://stub")
    assert out is None


@pytest.mark.asyncio
async def test_read_partial_sch_tool_call_raises_returns_none() -> None:
    transport_p, session_p, _ = _patch_mcp_call(
        call_tool_side_effect=TimeoutError("slow")
    )
    with transport_p, session_p:
        out = await read_partial_sch("d", 1, base_url="http://stub")
    assert out is None


@pytest.mark.asyncio
async def test_read_partial_sch_is_error_returns_none() -> None:
    result = _mk_call_tool_result(
        structured={"sch_text": "x", "bom": [], "net_count": 0}, is_error=True
    )
    transport_p, session_p, _ = _patch_mcp_call(call_tool_return=result)
    with transport_p, session_p:
        out = await read_partial_sch("d", 1, base_url="http://stub")
    assert out is None


@pytest.mark.asyncio
async def test_read_partial_sch_missing_keys_returns_none() -> None:
    # Server responded but omitted ``net_count``
    result = _mk_call_tool_result(structured={"sch_text": "x", "bom": []})
    transport_p, session_p, _ = _patch_mcp_call(call_tool_return=result)
    with transport_p, session_p:
        out = await read_partial_sch("d", 1, base_url="http://stub")
    assert out is None


@pytest.mark.asyncio
async def test_read_partial_sch_no_payload_returns_none() -> None:
    result = _mk_call_tool_result(structured=None, text=None)
    transport_p, session_p, _ = _patch_mcp_call(call_tool_return=result)
    with transport_p, session_p:
        out = await read_partial_sch("d", 1, base_url="http://stub")
    assert out is None


@pytest.mark.asyncio
async def test_read_partial_sch_uses_resolver_when_no_base_url(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CAD_MCP_HTTP_URL", "http://from-env:8022")
    payload = {"sch_text": "x", "bom": [], "net_count": 0}
    result = _mk_call_tool_result(structured=payload)
    seen_url = {}

    @asynccontextmanager
    async def fake_transport(url, **kwargs):
        seen_url["url"] = url
        yield (MagicMock(), MagicMock(), lambda: "sid")

    session = MagicMock()
    session.initialize = AsyncMock()
    session.call_tool = AsyncMock(return_value=result)

    @asynccontextmanager
    async def fake_session(read, write):
        yield session

    with patch.object(cad_mcp_client, "streamablehttp_client", fake_transport), \
            patch.object(cad_mcp_client, "ClientSession", fake_session):
        out = await read_partial_sch("d", 1)
    assert out == payload
    assert seen_url["url"] == "http://from-env:8022/mcp"


# ---------------------------------------------------------------------------
# format_partial_read_for_prompt — boundary sizes
# ---------------------------------------------------------------------------


def test_format_no_components() -> None:
    out = format_partial_read_for_prompt(
        {"sch_text": "", "bom": [], "net_count": 0}
    )
    assert "0 component(s)" in out
    assert "0 net(s)" in out
    assert "Components:" not in out
    assert "more)" not in out


def test_format_one_component() -> None:
    out = format_partial_read_for_prompt(
        {
            "sch_text": "",
            "bom": [{"reference": "R1", "value": "10k", "footprint": "0603"}],
            "net_count": 1,
        }
    )
    assert "1 component(s)" in out
    assert "1 net(s)" in out
    assert "R1: 10k [0603]" in out
    assert "more)" not in out


def test_format_twenty_components_no_overflow_marker() -> None:
    bom = [
        {"reference": f"R{i}", "value": f"{i}k", "footprint": "0603"}
        for i in range(20)
    ]
    out = format_partial_read_for_prompt(
        {"sch_text": "", "bom": bom, "net_count": 20}
    )
    assert "20 component(s)" in out
    assert "R0: 0k [0603]" in out
    assert "R19: 19k [0603]" in out
    assert "more)" not in out


def test_format_thirty_components_truncates_with_marker() -> None:
    bom = [
        {"reference": f"R{i}", "value": f"{i}k", "footprint": "0603"}
        for i in range(30)
    ]
    out = format_partial_read_for_prompt(
        {"sch_text": "", "bom": bom, "net_count": 30}
    )
    assert "30 component(s)" in out
    assert "R0: 0k [0603]" in out
    assert "R19: 19k [0603]" in out
    assert "R20" not in out  # truncated
    assert "... (10 more)" in out


def test_format_propagates_error_field() -> None:
    out = format_partial_read_for_prompt(
        {
            "sch_text": "raw",
            "bom": [],
            "net_count": 0,
            "error": "kiutils parse failed: foo",
        }
    )
    assert "kiutils parse warning: kiutils parse failed: foo" in out


def test_format_handles_missing_keys_gracefully() -> None:
    # No bom / net_count — defaults applied
    out = format_partial_read_for_prompt({"sch_text": ""})
    assert "0 component(s)" in out
    assert "0 net(s)" in out


def test_format_handles_bom_with_partial_fields() -> None:
    out = format_partial_read_for_prompt(
        {
            "sch_text": "",
            "bom": [{"reference": "U1"}, {"value": "100nF"}],
            "net_count": 0,
        }
    )
    assert "U1: ? []" in out
    assert "?: 100nF []" in out
