"""Tests for the cad-mcp HTTP client (Sprint 2 P2B)."""
from __future__ import annotations

import httpx
import pytest
import respx

from life_core.tools.cad_mcp_client import (
    DEFAULT_BASE_URL,
    _resolve_base_url,
    format_partial_read_for_prompt,
    read_partial_sch,
)


_HAPPY_BODY = {
    "sch_text": "(kicad_sch (version 20231120))",
    "bom": [
        {"reference": "U1", "value": "AMS1117-3.3", "footprint": "SOT-223-3"},
        {"reference": "C1", "value": "10uF", "footprint": "C_0603"},
    ],
    "net_count": 3,
}


@respx.mock
def test_read_partial_sch_happy_path():
    route = respx.post(
        "http://cad-mcp:8022/tools/read_partial_sch"
    ).mock(return_value=httpx.Response(200, json=_HAPPY_BODY))
    result = read_partial_sch("sensor-node-minimal-hardware", 1)
    assert route.called
    assert result is not None
    assert result["sch_text"].startswith("(kicad_sch")
    assert len(result["bom"]) == 2
    assert result["net_count"] == 3


@respx.mock
def test_read_partial_sch_returns_none_on_500():
    respx.post(
        "http://cad-mcp:8022/tools/read_partial_sch"
    ).mock(return_value=httpx.Response(500, text="boom"))
    assert read_partial_sch("any", 1) is None


@respx.mock
def test_read_partial_sch_returns_none_on_invalid_json():
    respx.post(
        "http://cad-mcp:8022/tools/read_partial_sch"
    ).mock(return_value=httpx.Response(200, text="not json"))
    assert read_partial_sch("any", 1) is None


@respx.mock
def test_read_partial_sch_returns_none_on_missing_keys():
    respx.post(
        "http://cad-mcp:8022/tools/read_partial_sch"
    ).mock(return_value=httpx.Response(200, json={"only": "this"}))
    assert read_partial_sch("any", 1) is None


@respx.mock
def test_read_partial_sch_returns_none_on_network_error():
    respx.post(
        "http://cad-mcp:8022/tools/read_partial_sch"
    ).mock(side_effect=httpx.ConnectError("refused"))
    assert read_partial_sch("any", 1) is None


@respx.mock
def test_read_partial_sch_uses_explicit_base_url():
    route = respx.post(
        "http://override:9999/tools/read_partial_sch"
    ).mock(return_value=httpx.Response(200, json=_HAPPY_BODY))
    result = read_partial_sch("s", 1, base_url="http://override:9999")
    assert route.called
    assert result is not None


def test_resolve_base_url_env_override(monkeypatch):
    monkeypatch.setenv("CAD_MCP_HTTP_URL", "http://envhost:1234/")
    assert _resolve_base_url() == "http://envhost:1234"


def test_resolve_base_url_from_catalog(monkeypatch):
    monkeypatch.delenv("CAD_MCP_HTTP_URL", raising=False)
    url = _resolve_base_url()
    assert url == "http://cad-mcp:8022"


def test_resolve_base_url_default_when_registry_broken(monkeypatch):
    monkeypatch.delenv("CAD_MCP_HTTP_URL", raising=False)
    import life_core.tools.cad_mcp_client as mod

    def _broken_registry():
        from life_core import providers as _p
        raise RuntimeError("registry exploded")

    # Force the import path inside _resolve_base_url to fail by stubbing
    # get_mcp_catalog to raise. Easiest: patch it after import.
    import life_core.providers.registry as registry

    def _raise():
        raise RuntimeError("explode")

    monkeypatch.setattr(registry, "get_mcp_catalog", _raise)
    # _resolve_base_url imports lazily and try/excepts, so it falls back.
    assert mod._resolve_base_url() == DEFAULT_BASE_URL


def test_format_partial_read_for_prompt_includes_components_and_nets():
    text = format_partial_read_for_prompt(_HAPPY_BODY)
    assert "2 component(s)" in text
    assert "3 net(s)" in text
    assert "U1" in text
    assert "AMS1117-3.3" in text
    assert "C1" in text


def test_format_partial_read_for_prompt_handles_kiutils_error():
    body = {
        "sch_text": "garbage",
        "bom": [],
        "net_count": 0,
        "error": "kiutils parse failed: bad token at line 7",
    }
    text = format_partial_read_for_prompt(body)
    assert "0 component(s)" in text
    assert "kiutils parse warning" in text


def test_format_partial_read_for_prompt_truncates_large_bom():
    bom = [
        {"reference": f"R{i}", "value": "10k", "footprint": "0402"}
        for i in range(50)
    ]
    text = format_partial_read_for_prompt(
        {"sch_text": "x", "bom": bom, "net_count": 1}
    )
    assert "(30 more)" in text
