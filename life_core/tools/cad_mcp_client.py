"""Async client for the cad-mcp ``read_partial_sch`` tool (Sprint 2 P2B).

This module talks to the ``cad-mcp`` FastMCP server over the **JSON-RPC**
streamable-http transport, using the official ``mcp`` Python SDK
(``mcp.client.streamable_http`` + ``mcp.ClientSession``). FastMCP 3.x does
**not** expose REST-style ``POST /tools/<name>`` endpoints — every HTTP
transport speaks JSON-RPC 2.0 over a single ``/mcp`` endpoint after an
``initialize`` handshake. An earlier REST-flavoured client (commit
``a48ddd2`` on ``feat/inner-trace-schema``) was wrong by construction; this
file replaces it.

Server-side contract (see ``cad-mcp/src/cad_mcp/server.py``):

    @mcp.tool()
    async def read_partial_sch(deliverable_slug: str, version: int) -> dict:
        ...

    Returns ``{"sch_text": str, "bom": list[{reference, value, footprint}],
    "net_count": int}`` — or the same dict plus ``"error": str`` when the
    schematic exists but kiutils failed to parse it (partial success).

Failure handling matches the legacy contract: any timeout, transport error,
JSON shape error, or missing-key result returns ``None`` and logs a warning.
The caller (``life_core.generators.kicad_generator``) falls back to the
``human_feedback`` reprompt path when ``None`` is returned, so the generator
stays usable when cad-mcp is offline.

URL resolution order:
    1. ``CAD_MCP_HTTP_URL`` env var (overrides everything; trailing
       ``/mcp`` is appended if missing).
    2. ``life_core.providers.registry.get_mcp_catalog()`` entry where
       ``name == "cad"``, fields ``http_url`` then ``url``. Currently a
       TODO — origin/main has no such helper. Wired defensively so it
       picks up automatically once the registry adds it.
    3. ``http://cad-mcp:8022/mcp`` default (Docker compose service name).
"""
from __future__ import annotations

import logging
import os
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://cad-mcp:8022/mcp"
DEFAULT_TIMEOUT_S = 5.0
_REQUIRED_KEYS = frozenset({"sch_text", "bom", "net_count"})


def _normalise_url(url: str) -> str:
    """Ensure the URL ends with ``/mcp`` (the FastMCP streamable-http path)."""
    url = url.rstrip("/")
    if not url.endswith("/mcp"):
        url = f"{url}/mcp"
    return url


def _resolve_base_url() -> str:
    """Resolve the cad-mcp endpoint URL.

    Resolution order is documented at module level. This function never
    raises — registry lookup failures are logged and fall through to the
    default. The returned URL always points at the FastMCP ``/mcp``
    JSON-RPC endpoint.
    """
    env_override = os.environ.get("CAD_MCP_HTTP_URL")
    if env_override:
        return _normalise_url(env_override)
    # TODO: integrate with mcp_catalog once life_core.providers.registry
    # exposes get_mcp_catalog() on main. Defensive import keeps origin/main
    # green today and auto-activates when the helper lands.
    try:
        from life_core.providers.registry import (  # type: ignore[attr-defined]
            get_mcp_catalog,
        )

        for entry in get_mcp_catalog() or []:
            if entry.get("name") == "cad":
                url = entry.get("http_url") or entry.get("url")
                if url:
                    return _normalise_url(str(url))
    except Exception as exc:  # noqa: BLE001 — resolver must never raise
        logger.debug("cad-mcp catalog lookup unavailable: %s", exc)
    return DEFAULT_BASE_URL


def _extract_tool_payload(result: Any) -> dict[str, Any] | None:
    """Pull the tool's structured dict out of an MCP ``CallToolResult``.

    FastMCP returns structured results in ``structuredContent`` when the
    tool's return annotation is a dict. Older / minimal servers may only
    populate ``content`` with a JSON-encoded ``TextContent`` block; we try
    both. ``isError`` short-circuits to ``None``.
    """
    if getattr(result, "isError", False):
        logger.warning(
            "cad-mcp read_partial_sch returned isError=True: %s",
            getattr(result, "content", None),
        )
        return None

    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP sometimes wraps the dict as {"result": {...}} when the
        # tool return type is annotated as a bare ``dict``. Unwrap if so.
        if (
            "result" in structured
            and isinstance(structured["result"], dict)
            and not _REQUIRED_KEYS.issubset(structured.keys())
        ):
            return structured["result"]
        return structured

    # Fallback: try to JSON-decode the first text content block.
    content = getattr(result, "content", None) or []
    for block in content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            import json

            decoded = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(decoded, dict):
            return decoded
    return None


async def read_partial_sch(
    deliverable_slug: str,
    version: int,
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Fetch parsed schematic for ``(slug, version)`` from cad-mcp.

    Performs a full MCP handshake (``streamablehttp_client`` → ``ClientSession``
    → ``initialize`` → ``call_tool``) for each invocation. The response is
    validated to contain ``sch_text``, ``bom`` and ``net_count`` keys; an
    optional ``error`` key signals a non-fatal kiutils parse failure (the raw
    text is still returned so the LLM can read it).

    Returns the response dict on success, ``None`` on **any** failure
    (timeout, MCP protocol error, JSON shape error, missing keys). Never
    raises — matches the legacy synchronous contract.
    """
    url = _normalise_url(base_url) if base_url else _resolve_base_url()
    arguments = {"deliverable_slug": deliverable_slug, "version": version}
    try:
        async with streamablehttp_client(url, timeout=timeout_s) as (
            read_stream,
            write_stream,
            _get_session_id,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(
                    "read_partial_sch", arguments=arguments
                )
    except Exception as exc:  # noqa: BLE001 — never raise out of client
        logger.warning(
            "cad-mcp read_partial_sch transport error (%s): %s",
            type(exc).__name__,
            exc,
        )
        return None

    data = _extract_tool_payload(result)
    if data is None:
        logger.warning("cad-mcp read_partial_sch returned no structured payload")
        return None
    if not _REQUIRED_KEYS.issubset(data.keys()):
        logger.warning(
            "cad-mcp read_partial_sch missing required keys (got %s)",
            sorted(data.keys()),
        )
        return None
    return data


def format_partial_read_for_prompt(data: dict[str, Any]) -> str:
    """Render a ``read_partial_sch`` result as a compact prompt-ready string.

    Capped at 20 BOM rows so the block stays small enough to drop into a
    reprompt alongside constraints and spec excerpts. Empty BOM produces a
    single summary line; ``error`` (kiutils partial-failure marker) is
    surfaced verbatim so the LLM can react to it.
    """
    bom = data.get("bom") or []
    net_count = data.get("net_count", 0)
    error = data.get("error")
    lines = [
        f"kiutils parsed {len(bom)} component(s) and {net_count} net(s) "
        "from the previous attempt.",
    ]
    if error:
        lines.append(f"kiutils parse warning: {error}")
    if bom:
        lines.append("Components:")
        for item in bom[:20]:
            ref = item.get("reference", "?")
            val = item.get("value", "?")
            fp = item.get("footprint", "")
            lines.append(f"  - {ref}: {val} [{fp}]")
        if len(bom) > 20:
            lines.append(f"  ... ({len(bom) - 20} more)")
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_TIMEOUT_S",
    "format_partial_read_for_prompt",
    "read_partial_sch",
]
