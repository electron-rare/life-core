"""Thin HTTP client for the cad-mcp ``read_partial_sch`` tool (Sprint 2 P2B).

Sprint 2 P2A landed the MCP-side tool in ``cad-mcp`` (see
``cad-mcp/src/cad_mcp/server.py::_read_partial_sch_impl``). This module gives
``life_core.generators.kicad_generator`` a backend-agnostic way to fetch the
parsed schematic (raw text + BOM + net count) for the most recent hardware
artifact version of a deliverable, so the auto-reprompt loop can inject
introspection into the next prompt.

We intentionally avoid the LiteLLM tool-calling round trip in V1: the existing
generator loop already orchestrates the retry, and threading tool calls through
``litellm.completion`` would force a heavy provider-side refactor. The same
*observable* behaviour (LLM sees what kiutils built before re-emitting JSON) is
achieved by enriching ``ctx.prompt_vars["partial_read"]`` between attempts; the
``kicad.j2`` template renders that block when present.

Endpoint convention (matches ``cad-mcp/src/cad_mcp/server.py``):
    POST {base_url}/tools/read_partial_sch
        body: {"deliverable_slug": str, "version": int}
        200: {"sch_text": str, "bom": list[dict], "net_count": int,
              "error": str | None}

Failures (non-2xx, network error, missing keys) return ``None`` — the caller
falls back to the legacy ``human_feedback`` reprompt path so the generator
remains usable when cad-mcp is offline.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://cad-mcp:8022"
DEFAULT_TIMEOUT_S = 5.0


def _resolve_base_url() -> str:
    """Resolve the cad-mcp base URL from the MCP catalog or env override.

    Resolution order: ``CAD_MCP_HTTP_URL`` env var, then the
    ``get_mcp_catalog()`` ``cad`` entry, then ``DEFAULT_BASE_URL``. Any
    failure to query the registry falls back silently — the cad-mcp client
    must never raise inside the resolver.
    """
    env_override = os.environ.get("CAD_MCP_HTTP_URL")
    if env_override:
        return env_override.rstrip("/")
    try:
        from life_core.providers.registry import get_mcp_catalog

        for entry in get_mcp_catalog():
            if entry.get("name") == "cad":
                url = (
                    entry.get("http_url") or entry.get("url") or DEFAULT_BASE_URL
                )
                return str(url).rstrip("/")
    except Exception as exc:  # noqa: BLE001 — never raise out of resolver
        logger.warning("cad-mcp catalog lookup failed: %s", exc)
    return DEFAULT_BASE_URL


def read_partial_sch(
    deliverable_slug: str,
    version: int,
    *,
    base_url: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any] | None:
    """Fetch parsed schematic for ``(slug, version)`` from cad-mcp.

    Returns the response dict on success, ``None`` on any failure. The
    response always carries ``sch_text``, ``bom``, ``net_count`` keys; an
    optional ``error`` key signals a non-fatal kiutils parse failure (the
    raw text is still returned).
    """
    url = (base_url or _resolve_base_url()) + "/tools/read_partial_sch"
    payload = {"deliverable_slug": deliverable_slug, "version": version}
    try:
        resp = httpx.post(url, json=payload, timeout=timeout_s)
    except httpx.HTTPError as exc:
        logger.warning("cad-mcp read_partial_sch network error: %s", exc)
        return None
    if resp.status_code != 200:
        logger.warning(
            "cad-mcp read_partial_sch returned %d: %s",
            resp.status_code,
            resp.text[:200],
        )
        return None
    try:
        data = resp.json()
    except ValueError as exc:
        logger.warning("cad-mcp read_partial_sch invalid JSON: %s", exc)
        return None
    if not isinstance(data, dict):
        return None
    required = {"sch_text", "bom", "net_count"}
    if not required.issubset(data.keys()):
        return None
    return data


def format_partial_read_for_prompt(data: dict[str, Any]) -> str:
    """Render ``read_partial_sch`` result as a compact prompt-ready string.

    Kept short so it fits inside a reprompt without crowding the existing
    constraints + spec excerpts.
    """
    bom = data.get("bom", []) or []
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
