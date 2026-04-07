"""MCP server exposing FineFab tools for Goose and other MCP clients."""
from __future__ import annotations
import json
import logging
import os

import httpx
from mcp.server.fastmcp import FastMCP

logger = logging.getLogger("life_core.mcp")
DOCSTORE_URL = os.environ.get("DOCSTORE_URL", "http://100.126.225.111:8200")
CAD_URL = os.environ.get("CAD_GATEWAY_URL", "http://makelife-cad:8001")

mcp = FastMCP("FineFab", stateless_http=True, json_response=True)


# ---------------------------------------------------------------------------
# RAG tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def finefab_rag_search(query: str, top_k: int = 5) -> str:
    """Search FineFab document store for relevant content."""
    from life_core.docstore_client import search_docstore
    results = await search_docstore(query, top_k=top_k)
    if not results:
        return "No results found."
    lines = []
    for r in results:
        score = r.get("score", 0)
        name = r.get("document_name", "unknown")
        content = r.get("content", "")[:300]
        lines.append(f"[{name}] (score={score:.2f}) {content}")
    return "\n---\n".join(lines)


@mcp.tool()
async def finefab_rag_ingest(repo_url: str, branch: str = "main", patterns: str = "*.md,*.py,*.ts") -> str:
    """Index a GitHub/Forgejo repo into the document store."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{DOCSTORE_URL}/ingest/github",
            params={"repo_url": repo_url, "branch": branch, "patterns": patterns},
        )
        if resp.status_code >= 400:
            return f"Ingest failed: HTTP {resp.status_code} - {resp.text[:200]}"
        data = resp.json()
    return f"Indexed {data.get('files_indexed', 0)} files from {repo_url} ({branch})"


# ---------------------------------------------------------------------------
# Chat tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def finefab_chat_complete(message: str, model: str | None = None, use_rag: bool = False) -> str:
    """Send a message to the FineFab LLM router."""
    from life_core.services.chat import ChatService
    svc = ChatService()
    messages = [{"role": "user", "content": message}]
    if use_rag:
        from life_core.docstore_client import augment_with_docstore
        context = await augment_with_docstore(message, top_k=3)
        if context:
            messages.insert(0, {"role": "system", "content": f"Context:\n{context}"})
    result = await svc.chat(messages=messages, model=model)
    return result.get("content", str(result))


# ---------------------------------------------------------------------------
# Monitoring / infra tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def finefab_infra_status() -> str:
    """Get infrastructure status: GPU, machines, containers."""
    from life_core.monitoring_api import gpu_stats, list_machines
    from life_core.infra_api import list_containers
    sections = []
    for label, coro in [("GPU", gpu_stats()), ("Machines", list_machines()), ("Containers", list_containers())]:
        try:
            data = await coro
            if label == "Containers":
                names = [c["name"] for c in data.get("containers", []) if not c.get("error")]
                sections.append(f"{label} ({len(names)}): {', '.join(names)}")
            else:
                sections.append(f"{label}: {json.dumps(data, indent=2)}")
        except Exception as e:
            sections.append(f"{label}: error - {e}")
    return "\n\n".join(sections)


@mcp.tool()
async def finefab_infra_alerts() -> str:
    """Get active infrastructure alerts."""
    from life_core.ws_alerts import _collect_alerts
    alerts = await _collect_alerts()
    if not alerts:
        return "No active alerts."
    return "\n".join(f"[{a['severity']}] {a['title']}: {a['message']}" for a in alerts)


# ---------------------------------------------------------------------------
# CAD tools (proxy to makelife-cad gateway)
# ---------------------------------------------------------------------------


@mcp.tool()
async def finefab_cad_drc(project_path: str) -> str:
    """Run KiCad DRC check via CAD gateway."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{CAD_URL}/api/design/drc", json={"project_path": project_path})
        return json.dumps(resp.json(), indent=2) if resp.status_code < 400 else f"DRC failed: HTTP {resp.status_code}"


@mcp.tool()
async def finefab_cad_bom(project_path: str) -> str:
    """Validate BOM for a KiCad project."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(f"{CAD_URL}/api/design/bom", json={"project_path": project_path})
        return json.dumps(resp.json(), indent=2) if resp.status_code < 400 else f"BOM failed: HTTP {resp.status_code}"


@mcp.tool()
async def finefab_cad_export(project_path: str, output_format: str = "pdf") -> str:
    """Export KiCad schematic or PCB to PDF/SVG/Gerber."""
    if output_format not in ("pdf", "svg", "gerber"):
        return f"Unsupported format: {output_format}. Use pdf, svg, or gerber."
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{CAD_URL}/api/design/export",
            json={"project_path": project_path, "format": output_format},
        )
        if resp.status_code >= 400:
            return f"Export failed: HTTP {resp.status_code}"
        data = resp.json()
        return f"Exported {project_path} as {output_format}: {data.get('output_path', 'unknown')}"


# ---------------------------------------------------------------------------
# Activepieces workflow tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def finefab_flows_list() -> str:
    """List Activepieces automation flows."""
    from life_core.monitoring_api import activepieces_flows
    try:
        data = await activepieces_flows()
        flows = data.get("flows", [])
        if not flows:
            return "No flows found."
        return "\n".join(
            f"- {f['name']}: {f.get('status', '?')} (last: {f.get('last_run_status', 'N/A')})"
            for f in flows
        )
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
async def finefab_flows_trigger(flow_name: str) -> str:
    """Trigger an Activepieces flow by name."""
    from life_core.monitoring_api import activepieces_flows
    try:
        data = await activepieces_flows()
        target = next((f for f in data.get("flows", []) if f["name"] == flow_name), None)
        if not target:
            return f"Flow '{flow_name}' not found."
        webhook_url = target.get("webhook_url")
        if not webhook_url:
            return f"Flow '{flow_name}' has no webhook."
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(webhook_url, json={"triggered_by": "goose"})
        return f"Triggered '{flow_name}': HTTP {resp.status_code}"
    except Exception as e:
        return f"Error: {e}"
