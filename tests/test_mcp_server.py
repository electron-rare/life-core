"""Tests for MCP server tools."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock


def test_mcp_server_registers_tools():
    from life_core.mcp_server import mcp
    tool_names = [t.name for t in mcp._tool_manager.list_tools()]
    assert "finefab_rag_search" in tool_names
    assert "finefab_chat_complete" in tool_names
    assert "finefab_infra_status" in tool_names
    assert "finefab_cad_drc" in tool_names
    assert "finefab_cad_export" in tool_names
    assert "finefab_flows_list" in tool_names


@pytest.mark.asyncio
async def test_rag_search_formatted():
    mock_results = [{"content": "STM32 GPIO", "score": 0.92, "document_name": "doc1"}]
    with patch("life_core.docstore_client.search_docstore", new_callable=AsyncMock, return_value=mock_results):
        # patch at the source so lazy import picks it up
        import life_core.docstore_client as dc
        original = dc.search_docstore
        dc.search_docstore = AsyncMock(return_value=mock_results)
        try:
            from life_core.mcp_server import finefab_rag_search
            result = await finefab_rag_search(query="STM32", top_k=3)
        finally:
            dc.search_docstore = original
    assert "STM32 GPIO" in result
    assert "0.92" in result


@pytest.mark.asyncio
async def test_rag_search_empty():
    import life_core.docstore_client as dc
    original = dc.search_docstore
    dc.search_docstore = AsyncMock(return_value=[])
    try:
        from life_core.mcp_server import finefab_rag_search
        result = await finefab_rag_search(query="nothing", top_k=3)
    finally:
        dc.search_docstore = original
    assert "No results" in result


@pytest.mark.asyncio
async def test_rag_ingest():
    with patch("life_core.mcp_server.httpx.AsyncClient") as mock_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock(status_code=200)
        mock_resp.json.return_value = {"files_indexed": 8}
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_cls.return_value = mock_client
        from life_core.mcp_server import finefab_rag_ingest
        result = await finefab_rag_ingest(repo_url="https://github.com/test/repo")
    assert "8 files" in result


@pytest.mark.asyncio
async def test_infra_alerts_empty():
    import life_core.ws_alerts as wa
    original = getattr(wa, "_collect_alerts", None)
    wa._collect_alerts = AsyncMock(return_value=[])
    try:
        from life_core.mcp_server import finefab_infra_alerts
        result = await finefab_infra_alerts()
    finally:
        if original is not None:
            wa._collect_alerts = original
    assert "No active alerts" in result
