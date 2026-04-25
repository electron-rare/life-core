"""Sprint 2 P2B — cad-mcp registered in the MCP catalog with read_partial_sch."""


def test_cad_mcp_entry_present():
    from life_core.providers.registry import get_mcp_catalog

    catalog = get_mcp_catalog()
    entry = next((e for e in catalog if e.get("name") == "cad"), None)
    assert entry is not None, "cad MCP server not registered"
    assert entry["transport"] == "sse"
    assert entry["url"].endswith("/sse")


def test_cad_mcp_entry_carries_http_url_8022():
    from life_core.providers.registry import get_mcp_catalog

    catalog = get_mcp_catalog()
    entry = next(e for e in catalog if e.get("name") == "cad")
    assert entry["http_url"].endswith(":8022")


def test_cad_mcp_capabilities_include_read_partial_sch():
    from life_core.providers.registry import get_mcp_catalog

    catalog = get_mcp_catalog()
    entry = next(e for e in catalog if e.get("name") == "cad")
    assert "read_partial_sch" in entry["capabilities"]
