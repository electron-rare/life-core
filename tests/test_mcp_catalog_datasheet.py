"""V1.8 Wave B axes 1+6 — datasheet-mcp registered in MCP catalog."""


def test_datasheet_mcp_entry_present():
    from life_core.providers.registry import get_mcp_catalog

    catalog = get_mcp_catalog()
    entry = next(
        (e for e in catalog if e.get("name") == "datasheet"), None
    )
    assert entry is not None, "datasheet MCP server not registered"
    assert entry["transport"] == "sse"
    assert entry["url"].endswith("/sse")


def test_datasheet_mcp_entry_carries_http_url():
    from life_core.providers.registry import get_mcp_catalog

    catalog = get_mcp_catalog()
    entry = next(e for e in catalog if e.get("name") == "datasheet")
    assert entry["http_url"].endswith(":8022") or (
        "/datasheets" in entry["http_url"]
    )
