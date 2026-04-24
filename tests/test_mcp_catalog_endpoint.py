"""V1.8 Wave B axes 1+6 — /mcp/catalog surface contract."""
from fastapi.testclient import TestClient

from life_core.api import app


def test_mcp_catalog_surface_lists_datasheet(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "t")
    client = TestClient(app)
    response = client.get(
        "/mcp/catalog", headers={"Authorization": "Bearer t"}
    )
    assert response.status_code == 200
    body = response.json()
    assert "servers" in body
    names = {entry["name"] for entry in body["servers"]}
    assert "datasheet" in names
