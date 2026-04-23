"""Tests for monitoring_api: machines inventory from YAML.

Note: the earlier test suite (extract_by_instance / query_prom helpers) was
removed in commit b999a02 when the module migrated to OTel text scrape.
The obsolete tests were pruned in 2026-04-23 as part of the P3 rewrite
that makes /infra/machines read its inventory from a YAML config.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.monitoring_api import monitoring_router


def test_machines_returns_five_hosts(tmp_path, monkeypatch):
    """L'endpoint /infra/machines doit retourner les 5 hôtes déclarés."""
    yaml_content = """
machines:
  - name: electron-server
    ip: via-kxkm-ai
    role: F4L primary host
    specs: {cores: 16, ram_gb: 64, storage_gb: 1000}
    services: [life-core]
  - name: Tower
    ip: 192.168.0.120
    role: Docker
    specs: {cores: 12, ram_gb: 31, storage_gb: 500}
    services: [litellm]
  - name: KXKM-AI
    ip: 100.87.54.119
    role: GPU
    specs: {cores: 28, ram_gb: 62, storage_gb: 2000, gpu: "RTX 4090"}
    services: [vllm]
  - name: VM
    ip: 192.168.0.119
    role: Docker
    specs: {cores: 4, ram_gb: 6.8, storage_gb: 100}
    services: [dify]
  - name: CILS
    ip: 192.168.0.210
    role: Edge
    specs: {cores: 4, ram_gb: 16, storage_gb: 250}
    services: [ollama]
"""
    cfg = tmp_path / "machines.yaml"
    cfg.write_text(yaml_content)
    monkeypatch.setenv("F4L_MACHINES_YAML", str(cfg))

    app = FastAPI()
    app.include_router(monitoring_router)
    client = TestClient(app)

    resp = client.get("/infra/machines")
    assert resp.status_code == 200
    body = resp.json()
    names = [m["name"] for m in body["machines"]]
    assert set(names) == {"electron-server", "Tower", "KXKM-AI", "VM", "CILS"}
    # Chaque machine a les champs de base attendus
    first = body["machines"][0]
    for field in (
        "name",
        "ip",
        "cpu_percent",
        "ram_used_gb",
        "ram_total_gb",
        "disk_used_gb",
        "disk_total_gb",
        "uptime_hours",
    ):
        assert field in first, f"field {field} missing"
