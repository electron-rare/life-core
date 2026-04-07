"""Extended tests for infra_api — storage, network, deploy endpoints."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.infra_api import infra_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(infra_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /infra/containers
# ---------------------------------------------------------------------------


def test_containers_docker_unreachable_returns_fallback(client):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("docker unavailable"))
    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.get("/infra/containers")
    assert resp.status_code == 200
    data = resp.json()
    assert "containers" in data
    assert len(data["containers"]) > 0


def test_containers_fallback_includes_life_core(client):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("docker unavailable"))
    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.get("/infra/containers")
    names = {c["name"] for c in resp.json()["containers"]}
    assert "life-core" in names


def test_containers_fallback_has_error_field(client):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("timeout"))
    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        resp = client.get("/infra/containers")
    for c in resp.json()["containers"]:
        assert c.get("error") == "docker_unreachable"


# ---------------------------------------------------------------------------
# GET /infra/storage
# ---------------------------------------------------------------------------


def test_storage_redis_error_returns_error_field(client):
    # redis is imported inline in storage_stats(), patch it in sys.modules
    with patch("redis.from_url", side_effect=Exception("connection refused"), create=True):
        resp = client.get("/infra/storage")
    assert resp.status_code == 200
    data = resp.json()
    assert "redis" in data
    assert "qdrant" in data


def test_storage_qdrant_error_returns_error_field(client):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("qdrant down"))

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        with patch("redis.from_url", side_effect=Exception("redis down"), create=True):
            resp = client.get("/infra/storage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["qdrant"]["status"] == "error"


def test_storage_qdrant_success_path(client):
    qdrant_data = {"result": {"collections": [{"name": "life_rag"}, {"name": "embeddings"}]}}
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = qdrant_data

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(return_value=mock_resp)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        with patch("redis.from_url", side_effect=Exception("redis down"), create=True):
            resp = client.get("/infra/storage")

    data = resp.json()
    assert data["qdrant"]["status"] == "connected"
    assert data["qdrant"]["collections"] == 2


# ---------------------------------------------------------------------------
# GET /infra/network
# ---------------------------------------------------------------------------


def test_network_no_env_vars_returns_empty_checks(client):
    env = {"OLLAMA_URL": "", "OLLAMA_REMOTE_URL": "", "VLLM_BASE_URL": ""}
    with patch.dict("os.environ", env, clear=False):
        resp = client.get("/infra/network")
    assert resp.status_code == 200
    data = resp.json()
    # Only jaeger check runs when no URLs set
    assert "jaeger" in data


def test_network_jaeger_down(client):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("jaeger down"))
    env = {"OLLAMA_URL": "", "OLLAMA_REMOTE_URL": "", "VLLM_BASE_URL": ""}
    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        with patch.dict("os.environ", env, clear=False):
            resp = client.get("/infra/network")
    data = resp.json()
    assert data["jaeger"]["status"] == "down"


def test_network_embed_server_error(client):
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
    env = {"EMBED_URL": "http://localhost:11437", "VLLM_BASE_URL": "", "LOCAL_LLM_URL": ""}
    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        with patch.dict("os.environ", env, clear=False):
            resp = client.get("/infra/network")
    data = resp.json()
    assert "embed_server" in data
    assert data["embed_server"]["status"] == "down"


# ---------------------------------------------------------------------------
# POST /infra/deploy
# ---------------------------------------------------------------------------


def test_deploy_requires_token(client):
    resp = client.post("/infra/deploy", json={"service": "life-core", "image": "img:latest"})
    # Missing X-Deploy-Token header → 422 (FastAPI validation)
    assert resp.status_code == 422


def test_deploy_invalid_token_returns_403(client):
    with patch.dict("os.environ", {"DEPLOY_TOKEN": "secret"}):
        with patch("life_core.infra_api.docker") as mock_docker:
            resp = client.post(
                "/infra/deploy",
                json={"service": "life-core", "image": "img:latest"},
                headers={"X-Deploy-Token": "wrong-token"},
            )
    assert resp.status_code == 403
