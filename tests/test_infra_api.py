"""Tests for infra API."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import FastAPI
from life_core.infra_api import infra_router


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(infra_router)
    return TestClient(app)


def test_containers(client):
    response = client.get("/infra/containers")
    assert response.status_code == 200
    containers = response.json()["containers"]
    assert len(containers) > 0
    assert any(c["name"] == "life-core" for c in containers)


def test_storage(client):
    response = client.get("/infra/storage")
    assert response.status_code == 200
    data = response.json()
    assert "redis" in data
    assert "qdrant" in data


def test_network(client):
    response = client.get("/infra/network")
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# list_containers — stats parsing (lines 51-71)
# ---------------------------------------------------------------------------

def _make_stats_payload(cpu_total=500_000_000, precpu_total=400_000_000,
                        sys_cpu=2_000_000_000, presys_cpu=1_000_000_000,
                        percpu=None, mem_usage=104857600, mem_limit=1073741824):
    """Build a minimal Docker stats JSON payload."""
    return {
        "cpu_stats": {
            "cpu_usage": {
                "total_usage": cpu_total,
                "percpu_usage": percpu if percpu is not None else [0, 0, 0, 0],
            },
            "system_cpu_usage": sys_cpu,
        },
        "precpu_stats": {
            "cpu_usage": {"total_usage": precpu_total},
            "system_cpu_usage": presys_cpu,
        },
        "memory_stats": {
            "usage": mem_usage,
            "limit": mem_limit,
        },
    }


def _make_container_json(cid="abc123", name="mycontainer", state="running",
                         status="Up 2 hours (healthy)", created=0):
    return {
        "Id": cid,
        "Names": [f"/{name}"],
        "Image": "myimage:latest",
        "State": state,
        "Status": status,
        "Created": created,
    }


@pytest.mark.anyio
async def test_list_containers_stats_parsing(client):
    """CPU% and memory are computed correctly from Docker stats."""
    container_json = [_make_container_json()]
    stats_payload = _make_stats_payload()

    containers_response = MagicMock()
    containers_response.status_code = 200
    containers_response.json.return_value = container_json
    containers_response.raise_for_status = MagicMock()

    stats_response = MagicMock()
    stats_response.status_code = 200
    stats_response.json.return_value = stats_payload

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[containers_response, stats_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/containers")

    assert response.status_code == 200
    containers = response.json()["containers"]
    assert len(containers) == 1
    c = containers[0]
    assert c["name"] == "mycontainer"
    assert c["health"] == "healthy"
    # cpu_delta=100M, sys_delta=1000M, num_cpus=4 → 4*100/1000*100 = 40.0%
    assert c["cpu_percent"] == 40.0
    # mem_usage=100MB
    assert c["memory_mb"] == pytest.approx(100.0, abs=1.0)
    assert c["memory_limit_mb"] == pytest.approx(1024.0, abs=1.0)


@pytest.mark.anyio
async def test_list_containers_unhealthy_status(client):
    """Health string is 'unhealthy' when status contains (unhealthy)."""
    container_json = [_make_container_json(status="Up 5 min (unhealthy)")]
    stats_payload = _make_stats_payload()

    containers_response = MagicMock()
    containers_response.status_code = 200
    containers_response.json.return_value = container_json
    containers_response.raise_for_status = MagicMock()

    stats_response = MagicMock()
    stats_response.status_code = 200
    stats_response.json.return_value = stats_payload

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[containers_response, stats_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/containers")

    assert response.status_code == 200
    c = response.json()["containers"][0]
    assert c["health"] == "unhealthy"


@pytest.mark.anyio
async def test_list_containers_stats_zero_sys_delta(client):
    """cpu_pct stays 0.0 when system_cpu_usage delta is zero."""
    container_json = [_make_container_json()]
    # sys_delta == 0 → no cpu calculation
    stats_payload = _make_stats_payload(sys_cpu=1_000_000_000, presys_cpu=1_000_000_000)

    containers_response = MagicMock()
    containers_response.status_code = 200
    containers_response.json.return_value = container_json
    containers_response.raise_for_status = MagicMock()

    stats_response = MagicMock()
    stats_response.status_code = 200
    stats_response.json.return_value = stats_payload

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[containers_response, stats_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/containers")

    assert response.status_code == 200
    c = response.json()["containers"][0]
    assert c["cpu_percent"] == 0.0


@pytest.mark.anyio
async def test_list_containers_stats_fetch_failure(client):
    """Stats fetch failure is gracefully ignored; container still listed."""
    container_json = [_make_container_json()]

    containers_response = MagicMock()
    containers_response.status_code = 200
    containers_response.json.return_value = container_json
    containers_response.raise_for_status = MagicMock()

    mock_client = AsyncMock()
    # stats call raises
    mock_client.get = AsyncMock(side_effect=[containers_response, Exception("timeout")])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/containers")

    assert response.status_code == 200
    containers = response.json()["containers"]
    assert len(containers) == 1
    assert containers[0]["cpu_percent"] == 0.0


# ---------------------------------------------------------------------------
# list_containers — Docker-unreachable fallback (lines 84-93)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_list_containers_docker_unreachable(client):
    """When Docker socket is unreachable, the fallback list is returned."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/containers")

    assert response.status_code == 200
    containers = response.json()["containers"]
    names = [c["name"] for c in containers]
    assert "life-core" in names
    assert "redis" in names
    for c in containers:
        assert c["status"] == "unknown"
        assert c["error"] == "docker_unreachable"


# ---------------------------------------------------------------------------
# storage_stats — error paths (lines 116-117, 134-135)
# ---------------------------------------------------------------------------

def test_storage_redis_error(client):
    """Redis error is captured and status=error returned."""
    with patch("redis.from_url", side_effect=Exception("redis connection refused")):
        response = client.get("/infra/storage")

    assert response.status_code == 200
    data = response.json()
    assert data["redis"]["status"] == "error"
    assert "redis connection refused" in data["redis"]["error"]


@pytest.mark.anyio
async def test_storage_qdrant_error(client):
    """Qdrant HTTP failure is captured and status=error returned."""
    # Redis succeeds with a minimal mock so we can isolate Qdrant
    mock_redis = MagicMock()
    mock_redis.info.return_value = {"used_memory_human": "1M"}
    mock_redis.info.side_effect = None
    mock_redis.dbsize.return_value = 0

    def redis_info(section):
        if section == "memory":
            return {"used_memory_human": "1M"}
        if section == "clients":
            return {"connected_clients": 1}
        return {}

    mock_redis.info = MagicMock(side_effect=redis_info)
    mock_redis.dbsize.return_value = 5
    mock_redis.close = MagicMock()

    mock_httpx_client = AsyncMock()
    mock_httpx_client.get = AsyncMock(side_effect=Exception("qdrant unreachable"))
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    with patch("redis.from_url", return_value=mock_redis), \
         patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_httpx_client):
        response = client.get("/infra/storage")

    assert response.status_code == 200
    data = response.json()
    assert data["qdrant"]["status"] == "error"
    assert "qdrant unreachable" in data["qdrant"]["error"]


@pytest.mark.anyio
async def test_storage_qdrant_success(client):
    """Qdrant 200 response with collections is parsed correctly."""
    mock_redis = MagicMock()

    def redis_info_success(section):
        if section == "memory":
            return {"used_memory_human": "3M"}
        return {"connected_clients": 2}

    mock_redis.info = MagicMock(side_effect=redis_info_success)
    mock_redis.dbsize.return_value = 12
    mock_redis.close = MagicMock()

    qdrant_response = MagicMock()
    qdrant_response.status_code = 200
    qdrant_response.json.return_value = {
        "result": {"collections": [{"name": "docs"}, {"name": "chunks"}]}
    }

    mock_httpx_client = AsyncMock()
    mock_httpx_client.get = AsyncMock(return_value=qdrant_response)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    with patch("redis.from_url", return_value=mock_redis), \
         patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_httpx_client):
        response = client.get("/infra/storage")

    assert response.status_code == 200
    data = response.json()
    assert data["redis"]["status"] == "connected"
    assert data["redis"]["keys"] == 12
    assert data["qdrant"]["status"] == "connected"
    assert data["qdrant"]["collections"] == 2
    assert "docs" in data["qdrant"]["collection_names"]


@pytest.mark.anyio
async def test_storage_qdrant_non_200(client):
    """Qdrant non-200 status is captured."""
    mock_redis = MagicMock()

    def redis_info(section):
        if section == "memory":
            return {"used_memory_human": "2M"}
        return {"connected_clients": 0}

    mock_redis.info = MagicMock(side_effect=redis_info)
    mock_redis.dbsize.return_value = 0
    mock_redis.close = MagicMock()

    qdrant_response = MagicMock()
    qdrant_response.status_code = 503

    mock_httpx_client = AsyncMock()
    mock_httpx_client.get = AsyncMock(return_value=qdrant_response)
    mock_httpx_client.__aenter__ = AsyncMock(return_value=mock_httpx_client)
    mock_httpx_client.__aexit__ = AsyncMock(return_value=False)

    with patch("redis.from_url", return_value=mock_redis), \
         patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_httpx_client):
        response = client.get("/infra/storage")

    assert response.status_code == 200
    data = response.json()
    assert data["qdrant"]["status"] == "error"
    assert data["qdrant"]["code"] == 503


# ---------------------------------------------------------------------------
# network_status — service-down scenarios (lines 146-185)
# ---------------------------------------------------------------------------

@pytest.mark.anyio
async def test_network_embed_server_down(client, monkeypatch):
    """embed_server down when request raises."""
    monkeypatch.setenv("EMBED_URL", "http://tower:11437")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["embed_server"]["status"] == "down"
    assert data["embed_server"]["url"] == "http://tower:11437"


@pytest.mark.anyio
async def test_network_embed_server_up(client, monkeypatch):
    """embed_server up when /health responds 200."""
    monkeypatch.setenv("EMBED_URL", "http://tower:11437")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)
    monkeypatch.delenv("LOCAL_LLM_URL", raising=False)

    health_response = MagicMock()
    health_response.status_code = 200

    jaeger_response = MagicMock()
    jaeger_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[health_response, jaeger_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["embed_server"]["status"] == "up"


@pytest.mark.anyio
async def test_network_local_llm_down(client, monkeypatch):
    """llm_local down when request raises."""
    monkeypatch.delenv("EMBED_URL", raising=False)
    monkeypatch.setenv("LOCAL_LLM_URL", "http://tower:8080")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_local"]["status"] == "down"


@pytest.mark.anyio
async def test_network_local_llm_up(client, monkeypatch):
    """llm_local up when /health responds 200."""
    monkeypatch.delenv("EMBED_URL", raising=False)
    monkeypatch.setenv("LOCAL_LLM_URL", "http://tower:8080")
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    health_response = MagicMock()
    health_response.status_code = 200

    jaeger_response = MagicMock()
    jaeger_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[health_response, jaeger_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["llm_local"]["status"] == "up"


@pytest.mark.anyio
async def test_network_vllm_down(client, monkeypatch):
    """vllm_gpu down when request raises."""
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_REMOTE_URL", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://kxkm-ai:8000")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["vllm_gpu"]["status"] == "down"


@pytest.mark.anyio
async def test_network_vllm_up(client, monkeypatch):
    """vllm_gpu up with model list when /health and /v1/models succeed."""
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_REMOTE_URL", raising=False)
    monkeypatch.setenv("VLLM_BASE_URL", "http://kxkm-ai:8000")

    health_response = MagicMock()
    models_response = MagicMock()
    models_response.json.return_value = {"data": [{"id": "Qwen/Qwen2-14B-Instruct-AWQ"}]}

    jaeger_response = MagicMock()
    jaeger_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=[health_response, models_response, jaeger_response])
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["vllm_gpu"]["status"] == "up"
    assert "Qwen/Qwen2-14B-Instruct-AWQ" in data["vllm_gpu"]["models"]


@pytest.mark.anyio
async def test_network_jaeger_down(client, monkeypatch):
    """jaeger down when request raises."""
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_REMOTE_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("jaeger unreachable"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["jaeger"]["status"] == "down"


@pytest.mark.anyio
async def test_network_jaeger_up(client, monkeypatch):
    """jaeger up when /api/services returns 200."""
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_REMOTE_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    jaeger_response = MagicMock()
    jaeger_response.status_code = 200

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=jaeger_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["jaeger"]["status"] == "up"


@pytest.mark.anyio
async def test_network_jaeger_non_200(client, monkeypatch):
    """jaeger down when /api/services returns non-200."""
    monkeypatch.delenv("OLLAMA_URL", raising=False)
    monkeypatch.delenv("OLLAMA_REMOTE_URL", raising=False)
    monkeypatch.delenv("VLLM_BASE_URL", raising=False)

    jaeger_response = MagicMock()
    jaeger_response.status_code = 503

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=jaeger_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/network")

    assert response.status_code == 200
    data = response.json()
    assert data["jaeger"]["status"] == "down"


# ---------------------------------------------------------------------------
# deploy — endpoint (lines 202-230)
# ---------------------------------------------------------------------------

def _deploy_payload(service="life-core", image="ghcr.io/org/life-core:latest"):
    return {"service": service, "image": image}


def test_deploy_invalid_token(client, monkeypatch):
    """deploy returns 403 with wrong token."""
    monkeypatch.setenv("DEPLOY_TOKEN", "secret-token")
    response = client.post(
        "/infra/deploy",
        json=_deploy_payload(),
        headers={"x-deploy-token": "wrong-token"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid deploy token"


def test_deploy_missing_token(client):
    """deploy returns 422 when token header is absent."""
    response = client.post("/infra/deploy", json=_deploy_payload())
    assert response.status_code == 422


def test_deploy_container_not_found(client, monkeypatch):
    """deploy returns 404 when container not found after image pull."""
    monkeypatch.setenv("DEPLOY_TOKEN", "secret-token")

    mock_docker = MagicMock()
    mock_docker.images.pull = MagicMock()
    mock_docker.containers.list.return_value = []

    with patch("life_core.infra_api.docker.from_env", return_value=mock_docker):
        response = client.post(
            "/infra/deploy",
            json=_deploy_payload(service="nonexistent"),
            headers={"x-deploy-token": "secret-token"},
        )

    assert response.status_code == 404
    assert "nonexistent" in response.json()["detail"]


def test_deploy_success(client, monkeypatch):
    """deploy returns deployed status when container is found and restarted."""
    monkeypatch.setenv("DEPLOY_TOKEN", "secret-token")

    mock_container = MagicMock()
    mock_container.attrs = {
        "Config": {"Env": ["FOO=bar"]},
        "HostConfig": {"PortBindings": {"8000/tcp": [{"HostPort": "8000"}]}},
        "NetworkSettings": {"Networks": {"bridge": {}}},
    }
    mock_container.stop = MagicMock()
    mock_container.remove = MagicMock()

    mock_docker = MagicMock()
    mock_docker.images.pull = MagicMock()
    mock_docker.containers.list.return_value = [mock_container]
    mock_docker.containers.run = MagicMock()

    with patch("life_core.infra_api.docker.from_env", return_value=mock_docker):
        response = client.post(
            "/infra/deploy",
            json=_deploy_payload(service="life-core", image="ghcr.io/org/life-core:v2"),
            headers={"x-deploy-token": "secret-token"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deployed"
    assert data["service"] == "life-core"
    assert data["image"] == "ghcr.io/org/life-core:v2"
    mock_container.stop.assert_called_once_with(timeout=10)
    mock_container.remove.assert_called_once()
    mock_docker.containers.run.assert_called_once()


def test_deploy_docker_pull_failure(monkeypatch):
    """deploy propagates exception when docker image pull fails."""
    monkeypatch.setenv("DEPLOY_TOKEN", "secret-token")

    app = FastAPI()
    app.include_router(infra_router)
    # raise_server_exceptions=False → unhandled server errors return 500 instead of re-raising
    test_client = TestClient(app, raise_server_exceptions=False)

    mock_docker = MagicMock()
    mock_docker.images.pull.side_effect = Exception("pull failed: image not found")

    with patch("life_core.infra_api.docker.from_env", return_value=mock_docker):
        response = test_client.post(
            "/infra/deploy",
            json=_deploy_payload(),
            headers={"x-deploy-token": "secret-token"},
        )

    assert response.status_code == 500


def test_list_containers_uses_env_compose_project(client, monkeypatch):
    """Le filtre label doit lire F4L_COMPOSE_PROJECT (pas la valeur hardcodée)."""
    monkeypatch.setenv("F4L_COMPOSE_PROJECT", "my-custom-project")

    captured = {}

    async def fake_get(url, params=None, *args, **kwargs):
        captured["params"] = params
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = []
        resp.raise_for_status = MagicMock()
        return resp

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=fake_get)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.infra_api.httpx.AsyncClient", return_value=mock_client):
        response = client.get("/infra/containers")

    assert response.status_code == 200
    filters_str = captured["params"]["filters"]
    assert "my-custom-project" in filters_str
    assert "finefab-life" not in filters_str
