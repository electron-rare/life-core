"""Infrastructure monitoring API endpoints."""

from __future__ import annotations

import logging
import os
import time as _time

import docker
import httpx
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

logger = logging.getLogger("life_core.infra_api")

infra_router = APIRouter(prefix="/infra", tags=["Infra"])


@infra_router.get("/containers")
async def list_containers():
    """List Docker containers via Tower's Docker socket proxy."""
    docker_host = os.environ.get("DOCKER_HOST", "http://host.docker.internal:2375")
    containers = []

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{docker_host}/containers/json?all=1")
            resp.raise_for_status()
            raw_containers = resp.json()

        async with httpx.AsyncClient(timeout=15.0) as client:
            for c in raw_containers:
                cid = c["Id"]
                name = c["Names"][0].lstrip("/") if c.get("Names") else cid[:12]
                image = c.get("Image", "")
                state = c.get("State", "unknown")
                health = c.get("Status", "")
                # Parse "healthy"/"unhealthy" from status string like "Up 2 days (healthy)"
                health_str = "healthy" if "(healthy)" in health else (
                    "unhealthy" if "(unhealthy)" in health else state
                )

                # Uptime from "Created" timestamp (Unix epoch)
                created_ts = c.get("Created", 0)
                uptime_hours = round((_time.time() - created_ts) / 3600, 1)

                # Stats (CPU + memory)
                cpu_pct = 0.0
                mem_mb = 0.0
                mem_limit_mb = 0.0
                try:
                    stats_resp = await client.get(f"{docker_host}/containers/{cid}/stats?stream=false")
                    if stats_resp.status_code == 200:
                        s = stats_resp.json()
                        cpu_delta = (
                            s["cpu_stats"]["cpu_usage"]["total_usage"]
                            - s["precpu_stats"]["cpu_usage"]["total_usage"]
                        )
                        sys_delta = (
                            s["cpu_stats"].get("system_cpu_usage", 0)
                            - s["precpu_stats"].get("system_cpu_usage", 0)
                        )
                        num_cpus = len(s["cpu_stats"]["cpu_usage"].get("percpu_usage", [1]))
                        if sys_delta > 0:
                            cpu_pct = round((cpu_delta / sys_delta) * num_cpus * 100, 2)
                        mem_usage = s.get("memory_stats", {}).get("usage", 0)
                        mem_limit = s.get("memory_stats", {}).get("limit", 1)
                        mem_mb = round(mem_usage / 1024**2, 1)
                        mem_limit_mb = round(mem_limit / 1024**2, 1)
                except Exception as stats_exc:
                    logger.debug("Stats fetch failed for %s: %s", name, stats_exc)

                containers.append({
                    "name": name,
                    "image": image,
                    "status": state,
                    "health": health_str,
                    "cpu_percent": cpu_pct,
                    "memory_mb": mem_mb,
                    "memory_limit_mb": mem_limit_mb,
                    "uptime_hours": uptime_hours,
                })

    except Exception as exc:
        logger.warning("Docker API unreachable at %s: %s", docker_host, exc)
        # Graceful fallback: return known containers with unknown stats
        containers = [
            {"name": n, "image": "", "status": "unknown", "health": "unknown",
             "cpu_percent": 0.0, "memory_mb": 0.0, "memory_limit_mb": 0.0,
             "uptime_hours": 0.0, "error": "docker_unreachable"}
            for n in ["life-core", "life-reborn", "life-web", "redis", "qdrant",
                      "forgejo", "langfuse", "jaeger", "otel-collector", "traefik"]
        ]

    return {"containers": containers}


@infra_router.get("/storage")
async def storage_stats():
    """Get storage stats from Redis and Qdrant."""
    result = {"redis": {}, "qdrant": {}}

    # Redis stats
    try:
        import redis as redis_lib
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        r = redis_lib.from_url(redis_url)
        info = r.info("memory")
        result["redis"] = {
            "status": "connected",
            "used_memory_human": info.get("used_memory_human", "?"),
            "connected_clients": r.info("clients").get("connected_clients", 0),
            "keys": r.dbsize(),
        }
        r.close()
    except Exception as e:
        result["redis"] = {"status": "error", "error": str(e)}

    # Qdrant stats
    try:
        qdrant_url = os.environ.get("QDRANT_URL", "http://localhost:6333")
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{qdrant_url}/collections")
            if resp.status_code == 200:
                data = resp.json()
                collections = data.get("result", {}).get("collections", [])
                result["qdrant"] = {
                    "status": "connected",
                    "collections": len(collections),
                    "collection_names": [c["name"] for c in collections],
                }
            else:
                result["qdrant"] = {"status": "error", "code": resp.status_code}
    except Exception as e:
        result["qdrant"] = {"status": "error", "error": str(e)}

    return result


@infra_router.get("/network")
async def network_status():
    """Check network connectivity to external services."""
    checks = {}

    # Ollama local
    ollama_url = os.environ.get("OLLAMA_URL", "")
    if ollama_url:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(f"{ollama_url}/api/tags")
                models = resp.json().get("models", [])
                checks["ollama_local"] = {"status": "up", "models": len(models), "url": ollama_url}
        except Exception as e:
            checks["ollama_local"] = {"status": "down", "error": str(e), "url": ollama_url}

    # Ollama remote (KXKM-AI)
    ollama_remote = os.environ.get("OLLAMA_REMOTE_URL", "")
    if ollama_remote:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{ollama_remote}/api/tags")
                models = resp.json().get("models", [])
                checks["ollama_gpu"] = {"status": "up", "models": len(models), "url": ollama_remote}
        except Exception as e:
            checks["ollama_gpu"] = {"status": "down", "error": str(e), "url": ollama_remote}

    # vLLM GPU (KXKM-AI via proxy)
    vllm_url = os.environ.get("VLLM_BASE_URL", "")
    if vllm_url:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.get(f"{vllm_url}/health")
                models_resp = await client.get(f"{vllm_url}/v1/models")
                model_list = [m["id"] for m in models_resp.json().get("data", [])]
                checks["vllm_gpu"] = {"status": "up", "models": model_list, "url": vllm_url}
        except Exception as e:
            checks["vllm_gpu"] = {"status": "down", "error": str(e), "url": vllm_url}

    # Jaeger
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get("http://jaeger:16686/api/services")
            checks["jaeger"] = {"status": "up" if resp.status_code == 200 else "down"}
    except Exception:
        checks["jaeger"] = {"status": "down"}

    return checks


class DeployRequest(BaseModel):
    service: str
    image: str


@infra_router.post("/deploy")
def deploy(req: DeployRequest, x_deploy_token: str = Header(...)):
    """Pull a new image and restart a named container."""
    deploy_token = os.getenv("DEPLOY_TOKEN", "change-me")
    if x_deploy_token != deploy_token:
        raise HTTPException(status_code=403, detail="Invalid deploy token")

    client = docker.from_env()

    # Pull latest image
    client.images.pull(req.image)

    # Find running container by name
    containers = client.containers.list(filters={"name": req.service})
    if not containers:
        raise HTTPException(status_code=404, detail=f"Container '{req.service}' not found")

    container = containers[0]

    # Capture current config before stop
    env = container.attrs["Config"]["Env"]
    ports = container.attrs["HostConfig"]["PortBindings"]
    network = list(container.attrs["NetworkSettings"]["Networks"].keys())[0]

    container.stop(timeout=10)
    container.remove()

    client.containers.run(
        image=req.image,
        name=req.service,
        detach=True,
        environment=env,
        ports=ports,
        network=network,
        restart_policy={"Name": "unless-stopped"},
    )

    return {"status": "deployed", "service": req.service, "image": req.image}
