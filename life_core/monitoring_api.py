# life-core/life_core/monitoring_api.py
"""Monitoring proxy endpoints: machines, GPU, Activepieces."""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx
from fastapi import APIRouter

logger = logging.getLogger("life_core.monitoring_api")

monitoring_router = APIRouter(prefix="/infra", tags=["Monitoring"])

MACHINE_DEFAULTS = [
    {"name": "Tower",   "ip": "192.168.0.120", "ram_total_gb": 31.0,  "disk_total_gb": 1800.0},
    {"name": "KXKM-AI", "ip": "100.87.54.119", "ram_total_gb": 32.0,  "disk_total_gb": 512.0},
    {"name": "Cils",    "ip": "100.126.225.111","ram_total_gb": 16.0,  "disk_total_gb": 256.0},
    {"name": "GrosMac", "ip": "100.80.178.42",  "ram_total_gb": 36.0,  "disk_total_gb": 1000.0},
]

_PROMQL_CPU    = '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)'
_PROMQL_RAMFREE = "node_memory_MemAvailable_bytes"
_PROMQL_RAMTOT  = "node_memory_MemTotal_bytes"
_PROMQL_DISKUSED = 'node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}'
_PROMQL_DISKTOT  = 'node_filesystem_size_bytes{mountpoint="/"}'
_PROMQL_UPTIME   = "time() - node_boot_time_seconds"


async def _query_prom(client: httpx.AsyncClient, grafana_url: str, api_key: str, promql: str) -> dict[str, Any]:
    """Run a PromQL instant query via Prometheus endpoint (OTEL collector or Grafana proxy)."""
    prom_url = os.environ.get("PROMETHEUS_URL", "")
    if prom_url:
        # Direct Prometheus query (preferred)
        resp = await client.get(
            f"{prom_url}/api/v1/query",
            params={"query": promql},
            timeout=5.0,
        )
    else:
        # Fallback: Grafana datasource proxy by UID
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        ds_uid = os.environ.get("GRAFANA_DS_UID", "prometheus")
        resp = await client.get(
            f"{grafana_url}/api/ds/query",
            params={"ds_type": "prometheus", "expression": promql},
            headers=headers,
            timeout=5.0,
        )
    resp.raise_for_status()
    return resp.json()


def _extract_by_instance(data: dict[str, Any]) -> dict[str, float]:
    """Return {instance_label: float_value} from a Prometheus vector result."""
    out: dict[str, float] = {}
    for item in data.get("data", {}).get("result", []):
        instance = item.get("metric", {}).get("instance", "")
        try:
            out[instance] = float(item["value"][1])
        except (KeyError, ValueError, IndexError):
            pass
    return out


def _read_host_stats() -> dict[str, float]:
    """Read CPU, RAM, disk, uptime from /proc (works inside Docker containers)."""
    stats: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            mem = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mem[parts[0].rstrip(":")] = int(parts[1]) * 1024  # kB → bytes
            stats["ram_total"] = mem.get("MemTotal", 0)
            stats["ram_available"] = mem.get("MemAvailable", 0)
    except Exception:
        pass
    try:
        with open("/proc/uptime") as f:
            stats["uptime_seconds"] = float(f.read().split()[0])
    except Exception:
        pass
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu "):
                    fields = [float(x) for x in line.split()[1:]]
                    idle = fields[3] if len(fields) > 3 else 0
                    total = sum(fields)
                    stats["cpu_idle_ratio"] = idle / total if total else 1.0
                    break
    except Exception:
        pass
    try:
        st = os.statvfs("/")
        stats["disk_total"] = st.f_frsize * st.f_blocks
        stats["disk_used"] = st.f_frsize * (st.f_blocks - st.f_bavail)
    except Exception:
        pass
    return stats


@monitoring_router.get("/machines")
async def list_machines():
    """Return machine stats. Tower reads /proc directly, remote machines use defaults."""
    host_stats = _read_host_stats()
    machines = []
    for m in MACHINE_DEFAULTS:
        if m["name"] == "Tower":
            ram_total = host_stats.get("ram_total", m["ram_total_gb"] * 1024**3)
            ram_avail = host_stats.get("ram_available", 0)
            machines.append({
                "name": m["name"],
                "ip": m["ip"],
                "cpu_percent": round((1 - host_stats.get("cpu_idle_ratio", 1.0)) * 100, 1),
                "ram_used_gb": round((ram_total - ram_avail) / 1024**3, 2),
                "ram_total_gb": round(ram_total / 1024**3, 1),
                "disk_used_gb": round(host_stats.get("disk_used", 0) / 1024**3, 1),
                "disk_total_gb": round(host_stats.get("disk_total", m["disk_total_gb"] * 1024**3) / 1024**3, 1),
                "uptime_hours": round(host_stats.get("uptime_seconds", 0) / 3600, 1),
            })
        else:
            machines.append({
                "name": m["name"],
                "ip": m["ip"],
                "cpu_percent": 0.0,
                "ram_used_gb": 0.0,
                "ram_total_gb": m["ram_total_gb"],
                "disk_used_gb": 0.0,
                "disk_total_gb": m["disk_total_gb"],
                "uptime_hours": 0.0,
                "error": "remote_no_agent",
            })

    return {"machines": machines}


# --- Task 3: GPU stats from vLLM Prometheus metrics ---

_VRAM_TOTAL_GB = 24.0  # RTX 4090 constant


def _parse_prometheus_text(text: str) -> dict[str, float]:
    """Parse Prometheus text exposition format into {metric_name: value}."""
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        # metric_name{labels} value [timestamp]
        match = re.match(r'^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([\d.eE+\-]+)', line)
        if match:
            name = re.sub(r'\{[^}]*\}', '', match.group(1))  # strip labels for simple lookup
            try:
                metrics[name] = float(match.group(2))
            except ValueError:
                pass
    return metrics


@monitoring_router.get("/gpu")
async def gpu_stats():
    """Parse vLLM /metrics Prometheus endpoint for VRAM, requests, throughput."""
    base_url = os.environ.get("VLLM_METRICS_URL") or (
        os.environ.get("VLLM_BASE_URL", "").rstrip("/") + "/metrics"
    )

    fallback = {
        "model": "unknown",
        "vram_used_gb": 0.0,
        "vram_total_gb": _VRAM_TOTAL_GB,
        "requests_active": 0,
        "tokens_per_sec": 0.0,
        "kv_cache_usage_percent": 0.0,
        "error": "vllm_unreachable",
    }

    if not base_url or base_url == "/metrics":
        return fallback

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(base_url)
            resp.raise_for_status()
            raw = resp.text

        metrics = _parse_prometheus_text(raw)

        # Extract model name from raw text (label on any vllm metric)
        model_match = re.search(r'model_name="([^"]+)"', raw)
        model_name = model_match.group(1) if model_match else "unknown"

        kv_cache_pct = metrics.get("vllm:gpu_cache_usage_perc", 0.0) * 100
        requests_active = int(metrics.get("vllm:num_requests_running", 0))
        # vram: use kv_cache as proxy for VRAM pressure
        vram_pct = kv_cache_pct / 100
        vram_used = round(_VRAM_TOTAL_GB * vram_pct, 2)

        # tokens/sec: vllm:generation_tokens_total is a counter
        tokens_total = metrics.get("vllm:generation_tokens_total", 0.0)

        return {
            "model": model_name,
            "vram_used_gb": vram_used,
            "vram_total_gb": _VRAM_TOTAL_GB,
            "requests_active": requests_active,
            "tokens_per_sec": round(tokens_total, 1),  # counter, not rate
            "kv_cache_usage_percent": round(kv_cache_pct, 1),
        }

    except Exception as exc:
        logger.warning("vLLM metrics unreachable: %s", exc)
        return fallback


# --- Task 5: Activepieces flows ---

@monitoring_router.get("/activepieces")
async def activepieces_flows():
    """Proxy Activepieces API for FineFab project flows."""
    ap_url = os.environ.get("ACTIVEPIECES_URL", "https://auto.saillant.cc")
    ap_token = os.environ.get("ACTIVEPIECES_TOKEN", "")
    project_id = os.environ.get("ACTIVEPIECES_PROJECT_ID", "QG09trLP4ICBvpCbyjVRw")

    if not ap_token:
        return {"flows": [], "error": "ACTIVEPIECES_TOKEN not configured"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{ap_url}/api/v1/flows",
                params={"projectId": project_id},
                headers={"Authorization": f"Bearer {ap_token}"},
            )
            resp.raise_for_status()
            data = resp.json()

        flows = []
        for f in data.get("data", data if isinstance(data, list) else []):
            last_run = f.get("lastRun") or {}
            flows.append({
                "id": f.get("id", ""),
                "name": f.get("version", {}).get("displayName", f.get("id", "")),
                "status": f.get("status", "UNKNOWN"),
                "trigger": f.get("version", {}).get("trigger", {}).get("type", "UNKNOWN"),
                "last_run_at": last_run.get("startTime", ""),
                "last_run_status": last_run.get("status", "UNKNOWN"),
            })
        return {"flows": flows}

    except Exception as exc:
        logger.warning("Activepieces API unreachable: %s", exc)
        return {"flows": [], "error": str(exc)}
