# life-core/life_core/monitoring_api.py
"""Monitoring proxy endpoints: machines, GPU, Activepieces."""

from __future__ import annotations

import logging
import os
import re
import time as _time
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import APIRouter, HTTPException

logger = logging.getLogger("life_core.monitoring_api")

monitoring_router = APIRouter(prefix="/infra", tags=["Monitoring"])

_DEFAULT_MACHINES_YAML = Path(__file__).parent / "config" / "machines.yaml"


def _load_machines_config() -> list[dict]:
    """Load machines inventory from YAML (env F4L_MACHINES_YAML overrides default)."""
    path = Path(os.environ.get("F4L_MACHINES_YAML", str(_DEFAULT_MACHINES_YAML)))
    if not path.exists():
        logger.warning("machines yaml not found at %s", path)
        return []
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return data.get("machines", [])


async def _ping_host(name: str, ip: str) -> dict:
    """Return dict enriched with runtime metrics (or 0 when host unreachable).

    Hook minimal: on essaie d'atteindre un exporter Prometheus configuré via
    F4L_METRICS_<NAME_UPPER>. L'intégration réelle (SSH agent ou scrape dédié)
    se fera en itération. Pour l'instant ce hook expose les champs obligatoires
    avec des zéros et un drapeau `error` explicite.
    """
    default = {
        "cpu_percent": 0.0,
        "ram_used_gb": 0.0,
        "disk_used_gb": 0.0,
        "uptime_hours": 0.0,
        "error": None,
    }
    env_key = f"F4L_METRICS_{name.upper().replace('-', '_')}"
    exporter_url = os.environ.get(env_key, "")
    if not exporter_url:
        default["error"] = "no_exporter_configured"
        return default
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            r = await client.get(f"{exporter_url}/metrics")
            if r.status_code == 200:
                default["error"] = None
            else:
                default["error"] = f"ping_http_{r.status_code}"
    except Exception as exc:  # pragma: no cover - network path
        default["error"] = f"ping_failed:{exc.__class__.__name__}"
    return default


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


async def _scrape_node_metrics() -> dict[str, dict[str, float]]:
    """Scrape otel-collector Prometheus exporter and extract node_* metrics by machine label."""
    prom_url = os.environ.get("PROMETHEUS_URL", "http://otel-collector:8889")
    result: dict[str, dict[str, float]] = {}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{prom_url}/metrics")
            resp.raise_for_status()
            for line in resp.text.splitlines():
                if line.startswith("#") or not line or "node_" not in line:
                    continue
                m = re.match(r'^(finefab_node_\w+)\{([^}]*)\}\s+([\d.eE+\-]+)', line)
                if not m:
                    continue
                metric, labels_str, val = m.group(1), m.group(2), float(m.group(3))
                labels = dict(re.findall(r'(\w+)="([^"]*)"', labels_str))
                machine = labels.get("machine", "")
                if not machine:
                    continue
                # For filesystem metrics, only keep root mountpoint
                if "filesystem" in metric and labels.get("mountpoint") != "/":
                    continue
                if machine not in result:
                    result[machine] = {}
                result[machine][metric] = val
    except Exception as exc:
        logger.warning("Failed to scrape node metrics: %s", exc)
    return result


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
    """Return the declarative hosts inventory with runtime metrics.

    Source of truth: `life_core/config/machines.yaml` (override via env
    F4L_MACHINES_YAML). Each machine is enriched with a best-effort ping
    to a per-host exporter (env F4L_METRICS_<NAME_UPPER>).
    """
    cfg = _load_machines_config()
    result: list[dict[str, Any]] = []
    for m in cfg:
        specs = m.get("specs", {}) or {}
        stats = await _ping_host(m["name"], m.get("ip", ""))
        result.append({
            "name": m["name"],
            "ip": m.get("ip", ""),
            "role": m.get("role", ""),
            "services": m.get("services", []),
            "specs": specs,
            "cpu_percent": stats["cpu_percent"],
            "ram_used_gb": stats["ram_used_gb"],
            "ram_total_gb": float(specs.get("ram_gb", 0) or 0),
            "disk_used_gb": stats["disk_used_gb"],
            "disk_total_gb": float(specs.get("storage_gb", 0) or 0),
            "uptime_hours": stats["uptime_hours"],
            "error": stats["error"],
        })
    return {"machines": result}


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
    base_url = os.environ.get("VLLM_METRICS_URL")
    if not base_url:
        vllm_base = os.environ.get("VLLM_BASE_URL", "").rstrip("/")
        # Strip /v1 suffix to get root URL for metrics endpoint
        if vllm_base.endswith("/v1"):
            vllm_base = vllm_base[:-3]
        base_url = vllm_base + "/metrics" if vllm_base else ""

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
        tokens_total = metrics.get("vllm:generation_tokens_total", 0.0)

        # Get real VRAM from nvidia-gpu-exporter via otel-collector
        node_metrics = await _scrape_node_metrics()
        gpu_metrics = {}
        prom_url = os.environ.get("PROMETHEUS_URL", "http://otel-collector:8889")
        try:
            async with httpx.AsyncClient(timeout=5.0) as gpu_client:
                gpu_resp = await gpu_client.get(f"{prom_url}/metrics")
                gpu_resp.raise_for_status()
                for line in gpu_resp.text.splitlines():
                    if "nvidia_smi_memory" not in line or line.startswith("#"):
                        continue
                    gm = re.match(r'^(finefab_nvidia_smi_\w+)\{[^}]*\}\s+([\d.eE+\-]+)', line)
                    if gm:
                        gpu_metrics[gm.group(1)] = float(gm.group(2))
        except Exception:
            pass

        vram_used_bytes = gpu_metrics.get("finefab_nvidia_smi_memory_used_bytes", 0)
        vram_total_bytes = gpu_metrics.get("finefab_nvidia_smi_memory_total_bytes", 0)
        vram_used = round(vram_used_bytes / 1024**3, 2) if vram_used_bytes else round(_VRAM_TOTAL_GB * kv_cache_pct / 100, 2)
        vram_total = round(vram_total_bytes / 1024**3, 1) if vram_total_bytes else _VRAM_TOTAL_GB

        return {
            "model": model_name,
            "vram_used_gb": vram_used,
            "vram_total_gb": vram_total,
            "requests_active": requests_active,
            "tokens_per_sec": round(tokens_total, 1),
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
        return {"flows": [], "note": "Activepieces flows available via Goose MCP at /goose"}

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


@monitoring_router.post("/activepieces/trigger")
async def trigger_activepieces_flow(body: dict):
    """Trigger an Activepieces flow by name via its webhook URL."""
    flow_name = body.get("flow_name", "")
    ap_url = os.environ.get("ACTIVEPIECES_URL", "https://auto.saillant.cc")
    ap_token = os.environ.get("ACTIVEPIECES_TOKEN", "")
    project_id = os.environ.get("ACTIVEPIECES_PROJECT_ID", "QG09trLP4ICBvpCbyjVRw")

    if not ap_token:
        raise HTTPException(status_code=503, detail="Activepieces not configured")

    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(
            f"{ap_url}/api/v1/flows",
            params={"projectId": project_id},
            headers={"Authorization": f"Bearer {ap_token}"},
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="Failed to list flows")

        flows = resp.json().get("data", [])
        if isinstance(flows, dict):
            flows = flows.get("data", [])
        target = next(
            (f for f in flows if f.get("version", {}).get("displayName") == flow_name),
            None,
        )
        if not target:
            raise HTTPException(status_code=404, detail=f"Flow '{flow_name}' not found")

        trigger = target.get("version", {}).get("trigger", {})
        webhook_url = trigger.get("settings", {}).get("webhookUrl")
        if not webhook_url:
            raise HTTPException(status_code=400, detail=f"Flow '{flow_name}' has no webhook trigger")

        trigger_resp = await client.post(webhook_url, json={"triggered_by": "cockpit"})

    return {"status": "triggered", "flow_name": flow_name, "http_status": trigger_resp.status_code}
