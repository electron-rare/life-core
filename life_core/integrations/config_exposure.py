"""V1.7 Track II Task 8 — read-only /config exposure.

Allowlists env vars that are safe to surface in the cockpit and
hard-blocks any name matching the secret regex
(`*_KEY | *_SECRET | *_TOKEN | *_PASSWORD`, case-insensitive)
regardless of whether it was explicitly allowlisted.

Also exposes the full model list (union of KIKI_FULL_MODELS,
VLLM_MODELS, LOCAL_LLM_MODELS) and the network host map derived
from the Prometheus scraper DEFAULT_TARGETS (spec §5.7, 7 hosts).
"""
from __future__ import annotations

import os
import re
from typing import Any

from life_core.monitoring.prometheus_scraper import DEFAULT_TARGETS

#: Env vars that are safe to surface verbatim in the cockpit. Any
#: name matching ``_SECRET_RE`` is hard-blocked below regardless of
#: whether it is listed here — the allowlist is a convenience, the
#: regex is the enforcement.
ALLOWLIST: tuple[str, ...] = (
    "VLLM_BASE_URL",
    "VLLM_MODELS",
    "LOCAL_LLM_BASE_URL",
    "LOCAL_LLM_MODELS",
    "KIKI_FULL_BASE_URL",
    "KIKI_FULL_MODELS",
    "OLLAMA_URL",
    "OLLAMA_EMBED_URL",
    "LANGFUSE_HOST",
    "F4L_CONTAINER_FILTER",
    "REDIS_URL",
    "QDRANT_URL",
    "OPENAI_API_BASE",
)

_SECRET_RE = re.compile(
    r"(?:_KEY|_SECRET|_TOKEN|_PASSWORD)$", re.IGNORECASE
)

REDACTED = "<redacted>"


def is_secret_key(name: str) -> bool:
    """True when the env var name ends in ``_KEY``, ``_SECRET``,
    ``_TOKEN``, or ``_PASSWORD`` (case-insensitive)."""
    return bool(_SECRET_RE.search(name))


def collect_env() -> dict[str, str]:
    """Return the allowlisted env vars with secret-like names
    replaced by the ``<redacted>`` sentinel — never the raw value.

    A name appears in the output only if it is in ``ALLOWLIST`` and
    the env var is actually set. Entries whose name matches the
    secret regex are emitted as ``<redacted>`` rather than dropped,
    so callers can tell the setting is configured without seeing
    the secret.
    """
    out: dict[str, str] = {}
    for name in ALLOWLIST:
        val = os.environ.get(name)
        if val is None:
            continue
        if is_secret_key(name):
            out[name] = REDACTED
        else:
            out[name] = val
    return out


def collect_models() -> list[str]:
    """Union of comma-separated model lists from the known env
    vars, sorted and de-duplicated. Same source as /providers."""
    models: set[str] = set()
    for env_name in ("KIKI_FULL_MODELS", "VLLM_MODELS", "LOCAL_LLM_MODELS"):
        raw = os.environ.get(env_name, "")
        for model_id in raw.split(","):
            model_id = model_id.strip()
            if model_id:
                models.add(model_id)
    return sorted(models)


def collect_network() -> dict[str, Any]:
    """Network host map derived from the Prometheus scraper
    DEFAULT_TARGETS. Exposes the list of host names only — no URLs,
    no credentials. Matches V1.7 spec §5.7 (7 hosts)."""
    return {"hosts": [target.host for target in DEFAULT_TARGETS]}


def collect() -> dict[str, Any]:
    """Full /config payload: env + models + network."""
    return {
        "env": collect_env(),
        "models": collect_models(),
        "network": collect_network(),
    }
