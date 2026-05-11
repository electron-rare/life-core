"""Pre-routing Kiki meta -> niche via Studio's /v1/route endpoint.

When a user requests a kiki-meta-* model, we first ask the Studio MetaRouter
which domain fits the query best. If the top domain has score > threshold,
we downgrade the request to the corresponding kiki-niche-<domain>. Otherwise
we keep the original meta (which will trigger Studio's internal adapter
selection).

Graceful: if /v1/route is 404 / timeout / error, we always fall back to the
original model. This lets life-core run safely against older Studio
deployments where /v1/route isn't exposed yet.
"""
from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = float(os.environ.get("KIKI_PRE_ROUTING_THRESHOLD", "0.5"))
_DEFAULT_TIMEOUT_S = float(os.environ.get("KIKI_PRE_ROUTING_TIMEOUT_S", "2.0"))


def _extract_query(messages: Sequence[dict]) -> str:
    """Pick the last user message as the routing query."""
    for msg in reversed(list(messages)):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
    return ""


async def resolve_model(
    model: str,
    messages: Sequence[dict],
    *,
    kiki_full_base_url: str | None = None,
    threshold: float | None = None,
    timeout_s: float | None = None,
) -> str:
    """Return the model to use — downgraded to niche if /v1/route is confident,
    else original.

    Only applies to kiki-meta-* models. Any other model passes through
    unchanged.
    """
    if not model.startswith("kiki-meta-"):
        return model
    if not kiki_full_base_url:
        return model
    query = _extract_query(messages)
    if not query.strip():
        return model

    threshold = threshold if threshold is not None else _DEFAULT_THRESHOLD
    timeout = timeout_s if timeout_s is not None else _DEFAULT_TIMEOUT_S

    route_url = kiki_full_base_url.rstrip("/") + "/route"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                route_url, json={"query": query, "top_k": 1}
            )
            if resp.status_code != 200:
                logger.debug(
                    "kiki /v1/route returned %s, keeping %s",
                    resp.status_code,
                    model,
                )
                return model
            data = resp.json()
            domains = data.get("domains", [])
            if not domains:
                return model
            top = domains[0]
            name = top.get("name")
            score = top.get("score", 0.0)
            if not name:
                return model
            if score < threshold:
                logger.debug(
                    "kiki /v1/route score %.3f < threshold %.3f, keeping %s",
                    score,
                    threshold,
                    model,
                )
                return model
            niche_model = f"kiki-niche-{name}"
            logger.info(
                "kiki pre-routing: %s -> %s (domain=%s score=%.3f)",
                model,
                niche_model,
                name,
                score,
            )
            return niche_model
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "kiki pre-routing failed, falling back to %s: %s", model, exc
        )
        return model
