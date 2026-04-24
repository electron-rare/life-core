"""GET /api/providers — expose providers LLM + status + models count."""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

providers_router = APIRouter()


def _get_router():
    """Indirection pour permettre monkeypatch en tests."""
    from life_core.api import router
    return router


@providers_router.get("/api/providers")
async def list_providers():
    """Liste les providers LLM + statut runtime + nombre de modèles."""
    router = _get_router()
    if router is None:
        return {"providers": []}

    provider_ids = router.list_available_providers()
    try:
        status_map = router.get_provider_status()
    except Exception as exc:
        logger.warning("get_provider_status failed: %s", exc)
        status_map = {}

    out = []
    for pid in provider_ids:
        provider = router.providers.get(pid)
        models: list[str] = []
        if provider is not None:
            try:
                models = await provider.list_models()
            except Exception as exc:
                logger.debug("list_models(%s) failed: %s", pid, exc)

        status = "up" if status_map.get(pid, False) else "down"
        out.append({
            "id": pid,
            "name": pid,
            "status": status,
            "models_count": len(models),
        })

    return {"providers": out}
