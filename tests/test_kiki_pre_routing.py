"""Tests for Kiki pre-routing via /v1/route before /v1/chat/completions."""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_pre_routing_downgrades_meta_to_niche_on_high_score():
    """Quand /v1/route retourne un domaine avec score > threshold,
    downgrade kiki-meta-* vers kiki-niche-<domain>.
    """
    from life_core.router.kiki_pre_routing import resolve_model

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": "écris une fonction python qui parse un json",
        "domains": [{"name": "python", "score": 0.87}],
        "fallback": False,
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "life_core.router.kiki_pre_routing.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await resolve_model(
            "kiki-meta-coding",
            [{"role": "user", "content": "écris une fonction python qui parse un json"}],
            kiki_full_base_url="http://kiki:9200/v1",
            threshold=0.5,
        )
    assert result == "kiki-niche-python"


@pytest.mark.asyncio
async def test_pre_routing_keeps_meta_when_score_below_threshold():
    """Score trop bas -> garde le meta d'origine (confiance insuffisante)."""
    from life_core.router.kiki_pre_routing import resolve_model

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "query": "bonjour",
        "domains": [{"name": "chat-fr", "score": 0.15}],
        "fallback": True,
    }
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "life_core.router.kiki_pre_routing.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await resolve_model(
            "kiki-meta-coding",
            [{"role": "user", "content": "bonjour"}],
            kiki_full_base_url="http://kiki:9200/v1",
            threshold=0.5,
        )
    assert result == "kiki-meta-coding"


@pytest.mark.asyncio
async def test_pre_routing_graceful_on_404():
    """Si /v1/route renvoie 404 (endpoint pas déployé), fallback sur le
    model original.
    """
    from life_core.router.kiki_pre_routing import resolve_model

    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "life_core.router.kiki_pre_routing.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await resolve_model(
            "kiki-meta-coding",
            [{"role": "user", "content": "x"}],
            kiki_full_base_url="http://kiki:9200/v1",
        )
    assert result == "kiki-meta-coding"


@pytest.mark.asyncio
async def test_pre_routing_graceful_on_timeout():
    """Timeout -> fallback."""
    from life_core.router.kiki_pre_routing import resolve_model

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=Exception("timeout"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "life_core.router.kiki_pre_routing.httpx.AsyncClient",
        return_value=mock_client,
    ):
        result = await resolve_model(
            "kiki-meta-coding",
            [{"role": "user", "content": "x"}],
            kiki_full_base_url="http://kiki:9200/v1",
        )
    assert result == "kiki-meta-coding"


@pytest.mark.asyncio
async def test_pre_routing_skips_non_meta_models():
    """Only kiki-meta-* models are pre-routed. kiki-niche-* and other models
    pass through.
    """
    from life_core.router.kiki_pre_routing import resolve_model

    with patch("life_core.router.kiki_pre_routing.httpx.AsyncClient") as mock_c:
        result1 = await resolve_model(
            "openai/gpt-4o", [], kiki_full_base_url="http://kiki:9200/v1"
        )
        assert result1 == "openai/gpt-4o"
        result2 = await resolve_model(
            "kiki-niche-python", [], kiki_full_base_url="http://kiki:9200/v1"
        )
        assert result2 == "kiki-niche-python"
        mock_c.assert_not_called()
