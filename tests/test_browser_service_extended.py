"""Extended tests for browser service — _run_remote, _run_http_fetch, fallback-disabled."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from life_core.services.browser import (
    BrowserDependencyMissingError,
    BrowserRemoteRunnerError,
    BrowserService,
    BrowserServiceError,
)


# ---------------------------------------------------------------------------
# Fallback disabled — exceptions propagate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_no_fallback_dependency_missing_propagates():
    """With fallback disabled, BrowserDependencyMissingError must propagate."""
    service = BrowserService(force_local=True)
    service.enable_http_fallback = False
    service._run_camoufox = AsyncMock(
        side_effect=BrowserDependencyMissingError("no camoufox")
    )

    with pytest.raises(BrowserDependencyMissingError, match="no camoufox"):
        await service.scrape(url="https://example.com", timeout_ms=2000)


@pytest.mark.asyncio
async def test_scrape_no_fallback_generic_exception_propagates():
    """With fallback disabled, generic exceptions must propagate."""
    service = BrowserService(force_local=True)
    service.enable_http_fallback = False
    service._run_camoufox = AsyncMock(side_effect=RuntimeError("crash"))

    with pytest.raises(RuntimeError, match="crash"):
        await service.scrape(url="https://example.com", timeout_ms=2000)


# ---------------------------------------------------------------------------
# _run_remote
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_remote_success():
    """Successful remote scrape returns parsed JSON."""
    service = BrowserService(runner_url="http://runner:8123")

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "url": "https://example.com",
        "title": "Remote Page",
        "content": "Remote Content",
    }

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_remote(
            url="https://example.com", selector="h1", timeout_ms=5000
        )

    assert result["url"] == "https://example.com"
    assert result["title"] == "Remote Page"
    assert result["content"] == "Remote Content"


@pytest.mark.asyncio
async def test_run_remote_http_error():
    """HTTP >= 400 from runner raises BrowserRemoteRunnerError."""
    service = BrowserService(runner_url="http://runner:8123")

    mock_response = MagicMock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(BrowserRemoteRunnerError, match="HTTP 503"):
            await service._run_remote(
                url="https://example.com", selector=None, timeout_ms=5000
            )


@pytest.mark.asyncio
async def test_run_remote_network_error():
    """Network failure raises BrowserRemoteRunnerError."""
    service = BrowserService(runner_url="http://runner:8123")

    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(BrowserRemoteRunnerError, match="request failed"):
            await service._run_remote(
                url="https://example.com", selector=None, timeout_ms=5000
            )


# ---------------------------------------------------------------------------
# _run_http_fetch
# ---------------------------------------------------------------------------


def _make_http_response(
    text: str,
    content_type: str = "text/html",
    url: str = "https://example.com",
) -> MagicMock:
    resp = MagicMock()
    resp.text = text
    resp.url = url
    resp.headers = {"content-type": content_type}
    resp.raise_for_status = MagicMock()
    return resp


@pytest.mark.asyncio
async def test_run_http_fetch_html_no_selector():
    """HTML response without selector returns raw HTML."""
    service = BrowserService(force_local=True)
    html = "<html><head><title>Test Page</title></head><body><p>Hello</p></body></html>"
    mock_resp = _make_http_response(html)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_http_fetch(
            url="https://example.com", selector=None, timeout_ms=5000
        )

    assert result["title"] == "Test Page"
    assert result["content"] == html


@pytest.mark.asyncio
async def test_run_http_fetch_html_with_selector():
    """HTML response with CSS selector extracts matched text."""
    service = BrowserService(force_local=True)
    html = "<html><head><title>Page</title></head><body><main>Main Content</main></body></html>"
    mock_resp = _make_http_response(html)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_http_fetch(
            url="https://example.com", selector="main", timeout_ms=5000
        )

    assert result["title"] == "Page"
    assert result["content"] == "Main Content"


@pytest.mark.asyncio
async def test_run_http_fetch_html_selector_no_match():
    """When CSS selector matches nothing, content is empty."""
    service = BrowserService(force_local=True)
    html = "<html><head><title>Page</title></head><body><p>Text</p></body></html>"
    mock_resp = _make_http_response(html)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_http_fetch(
            url="https://example.com", selector="article", timeout_ms=5000
        )

    assert result["content"] == ""


@pytest.mark.asyncio
async def test_run_http_fetch_invalid_selector():
    """Invalid CSS selector raises BrowserServiceError."""
    service = BrowserService(force_local=True)
    html = "<html><body><p>Text</p></body></html>"
    mock_resp = _make_http_response(html)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        with pytest.raises(BrowserServiceError, match="invalid CSS selector"):
            await service._run_http_fetch(
                url="https://example.com", selector="[[[invalid", timeout_ms=5000
            )


@pytest.mark.asyncio
async def test_run_http_fetch_non_html_no_selector():
    """Non-HTML content without selector returns raw content."""
    service = BrowserService(force_local=True)
    mock_resp = _make_http_response("plain text data", content_type="text/plain")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_http_fetch(
            url="https://example.com/data.txt", selector=None, timeout_ms=5000
        )

    assert result["content"] == "plain text data"
    assert result["title"] == ""


@pytest.mark.asyncio
async def test_run_http_fetch_non_html_with_selector():
    """Non-HTML content with selector returns empty content."""
    service = BrowserService(force_local=True)
    mock_resp = _make_http_response("json data", content_type="application/json")

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_http_fetch(
            url="https://example.com/api", selector="main", timeout_ms=5000
        )

    assert result["content"] == ""


@pytest.mark.asyncio
async def test_run_http_fetch_no_title_tag():
    """HTML without <title> returns empty title."""
    service = BrowserService(force_local=True)
    html = "<html><body><p>No title</p></body></html>"
    mock_resp = _make_http_response(html)

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("httpx.AsyncClient", return_value=mock_client):
        result = await service._run_http_fetch(
            url="https://example.com", selector=None, timeout_ms=5000
        )

    assert result["title"] == ""
