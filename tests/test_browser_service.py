"""Tests for browser scraping PoC."""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from life_core.services.browser import (
    BrowserDependencyMissingError,
    BrowserRemoteRunnerError,
    BrowserService,
    BrowserServiceError,
)


@pytest.mark.asyncio
async def test_browser_service_validates_url():
    service = BrowserService()

    with pytest.raises(BrowserServiceError):
        await service.scrape(url="file:///etc/passwd")


@pytest.mark.asyncio
async def test_browser_service_scrape_with_mock(monkeypatch):
    service = BrowserService()

    async def _fake_run_camoufox(*, url: str, selector: str | None, timeout_ms: int):
        assert url == "https://example.com"
        assert selector == "h1"
        assert timeout_ms == 5000
        return {
            "url": "https://example.com",
            "title": "Example Domain",
            "content": "Example Domain",
        }

    monkeypatch.setattr(service, "_run_camoufox", _fake_run_camoufox)

    result = await service.scrape(url="https://example.com", selector="h1", timeout_ms=5000)
    assert result["title"] == "Example Domain"


@pytest.mark.asyncio
async def test_scrape_endpoint_error_mapping(monkeypatch):
    import life_core.api as api

    class _StubBrowserService:
        async def scrape(self, **kwargs):
            raise BrowserDependencyMissingError("camoufox is missing")

    api.browser_service = _StubBrowserService()

    with pytest.raises(HTTPException) as err:
        await api.scrape(api.ScrapeRequest(url="https://example.com"))

    assert err.value.status_code == 503


@pytest.mark.asyncio
async def test_scrape_endpoint_remote_runner_errors_map_to_502():
    import life_core.api as api

    class _StubBrowserService:
        async def scrape(self, **kwargs):
            raise BrowserRemoteRunnerError("runner unreachable")

    api.browser_service = _StubBrowserService()

    with pytest.raises(HTTPException) as err:
        await api.scrape(api.ScrapeRequest(url="https://example.com"))

    assert err.value.status_code == 502


# ---------------------------------------------------------------------------
# New tests for increased coverage
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_scrape_timeout_ms_zero_raises():
    """timeout_ms <= 0 must raise BrowserServiceError (lines 33-34)."""
    service = BrowserService()
    with pytest.raises(BrowserServiceError, match="timeout_ms must be > 0"):
        await service.scrape(url="https://example.com", timeout_ms=0)


@pytest.mark.asyncio
async def test_scrape_timeout_ms_negative_raises():
    """Negative timeout_ms must also raise BrowserServiceError."""
    service = BrowserService()
    with pytest.raises(BrowserServiceError, match="timeout_ms must be > 0"):
        await service.scrape(url="https://example.com", timeout_ms=-1)


@pytest.mark.asyncio
async def test_scrape_rejects_host_outside_allowlist():
    service = BrowserService(allowed_hosts={"example.com"})
    with pytest.raises(BrowserServiceError, match="not allowed"):
        await service.scrape(url="https://not-example.org")


@pytest.mark.asyncio
async def test_scrape_uses_remote_runner_when_configured():
    service = BrowserService(runner_url="http://browser-runner:8123")
    service._run_remote = AsyncMock(return_value={
        "url": "https://example.com",
        "title": "Remote",
        "content": "Remote Content",
    })

    result = await service.scrape(url="https://example.com", timeout_ms=2500)

    assert result["title"] == "Remote"
    service._run_remote.assert_awaited_once()


@pytest.mark.asyncio
async def test_scrape_uses_http_engine_when_configured():
    service = BrowserService(force_local=True)
    service.engine = "http"
    service._run_http_fetch = AsyncMock(return_value={
        "url": "https://example.com",
        "title": "HTTP Title",
        "content": "HTTP Content",
    })
    service._run_camoufox = AsyncMock()

    result = await service.scrape(url="https://example.com", timeout_ms=2500)

    assert result["title"] == "HTTP Title"
    service._run_http_fetch.assert_awaited_once_with(
        url="https://example.com", selector=None, timeout_ms=2500
    )
    service._run_camoufox.assert_not_called()


@pytest.mark.asyncio
async def test_scrape_falls_back_to_http_when_camoufox_fails():
    service = BrowserService(force_local=True)
    service.enable_http_fallback = True
    service._run_camoufox = AsyncMock(side_effect=RuntimeError("camoufox hung"))
    service._run_http_fetch = AsyncMock(return_value={
        "url": "https://example.com",
        "title": "Fallback Title",
        "content": "Fallback Content",
    })

    result = await service.scrape(url="https://example.com", selector="h1", timeout_ms=2500)

    assert result["title"] == "Fallback Title"
    service._run_camoufox.assert_awaited_once_with(
        url="https://example.com", selector="h1", timeout_ms=2500
    )
    service._run_http_fetch.assert_awaited_once_with(
        url="https://example.com", selector="h1", timeout_ms=2500
    )


@pytest.mark.asyncio
async def test_run_camoufox_import_error_raises_dependency_missing():
    """When camoufox is not installed, _run_camoufox must raise BrowserDependencyMissingError."""
    service = BrowserService()

    # Simulate camoufox not being importable by patching the import inside the module
    with patch.dict(sys.modules, {"camoufox": None, "camoufox.async_api": None}):
        with pytest.raises(BrowserDependencyMissingError, match="camoufox is not installed"):
            await service._run_camoufox(url="https://example.com", selector=None, timeout_ms=5000)


@pytest.mark.asyncio
async def test_run_camoufox_successful_no_selector():
    """Full path through _run_camoufox without selector — returns url/title/content."""
    service = BrowserService()

    # Build a mock page
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.title = AsyncMock(return_value="My Title")
    mock_page.content = AsyncMock(return_value="<html>body</html>")
    mock_page.url = "https://example.com/final"

    # Build a mock browser acting as async context manager
    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
    mock_browser.__aexit__ = AsyncMock(return_value=False)

    mock_async_camoufox_cls = MagicMock(return_value=mock_browser)

    mock_module = MagicMock()
    mock_module.AsyncCamoufox = mock_async_camoufox_cls

    with patch.dict(sys.modules, {"camoufox": MagicMock(), "camoufox.async_api": mock_module}):
        result = await service._run_camoufox(
            url="https://example.com", selector=None, timeout_ms=5000
        )

    assert result["url"] == "https://example.com/final"
    assert result["title"] == "My Title"
    assert result["content"] == "<html>body</html>"
    mock_page.goto.assert_awaited_once_with(
        "https://example.com", timeout=5000, wait_until="domcontentloaded"
    )


@pytest.mark.asyncio
async def test_run_camoufox_successful_with_selector():
    """Full path through _run_camoufox with a CSS selector — uses locator.text_content()."""
    service = BrowserService()

    # locator().first returns an object whose text_content() is awaitable
    mock_element = AsyncMock()
    mock_element.text_content = AsyncMock(return_value="  Hello World  ")

    mock_locator = MagicMock()
    mock_locator.first = mock_element

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.title = AsyncMock(return_value="Selector Title")
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.url = "https://example.com"

    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
    mock_browser.__aexit__ = AsyncMock(return_value=False)

    mock_async_camoufox_cls = MagicMock(return_value=mock_browser)

    mock_module = MagicMock()
    mock_module.AsyncCamoufox = mock_async_camoufox_cls

    with patch.dict(sys.modules, {"camoufox": MagicMock(), "camoufox.async_api": mock_module}):
        result = await service._run_camoufox(
            url="https://example.com", selector="h1", timeout_ms=3000
        )

    assert result["title"] == "Selector Title"
    assert result["content"] == "Hello World"  # stripped
    mock_page.locator.assert_called_once_with("h1")


@pytest.mark.asyncio
async def test_run_camoufox_selector_returns_none_content():
    """When element.text_content() returns None, content must be empty string."""
    service = BrowserService()

    mock_element = AsyncMock()
    mock_element.text_content = AsyncMock(return_value=None)

    mock_locator = MagicMock()
    mock_locator.first = mock_element

    mock_page = AsyncMock()
    mock_page.goto = AsyncMock()
    mock_page.title = AsyncMock(return_value="")
    mock_page.locator = MagicMock(return_value=mock_locator)
    mock_page.url = "https://example.com"

    mock_browser = AsyncMock()
    mock_browser.new_page = AsyncMock(return_value=mock_page)
    mock_browser.__aenter__ = AsyncMock(return_value=mock_browser)
    mock_browser.__aexit__ = AsyncMock(return_value=False)

    mock_async_camoufox_cls = MagicMock(return_value=mock_browser)
    mock_module = MagicMock()
    mock_module.AsyncCamoufox = mock_async_camoufox_cls

    with patch.dict(sys.modules, {"camoufox": MagicMock(), "camoufox.async_api": mock_module}):
        result = await service._run_camoufox(
            url="https://example.com", selector=".missing", timeout_ms=5000
        )

    assert result["content"] == ""
    assert result["title"] == ""
