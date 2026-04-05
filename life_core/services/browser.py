"""Camoufox browser service used by scraping endpoints."""

from __future__ import annotations

import asyncio
import os
from urllib.parse import urlparse


class BrowserServiceError(RuntimeError):
    """Base error for browser service failures."""


class BrowserDependencyMissingError(BrowserServiceError):
    """Raised when camoufox package is not installed."""


class BrowserRemoteRunnerError(BrowserServiceError):
    """Raised when the dedicated browser runner fails."""


class BrowserService:
    """Minimal browser wrapper for Camoufox-powered page extraction."""

    def __init__(
        self,
        *,
        runner_url: str | None = None,
        allowed_hosts: set[str] | None = None,
        force_local: bool = False,
    ) -> None:
        self.runner_url = runner_url or os.environ.get("BROWSER_RUNNER_URL")
        self.engine = (os.environ.get("BROWSER_ENGINE", "camoufox").strip().lower() or "camoufox")
        self.enable_http_fallback = os.environ.get(
            "BROWSER_ENABLE_HTTP_FALLBACK", "true"
        ).strip().lower() not in {"0", "false", "no"}
        env_hosts = os.environ.get("BROWSER_ALLOWED_HOSTS", "")
        configured_hosts = allowed_hosts if allowed_hosts is not None else {
            host.strip().lower()
            for host in env_hosts.split(",")
            if host.strip()
        }
        self.allowed_hosts = configured_hosts
        self.force_local = force_local

    def _validate_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise BrowserServiceError("url must be a valid http(s) URL")

        if self.allowed_hosts:
            hostname = (parsed.hostname or "").lower()
            allowed = any(
                hostname == candidate or hostname.endswith(f".{candidate}")
                for candidate in self.allowed_hosts
            )
            if not allowed:
                raise BrowserServiceError("url host is not allowed by browser runner policy")

    async def scrape(
        self,
        *,
        url: str,
        selector: str | None = None,
        timeout_ms: int = 15000,
    ) -> dict[str, str]:
        self._validate_url(url)
        if timeout_ms <= 0:
            raise BrowserServiceError("timeout_ms must be > 0")

        if self.runner_url and not self.force_local:
            return await self._run_remote(
                url=url,
                selector=selector,
                timeout_ms=timeout_ms,
            )

        if self.engine == "http":
            return await self._run_http_fetch(url=url, selector=selector, timeout_ms=timeout_ms)

        camoufox_budget_s = min((timeout_ms / 1000) + 1.0, 8.0)
        try:
            return await asyncio.wait_for(
                self._run_camoufox(url=url, selector=selector, timeout_ms=timeout_ms),
                timeout=camoufox_budget_s,
            )
        except BrowserDependencyMissingError:
            if not self.enable_http_fallback:
                raise
        except Exception:
            if not self.enable_http_fallback:
                raise

        return await self._run_http_fetch(url=url, selector=selector, timeout_ms=timeout_ms)

    async def _run_remote(
        self,
        *,
        url: str,
        selector: str | None,
        timeout_ms: int,
    ) -> dict[str, str]:
        import httpx

        target = self.runner_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=(timeout_ms / 1000) + 20.0) as client:
                response = await client.post(
                    f"{target}/scrape",
                    json={
                        "url": url,
                        "selector": selector,
                        "timeout_ms": timeout_ms,
                    },
                )
        except Exception as exc:  # noqa: BLE001
            raise BrowserRemoteRunnerError(f"browser runner request failed: {exc}") from exc

        if response.status_code >= 400:
            raise BrowserRemoteRunnerError(
                f"browser runner returned HTTP {response.status_code}: {response.text}"
            )

        data = response.json()
        return {
            "url": str(data.get("url", "")),
            "title": str(data.get("title", "")),
            "content": str(data.get("content", "")),
        }

    async def _run_http_fetch(
        self,
        *,
        url: str,
        selector: str | None,
        timeout_ms: int,
    ) -> dict[str, str]:
        import httpx
        from bs4 import BeautifulSoup
        from soupsieve import SelectorSyntaxError

        async with httpx.AsyncClient(
            follow_redirects=True,
            timeout=(timeout_ms / 1000) + 5.0,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        final_url = str(response.url)
        raw_content = response.text
        title = ""

        if "html" not in response.headers.get("content-type", "").lower():
            return {
                "url": final_url,
                "title": "",
                "content": "" if selector else raw_content,
            }

        soup = BeautifulSoup(raw_content, "html.parser")
        if soup.title:
            title = soup.title.get_text(" ", strip=True)

        if selector:
            try:
                match = soup.select_one(selector)
            except SelectorSyntaxError as exc:
                raise BrowserServiceError(f"invalid CSS selector: {selector}") from exc
            content = match.get_text(" ", strip=True) if match else ""
        else:
            content = raw_content

        return {
            "url": final_url,
            "title": title,
            "content": content,
        }

    async def _run_camoufox(
        self,
        *,
        url: str,
        selector: str | None,
        timeout_ms: int,
    ) -> dict[str, str]:
        try:
            from camoufox.async_api import AsyncCamoufox
        except ImportError as exc:
            raise BrowserDependencyMissingError(
                "camoufox is not installed in life-core environment"
            ) from exc

        async with AsyncCamoufox(headless=True) as browser:
            page = await browser.new_page()
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            title = await page.title()

            if selector:
                element = page.locator(selector).first
                extracted = await element.text_content()
                content = (extracted or "").strip()
            else:
                content = await page.content()

            return {
                "url": page.url,
                "title": title or "",
                "content": content or "",
            }
