"""Dedicated browser runner API for isolated scrape execution."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from life_core.services.browser import (
    BrowserDependencyMissingError,
    BrowserRemoteRunnerError,
    BrowserService,
    BrowserServiceError,
)

logger = logging.getLogger(__name__)


class ScrapeRequest(BaseModel):
    """Browser runner request payload."""

    url: str = Field(..., min_length=1)
    selector: str | None = None
    timeout_ms: int = Field(default=15000, ge=1, le=120000)


class ScrapeResponse(BaseModel):
    """Browser runner response payload."""

    url: str
    title: str
    content: str


app = FastAPI(title="life-core browser-runner", version="0.1.0")
browser_service = BrowserService(force_local=True)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "browser-runner"}


@app.post("/scrape", response_model=ScrapeResponse)
async def scrape(request: ScrapeRequest) -> ScrapeResponse:
    try:
        result = await browser_service.scrape(
            url=request.url,
            selector=request.selector,
            timeout_ms=request.timeout_ms,
        )
        return ScrapeResponse(**result)
    except BrowserDependencyMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except BrowserRemoteRunnerError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except BrowserServiceError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("browser runner failed")
        raise HTTPException(status_code=500, detail="browser runner failed") from exc
