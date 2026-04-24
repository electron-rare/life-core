"""V1.7 Task 5 — 7-host Prometheus scraper.

Polls `/metrics` on the 7 known hosts every SCRAPE_INTERVAL_SECONDS
and publishes one `infra.network.host` SSE event per host per tick.

The set of seven matches spec §5.7:

  electron-server, VM 192.168.0.119, CILS (macOS i7), GrosMac (M5),
  Studio (M3 Ultra, hosts kiki-router :9200),
  Tower (already monitored), KXKM-AI (already monitored).

Wired as an asyncio startup task by life_core.api.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

import httpx

# Import the event type enum defined in Plan 2 Task 1. INFRA_NETWORK_HOST
# is the exact member expected by the frontend useSSE hook and the
# /health aggregator.
from life_core.events.broker import EventBroker, get_broker
from life_core.events.schema import Event, EventType

logger = logging.getLogger(__name__)


# Spec §5.7 — 30 s tick for infra.network.host. Pinned in a test.
SCRAPE_INTERVAL_SECONDS: float = 30.0

# Per-host HTTP timeout. Anything slower than this counts as `up=false`
# — the cockpit treats latency > 3 s as effectively down anyway.
SCRAPE_TIMEOUT_SECONDS: float = 3.0


@dataclass(frozen=True)
class Target:
    host: str
    url: str


DEFAULT_TARGETS: tuple[Target, ...] = (
    Target("tower", "http://192.168.0.120:9100/metrics"),
    Target("kxkm-ai", "http://kxkm-ai:9100/metrics"),
    Target("electron-server", "http://electron-server:9100/metrics"),
    Target("vm-119", "http://192.168.0.119:9100/metrics"),
    Target("cils", "http://192.168.0.210:9100/metrics"),
    Target("grosmac", "http://grosmac:9100/metrics"),
    Target("studio", "http://studio:9100/metrics"),
)


async def scrape_host(
    client: httpx.AsyncClient, target: Target
) -> tuple[str, bool, float | None]:
    """Probe one host's /metrics endpoint.

    Returns `(host, up, latency_ms)`. `up` is True when the response
    is 2xx. `latency_ms` is the wall-clock round-trip measured in
    milliseconds, or `None` on timeout / error.
    """
    t0 = time.perf_counter()
    try:
        response = await client.get(target.url, timeout=SCRAPE_TIMEOUT_SECONDS)
    except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
        logger.debug("scrape: %s unreachable (%s)", target.host, exc)
        return (target.host, False, None)
    except Exception as exc:  # defensive — never break the loop
        logger.warning("scrape: %s unexpected error: %s", target.host, exc)
        return (target.host, False, None)

    latency_ms = (time.perf_counter() - t0) * 1000.0
    up = 200 <= response.status_code < 300
    return (target.host, up, latency_ms if up else None)


async def scrape_once(
    broker: EventBroker,
    targets: Iterable[Target] | None = None,
) -> None:
    """Run one scrape pass against all targets and publish events."""
    picked = tuple(targets) if targets is not None else DEFAULT_TARGETS
    async with httpx.AsyncClient() as client:
        results = await asyncio.gather(
            *(scrape_host(client, t) for t in picked),
            return_exceptions=False,
        )

    now = datetime.now(timezone.utc)
    for host, up, latency_ms in results:
        event = Event(
            type=EventType.INFRA_NETWORK_HOST,
            data={"host": host, "up": up, "latency_ms": latency_ms},
            timestamp=now,
        )
        await broker.publish(event)


async def run_scraper(broker: EventBroker) -> None:
    """Scrape loop. Cancelled on app shutdown.

    Any exception in `scrape_once` is logged and the loop continues —
    the cockpit should never lose its infra heartbeat because of a
    single-pass error.
    """
    logger.info(
        "prometheus scraper started (%d hosts, interval=%.1fs)",
        len(DEFAULT_TARGETS),
        SCRAPE_INTERVAL_SECONDS,
    )
    while True:
        try:
            await scrape_once(broker)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("scrape_once raised: %s", exc)
        await asyncio.sleep(SCRAPE_INTERVAL_SECONDS)


def install_startup_hook(app) -> None:  # pragma: no cover — wiring glue
    """Attach the scraper as a FastAPI startup task.

    Exposed for the life_core.api module so the wiring lives in one
    place. The created task is stored on `app.state._scrape_task`
    so the shutdown hook can cancel it.
    """

    @app.on_event("startup")
    async def _start_scrape() -> None:
        broker = get_broker()
        app.state._scrape_task = asyncio.create_task(run_scraper(broker))

    @app.on_event("shutdown")
    async def _stop_scrape() -> None:
        task = getattr(app.state, "_scrape_task", None)
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
