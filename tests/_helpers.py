"""Shared helpers for life-core tests.

Centralised utilities used across the test suite to avoid duplication.
"""
from __future__ import annotations

from pathlib import Path


def docker_available() -> bool:
    """Return True when a Docker socket is reachable on the host.

    Used by integration tests that hit the real Docker daemon. CI runners
    (GitHub Actions, Forgejo) typically do not bind `/var/run/docker.sock`
    into the test container, so the socket is absent and the test must
    skip instead of failing on an empty container list.
    """
    return Path("/var/run/docker.sock").exists()
