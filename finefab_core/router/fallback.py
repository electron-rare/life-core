"""Fallback state tracker for provider failures."""

from __future__ import annotations


class FallbackState:
    def __init__(self) -> None:
        self.total_failures = 0

    def record_failure(self) -> None:
        self.total_failures += 1

    def get_failure_stats(self) -> dict[str, int]:
        return {"total_failures": self.total_failures}