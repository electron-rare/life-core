"""Minimal circuit breaker used by the phase 1 router scaffold."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CircuitBreaker:
    fail_max: int = 3
    failure_count: int = 0

    def allow_request(self) -> bool:
        return self.failure_count < self.fail_max

    def record_success(self) -> None:
        self.failure_count = 0

    def record_failure(self) -> None:
        self.failure_count += 1