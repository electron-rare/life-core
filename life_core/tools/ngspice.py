"""ngspice wrapper that delegates to the sibling ``spice_life`` package.

The sibling submodule ``spice-life`` exposes ``spice_life.ngspice_wrapper.run``.
When ``spice-life`` is not installed as editable in the current environment we
fall back to a stub that raises at call time — tests patch ``_spice_life_run``
directly so import-time absence of the sibling does not prevent collection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:  # pragma: no cover - exercised only with spice-life installed
    from spice_life.ngspice_wrapper import run as _spice_life_run  # type: ignore
except ImportError:  # pragma: no cover - exercised only without spice-life

    def _spice_life_run(netlist_path: str, timeout: int = 60):  # type: ignore[override]
        raise NotImplementedError(
            "spice-life not installed; patch _spice_life_run in tests"
        )


@dataclass
class SimulationResult:
    """Structured result bridging spice_life's return object to life_core."""

    converged: bool
    operating_points: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    raw: Any = None


def simulate(netlist_path: str, timeout_s: int = 60) -> SimulationResult:
    """Run an ngspice simulation by delegating to ``spice_life``."""

    raw = _spice_life_run(netlist_path, timeout=timeout_s)
    return SimulationResult(
        converged=bool(getattr(raw, "converged", False)),
        operating_points=dict(getattr(raw, "operating_points", {}) or {}),
        errors=list(getattr(raw, "errors", []) or []),
        raw=raw,
    )
