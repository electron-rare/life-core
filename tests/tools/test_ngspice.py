"""Tests for life_core.tools.ngspice — delegates to spice_life.ngspice_wrapper."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from life_core.tools.ngspice import SimulationResult, simulate


def test_simulate_delegates_to_spice_life(tmp_path) -> None:
    raw = MagicMock(
        converged=True,
        operating_points={"VCC": 3.3, "GND": 0.0},
        errors=[],
    )
    with patch(
        "life_core.tools.ngspice._spice_life_run",
        return_value=raw,
    ) as mock_run:
        result = simulate(str(tmp_path / "circuit.cir"), timeout_s=45)

    assert isinstance(result, SimulationResult)
    assert result.converged is True
    assert result.operating_points == {"VCC": 3.3, "GND": 0.0}
    assert result.errors == []
    assert result.raw is raw
    mock_run.assert_called_once()
    # timeout must be forwarded as kwarg to match spice_life signature.
    assert mock_run.call_args.kwargs.get("timeout") == 45
