"""Tests for ``life_core.evaluations.comparators.simulation_diff`` (T1.9b)."""
from __future__ import annotations

from unittest.mock import patch

from life_core.evaluations.comparators import simulation_diff
from life_core.evaluations.comparators.simulation_diff import _rmse, compare
from life_core.tools.ngspice import SimulationResult


def test_rmse_empty_key_intersection_is_inf():
    assert _rmse({"a": 1.0}, {"b": 2.0}) == float("inf")


def test_rmse_exact_match_is_zero():
    assert _rmse({"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}) == 0.0


def test_rmse_known_value():
    """RMSE of (0,1) vs (1,2) over two keys is ``sqrt(2/2) = 1.0``."""
    assert _rmse({"a": 0.0, "b": 1.0}, {"a": 1.0, "b": 2.0}) == 1.0


def test_compare_non_converged_returns_zero():
    """If either simulation fails to converge → score 0."""
    with patch.object(
        simulation_diff,
        "simulate",
        return_value=SimulationResult(converged=False),
    ):
        result = compare("/h.cir", "/l.cir")
    assert result["score"] == 0.0
    assert result["details"]["converged"] is False


def test_compare_converged_within_tolerance():
    """Identical operating points → score 1.0."""
    sim = SimulationResult(
        converged=True,
        operating_points={"v_in": 5.0, "v_out": 3.3},
    )
    with patch.object(simulation_diff, "simulate", return_value=sim):
        result = compare("/h.cir", "/l.cir")
    assert result["score"] == 1.0
    assert "rmse" in result["details"]
    assert "tolerance" in result["details"]


def test_compare_converged_outside_tolerance():
    """Noticeable divergence → score < 1.0 but bounded ≥ 0."""
    human = SimulationResult(
        converged=True,
        operating_points={"v_in": 5.0, "v_out": 3.3},
    )
    llm = SimulationResult(
        converged=True,
        operating_points={"v_in": 5.0, "v_out": 1.0},
    )
    results = iter([human, llm])

    def _fake_simulate(_path: str) -> SimulationResult:
        return next(results)

    with patch.object(simulation_diff, "simulate", side_effect=_fake_simulate):
        result = compare("/h.cir", "/l.cir")
    assert 0.0 <= result["score"] < 1.0
    assert result["details"]["rmse"] > 0.0


def test_compare_converged_one_side_fails():
    """Human converges but LLM does not → score 0."""
    results = iter(
        [
            SimulationResult(
                converged=True, operating_points={"v": 1.0}
            ),
            SimulationResult(converged=False),
        ]
    )

    def _fake_simulate(_path: str) -> SimulationResult:
        return next(results)

    with patch.object(simulation_diff, "simulate", side_effect=_fake_simulate):
        result = compare("/h.cir", "/l.cir")
    assert result["score"] == 0.0
