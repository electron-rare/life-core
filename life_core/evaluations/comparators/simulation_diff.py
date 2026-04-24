"""SPICE simulation-diff comparator (P4 T1.9b).

Runs ngspice on both the human and LLM netlists, computes the RMSE of
their operating points, and scores it against a ±5 % tolerance of the
average operating-point magnitude.
"""
from __future__ import annotations

from life_core.tools.ngspice import simulate


def _rmse(a: dict, b: dict) -> float:
    """Root-mean-square error across shared keys; ``inf`` on empty set."""

    keys = set(a) & set(b)
    if not keys:
        return float("inf")
    return (sum((a[k] - b[k]) ** 2 for k in keys) / len(keys)) ** 0.5


def compare(human_netlist: str, llm_netlist: str) -> dict:
    """Score the LLM netlist against the human one."""

    hr = simulate(human_netlist)
    lr = simulate(llm_netlist)
    if not (hr.converged and lr.converged):
        return {"score": 0.0, "details": {"converged": False}}

    rmse = _rmse(hr.operating_points, lr.operating_points)
    avg_abs = sum(abs(v) for v in hr.operating_points.values()) / max(
        len(hr.operating_points), 1
    )
    tol = 0.05 * max(avg_abs, 1e-9)
    if rmse <= tol:
        score = 1.0
    else:
        score = max(0.0, 1.0 - (rmse - tol) / max(avg_abs, 1e-9))
    return {
        "score": round(score, 4),
        "details": {"rmse": rmse, "tolerance": tol},
    }
