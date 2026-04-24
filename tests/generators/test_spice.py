"""Tests for the ngspice SPICE generator (T1.7b)."""
from __future__ import annotations

from unittest.mock import patch

from life_core.generators.base import GenerationContext
from life_core.generators.spice_generator import SpiceGenerator
from life_core.tools.ngspice import SimulationResult


def _ctx() -> GenerationContext:
    return GenerationContext(
        deliverable_slug="s-sim",
        deliverable_type="simulation",
        prompt_template="spice.j2",
        llm_model="mascarade-spice",
        prompt_vars={"brief": "b", "constraints": [], "upstream": []},
        max_reprompts=0,
    )


def test_spice_generator_ok_when_simulation_converges() -> None:
    net = "Title\n.op\nV1 1 0 5\nR1 1 0 1k\n.end"
    with patch(
        "life_core.generators.spice_generator.completion",
        return_value={
            "choices": [{"message": {"content": net}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.spice_generator.simulate",
        return_value=SimulationResult(
            converged=True, operating_points={"v(1)": 5.0}, errors=[]
        ),
    ):
        gen = SpiceGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is True


def test_spice_generator_fails_on_non_convergence() -> None:
    net = "Title\n.op\n.end"
    with patch(
        "life_core.generators.spice_generator.completion",
        return_value={
            "choices": [{"message": {"content": net}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.spice_generator.simulate",
        return_value=SimulationResult(
            converged=False, errors=["singular matrix"]
        ),
    ):
        gen = SpiceGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any(
        "converge" in e.lower() or "singular" in e for e in outcome.errors
    )


def test_spice_generator_rejects_missing_end_card() -> None:
    net = "Title\n.op\nV1 1 0 5"  # no .end
    with patch(
        "life_core.generators.spice_generator.completion",
        return_value={
            "choices": [{"message": {"content": net}}],
            "usage": {},
        },
    ):
        gen = SpiceGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any(".end" in e for e in outcome.errors)


def test_spice_generator_strips_triple_backticks() -> None:
    net = "```\nTitle\n.op\n.end\n```"
    with patch(
        "life_core.generators.spice_generator.completion",
        return_value={
            "choices": [{"message": {"content": net}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.spice_generator.simulate",
        return_value=SimulationResult(
            converged=True, operating_points={}, errors=[]
        ),
    ):
        gen = SpiceGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is True
