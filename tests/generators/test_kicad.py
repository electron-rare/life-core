"""Tests for the KiCad generator (ADR-006, kiutils-backed)."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, patch

from life_core.generators.base import GenerationContext
from life_core.generators.kicad_generator import (
    KicadGenerator,
    _render_kiutils_from_json,
)
from life_core.tools.kicad_cli import DRCResult


def _ctx() -> GenerationContext:
    return GenerationContext(
        deliverable_slug="s-hw",
        deliverable_type="hardware",
        prompt_template="kicad.j2",
        llm_model="mascarade-kicad",
        prompt_vars={
            "brief": "b",
            "constraints": [],
            "upstream": [
                {
                    "type": "spec",
                    "version": 1,
                    "deliverable_slug": "s-spec",
                    "content": "# s",
                }
            ],
        },
        max_reprompts=0,
    )


def _json_payload() -> str:
    return json.dumps(
        {
            "components": [
                {
                    "reference": "U1",
                    "lib_id": "Regulator_Linear:AMS1117-3.3",
                    "value": "AMS1117-3.3",
                    "footprint": "Package_TO_SOT_SMD:SOT-223-3_TabPin2",
                    "at": [90, 50, 0],
                },
                {
                    "reference": "C1",
                    "lib_id": "Device:C",
                    "value": "10uF",
                    "footprint": "Capacitor_SMD:C_0603_1608Metric",
                    "at": [50, 50, 0],
                },
                {
                    "reference": "C2",
                    "lib_id": "Device:C",
                    "value": "22uF",
                    "footprint": "Capacitor_SMD:C_0603_1608Metric",
                    "at": [130, 50, 0],
                },
            ],
            "wires": [
                {"pts": [[60, 50], [82, 50]]},
                {"pts": [[98, 50], [122, 50]]},
            ],
            "labels": [
                {"at": [55, 40], "text": "VIN_5V"},
                {"at": [135, 40], "text": "VCC_3V3"},
            ],
        }
    )


def test_kicad_generator_ok_on_clean_drc() -> None:
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=DRCResult(passed=True, errors=[]),
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is True


def test_kicad_generator_fails_on_malformed_json() -> None:
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": "not valid JSON"}}],
            "usage": {},
        },
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("JSON" in e for e in outcome.errors)


def test_kicad_generator_fails_on_missing_keys() -> None:
    payload = json.dumps({"components": []})  # missing wires + labels
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": payload}}],
            "usage": {},
        },
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("missing keys" in e for e in outcome.errors)


def test_kicad_generator_rejects_empty_components() -> None:
    """Trivially empty schema (zero components) must not pass validation."""
    payload = json.dumps({"components": [], "wires": [], "labels": []})
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": payload}}],
            "usage": {},
        },
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("components" in e.lower() and "0" in e for e in outcome.errors)


def test_kicad_generator_rejects_below_min_components() -> None:
    """When ``min_components`` is overridden, fewer must be rejected."""
    # _json_payload has 3 components; require 30 via prompt_vars.
    ctx = _ctx()
    ctx.prompt_vars["min_components"] = 30
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ):
        gen = KicadGenerator()
        outcome = gen.generate(ctx)
    assert outcome.ok is False
    assert any("below threshold" in e for e in outcome.errors)


def test_kicad_generator_strips_triple_backticks() -> None:
    wrapped = f"```json\n{_json_payload()}\n```"
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": wrapped}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=DRCResult(passed=True, errors=[]),
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is True


def test_kicad_generator_reports_render_failure() -> None:
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        side_effect=RuntimeError("boom"),
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("kiutils render failed" in e for e in outcome.errors)


def test_render_kiutils_from_json_writes_valid_file(tmp_path) -> None:
    payload = json.loads(_json_payload())
    path = _render_kiutils_from_json(payload)
    from pathlib import Path

    content = Path(path).read_text()
    assert content.startswith("(kicad_sch")
    assert "Regulator_Linear:AMS1117-3.3" in content
    assert "VIN_5V" in content


def test_render_kiutils_from_json_tolerates_minimal_payload() -> None:
    path = _render_kiutils_from_json(
        {"components": [], "wires": [], "labels": []}
    )
    from pathlib import Path

    assert Path(path).exists()


def test_kicad_generator_fails_on_drc_errors() -> None:
    drc = DRCResult(
        passed=False,
        errors=[{"severity": "error", "description": "short"}],
    )
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=drc,
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("short" in e for e in outcome.errors)


def test_kicad_validate_score_decreases_with_drc_errors():
    """validate() score reflects ERC error count.

    Payload must clear the non-trivial-output check (>=1 component) so the
    test actually exercises the score-from-erc-errors branch.
    """
    payload = json.dumps(
        {
            "components": [
                {
                    "reference": "R1",
                    "lib_id": "Device:R",
                    "value": "1k",
                    "footprint": "Resistor_SMD:R_0603_1608Metric",
                    "at": [10, 10, 0],
                }
            ],
            "wires": [],
            "labels": [],
        }
    )
    with patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/x.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=DRCResult(
            passed=False,
            errors=[{"description": "e1"}, {"description": "e2"}],
        ),
    ):
        gen = KicadGenerator()
        result = gen.validate(payload.encode(), _ctx())
    assert len(result) == 3
    ok, errors, score = result
    assert ok is False
    assert score == 0.8  # 1.0 - 0.1*2


# ---------------------------------------------------------------------------
# Sprint 2 P2B — cad-mcp partial-read wiring tests.
# ---------------------------------------------------------------------------


def _ctx_with_partial_read(*, allow: bool = True) -> GenerationContext:
    ctx = _ctx()
    ctx.max_reprompts = 1  # one reprompt → two attempts → one bridge step
    if allow:
        ctx.prompt_vars["allow_partial_read"] = True
    return ctx


_PARTIAL_DATA = {
    "sch_text": "(kicad_sch ...)",
    "bom": [
        {"reference": "C1", "value": "10uF", "footprint": "C_0603"},
        {"reference": "U1", "value": "AMS1117", "footprint": "SOT-223"},
    ],
    "net_count": 7,
}


def test_kicad_happy_path_skips_partial_read() -> None:
    """First attempt succeeds → read_partial_sch is never awaited."""
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=DRCResult(passed=True, errors=[]),
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        ctx = _ctx_with_partial_read()
        outcome = gen.generate(ctx)
    assert outcome.ok is True
    assert outcome.attempts == 1
    fake_read.assert_not_awaited()
    assert "partial_read" not in ctx.prompt_vars


def test_kicad_failure_then_partial_read_dict_then_success() -> None:
    """First fails → partial_read returns dict → second sees it → succeeds."""
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "short to GND"}]
    )
    good_drc = DRCResult(passed=True, errors=[])
    captured: list[dict] = []

    def _completion(**_kwargs):
        # Snapshot the prompt_vars-derived prompt for inspection.
        captured.append({"prompt": _kwargs.get("messages")[0]["content"]})
        return {
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        }

    with patch(
        "life_core.generators.kicad_generator.completion",
        side_effect=_completion,
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        side_effect=[bad_drc, good_drc],
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        ctx = _ctx_with_partial_read()
        outcome = gen.generate(ctx)
    assert outcome.ok is True
    assert outcome.attempts == 2
    fake_read.assert_awaited_once()
    # Default version is 1 when partial_read_version is unset.
    args, kwargs = fake_read.call_args
    assert args[0] == ctx.deliverable_slug
    assert args[1] == 1
    assert kwargs.get("base_url") is None
    assert kwargs.get("timeout_s") == 5.0
    # The injected key survives in prompt_vars and the second prompt
    # contains the formatted block.
    assert "partial_read" in ctx.prompt_vars
    assert "kiutils parsed 2 component" in ctx.prompt_vars["partial_read"]
    assert len(captured) == 2
    assert "Partial read of the previous schematic" in captured[1]["prompt"]


def test_kicad_failure_then_partial_read_none_keeps_loop_going() -> None:
    """First fails → partial_read returns None → loop continues w/o key."""
    fake_read = AsyncMock(return_value=None)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "short to GND"}]
    )
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=bad_drc,
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        ctx = _ctx_with_partial_read()
        outcome = gen.generate(ctx)
    assert outcome.ok is False
    assert outcome.attempts == 2  # max_reprompts=1 → 2 attempts total
    fake_read.assert_awaited_once()
    assert "partial_read" not in ctx.prompt_vars


def test_kicad_partial_read_skipped_when_flag_falsy() -> None:
    """allow_partial_read=False → read_partial_sch is never called."""
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "open net"}]
    )
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=bad_drc,
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        ctx = _ctx_with_partial_read(allow=False)
        outcome = gen.generate(ctx)
    assert outcome.ok is False
    fake_read.assert_not_awaited()
    assert "partial_read" not in ctx.prompt_vars


def test_kicad_partial_read_uses_explicit_version_and_overrides() -> None:
    """``partial_read_version/base_url/timeout_s`` overrides are respected."""
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "missing footprint"}]
    )
    good_drc = DRCResult(passed=True, errors=[])
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        side_effect=[bad_drc, good_drc],
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        ctx = _ctx_with_partial_read()
        ctx.prompt_vars["partial_read_version"] = 4
        ctx.prompt_vars["partial_read_base_url"] = "http://other:9999/mcp"
        ctx.prompt_vars["partial_read_timeout_s"] = 1.5
        outcome = gen.generate(ctx)
    assert outcome.ok is True
    args, kwargs = fake_read.call_args
    assert args[1] == 4
    assert kwargs["base_url"] == "http://other:9999/mcp"
    assert kwargs["timeout_s"] == 1.5


def test_kicad_partial_read_not_invoked_after_last_attempt() -> None:
    """No partial read is fired *after* the final attempt's failure."""
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "open net"}]
    )
    ctx = _ctx_with_partial_read()
    ctx.max_reprompts = 0  # one attempt total → never bridge
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        return_value=bad_drc,
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        outcome = gen.generate(ctx)
    assert outcome.ok is False
    assert outcome.attempts == 1
    fake_read.assert_not_awaited()


def test_kicad_agenerate_runs_under_existing_event_loop() -> None:
    """Sync ``generate`` works when called from inside a running loop.

    Reproduces the orchestrator path (``async def run_agent`` → sync
    ``gen.generate(ctx)``). Without the worker-thread bridge this would
    raise ``RuntimeError: asyncio.run() cannot be called from a running
    event loop``.
    """
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "short"}]
    )
    good_drc = DRCResult(passed=True, errors=[])

    async def _driver():
        with patch(
            "life_core.generators.kicad_generator.completion",
            return_value={
                "choices": [{"message": {"content": _json_payload()}}],
                "usage": {},
            },
        ), patch(
            "life_core.generators.kicad_generator._render_kiutils_from_json",
            return_value="/tmp/out.kicad_sch",
        ), patch(
            "life_core.generators.kicad_generator.run_erc",
            side_effect=[bad_drc, good_drc],
        ), patch(
            "life_core.generators.kicad_generator.read_partial_sch",
            fake_read,
        ):
            gen = KicadGenerator()
            ctx = _ctx_with_partial_read()
            return gen.generate(ctx)

    outcome = asyncio.run(_driver())
    assert outcome.ok is True
    assert outcome.attempts == 2
    fake_read.assert_awaited_once()


def test_kicad_agenerate_direct_via_asyncio_run() -> None:
    """Direct ``asyncio.run(gen.agenerate(ctx))`` works without bridging."""
    fake_read = AsyncMock(return_value=_PARTIAL_DATA)
    bad_drc = DRCResult(
        passed=False, errors=[{"description": "short"}]
    )
    good_drc = DRCResult(passed=True, errors=[])
    with patch(
        "life_core.generators.kicad_generator.completion",
        return_value={
            "choices": [{"message": {"content": _json_payload()}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.kicad_generator._render_kiutils_from_json",
        return_value="/tmp/out.kicad_sch",
    ), patch(
        "life_core.generators.kicad_generator.run_erc",
        side_effect=[bad_drc, good_drc],
    ), patch(
        "life_core.generators.kicad_generator.read_partial_sch", fake_read
    ):
        gen = KicadGenerator()
        ctx = _ctx_with_partial_read()
        outcome = asyncio.run(gen.agenerate(ctx))
    assert outcome.ok is True
    assert outcome.attempts == 2
    fake_read.assert_awaited_once()
