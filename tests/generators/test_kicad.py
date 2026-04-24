"""Tests for the KiCad generator (ADR-006, kiutils-backed)."""
from __future__ import annotations

import json
from unittest.mock import patch

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
        "life_core.generators.kicad_generator.run_drc",
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
        "life_core.generators.kicad_generator.run_drc",
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
        "life_core.generators.kicad_generator.run_drc",
        return_value=drc,
    ):
        gen = KicadGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("short" in e for e in outcome.errors)
