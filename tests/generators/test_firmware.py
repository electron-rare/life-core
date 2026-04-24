"""Tests for the PlatformIO firmware generator (T1.7a)."""
from __future__ import annotations

import json
from unittest.mock import patch

from life_core.generators.base import GenerationContext
from life_core.generators.firmware_generator import FirmwareGenerator
from life_core.tools.platformio import BuildResult


def _ctx() -> GenerationContext:
    return GenerationContext(
        deliverable_slug="s-fw",
        deliverable_type="firmware",
        prompt_template="firmware.j2",
        llm_model="mascarade-platformio",
        prompt_vars={"brief": "b", "constraints": [], "upstream": []},
        max_reprompts=0,
    )


def test_firmware_generator_ok_when_build_passes() -> None:
    payload = {
        "platformio_ini": "[env:native]\nplatform=native\n",
        "src_main_cpp": "int main(){return 0;}",
    }
    with patch(
        "life_core.generators.firmware_generator.completion",
        return_value={
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.firmware_generator.build_native",
        return_value=BuildResult(ok=True, stdout="Success", stderr=""),
    ):
        gen = FirmwareGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is True


def test_firmware_generator_fails_on_malformed_json() -> None:
    with patch(
        "life_core.generators.firmware_generator.completion",
        return_value={
            "choices": [{"message": {"content": "not json"}}],
            "usage": {},
        },
    ):
        gen = FirmwareGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any("JSON" in e for e in outcome.errors)


def test_firmware_generator_fails_on_missing_keys() -> None:
    payload = {"platformio_ini": "[env:native]\n"}  # missing src_main_cpp
    with patch(
        "life_core.generators.firmware_generator.completion",
        return_value={
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {},
        },
    ):
        gen = FirmwareGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any(
        "src_main_cpp" in e or "missing" in e.lower() for e in outcome.errors
    )


def test_firmware_generator_fails_on_compile_error() -> None:
    payload = {
        "platformio_ini": "[env:native]\n",
        "src_main_cpp": "broken",
    }
    with patch(
        "life_core.generators.firmware_generator.completion",
        return_value={
            "choices": [{"message": {"content": json.dumps(payload)}}],
            "usage": {},
        },
    ), patch(
        "life_core.generators.firmware_generator.build_native",
        return_value=BuildResult(
            ok=False, stdout="", stderr="error: undefined reference"
        ),
    ):
        gen = FirmwareGenerator()
        outcome = gen.generate(_ctx())
    assert outcome.ok is False
    assert any(
        "pio" in e.lower() or "undefined" in e for e in outcome.errors
    )
