"""Contract test: all 4 generators succeed when the LLM backend returns
content consistent with what ClaudeRunnerProvider.send() delivers.

Architecture note
-----------------
Generators (SpecGenerator / KicadGenerator / FirmwareGenerator /
SpiceGenerator) call ``litellm.completion`` directly inside
``call_llm()``.  ClaudeRunnerProvider is the router-layer sidecar that
also exposes ``/v1/chat/completions``; it is not wired into the
generators yet (that is Task 10's backend-resolution wiring).

This test therefore:
  1. Patches ``litellm.completion`` in each generator module with a
     fixture that matches the content shape ClaudeRunnerProvider.send()
     would surface (plain text / JSON string).
  2. Patches secondary validators that make external syscalls (kiutils,
     kicad-cli ERC, PlatformIO pio run, ngspice simulate) so the test
     is self-contained and deterministic.
  3. Asserts GenerationOutcome.ok is True and data is non-empty for
     every generator — proving the end-to-end generator loop works with
     claude-runner-compatible content.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from life_core.generators.base import GenerationContext, GenerationOutcome
from life_core.generators.spec_generator import SpecGenerator
from life_core.generators.kicad_generator import KicadGenerator
from life_core.generators.firmware_generator import FirmwareGenerator
from life_core.generators.spice_generator import SpiceGenerator
from life_core.tools.kicad_cli import DRCResult
from life_core.tools.platformio import BuildResult
from life_core.tools.ngspice import SimulationResult

# ---------------------------------------------------------------------------
# Gold-fixture content — what ClaudeRunnerProvider.send().content looks like
# ---------------------------------------------------------------------------

_SPEC_CONTENT = (
    "---\n"
    "description: Sensor Node EU868\n"
    "inputs: [power_rail_3v3, i2c_bus]\n"
    "outputs: [lora_packet, uart_debug]\n"
    "constraints: [CE_RED, RoHS, REACH]\n"
    "acceptance_criteria: [rssi_gt_minus120, bme280_accuracy_0_5C]\n"
    "compliance: CE_RED\n"
    "---\n"
    "# Sensor Node EU868\n\n"
    "LoRa 868 MHz, ESP32-S3, BME280 environmental sensor.\n"
)

_KICAD_CONTENT = json.dumps(
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
        ],
        "wires": [{"pts": [[60, 50], [82, 50]]}],
        "labels": [{"at": [55, 40], "text": "VIN_5V"}],
    }
)

_FIRMWARE_CONTENT = json.dumps(
    {
        "platformio_ini": "[env:native]\nplatform = native\n",
        "src_main_cpp": "int main() { return 0; }",
    }
)

_SPICE_CONTENT = (
    "Sensor Node TB\n"
    ".op\n"
    "V1 vdd 0 3.3\n"
    "R1 vdd out 1k\n"
    "R2 out 0 1k\n"
    ".end"
)

# Map each generator class to its fixture
_FIXTURES: dict[type, str] = {
    SpecGenerator: _SPEC_CONTENT,
    KicadGenerator: _KICAD_CONTENT,
    FirmwareGenerator: _FIRMWARE_CONTENT,
    SpiceGenerator: _SPICE_CONTENT,
}


def _completion_mock(content: str):
    """Return a litellm-compatible completion dict for the given content."""
    return {
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 200,
            "total_tokens": 300,
        },
    }


def _make_ctx(deliverable_type: str, prompt_template: str) -> GenerationContext:
    return GenerationContext(
        deliverable_slug=f"contract-{deliverable_type}",
        deliverable_type=deliverable_type,
        prompt_template=prompt_template,
        llm_model="claude-runner/claude-sonnet-4-7",
        prompt_vars={"brief": "Sensor Node EU868", "constraints": [], "upstream": []},
        max_reprompts=0,
    )


# ---------------------------------------------------------------------------
# Parametrised contract test — 4 generators × 1 scenario each
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "generator_cls,deliverable_type,prompt_template",
    [
        (SpecGenerator, "spec", "spec.j2"),
        (KicadGenerator, "hardware", "kicad.j2"),
        (FirmwareGenerator, "firmware", "firmware.j2"),
        (SpiceGenerator, "simulation", "spice.j2"),
    ],
    ids=["spec", "kicad", "firmware", "spice"],
)
def test_generator_works_with_claude_runner_content(
    generator_cls: type,
    deliverable_type: str,
    prompt_template: str,
) -> None:
    """Each generator succeeds when fed claude-runner-compatible content.

    The test patches litellm.completion (the actual call site inside each
    generator's call_llm) with the gold fixture for that generator type,
    and also patches external validators so the test is fully hermetic.
    """
    content = _FIXTURES[generator_cls]
    completion_rv = _completion_mock(content)
    ctx = _make_ctx(deliverable_type, prompt_template)

    completion_module = f"life_core.generators.{generator_cls.__module__.split('.')[-1]}.completion"

    if generator_cls is SpecGenerator:
        with patch(completion_module, return_value=completion_rv):
            outcome: GenerationOutcome = generator_cls().generate(ctx)

    elif generator_cls is KicadGenerator:
        with patch(completion_module, return_value=completion_rv), patch(
            "life_core.generators.kicad_generator._render_kiutils_from_json",
            return_value="/tmp/contract_test.kicad_sch",
        ), patch(
            "life_core.generators.kicad_generator.run_erc",
            return_value=DRCResult(passed=True, errors=[]),
        ):
            outcome = generator_cls().generate(ctx)

    elif generator_cls is FirmwareGenerator:
        with patch(completion_module, return_value=completion_rv), patch(
            "life_core.generators.firmware_generator.build_native",
            return_value=BuildResult(ok=True, stdout="Success", stderr=""),
        ):
            outcome = generator_cls().generate(ctx)

    elif generator_cls is SpiceGenerator:
        with patch(completion_module, return_value=completion_rv), patch(
            "life_core.generators.spice_generator.simulate",
            return_value=SimulationResult(
                converged=True, operating_points={"v(out)": 1.65}, errors=[]
            ),
        ):
            outcome = generator_cls().generate(ctx)

    else:
        pytest.fail(f"Unhandled generator class: {generator_cls}")
        return  # unreachable — satisfies type checker

    assert outcome.ok is True, (
        f"{generator_cls.__name__} failed: {outcome.errors}"
    )
    assert len(outcome.data) > 0, (
        f"{generator_cls.__name__} returned empty data"
    )
