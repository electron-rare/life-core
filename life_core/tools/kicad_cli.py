"""Thin subprocess wrapper around ``kicad-cli`` for DRC and netlist export."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field


@dataclass
class DRCResult:
    """Result of a ``kicad-cli pcb drc`` invocation."""

    passed: bool
    errors: list[dict] = field(default_factory=list)
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""


def run_drc(project_path: str, timeout_s: int = 60) -> DRCResult:
    """Run DRC on a KiCad PCB file and return a structured result.

    The wrapper is intentionally thin: it invokes ``kicad-cli pcb drc
    --format json`` and parses the JSON payload for violations. No attempt is
    made to second-guess the CLI's return code — ``passed`` is ``True`` iff the
    exit code is zero and no violations were reported.
    """

    cp = subprocess.run(
        ["kicad-cli", "pcb", "drc", "--format", "json", project_path],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )

    errors: list[dict] = []
    if cp.stdout:
        try:
            payload = json.loads(cp.stdout)
        except json.JSONDecodeError:
            payload = {}
        errors = list(payload.get("violations", []))
        errors.extend(payload.get("unconnected_items", []))

    return DRCResult(
        passed=(cp.returncode == 0 and not errors),
        errors=errors,
        returncode=cp.returncode,
        stdout=cp.stdout,
        stderr=cp.stderr,
    )


def export_netlist(sch_path: str, out_path: str, timeout_s: int = 60) -> bool:
    """Export a schematic netlist via ``kicad-cli sch export netlist``."""

    cp = subprocess.run(
        [
            "kicad-cli",
            "sch",
            "export",
            "netlist",
            "--output",
            out_path,
            sch_path,
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return cp.returncode == 0
