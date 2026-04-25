"""Tool wrappers for KiCad, PlatformIO, ngspice, EMC, and cad-mcp client."""
from __future__ import annotations

from .cad_mcp_client import format_partial_read_for_prompt, read_partial_sch
from .emc_analyzer import EMCReport, analyze
from .kicad_cli import DRCResult, export_netlist, run_drc
from .ngspice import SimulationResult, simulate
from .platformio import BuildResult, build_native

__all__ = [
    "run_drc",
    "export_netlist",
    "DRCResult",
    "build_native",
    "BuildResult",
    "simulate",
    "SimulationResult",
    "analyze",
    "EMCReport",
    "read_partial_sch",
    "format_partial_read_for_prompt",
]
