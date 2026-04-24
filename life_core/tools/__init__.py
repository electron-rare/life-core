"""Tool wrappers for KiCad, PlatformIO, ngspice, and EMC analysis."""
from __future__ import annotations

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
]
