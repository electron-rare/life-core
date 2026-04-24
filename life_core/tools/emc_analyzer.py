"""Placeholder bridge to the kicad-happy ``emc`` skill. Wired up in Sprint 2+."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EMCReport:
    rules_checked: int = 0
    violations: list[dict] = field(default_factory=list)


def analyze(pcb_path: str) -> EMCReport:
    """Return an empty report in Sprint 1 (generator does not invoke EMC yet)."""
    return EMCReport()
