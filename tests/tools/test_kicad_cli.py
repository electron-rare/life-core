"""Tests for life_core.tools.kicad_cli — thin subprocess wrappers."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from life_core.tools.kicad_cli import DRCResult, export_netlist, run_drc


def _mk_completed(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    cp = MagicMock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def test_run_drc_success_parses_empty_errors(tmp_path) -> None:
    payload = {"violations": [], "unconnected_items": []}
    with patch(
        "life_core.tools.kicad_cli.subprocess.run",
        return_value=_mk_completed(0, stdout=json.dumps(payload)),
    ) as mock_run:
        result = run_drc(str(tmp_path / "board.kicad_pcb"), timeout_s=30)

    assert isinstance(result, DRCResult)
    assert result.passed is True
    assert result.errors == []
    assert result.returncode == 0
    mock_run.assert_called_once()
    args = mock_run.call_args.args[0]
    assert args[0] == "kicad-cli"
    assert "drc" in args
    assert "--format" in args and "json" in args


def test_run_drc_failure_captures_errors(tmp_path) -> None:
    payload = {
        "violations": [{"type": "clearance", "description": "track too close"}],
        "unconnected_items": [],
    }
    with patch(
        "life_core.tools.kicad_cli.subprocess.run",
        return_value=_mk_completed(1, stdout=json.dumps(payload)),
    ):
        result = run_drc(str(tmp_path / "board.kicad_pcb"))

    assert result.passed is False
    assert len(result.errors) == 1
    assert result.errors[0]["type"] == "clearance"
    assert result.returncode == 1


def test_export_netlist_invokes_sch_export() -> None:
    with patch(
        "life_core.tools.kicad_cli.subprocess.run",
        return_value=_mk_completed(0, stdout="ok"),
    ) as mock_run:
        ok = export_netlist("/tmp/design.kicad_sch", "/tmp/out.net", timeout_s=15)

    assert ok is True
    args = mock_run.call_args.args[0]
    assert args[:4] == ["kicad-cli", "sch", "export", "netlist"]
    assert "/tmp/design.kicad_sch" in args
    assert "/tmp/out.net" in args
