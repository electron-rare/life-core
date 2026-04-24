"""Tests for life_core.tools.platformio native builds."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from life_core.tools.platformio import BuildResult, build_native


def _mk(returncode: int, stdout: str = "", stderr: str = "") -> MagicMock:
    cp = MagicMock()
    cp.returncode = returncode
    cp.stdout = stdout
    cp.stderr = stderr
    return cp


def test_build_native_success(tmp_path) -> None:
    with patch(
        "life_core.tools.platformio.subprocess.run",
        return_value=_mk(0, stdout="Linking .pio/build/native/program\n"),
    ) as mock_run:
        result = build_native(str(tmp_path))

    assert isinstance(result, BuildResult)
    assert result.ok is True
    assert "Linking" in result.stdout
    assert result.stderr == ""
    args = mock_run.call_args.args[0]
    assert args[:2] == ["pio", "run"]
    assert "-e" in args and "native" in args
    assert mock_run.call_args.kwargs["cwd"] == str(tmp_path)


def test_build_native_failure_captures_stderr(tmp_path) -> None:
    with patch(
        "life_core.tools.platformio.subprocess.run",
        return_value=_mk(1, stdout="", stderr="undefined reference to `main`\n"),
    ):
        result = build_native(str(tmp_path), timeout_s=120)

    assert result.ok is False
    assert "undefined reference" in result.stderr
