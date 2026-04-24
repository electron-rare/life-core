"""Tests for ``life_core.evaluations.comparators.firmware_behavior`` (T1.9b)."""
from __future__ import annotations

from unittest.mock import patch

from life_core.evaluations.comparators import firmware_behavior
from life_core.evaluations.comparators.firmware_behavior import compare
from life_core.tools.platformio import BuildResult


def test_compare_both_ok_same_size():
    """Both builds succeed and produce identical sizes → 1.0."""
    with patch.object(
        firmware_behavior,
        "build_native",
        return_value=BuildResult(ok=True, stdout="", stderr=""),
    ):
        with patch.object(
            firmware_behavior, "_flash_size_bytes", return_value=1024
        ):
            result = compare("/human", "/llm")

    assert result["score"] == 1.0
    assert result["details"]["human_build_ok"] is True
    assert result["details"]["llm_build_ok"] is True
    assert result["details"]["human_size"] == 1024
    assert result["details"]["llm_size"] == 1024


def test_compare_both_ok_size_differ():
    """Both builds OK but sizes differ → size_score reduced."""
    sizes = iter([1000, 1500])

    def _fake_size(_: str) -> int:
        return next(sizes)

    with patch.object(
        firmware_behavior,
        "build_native",
        return_value=BuildResult(ok=True, stdout="", stderr=""),
    ):
        with patch.object(
            firmware_behavior, "_flash_size_bytes", side_effect=_fake_size
        ):
            result = compare("/human", "/llm")

    # compile=1.0, size_score = 1 - 500/1500 ≈ 0.6667 ; total = 0.8333.
    assert 0.83 <= result["score"] <= 0.84


def test_compare_human_only_ok():
    """Only human compiles → compile_score = 0.5 ; sizes 0 → total 0.25."""
    builds = iter(
        [
            BuildResult(ok=True, stdout="", stderr=""),
            BuildResult(ok=False, stdout="", stderr="err"),
        ]
    )

    def _fake_build(_: str) -> BuildResult:
        return next(builds)

    with patch.object(firmware_behavior, "build_native", side_effect=_fake_build):
        with patch.object(
            firmware_behavior, "_flash_size_bytes", return_value=0
        ):
            result = compare("/human", "/llm")

    assert result["score"] == 0.25
    assert result["details"]["human_build_ok"] is True
    assert result["details"]["llm_build_ok"] is False


def test_compare_both_fail():
    """Neither builds → compile_score 0, size 0 → total 0."""
    with patch.object(
        firmware_behavior,
        "build_native",
        return_value=BuildResult(ok=False, stdout="", stderr="err"),
    ):
        with patch.object(
            firmware_behavior, "_flash_size_bytes", return_value=0
        ):
            result = compare("/human", "/llm")

    assert result["score"] == 0.0


def test_flash_size_uses_rglob(tmp_path):
    """_flash_size_bytes inspects ``program`` files via rglob."""
    pio_out = tmp_path / ".pio" / "build" / "native"
    pio_out.mkdir(parents=True)
    program = pio_out / "program"
    program.write_bytes(b"x" * 2048)

    assert (
        firmware_behavior._flash_size_bytes(str(tmp_path))
        == 2048
    )


def test_flash_size_returns_zero_when_missing(tmp_path):
    assert firmware_behavior._flash_size_bytes(str(tmp_path)) == 0
