"""Firmware behaviour comparator (P4 T1.9b).

Builds both the human and LLM PlatformIO projects with the ``native``
environment, then scores a 50/50 mix of compile success and flash-size
similarity.
"""
from __future__ import annotations

from pathlib import Path

from life_core.tools.platformio import build_native


def _flash_size_bytes(project_dir: str) -> int:
    """Return the size of the native build artifact (``program``)."""

    candidates = list(Path(project_dir).rglob("program"))
    if not candidates:
        return 0
    return candidates[0].stat().st_size


def compare(human_project: str, llm_project: str) -> dict:
    """Score the LLM firmware project against the human one."""

    hb = build_native(human_project)
    lb = build_native(llm_project)
    if hb.ok and lb.ok:
        compile_score = 1.0
    elif hb.ok != lb.ok:
        compile_score = 0.5
    else:
        compile_score = 0.0

    hs = _flash_size_bytes(human_project)
    ls = _flash_size_bytes(llm_project)
    if hs == 0 or ls == 0:
        size_score = 0.0
    else:
        size_score = max(0.0, 1.0 - abs(hs - ls) / max(hs, ls))

    score = 0.5 * compile_score + 0.5 * size_score
    return {
        "score": round(score, 4),
        "details": {
            "human_build_ok": hb.ok,
            "llm_build_ok": lb.ok,
            "human_size": hs,
            "llm_size": ls,
        },
    }
