"""Thin subprocess wrapper around PlatformIO (``pio run``)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass
class BuildResult:
    """Outcome of a PlatformIO build invocation."""

    ok: bool
    stdout: str
    stderr: str


def build_native(project_dir: str, timeout_s: int = 300) -> BuildResult:
    """Run ``pio run -e native`` inside ``project_dir`` and capture output.

    The wrapper forces ``text=True`` so callers receive decoded strings and
    can grep stderr for compiler diagnostics.
    """

    cp = subprocess.run(
        ["pio", "run", "-e", "native"],
        cwd=project_dir,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    return BuildResult(ok=(cp.returncode == 0), stdout=cp.stdout, stderr=cp.stderr)
