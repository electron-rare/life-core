"""Shape check for scripts/test-all-providers.sh."""
from __future__ import annotations

import stat
from pathlib import Path


def test_harness_exists_and_is_executable():
    p = Path(__file__).resolve().parents[2] / "scripts" / "test-all-providers.sh"
    assert p.exists(), "scripts/test-all-providers.sh missing"
    mode = p.stat().st_mode
    assert mode & stat.S_IXUSR, "script not executable"
    body = p.read_text()
    assert "/v1/models" in body
    assert "/v1/chat/completions" in body
    assert "JSON_REPORT" in body
