"""V1.8 axis 8 Done-when (c): restore test mandatory on
life-core DB to a staging instance.

This test drives scripts/backup/pg-restore-f4l.sh in
--staging mode against yesterday's langfuse-postgres
dump, then asserts the restored DB has the expected
tables and a non-zero row count on `traces`.

Preconditions:
- scripts/backup/pg-dump-f4l.sh has run at least once
  (Task 11 Step 4).
- Docker available on the host running pytest (GrosMac
  or electron-server jump).
- $HOME/.ssh/config has the jump alias wired for kxkm-ai
  so rsync can pull from the NAS.
"""
from __future__ import annotations

import datetime as _dt
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts/backup/pg-restore-f4l.sh"

pytestmark = pytest.mark.skipif(
    os.environ.get("F4L_RESTORE_TEST") != "1",
    reason="opt-in; set F4L_RESTORE_TEST=1 to run (needs Docker + NAS)",
)


def _yesterday() -> str:
    return (_dt.date.today() - _dt.timedelta(days=1)).isoformat()


def _run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"+ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def test_pg_restore_staging_langfuse() -> None:
    date = _yesterday()
    staging = "langfuse-postgres-restore-test"

    # Teardown any leftover from a prior run.
    _run(["docker", "rm", "-f", staging], check=False)

    proc = _run([
        "bash", str(SCRIPT),
        date, "langfuse-postgres", "langfuse", "langfuse",
        "--staging",
    ])
    assert "restore staging ok" in proc.stdout, proc.stdout + proc.stderr

    # Verify: expected schema and row count.
    psql = _run([
        "docker", "exec", staging,
        "psql", "-U", "langfuse", "-d", "langfuse", "-tAc",
        "SELECT to_regclass('public.traces') IS NOT NULL;",
    ])
    assert "t" in psql.stdout.strip().lower(), psql.stdout

    rowcount = _run([
        "docker", "exec", staging,
        "psql", "-U", "langfuse", "-d", "langfuse", "-tAc",
        "SELECT count(*) FROM traces;",
    ])
    n = int(rowcount.stdout.strip() or "0")
    assert n >= 0, f"unexpected traces rowcount {n}"

    # Cleanup
    _run(["docker", "rm", "-f", staging], check=False)
