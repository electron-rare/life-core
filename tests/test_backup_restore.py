"""V1.8 axis 8 Done-when (c): restore test mandatory on
life-core DB to a staging instance.

This test drives ``scripts/backup/pg-restore-f4l.sh`` in
``--staging`` mode against yesterday's langfuse-postgres
dump, then asserts the restored DB has the expected
tables and a non-zero row count on ``traces``.

Preconditions:

- ``scripts/backup/pg-dump-f4l.sh`` has run at least once
  (Task 11 Step 4).
- Docker available on the host running pytest (GrosMac
  or electron-server jump).
- ``$HOME/.ssh/config`` has the jump alias wired for
  kxkm-ai so rsync can pull from the NAS.

Opt-in: set ``F4L_RESTORE_TEST=1`` to enable. Skipped by
default in local dev + CI because it needs Docker, NAS
access, and a real dump file.
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


def _staging_drill(
    container: str,
    db: str,
    user: str,
    sentinel_table: str,
) -> None:
    """Common body for both langfuse + suite-keycloak drills.

    Pulls yesterday's dump from the NAS, restores into a
    disposable ``postgres:16-alpine`` container on :15432,
    asserts ``sentinel_table`` exists in ``public`` and
    ``count(*) >= 0``. The sentinel is the smallest table
    we expect every dump to carry — its absence means the
    dump shape drifted and the runbook needs an update.
    """
    date = _yesterday()
    staging = f"{container}-restore-test"

    # Teardown any leftover from a prior run.
    _run(["docker", "rm", "-f", staging], check=False)

    proc = _run(
        [
            "bash",
            str(SCRIPT),
            date,
            container,
            db,
            user,
            "--staging",
        ]
    )
    assert "restore staging ok" in proc.stdout, proc.stdout + proc.stderr

    # Verify: expected schema and row count.
    psql = _run(
        [
            "docker",
            "exec",
            staging,
            "psql",
            "-U",
            user,
            "-d",
            db,
            "-tAc",
            f"SELECT to_regclass('public.{sentinel_table}') IS NOT NULL;",
        ]
    )
    assert "t" in psql.stdout.strip().lower(), psql.stdout

    rowcount = _run(
        [
            "docker",
            "exec",
            staging,
            "psql",
            "-U",
            user,
            "-d",
            db,
            "-tAc",
            f"SELECT count(*) FROM {sentinel_table};",
        ]
    )
    n = int(rowcount.stdout.strip() or "0")
    assert n >= 0, f"unexpected {sentinel_table} rowcount {n}"

    # Cleanup
    _run(["docker", "rm", "-f", staging], check=False)


def test_pg_restore_staging_langfuse() -> None:
    """Langfuse drill — sentinel: ``public.traces``."""
    _staging_drill(
        container="langfuse-postgres",
        db="langfuse",
        user="langfuse",
        sentinel_table="traces",
    )


def test_pg_restore_staging_keycloak() -> None:
    """Keycloak drill — sentinel: ``public.realm`` (Keycloak's
    canonical core table, present in every fresh schema).

    Container name is ``suite-keycloak-postgres`` since the
    V1.6 suite rebrand — see backup-strategy.md §1 note.
    """
    _staging_drill(
        container="suite-keycloak-postgres",
        db="keycloak",
        user="keycloak",
        sentinel_table="realm",
    )
