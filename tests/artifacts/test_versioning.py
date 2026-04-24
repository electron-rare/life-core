from unittest.mock import MagicMock

from life_core.artifacts.versioning import next_version


def test_next_version_starts_at_1_when_no_prior():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = None
    assert next_version(session, "slug", "spec") == 1


def test_next_version_increments_when_prior_exists():
    session = MagicMock()
    session.execute.return_value.scalar.return_value = 3
    assert next_version(session, "slug", "spec") == 4
