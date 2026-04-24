from pathlib import Path
from unittest.mock import MagicMock

import pytest

from life_core.artifacts.storage import read, write


@pytest.fixture
def volume(tmp_path) -> Path:
    return tmp_path / "artifacts"


def test_write_creates_file_with_versioned_path(volume):
    session = MagicMock()
    session.execute.return_value.scalar.return_value = None  # version will be 1
    ref = write(session, volume, "my-slug", "spec", b"content bytes", source="llm")
    assert ref.version == 1
    assert ref.storage_path.exists()
    assert ref.storage_path.read_bytes() == b"content bytes"
    assert ref.storage_path.is_relative_to(volume / "my-slug" / "spec" / "v1")


def test_write_hashes_content(volume):
    from life_core.artifacts.hash import content_hash

    session = MagicMock()
    session.execute.return_value.scalar.return_value = None
    ref = write(session, volume, "s", "spec", b"hello", source="llm")
    assert ref.content_hash == content_hash(b"hello")


def test_read_round_trip(volume):
    session = MagicMock()
    session.execute.return_value.scalar.return_value = None
    ref = write(session, volume, "s", "spec", b"roundtrip", source="llm")
    assert read(ref) == b"roundtrip"
