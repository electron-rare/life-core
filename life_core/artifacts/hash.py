"""Content-addressed hashing for immutable artifacts."""
from __future__ import annotations

import hashlib


def content_hash(data: bytes) -> str:
    """Stable SHA-256 hex digest of the given bytes."""
    return hashlib.sha256(data).hexdigest()
