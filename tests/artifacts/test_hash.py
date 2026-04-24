from life_core.artifacts.hash import content_hash


def test_content_hash_deterministic():
    assert content_hash(b"abc") == content_hash(b"abc")


def test_content_hash_differs_for_different_bytes():
    assert content_hash(b"abc") != content_hash(b"abd")


def test_content_hash_format_sha256_hex():
    h = content_hash(b"abc")
    assert len(h) == 64
    assert set(h).issubset(set("0123456789abcdef"))
