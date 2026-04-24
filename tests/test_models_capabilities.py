"""V1.7 Track II Task 13 — /models capability typing."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")


def test_heuristic_embed():
    from life_core.models.capabilities import guess_capabilities

    assert guess_capabilities("nomic-embed-text") == ["embedding"]
    assert guess_capabilities("bge-m3-embed") == ["embedding"]


def test_heuristic_vision():
    from life_core.models.capabilities import guess_capabilities

    assert guess_capabilities("qwen-vl-7b") == ["vision"]
    assert guess_capabilities("some-vision-model") == ["vision"]
    assert guess_capabilities("mistral-small-vl") == ["vision"]


def test_heuristic_chat_default():
    from life_core.models.capabilities import guess_capabilities

    assert guess_capabilities("qwen-14b-awq-kxkm") == ["chat"]
    assert guess_capabilities("llama-3-70b") == ["chat"]


def test_explicit_override_wins():
    from life_core.models.capabilities import (
        CAPABILITY_OVERRIDES,
        guess_capabilities,
    )

    CAPABILITY_OVERRIDES["qwen-14b-awq-kxkm"] = ["chat", "tool"]
    try:
        assert guess_capabilities("qwen-14b-awq-kxkm") == [
            "chat",
            "tool",
        ]
    finally:
        CAPABILITY_OVERRIDES.pop("qwen-14b-awq-kxkm", None)


def test_v1_models_response_includes_capabilities(monkeypatch):
    from life_core.api import app

    monkeypatch.setenv(
        "KIKI_FULL_MODELS",
        "qwen-14b-awq-kxkm,nomic-embed-text,qwen-vl-7b",
    )
    monkeypatch.setenv(
        "KIKI_FULL_BASE_URL", "http://kxkm-ai:8000"
    )

    with TestClient(app) as client:
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer sekret"},
        )
    assert resp.status_code == 200
    body = resp.json()
    by_id = {m["id"]: m for m in body["data"]}
    assert by_id["qwen-14b-awq-kxkm"]["capabilities"] == ["chat"]
    assert by_id["nomic-embed-text"]["capabilities"] == ["embedding"]
    assert by_id["qwen-vl-7b"]["capabilities"] == ["vision"]
