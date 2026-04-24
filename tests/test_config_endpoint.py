"""V1.7 Track II Task 8 — /config exposure with secret scrubbing.

Tests the read-only /config endpoint: allowlisted env vars, full
models list, and network host map. Secret-like names
(*_KEY | *_SECRET | *_TOKEN | *_PASSWORD) are hard-blocked and their
values must never appear anywhere in the response body.
"""
from __future__ import annotations

import json
import os
import re

import pytest
from fastapi.testclient import TestClient


_SECRET_RE = re.compile(
    r"(?:_KEY|_SECRET|_TOKEN|_PASSWORD)$", re.IGNORECASE
)


@pytest.fixture(autouse=True)
def _bearer(monkeypatch):
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")


def _client() -> TestClient:
    from life_core.api import app

    return TestClient(app)


def test_config_exposes_known_safe_vars(monkeypatch):
    """Allowlisted non-secret env vars must surface verbatim."""
    monkeypatch.setenv("VLLM_BASE_URL", "http://kxkm-ai:8000")
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse:3000")
    monkeypatch.setenv(
        "KIKI_FULL_MODELS", "kiki-meta-code,kiki-niche-embedded"
    )

    resp = _client().get(
        "/config", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["env"]["VLLM_BASE_URL"] == "http://kxkm-ai:8000"
    assert body["env"]["LANGFUSE_HOST"] == "http://langfuse:3000"


def test_config_never_leaks_secret_values(monkeypatch):
    """No env value under a *_KEY|SECRET|TOKEN|PASSWORD name may
    appear anywhere in the response body, ever."""
    secret_values = {
        "OPENAI_API_KEY": "sk-should-not-leak",
        "OLLAMA_BEARER_TOKEN": "tok-should-not-leak",
        "LITELLM_MASTER_KEY": "mk-should-not-leak",
        "DB_PASSWORD": "p4ss-should-not-leak",
        "ANTHROPIC_API_KEY": "ant-should-not-leak",
        "JWT_SECRET": "jwt-should-not-leak",
    }
    for name, val in secret_values.items():
        monkeypatch.setenv(name, val)

    resp = _client().get(
        "/config", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    blob = json.dumps(resp.json())
    for val in secret_values.values():
        assert val not in blob, f"secret {val!r} leaked in response"

    # Secondary defense — scan every env value matching the
    # secret regex currently in os.environ and assert none appears
    # in the body.
    for name, val in os.environ.items():
        if not val:
            continue
        if _SECRET_RE.search(name):
            assert val not in blob, (
                f"secret env value for {name!r} leaked in response"
            )


def test_config_redacts_allowlisted_secret_names(monkeypatch):
    """If a secret-like name sneaks into the env, it must either be
    absent or explicitly redacted — never surfaced."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "sekret")

    resp = _client().get(
        "/config", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    env = resp.json()["env"]
    # LIFE_INTERNAL_BEARER is not in the allowlist and would match
    # no secret pattern (BEARER != KEY|SECRET|TOKEN|PASSWORD), so
    # it simply should not appear. This documents the allowlist.
    assert "LIFE_INTERNAL_BEARER" not in env
    # Any key present must not match the secret regex.
    for name in env:
        assert not _SECRET_RE.search(name), (
            f"secret-like name {name!r} surfaced in env"
        )


def test_config_includes_models_and_network(monkeypatch):
    """models is a list; network.hosts is a non-empty list of the
    7 Prometheus DEFAULT_TARGETS hosts."""
    monkeypatch.setenv(
        "KIKI_FULL_MODELS", "kiki-meta-code,kiki-meta-embedded"
    )

    resp = _client().get(
        "/config", headers={"Authorization": "Bearer sekret"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    assert isinstance(body["models"], list)
    assert "kiki-meta-code" in body["models"]
    assert "kiki-meta-embedded" in body["models"]

    assert "network" in body
    assert "hosts" in body["network"]
    hosts = body["network"]["hosts"]
    assert isinstance(hosts, list)
    # Must match Prometheus scraper DEFAULT_TARGETS (spec §5.7, 7 hosts).
    from life_core.monitoring.prometheus_scraper import DEFAULT_TARGETS

    assert set(hosts) == {t.host for t in DEFAULT_TARGETS}
    assert len(hosts) == len(DEFAULT_TARGETS) == 7


def test_config_secret_pattern_helper():
    """Unit-level coverage of the is_secret_key helper."""
    from life_core.integrations.config_exposure import is_secret_key

    assert is_secret_key("OPENAI_API_KEY")
    assert is_secret_key("LITELLM_MASTER_KEY")
    assert is_secret_key("OLLAMA_BEARER_TOKEN")
    assert is_secret_key("DB_PASSWORD")
    assert is_secret_key("SOME_SECRET")
    assert is_secret_key("openai_api_key")  # case-insensitive
    assert not is_secret_key("VLLM_BASE_URL")
    assert not is_secret_key("LANGFUSE_HOST")
    assert not is_secret_key("OLLAMA_URL")


def test_config_requires_auth():
    """Missing bearer must 401."""
    resp = _client().get("/config")
    assert resp.status_code == 401


def test_config_wrong_bearer_rejected():
    """Wrong bearer must 401."""
    resp = _client().get(
        "/config", headers={"Authorization": "Bearer nope"}
    )
    assert resp.status_code == 401
