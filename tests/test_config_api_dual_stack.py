"""Tests for LIFE_CORE_LLM_BACKEND dual-stack config selection."""
import pytest
from life_core.config_api import resolve_llm_config


def test_backend_auto_uses_claude_runner_with_qwen_fallback(monkeypatch):
    monkeypatch.setenv("LIFE_CORE_LLM_BACKEND", "auto")
    cfg = resolve_llm_config()
    assert cfg["primary_model"] == "claude-runner-sonnet-4-7"
    assert "qwen-14b-awq-kxkm" in cfg["fallback_models"]


def test_backend_qwen_only(monkeypatch):
    monkeypatch.setenv("LIFE_CORE_LLM_BACKEND", "qwen")
    cfg = resolve_llm_config()
    assert cfg["primary_model"] == "qwen-14b-awq-kxkm"
    assert cfg["fallback_models"] == []


def test_backend_claude_runner_only(monkeypatch):
    monkeypatch.setenv("LIFE_CORE_LLM_BACKEND", "claude-runner")
    cfg = resolve_llm_config()
    assert cfg["primary_model"] == "claude-runner-sonnet-4-7"
    assert cfg["fallback_models"] == []
