"""Tests for POST /audit/analyze endpoint — LLM calls mocked."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# Lazy import to avoid startup errors if makelife is not installed
@pytest.fixture()
def client():
    from life_core.api import app
    return TestClient(app)


MOCK_ANALYSIS = {
    "issues": [
        {
            "type": "stale_recommendation",
            "severity": "high",
            "location": "Section 2",
            "description": "Uses deprecated tool",
            "suggestion": "Migrate to new tool",
        }
    ],
    "summary": "One stale recommendation found.",
}


def _mock_analyzer_single(result: dict):
    """Context manager that mocks AuditAnalyzer.analyze_single."""
    mock = MagicMock(return_value=result)
    return patch("life_core.audit_analyze_handler.AuditAnalyzer.analyze_single", mock)


def _mock_analyzer_cross(result: dict):
    """Context manager that mocks AuditAnalyzer.analyze_cross."""
    mock = MagicMock(return_value=result)
    return patch("life_core.audit_analyze_handler.AuditAnalyzer.analyze_cross", mock)


def test_analyze_single_returns_200(client, tmp_path):
    audit_file = tmp_path / "audit.md"
    audit_file.write_text("# Audit\n")

    with _mock_analyzer_single(MOCK_ANALYSIS):
        response = client.post(
            "/audit/analyze",
            json={"file_path": str(audit_file)},
        )

    assert response.status_code == 200
    data = response.json()
    assert "issues" in data
    assert len(data["issues"]) == 1
    assert data["mode"] == "single"


def test_analyze_single_returns_summary(client, tmp_path):
    audit_file = tmp_path / "audit.md"
    audit_file.write_text("# Audit\n")

    with _mock_analyzer_single(MOCK_ANALYSIS):
        response = client.post(
            "/audit/analyze",
            json={"file_path": str(audit_file)},
        )

    data = response.json()
    assert data["summary"] == MOCK_ANALYSIS["summary"]


def test_analyze_cross_returns_200(client, tmp_path):
    audit_a = tmp_path / "a.md"
    audit_b = tmp_path / "b.md"
    audit_a.write_text("# A\n")
    audit_b.write_text("# B\n")

    cross_result = {
        "issues": [
            {
                "type": "contradiction",
                "severity": "high",
                "files": [str(audit_a), str(audit_b)],
                "description": "Conflicting guidance",
                "suggestion": "Align both docs",
            }
        ],
        "summary": "One contradiction.",
    }

    with _mock_analyzer_cross(cross_result):
        response = client.post(
            "/audit/analyze",
            json={"file_path": str(audit_a), "cross_paths": [str(audit_b)]},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "cross"
    assert any(i["type"] == "contradiction" for i in data["issues"])


def test_analyze_file_not_found_returns_404(client):
    with patch(
        "life_core.audit_analyze_handler.AuditAnalyzer.analyze_single",
        side_effect=FileNotFoundError("not found"),
    ):
        response = client.post(
            "/audit/analyze",
            json={"file_path": "/nonexistent/audit.md"},
        )
    assert response.status_code == 404


def test_analyze_llm_error_returns_502(client, tmp_path):
    from makelife.audit_analyzer import AnalysisError

    audit_file = tmp_path / "audit.md"
    audit_file.write_text("# Audit\n")

    with patch(
        "life_core.audit_analyze_handler.AuditAnalyzer.analyze_single",
        side_effect=AnalysisError("LLM unavailable"),
    ):
        response = client.post(
            "/audit/analyze",
            json={"file_path": str(audit_file)},
        )
    assert response.status_code == 502
    assert "LLM analysis failed" in response.json()["detail"]
