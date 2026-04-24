"""V1.8 Wave B — OpenAI-compat /v1/embeddings contract tests."""
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from life_core.api import app


def _auth_headers() -> dict:
    # V1_AUTH_DEPS accepts LIFE_INTERNAL_BEARER or a Keycloak JWT;
    # tests run with the internal bearer path.
    return {"Authorization": "Bearer test-internal-bearer"}


@patch("life_core.api.embed_backend", new_callable=AsyncMock)
def test_embeddings_accepts_string_input(mock_embed, monkeypatch):
    """V1.8 axis 10: input=str must return one data[] row."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "test-internal-bearer")
    mock_embed.return_value = [[0.1, 0.2, 0.3]]

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={"input": "hello", "model": "tei/bge-small"},
        headers=_auth_headers(),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == "tei/bge-small"
    assert len(body["data"]) == 1
    assert body["data"][0]["object"] == "embedding"
    assert body["data"][0]["index"] == 0
    assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert "usage" in body
    assert body["usage"]["prompt_tokens"] >= 1
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"]


@patch("life_core.api.embed_backend", new_callable=AsyncMock)
def test_embeddings_accepts_array_input(mock_embed, monkeypatch):
    """V1.8 axis 10: input=list[str] must return one data[] row per string."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "test-internal-bearer")
    mock_embed.return_value = [[0.1, 0.2], [0.3, 0.4]]

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={"input": ["a", "b"], "model": "tei/bge-small"},
        headers=_auth_headers(),
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["data"]) == 2
    assert [row["index"] for row in body["data"]] == [0, 1]
    assert body["data"][1]["embedding"] == [0.3, 0.4]
    assert body["usage"]["prompt_tokens"] >= 2


@patch("life_core.api.embed_backend", new_callable=AsyncMock)
def test_embeddings_rejects_empty_list(mock_embed, monkeypatch):
    """V1.8 axis 10: input=[] must return HTTP 400."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "test-internal-bearer")
    mock_embed.return_value = []

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={"input": []},
        headers=_auth_headers(),
    )
    assert response.status_code == 400
    assert "non-empty" in response.json()["detail"].lower()


@patch("life_core.api.embed_backend", new_callable=AsyncMock)
def test_embeddings_wraps_backend_failure_as_502(mock_embed, monkeypatch):
    """V1.8 axis 10: backend exception becomes HTTP 502."""
    monkeypatch.setenv("LIFE_INTERNAL_BEARER", "test-internal-bearer")
    mock_embed.side_effect = RuntimeError("TEI unreachable")

    client = TestClient(app)
    response = client.post(
        "/v1/embeddings",
        json={"input": "hello"},
        headers=_auth_headers(),
    )
    assert response.status_code == 502
    assert "TEI unreachable" in response.json()["detail"]


@patch("life_core.api.embed_backend", new_callable=AsyncMock)
def test_embeddings_requires_auth(mock_embed):
    """V1.8 axis 10: missing bearer must not reach the handler."""
    mock_embed.return_value = [[0.0]]
    client = TestClient(app)
    response = client.post("/v1/embeddings", json={"input": "x"})
    assert response.status_code in (401, 403)
    assert mock_embed.call_count == 0
