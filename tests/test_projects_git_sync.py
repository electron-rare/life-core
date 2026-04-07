"""Tests for projects/git_sync.py — fetch/push YAML from/to GitHub."""
from __future__ import annotations

import base64
import json as json_module
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from life_core.projects.git_sync import (
    fetch_remote_yaml,
    push_yaml,
    project_to_yaml,
    _headers,
)


# ---------------------------------------------------------------------------
# _headers
# ---------------------------------------------------------------------------


def test_headers_with_token():
    with patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test"}):
        h = _headers()
    assert h["Authorization"] == "Bearer ghp_test"
    assert "application/vnd.github" in h["Accept"]


def test_headers_without_token():
    with patch.dict("os.environ", {}, clear=True):
        h = _headers()
    assert "Authorization" not in h


# ---------------------------------------------------------------------------
# fetch_remote_yaml
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_remote_yaml_success():
    yaml_content = "kill_life:\n  project: test\n"
    encoded = base64.b64encode(yaml_content.encode()).decode()

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "content": encoded,
        "sha": "abc123sha",
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.git_sync.httpx.AsyncClient", return_value=mock_client):
        content, sha = await fetch_remote_yaml("test-project")

    assert content == yaml_content
    assert sha == "abc123sha"


@pytest.mark.asyncio
async def test_fetch_remote_yaml_with_newlines_in_base64():
    """GitHub base64 content includes newlines that must be stripped."""
    yaml_content = "kill_life:\n  project: test\n  client: ACME\n"
    encoded_with_newlines = base64.b64encode(yaml_content.encode()).decode()
    # Insert newlines like GitHub does
    encoded_with_newlines = encoded_with_newlines[:20] + "\n" + encoded_with_newlines[20:]

    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "content": encoded_with_newlines,
        "sha": "sha456",
    }

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.git_sync.httpx.AsyncClient", return_value=mock_client):
        content, sha = await fetch_remote_yaml("test")

    assert content == yaml_content


# ---------------------------------------------------------------------------
# push_yaml
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_yaml_success():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "commit": {"sha": "new_commit_sha"},
    }

    mock_client = AsyncMock()
    mock_client.put = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("life_core.projects.git_sync.httpx.AsyncClient", return_value=mock_client):
        commit_sha = await push_yaml("test-project", "yaml content", "old_sha", "chore: update")

    assert commit_sha == "new_commit_sha"
    # Verify the PUT payload
    call_args = mock_client.put.call_args
    payload = json_module.loads(call_args.kwargs["content"])
    assert payload["message"] == "chore: update"
    assert payload["sha"] == "old_sha"
    decoded = base64.b64decode(payload["content"]).decode()
    assert decoded == "yaml content"


# ---------------------------------------------------------------------------
# project_to_yaml
# ---------------------------------------------------------------------------


def test_project_to_yaml():
    project = {"project": "test", "client": "ACME", "gates": {}}
    result = project_to_yaml(project)
    assert "kill_life:" in result
    assert "project: test" in result
    assert "client: ACME" in result


def test_project_to_yaml_roundtrip():
    import yaml
    project = {"name": "round", "client": "Corp", "gates": {"s0": {"status": "passed"}}}
    yaml_str = project_to_yaml(project)
    parsed = yaml.safe_load(yaml_str)
    assert parsed["kill_life"]["name"] == "round"
    assert parsed["kill_life"]["gates"]["s0"]["status"] == "passed"
