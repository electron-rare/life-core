"""Tests for projects/team.py — Keycloak users + AI agents."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from life_core.projects.team import get_team_members, _AI_AGENTS


@pytest.mark.asyncio
async def test_get_team_members_no_keycloak():
    """Without KEYCLOAK_ADMIN_TOKEN, only AI agents are returned."""
    with patch.dict("os.environ", {"KEYCLOAK_ADMIN_TOKEN": "", "KEYCLOAK_URL": ""}):
        members = await get_team_members()

    assert len(members) == len(_AI_AGENTS)
    types = {m.type for m in members}
    assert types == {"agent"}


@pytest.mark.asyncio
async def test_get_team_members_with_keycloak():
    """With Keycloak configured, returns users + AI agents."""
    keycloak_users = [
        {"id": "u1", "firstName": "Alice", "lastName": "Smith", "username": "alice"},
        {"id": "u2", "firstName": "", "lastName": "", "username": "bob"},
    ]
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = keycloak_users

    mock_client = AsyncMock()
    mock_client.get = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    env = {
        "KEYCLOAK_ADMIN_TOKEN": "admin-token",
        "KEYCLOAK_URL": "http://keycloak:8080",
        "KEYCLOAK_REALM": "finefab",
    }
    with patch.dict("os.environ", env):
        with patch("life_core.projects.team.httpx.AsyncClient", return_value=mock_client):
            members = await get_team_members()

    humans = [m for m in members if m.type == "human"]
    agents = [m for m in members if m.type == "agent"]

    assert len(humans) == 2
    assert humans[0].name == "Alice Smith"
    # bob has no first/last name, falls back to username
    assert humans[1].name == "bob"
    assert len(agents) == len(_AI_AGENTS)


@pytest.mark.asyncio
async def test_get_team_members_keycloak_error():
    """Keycloak failure is caught, still returns AI agents."""
    mock_client = AsyncMock()
    mock_client.get = AsyncMock(side_effect=Exception("connection refused"))
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    env = {
        "KEYCLOAK_ADMIN_TOKEN": "token",
        "KEYCLOAK_URL": "http://keycloak:8080",
    }
    with patch.dict("os.environ", env):
        with patch("life_core.projects.team.httpx.AsyncClient", return_value=mock_client):
            members = await get_team_members()

    assert len(members) == len(_AI_AGENTS)
    assert all(m.type == "agent" for m in members)


@pytest.mark.asyncio
async def test_ai_agents_include_expected_ids():
    """AI agents list includes forge, goose, etc."""
    ids = {a.id for a in _AI_AGENTS}
    assert "forge" in ids
    assert "goose" in ids
    assert "qa-agent" in ids
