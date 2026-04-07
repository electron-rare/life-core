"""Team members: Keycloak users + hardcoded AI agents."""
from __future__ import annotations

import logging
import os

import httpx

from life_core.projects.models import TeamMember

logger = logging.getLogger("life_core.projects.team")

_AI_AGENTS: list[TeamMember] = [
    TeamMember(id="forge", name="Forge", type="agent"),
    TeamMember(id="firmware-agent", name="Firmware Agent", type="agent"),
    TeamMember(id="qa-agent", name="QA Agent", type="agent"),
    TeamMember(id="goose", name="Goose", type="agent"),
]


async def get_team_members() -> list[TeamMember]:
    """Return team members: Keycloak users (if token set) + AI agents."""
    members: list[TeamMember] = []

    token = os.getenv("KEYCLOAK_ADMIN_TOKEN")
    keycloak_url = os.getenv("KEYCLOAK_URL", "")
    realm = os.getenv("KEYCLOAK_REALM", "master")

    if token and keycloak_url:
        try:
            url = f"{keycloak_url}/admin/realms/{realm}/users"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    url,
                    headers={"Authorization": f"Bearer {token}"},
                )
                resp.raise_for_status()
                users = resp.json()
            for u in users:
                members.append(
                    TeamMember(
                        id=u.get("id", ""),
                        name=f"{u.get('firstName', '')} {u.get('lastName', '')}".strip()
                        or u.get("username", u.get("id", "")),
                        type="human",
                        avatar_url=None,
                    )
                )
        except Exception as e:
            logger.warning("Keycloak users fetch failed: %s", e)

    members.extend(_AI_AGENTS)
    return members
