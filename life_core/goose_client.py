"""ACP (Agent Communication Protocol) client for goosed."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from itertools import count
from typing import AsyncIterator

import httpx

logger = logging.getLogger("life_core.goose")

GOOSED_URL = os.environ.get("GOOSED_URL", "http://goosed:3000")


@dataclass
class GooseSession:
    session_id: str
    working_dir: str = "."


class GooseClient:
    """Client for the goosed ACP endpoint (JSON-RPC 2.0 over HTTP SSE)."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = base_url or GOOSED_URL
        self._id_counter = count(1)

    def _next_id(self) -> int:
        return next(self._id_counter)

    async def _rpc(
        self,
        method: str,
        params: dict | None = None,
        session_id: str | None = None,
    ) -> tuple[str | None, dict]:
        """Send a JSON-RPC 2.0 request, return (session_id_header, response_json)."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params:
            payload["params"] = params

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if session_id:
            headers["Acp-Session-Id"] = session_id

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self.base_url}/acp",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            sid = resp.headers.get("acp-session-id")
            return sid, resp.json()

    async def _stream_rpc(
        self,
        method: str,
        params: dict | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[dict]:
        """Send a JSON-RPC 2.0 request and stream SSE notifications back."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "id": self._next_id(),
        }
        if params:
            payload["params"] = params

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if session_id:
            headers["Acp-Session-Id"] = session_id

        async with httpx.AsyncClient(timeout=300.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/acp",
                json=payload,
                headers=headers,
            ) as resp:
                resp.raise_for_status()
                buf = ""
                async for chunk in resp.aiter_text():
                    buf += chunk
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                return
                            try:
                                yield json.loads(data_str)
                            except json.JSONDecodeError:
                                logger.warning("Malformed SSE data: %s", data_str[:100])

    async def create_session(self, working_dir: str = ".") -> GooseSession:
        """Create a new goosed session."""
        sid, _resp = await self._rpc("session/new", {"cwd": working_dir})
        return GooseSession(session_id=sid or "", working_dir=working_dir)

    async def prompt(self, session_id: str, text: str) -> AsyncIterator[dict]:
        """Send a prompt and stream ACP notifications (AgentMessageChunk, ToolCall, etc.)."""
        async for event in self._stream_rpc(
            "session/prompt",
            {"prompt": text},
            session_id=session_id,
        ):
            yield event

    async def prompt_sync(self, session_id: str, text: str) -> str:
        """Send a prompt and collect the full text response."""
        parts: list[str] = []
        async for event in self.prompt(session_id, text):
            method = event.get("method", "")
            if method == "AgentMessageChunk":
                content = event.get("params", {}).get("content", "")
                parts.append(content)
        return "".join(parts)

    async def cancel(self, session_id: str) -> None:
        """Cancel an in-progress prompt."""
        await self._rpc("session/cancel", {"session_id": session_id}, session_id=session_id)

    async def health(self) -> dict:
        """Check goosed health."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
