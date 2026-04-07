"""ACP client for goose — subprocess stdio transport (NDJSON JSON-RPC 2.0)."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from dataclasses import dataclass
from itertools import count
from typing import AsyncIterator

logger = logging.getLogger("life_core.goose")

GOOSE_BIN = os.environ.get("GOOSE_BIN", "goose")


@dataclass
class GooseSession:
    session_id: str
    working_dir: str = "."


class GooseClient:
    """Client that spawns `goose acp` as a subprocess and talks NDJSON JSON-RPC 2.0."""

    def __init__(self, goose_bin: str | None = None) -> None:
        self.goose_bin = goose_bin or GOOSE_BIN
        self._id_counter = count(1)
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()
        self._initialized = False

    @property
    def base_url(self) -> str:
        """Compat property for tests."""
        return f"stdio://{self.goose_bin}"

    async def _ensure_process(self) -> asyncio.subprocess.Process:
        """Start goose acp subprocess and initialize if needed. Caller must hold _lock."""
        if self._process is not None and self._process.returncode is None:
            return self._process
        env = {**os.environ}
        env.setdefault("GOOSE_PROVIDER", "anthropic")
        env.setdefault("GOOSE_MODEL", "claude-sonnet-4-20250514")
        self._process = await asyncio.create_subprocess_exec(
            self.goose_bin, "acp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        logger.info("Started goose acp subprocess (pid=%s)", self._process.pid)
        self._initialized = False
        return self._process

    def _next_id(self) -> int:
        return next(self._id_counter)

    async def _send(self, proc: asyncio.subprocess.Process, msg: dict) -> None:
        """Write a JSON-RPC message to stdin (NDJSON)."""
        line = json.dumps(msg, separators=(",", ":")) + "\n"
        proc.stdin.write(line.encode())  # type: ignore[union-attr]
        await proc.stdin.drain()  # type: ignore[union-attr]

    async def _read_line(self, proc: asyncio.subprocess.Process) -> dict | None:
        """Read one NDJSON line from stdout."""
        line = await proc.stdout.readline()  # type: ignore[union-attr]
        if not line:
            return None
        return json.loads(line)

    async def _send_and_receive(self, proc: asyncio.subprocess.Process, method: str, params: dict | None = None) -> dict:
        """Send request and wait for matching response. Caller must hold _lock."""
        req_id = self._next_id()
        msg: dict = {"jsonrpc": "2.0", "method": method, "id": req_id}
        if params:
            msg["params"] = params
        await self._send(proc, msg)

        while True:
            resp = await self._read_line(proc)
            if resp is None:
                raise ConnectionError("goose acp process closed stdout")
            if resp.get("id") == req_id:
                if "error" in resp:
                    raise RuntimeError(f"ACP error: {resp['error']}")
                return resp.get("result", {})
            # skip notifications during init

    async def _init_if_needed(self, proc: asyncio.subprocess.Process) -> None:
        """Send ACP initialize handshake if not done yet. Caller must hold _lock."""
        if self._initialized:
            return
        await self._send_and_receive(proc, "initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "life-core", "version": "1.0.0"},
        })
        self._initialized = True

    async def _rpc(self, method: str, params: dict | None = None) -> dict:
        """Send request, collect response. Thread-safe via lock."""
        async with self._lock:
            proc = await self._ensure_process()
            await self._init_if_needed(proc)
            return await self._send_and_receive(proc, method, params)

    async def _rpc_stream(
        self, method: str, params: dict | None = None,
    ) -> AsyncIterator[dict]:
        """Send request, yield notifications until response arrives. Holds lock for duration."""
        async with self._lock:
            proc = await self._ensure_process()
            await self._init_if_needed(proc)
            req_id = self._next_id()
            msg: dict = {"jsonrpc": "2.0", "method": method, "id": req_id}
            if params:
                msg["params"] = params
            await self._send(proc, msg)

        # Release lock for streaming — only one prompt at a time is supported by goose anyway
        proc = self._process
        while True:
            resp = await self._read_line(proc)  # type: ignore[arg-type]
            if resp is None:
                return
            if resp.get("id") == req_id:
                return  # final response
            if "method" in resp:
                yield resp  # notification

    async def initialize(self) -> dict:
        """ACP handshake (public, for tests)."""
        return await self._rpc("initialize", {
            "protocolVersion": 1,
            "clientCapabilities": {},
            "clientInfo": {"name": "life-core", "version": "1.0.0"},
        })

    async def create_session(self, working_dir: str = ".") -> GooseSession:
        """Create a new goose session."""
        result = await self._rpc("session/new", {"cwd": working_dir, "mcpServers": []})
        sid = result.get("sessionId", "") or result.get("session_id", "")
        return GooseSession(session_id=sid, working_dir=working_dir)

    async def load_session(self, session_id: str) -> GooseSession:
        """Resume an existing goose session."""
        await self._rpc("session/load", {"session_id": session_id})
        return GooseSession(session_id=session_id)

    async def prompt(self, session_id: str, text: str) -> AsyncIterator[dict]:
        """Send a prompt and stream notifications."""
        async for event in self._rpc_stream(
            "session/prompt",
            {"session_id": session_id, "prompt": [{"type": "text", "text": text}]},
        ):
            # Normalize to match our event format
            update = event.get("params", {}).get("update", {})
            kind = update.get("kind", "")
            if kind == "message":
                content_parts = update.get("content", [])
                for part in content_parts:
                    if part.get("type") == "text":
                        yield {
                            "jsonrpc": "2.0",
                            "method": "AgentMessageChunk",
                            "params": {"content": part["text"]},
                        }
            elif kind == "tool_use":
                yield {
                    "jsonrpc": "2.0",
                    "method": "ToolCall",
                    "params": {
                        "name": update.get("name", "unknown"),
                        "arguments": update.get("input", {}),
                    },
                }
            elif kind == "tool_result":
                yield {
                    "jsonrpc": "2.0",
                    "method": "ToolCallUpdate",
                    "params": {
                        "name": update.get("name", "unknown"),
                        "status": "done",
                        "result": str(update.get("output", "")),
                    },
                }
            else:
                yield event

    async def prompt_sync(self, session_id: str, text: str) -> str:
        """Send a prompt and collect the full text response."""
        parts: list[str] = []
        async for event in self.prompt(session_id, text):
            method = event.get("method", "")
            if method == "AgentMessageChunk":
                parts.append(event.get("params", {}).get("content", ""))
        return "".join(parts)

    async def cancel(self, session_id: str) -> None:
        """Cancel an in-progress prompt."""
        await self._rpc("session/cancel", {"session_id": session_id})

    async def health(self) -> dict:
        """Check if goose binary is available."""
        binary = shutil.which(self.goose_bin)
        if binary:
            return {"status": "ok", "binary": binary}
        return {"status": "error", "detail": f"{self.goose_bin} not found in PATH"}

    async def close(self) -> None:
        """Terminate the subprocess."""
        if self._process and self._process.returncode is None:
            self._process.terminate()
            await self._process.wait()
            self._process = None
