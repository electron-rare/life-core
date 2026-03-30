"""Tests de couverture pour ChatService et API FastAPI."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from life_core.router.providers.base import LLMResponse, LLMStreamChunk
from life_core.services.chat import ChatService


class _StubCache:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.set_calls = 0

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: str, ttl: int = 3600):
        self.store[key] = value
        self.set_calls += 1

    def get_stats(self):
        return {"hits": 0, "misses": 0, "size": len(self.store), "max_size": 1000}


class _StubRouter:
    def __init__(self, fail: bool = False):
        self.fail = fail
        class _StubProvider:
            async def list_models(self):
                return ["m1"]

        self.providers = {"mock": _StubProvider()}

    async def send(self, messages, model, provider=None, **kwargs):
        if self.fail:
            raise RuntimeError("router failure")
        return LLMResponse(content=f"ok:{messages[-1]['content']}", model=model, provider=provider or "mock")

    async def stream(self, messages, model, provider=None, **kwargs):
        if self.fail:
            raise RuntimeError("stream failure")
        yield LLMStreamChunk(content="part1", model=model)
        yield LLMStreamChunk(content="part2", model=model, finish_reason="stop")

    async def health_check_all(self):
        return {"mock": True}

    def list_available_providers(self):
        return ["mock"]

    def get_provider_status(self):
        return {"mock": True}


class _StubRag:
    async def augment_context(self, query: str, top_k: int = 3):
        return f"ctx-for:{query}"

    def get_stats(self):
        return {"indexed": 0}


@pytest.mark.asyncio
async def test_chat_service_cache_then_hit():
    cache = _StubCache()
    router = _StubRouter()
    svc = ChatService(router=router, cache=cache)

    messages = [{"role": "user", "content": "hello"}]
    first = await svc.chat(messages=messages, model="m1")
    second = await svc.chat(messages=messages, model="m1")

    assert first == "ok:hello"
    assert second == "ok:hello"
    assert cache.set_calls == 1
    assert svc.stats["requests"] == 2
    assert svc.stats["cache_hits"] == 1


@pytest.mark.asyncio
async def test_chat_service_rag_augmentation():
    cache = _StubCache()
    router = _StubRouter()
    svc = ChatService(router=router, cache=cache, rag=_StubRag())

    content = await svc.chat(
        messages=[{"role": "user", "content": "question"}],
        model="m1",
        use_rag=True,
    )

    assert "Context:" in content


@pytest.mark.asyncio
async def test_chat_service_stream_and_stats():
    svc = ChatService(router=_StubRouter(), cache=_StubCache(), rag=_StubRag())

    chunks = [
        chunk async for chunk in svc.stream_chat(
            messages=[{"role": "user", "content": "q"}], model="m1"
        )
    ]

    stats = svc.get_stats()
    assert [c.content for c in chunks] == ["part1", "part2"]
    assert stats["requests"] == 1
    assert "cache_stats" in stats
    assert "rag_stats" in stats


@pytest.mark.asyncio
async def test_api_routes_without_lifespan(monkeypatch):
    import life_core.api as api

    api.router = _StubRouter()
    api.cache = _StubCache()
    api.rag = _StubRag()
    api.chat_service = ChatService(router=api.router, cache=api.cache, rag=api.rag)

    health = await api.health()
    assert health.status == "ok"
    assert health.providers == ["mock"]

    models = await api.list_models()
    assert isinstance(models.models, list)

    chat = await api.chat(api.ChatRequest(messages=[{"role": "user", "content": "x"}]))
    assert chat.content.startswith("ok:")

    s = await api.stats()
    assert "chat_service" in s
    assert "router" in s


@pytest.mark.asyncio
async def test_api_error_paths(monkeypatch):
    import life_core.api as api

    api.router = None
    api.chat_service = None

    with pytest.raises(HTTPException):
        await api.health()

    with pytest.raises(HTTPException):
        await api.list_models()

    with pytest.raises(HTTPException):
        await api.chat(api.ChatRequest(messages=[{"role": "user", "content": "x"}]))


@pytest.mark.asyncio
async def test_api_lifespan_with_mocked_providers(monkeypatch):
    import life_core.api as api

    class _P:
        def __init__(self, api_key=None):
            self.provider_id = "p"
            self.is_available = True

        async def send(self, messages, model, **kwargs):
            return LLMResponse(content="ok", model=model, provider="p")

        async def stream(self, messages, model, **kwargs):
            yield LLMStreamChunk(content="ok", model=model)

        async def health_check(self):
            return True

        async def list_models(self):
            return ["m"]

    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")
    monkeypatch.setenv("OPENAI_API_KEY", "b")
    monkeypatch.setenv("GOOGLE_API_KEY", "c")
    monkeypatch.setattr(api, "ClaudeProvider", _P)
    monkeypatch.setattr(api, "OpenAIProvider", _P)
    monkeypatch.setattr(api, "GoogleProvider", _P)

    async with api.lifespan(api.app):
        assert api.router is not None
        assert api.chat_service is not None
