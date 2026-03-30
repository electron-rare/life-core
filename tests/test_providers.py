"""Tests de couverture des providers Claude/OpenAI/Google (mockés)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from life_core.router.providers.claude import ClaudeProvider
from life_core.router.providers.groq import GroqProvider
from life_core.router.providers.google import GoogleProvider
from life_core.router.providers.mistral import MistralProvider
from life_core.router.providers.openai import OpenAIProvider


class _ClaudeMessages:
    def create(self, **kwargs):
        return SimpleNamespace(
            content=[SimpleNamespace(text="claude-ok")],
            usage=SimpleNamespace(input_tokens=11, output_tokens=22),
            stop_reason="end_turn",
        )

    def stream(self, **kwargs):
        class _Ctx:
            def __enter__(self):
                self.text_stream = ["a", "b"]
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

        return _Ctx()


class _OpenAICompletions:
    async def create(self, **kwargs):
        if kwargs.get("stream"):
            class _Stream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                def __aiter__(self):
                    async def _gen():
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="x"), finish_reason=None)]
                        )
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="y"), finish_reason="stop")]
                        )

                    return _gen()

            return _Stream()

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="openai-ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=7, completion_tokens=9),
        )


class _GoogleModel:
    async def generate_content_async(self, messages, **kwargs):
        if kwargs.get("stream"):
            class _AIter:
                def __aiter__(self):
                    async def _gen():
                        yield SimpleNamespace(text="g1")
                        yield SimpleNamespace(text="g2")

                    return _gen()

            return _AIter()

        return SimpleNamespace(
            text="google-ok",
            usage_metadata=SimpleNamespace(prompt_token_count=5, candidates_token_count=6),
            candidates=[SimpleNamespace(finish_reason=SimpleNamespace(name="STOP"))],
        )


class _MistralChat:
    async def complete_async(self, **kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="mistral-ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=4, completion_tokens=8),
        )

    async def stream_async(self, **kwargs):
        class _AIter:
            def __aiter__(self):
                async def _gen():
                    yield SimpleNamespace(data=SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="m1"))]))
                    yield SimpleNamespace(data=SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="m2"))]))

                return _gen()

        return _AIter()


class _GroqCompletions:
    async def create(self, **kwargs):
        if kwargs.get("stream"):
            class _Stream:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc, tb):
                    return False

                def __aiter__(self):
                    async def _gen():
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="g1"), finish_reason=None)]
                        )
                        yield SimpleNamespace(
                            choices=[SimpleNamespace(delta=SimpleNamespace(content="g2"), finish_reason="stop")]
                        )

                    return _gen()

            return _Stream()

        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="groq-ok"), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=7),
        )


@pytest.mark.asyncio
async def test_claude_provider_send_stream_health_success():
    provider = ClaudeProvider(api_key="k")
    provider._client = SimpleNamespace(messages=_ClaudeMessages())

    send_resp = await provider.send(messages=[{"role": "user", "content": "hi"}], model="m")
    assert send_resp.content == "claude-ok"

    chunks = [
        c async for c in provider.stream(messages=[{"role": "user", "content": "hi"}], model="m")
    ]
    assert [c.content for c in chunks] == ["a", "b"]

    assert await provider.health_check() is True
    assert await provider.list_models()


@pytest.mark.asyncio
async def test_openai_provider_send_stream_health_success():
    provider = OpenAIProvider(api_key="k")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=_OpenAICompletions()))

    send_resp = await provider.send(messages=[{"role": "user", "content": "hi"}], model="m")
    assert send_resp.content == "openai-ok"

    chunks = [
        c async for c in provider.stream(messages=[{"role": "user", "content": "hi"}], model="m")
    ]
    assert [c.content for c in chunks] == ["x", "y"]

    assert await provider.health_check() is True
    assert await provider.list_models()


@pytest.mark.asyncio
async def test_google_provider_send_stream_health_success():
    provider = GoogleProvider(api_key="k")
    provider._client = SimpleNamespace(
        GenerativeModel=lambda model: _GoogleModel(),
        types=SimpleNamespace(GenerationConfig=lambda **kwargs: kwargs),
    )

    send_resp = await provider.send(messages=[{"role": "user", "content": "hi"}], model="m")
    assert send_resp.content == "google-ok"

    chunks = [
        c async for c in provider.stream(messages=[{"role": "user", "content": "hi"}], model="m")
    ]
    assert [c.content for c in chunks] == ["g1", "g2"]

    assert await provider.health_check() is True
    assert await provider.list_models()


@pytest.mark.asyncio
async def test_provider_error_paths_set_unavailable():
    provider = ClaudeProvider(api_key="k")

    class _BrokenMessages:
        def create(self, **kwargs):
            raise RuntimeError("boom")

        def stream(self, **kwargs):
            raise RuntimeError("boom")

    provider._client = SimpleNamespace(messages=_BrokenMessages())

    with pytest.raises(RuntimeError):
        await provider.send(messages=[{"role": "user", "content": "hi"}], model="m")
    assert provider.is_available is False

    with pytest.raises(RuntimeError):
        chunks = [
            c
            async for c in provider.stream(
                messages=[{"role": "user", "content": "hi"}],
                model="m",
            )
        ]
        assert chunks

    assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_mistral_provider_send_stream_health_success():
    provider = MistralProvider(api_key="k")
    provider._client = SimpleNamespace(chat=_MistralChat())

    send_resp = await provider.send(messages=[{"role": "user", "content": "hi"}], model="m")
    assert send_resp.content == "mistral-ok"

    chunks = [
        c async for c in provider.stream(messages=[{"role": "user", "content": "hi"}], model="m")
    ]
    assert [c.content for c in chunks] == ["m1", "m2"]

    assert await provider.health_check() is True
    assert await provider.list_models()


@pytest.mark.asyncio
async def test_groq_provider_send_stream_health_success():
    provider = GroqProvider(api_key="k")
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=_GroqCompletions()))

    send_resp = await provider.send(messages=[{"role": "user", "content": "hi"}], model="m")
    assert send_resp.content == "groq-ok"

    chunks = [
        c async for c in provider.stream(messages=[{"role": "user", "content": "hi"}], model="m")
    ]
    assert [c.content for c in chunks] == ["g1", "g2"]

    assert await provider.health_check() is True
    assert await provider.list_models()
