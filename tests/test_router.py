"""Tests pour le routeur LLM."""

import pytest

from life_core.router import LiteLLMProvider, LLMResponse, Router


class MockProvider:
    """Mock provider pour les tests."""

    def __init__(
        self,
        provider_id: str,
        fail_send: bool = False,
        fail_stream: bool = False,
        fail_health_exception: bool = False,
    ):
        self.provider_id = provider_id
        self.is_available = True
        self.fail_send = fail_send
        self.fail_stream = fail_stream
        self.fail_health_exception = fail_health_exception

    async def send(self, messages, model, **kwargs):
        if self.fail_send:
            raise RuntimeError(f"send failed for {self.provider_id}")
        return LLMResponse(
            content=f"Mock response from {self.provider_id}",
            model=model,
            provider=self.provider_id,
            usage={"input_tokens": 10, "output_tokens": 20},
        )

    async def stream(self, messages, model, **kwargs):
        if self.fail_stream:
            raise RuntimeError(f"stream failed for {self.provider_id}")
        yield type("Chunk", (), {
            "content": f"Mock chunk from {self.provider_id}",
            "model": model,
            "finish_reason": None,
        })()

    async def health_check(self):
        if self.fail_health_exception:
            raise RuntimeError("health check explosion")
        return self.is_available


def test_router_creation():
    """Test la création du routeur."""
    router = Router()
    assert router.providers == {}
    assert router.primary_provider is None


def test_register_provider():
    """Test l'enregistrement d'un provider."""
    router = Router()
    mock = MockProvider("test")
    
    router.register_provider(mock, is_primary=True)
    
    assert "test" in router.providers
    assert router.primary_provider == "test"
    assert router._health_status["test"] is True


def test_list_available_providers():
    """Test la liste des providers."""
    router = Router()
    mock1 = MockProvider("mock1")
    mock2 = MockProvider("mock2")
    
    router.register_provider(mock1)
    router.register_provider(mock2, is_primary=True)
    
    providers = router.list_available_providers()
    assert "mock1" in providers
    assert "mock2" in providers


@pytest.mark.asyncio
async def test_send_with_primary_provider():
    """Test l'envoi via le provider primaire."""
    router = Router()
    mock = MockProvider("mock")
    router.register_provider(mock, is_primary=True)
    
    response = await router.send(
        messages=[{"role": "user", "content": "test"}],
        model="test-model"
    )
    
    assert response.provider == "mock"
    assert "Mock response" in response.content
    assert response.model == "test-model"


@pytest.mark.asyncio
async def test_send_with_specific_provider():
    """Test l'envoi via un provider spécifique."""
    router = Router()
    mock1 = MockProvider("mock1")
    mock2 = MockProvider("mock2")
    router.register_provider(mock1, is_primary=True)
    router.register_provider(mock2)
    
    response = await router.send(
        messages=[{"role": "user", "content": "test"}],
        model="test-model",
        provider="mock2"
    )
    
    assert response.provider == "mock2"


@pytest.mark.asyncio
async def test_health_check_all():
    """Test la vérification de santé globale."""
    router = Router()
    mock1 = MockProvider("mock1")
    mock2 = MockProvider("mock2")
    
    router.register_provider(mock1)
    router.register_provider(mock2)
    
    status = await router.health_check_all()
    assert status["mock1"] is True
    assert status["mock2"] is True


def test_get_provider_status():
    """Test l'obtention du statut."""
    router = Router()
    mock = MockProvider("mock")
    router.register_provider(mock)
    
    status = router.get_provider_status()
    assert status["mock"] is True


def test_no_primary_provider_error():
    """Test l'erreur quand aucun primary provider."""
    router = Router()
    
    with pytest.raises(ValueError, match="No primary provider configured"):
        import asyncio
        asyncio.run(router.send(
            messages=[{"role": "user", "content": "test"}],
            model="test"
        ))


def test_invalid_provider_error():
    """Test l'erreur avec un provider invalide."""
    router = Router()
    mock = MockProvider("mock")
    router.register_provider(mock, is_primary=True)
    
    with pytest.raises(ValueError, match="Provider invalid not registered"):
        import asyncio
        asyncio.run(router.send(
            messages=[{"role": "user", "content": "test"}],
            model="test",
            provider="invalid"
        ))


def test_litellm_provider_creation():
    """Test la création du provider LiteLLM."""
    provider = LiteLLMProvider(models=["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"])
    assert provider.provider_id == "litellm"
    assert provider.models == ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514"]
    assert provider.is_available is True


def test_litellm_provider_with_ollama():
    """Test la création du provider LiteLLM avec Ollama."""
    provider = LiteLLMProvider(
        models=["ollama/llama3"],
        ollama_api_base="http://localhost:11434",
    )
    assert provider.ollama_api_base == "http://localhost:11434"


def test_register_litellm_provider():
    """Test l'enregistrement du provider LiteLLM."""
    router = Router()
    provider = LiteLLMProvider(models=["openai/gpt-4o"])
    router.register_provider(provider, is_primary=True)

    assert router.list_available_providers() == ["litellm"]
    assert router.primary_provider == "litellm"


@pytest.mark.asyncio
async def test_stream_with_primary_provider():
    """Test le streaming avec provider primaire."""
    router = Router()
    router.register_provider(MockProvider("primary"), is_primary=True)

    chunks = [
        c
        async for c in router.stream(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Mock chunk from primary"


@pytest.mark.asyncio
async def test_stream_fallback_on_primary_failure():
    """Test le fallback streaming si le provider primaire échoue."""
    router = Router()
    router.register_provider(MockProvider("primary", fail_stream=True), is_primary=True)
    router.register_provider(MockProvider("secondary"))

    chunks = [
        c
        async for c in router.stream(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )
    ]

    assert len(chunks) == 1
    assert chunks[0].content == "Mock chunk from secondary"


@pytest.mark.asyncio
async def test_send_all_providers_fail():
    """Test l'erreur quand tous les providers échouent en send."""
    router = Router()
    router.register_provider(MockProvider("p1", fail_send=True), is_primary=True)
    router.register_provider(MockProvider("p2", fail_send=True))

    with pytest.raises(RuntimeError, match="All providers failed"):
        await router.send(
            messages=[{"role": "user", "content": "test"}],
            model="test-model",
        )


@pytest.mark.asyncio
async def test_stream_all_providers_fail():
    """Test l'erreur quand tous les providers échouent en stream."""
    router = Router()
    router.register_provider(MockProvider("p1", fail_stream=True), is_primary=True)
    router.register_provider(MockProvider("p2", fail_stream=True))

    with pytest.raises(RuntimeError, match="All providers failed"):
        _ = [
            c
            async for c in router.stream(
                messages=[{"role": "user", "content": "test"}],
                model="test-model",
            )
        ]


@pytest.mark.asyncio
async def test_health_check_exception_marks_provider_unhealthy():
    """Test qu'une exception de health_check marque le provider en unhealthy."""
    router = Router()
    router.register_provider(MockProvider("boom", fail_health_exception=True), is_primary=True)

    status = await router.health_check_all()
    assert status["boom"] is False
