"""Tests for models catalog API."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from life_core.models_api import models_router, MODEL_CATALOG, DOMAIN_LABELS


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(models_router)
    return TestClient(app)


def test_catalog_returns_200(client):
    response = client.get("/models/catalog")
    assert response.status_code == 200


def test_catalog_contains_models_key(client):
    data = client.get("/models/catalog").json()
    assert "models" in data
    assert isinstance(data["models"], list)


def test_catalog_contains_domains_key(client):
    data = client.get("/models/catalog").json()
    assert "domains" in data
    assert isinstance(data["domains"], dict)


def test_catalog_models_are_not_empty(client):
    data = client.get("/models/catalog").json()
    assert len(data["models"]) > 0


def test_catalog_each_model_has_id(client):
    data = client.get("/models/catalog").json()
    for model in data["models"]:
        assert "id" in model, f"Model missing 'id': {model}"


def test_catalog_each_model_has_provider(client):
    data = client.get("/models/catalog").json()
    for model in data["models"]:
        assert "provider" in model, f"Model missing 'provider': {model}"


def test_catalog_each_model_has_domain(client):
    data = client.get("/models/catalog").json()
    for model in data["models"]:
        assert "domain" in model, f"Model missing 'domain': {model}"


def test_catalog_domains_cover_all_model_domains(client):
    data = client.get("/models/catalog").json()
    model_domains = {m["domain"] for m in data["models"]}
    catalog_domains = set(data["domains"].keys())
    assert model_domains.issubset(catalog_domains)


def test_catalog_contains_vllm_model(client):
    data = client.get("/models/catalog").json()
    vllm_models = [m for m in data["models"] if m["provider"] == "vllm"]
    assert len(vllm_models) > 0


def test_catalog_contains_local_llm_model(client):
    data = client.get("/models/catalog").json()
    local_models = [m for m in data["models"] if m["provider"] in ("llama.cpp", "tei")]
    assert len(local_models) > 0


def test_domain_labels_are_strings(client):
    data = client.get("/models/catalog").json()
    for key, label in data["domains"].items():
        assert isinstance(label, str), f"Domain label not string: {key}={label}"


def test_catalog_model_catalog_constant_matches_response(client):
    data = client.get("/models/catalog").json()
    assert len(data["models"]) == len(MODEL_CATALOG)
