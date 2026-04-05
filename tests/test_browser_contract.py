"""Contract guards for the shared browser scrape schema."""

from __future__ import annotations

import json
from pathlib import Path

import life_core.api as api
import life_core.browser_runner_api as runner_api


def _load_browser_scrape_contract() -> dict:
    candidates = [
        (
            Path(__file__).resolve().parents[2]
            / "finefab-shared"
            / "schemas"
            / "browser_scrape.schema.json"
        ),
        Path(__file__).resolve().parent / "fixtures" / "browser_scrape.schema.json",
    ]
    for contract_path in candidates:
        if contract_path.exists():
            return json.loads(contract_path.read_text())
    raise FileNotFoundError("Unable to locate browser scrape contract snapshot")


def _required_fields(model: type[api.BaseModel]) -> set[str]:
    return {
        field_name
        for field_name, field in model.model_fields.items()
        if field.is_required()
    }


def _metadata_value(field: object, attr: str) -> int | None:
    for metadata in getattr(field, "metadata", []):
        value = getattr(metadata, attr, None)
        if value is not None:
            return value
    return None


def test_scrape_request_matches_shared_contract() -> None:
    contract = _load_browser_scrape_contract()["properties"]["request"]
    for model in (api.ScrapeRequest, runner_api.ScrapeRequest):
        fields = model.model_fields
        assert set(fields) == set(contract["properties"])
        assert _required_fields(model) == set(contract["required"])
        assert _metadata_value(fields["url"], "min_length") == contract["properties"]["url"]["minLength"]
        assert _metadata_value(fields["timeout_ms"], "ge") == contract["properties"]["timeout_ms"]["minimum"]
        assert _metadata_value(fields["timeout_ms"], "le") == contract["properties"]["timeout_ms"]["maximum"]
        assert fields["selector"].default is None
        assert fields["timeout_ms"].default == 15000


def test_scrape_response_matches_shared_contract() -> None:
    contract = _load_browser_scrape_contract()["properties"]["response"]
    for model in (api.ScrapeResponse, runner_api.ScrapeResponse):
        fields = model.model_fields
        assert set(fields) == set(contract["properties"])
        assert _required_fields(model) == set(contract["required"])
