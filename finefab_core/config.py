"""Centralized runtime settings for life-core."""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


def secret_value(value: str | SecretStr) -> str:
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    return value


def is_secret_configured(value: str | SecretStr) -> bool:
    normalized = secret_value(value).strip()
    return bool(normalized and not normalized.endswith("..."))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "life-core"
    environment: str = "development"
    router_default_strategy: str = "best"
    request_timeout_seconds: float = 30.0
    redis_url: str = "redis://localhost:6379/0"
    anthropic_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    openai_api_key: SecretStr = Field(default=SecretStr(""), repr=False)
    mistral_api_key: SecretStr = Field(default=SecretStr(""), repr=False)


settings = Settings()