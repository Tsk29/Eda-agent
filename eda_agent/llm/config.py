from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """LLM provider configuration, sourced from environment variables.

    See .env.example at the repo root for the expected shape. The provider
    is always a config value -- never hardcode a provider or model in code.
    Since Ollama exposes an OpenAI-compatible API, the same `base_url` /
    `api_key` pair works transparently against Ollama, Groq, or real OpenAI.
    """

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        extra="ignore",
    )

    provider: str = "ollama"
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5-coder:14b"
