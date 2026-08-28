from __future__ import annotations

import json

from openai import OpenAI
from pydantic import BaseModel, ValidationError

from eda_agent.llm.config import LLMConfig
from eda_agent.log import get_logger

logger = get_logger(__name__)


class LLMOutputError(Exception):
    """Raised when an LLM response cannot be parsed into the requested schema.

    Raised only after a retry has already been attempted. The project's
    verification rules require failing loudly rather than silently returning
    a partial or default object, so callers must handle this explicitly.
    """


def _schema_instructions(schema: type[BaseModel]) -> str:
    """Describe the target JSON shape in plain text.

    `response_format={"type": "json_object"}` is requested on every call,
    but not every OpenAI-compatible backend (in particular, some Ollama
    builds) honors it reliably. Spelling out the schema in the system
    prompt is the fallback that makes JSON output likely even when the
    backend ignores `response_format` entirely.
    """
    json_schema = json.dumps(schema.model_json_schema(), indent=2)
    return (
        "You must respond with a single JSON object and nothing else -- "
        "no prose, no markdown code fences, no explanation.\n"
        f"The JSON object must validate against this JSON Schema:\n{json_schema}"
    )


class OpenAICompatibleClient:
    """LLMClient implementation backed by any OpenAI-compatible chat API.

    Works unmodified against Ollama's OpenAI-compatible endpoint, Groq, or
    real OpenAI -- the provider is selected entirely via `LLMConfig`
    (base_url / api_key / model), never hardcoded here.
    """

    def __init__(self, config: LLMConfig | None = None, client: OpenAI | None = None) -> None:
        self._config = config or LLMConfig()
        self._client = client or OpenAI(
            base_url=self._config.base_url,
            api_key=self._config.api_key,
        )

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        full_system = f"{system}\n\n{_schema_instructions(schema)}"
        messages: list[dict[str, str]] = [
            {"role": "system", "content": full_system},
            {"role": "user", "content": user},
        ]

        content = self._request(messages)
        try:
            return schema.model_validate_json(content)
        except ValidationError as first_error:
            logger.warning(
                "LLM response failed schema validation for %s, retrying once: %s",
                schema.__name__,
                first_error,
            )
            retry_user = (
                f"{user}\n\n"
                "Your previous response was invalid. It failed with this error:\n"
                f"{first_error}\n\n"
                "Respond again with ONLY a corrected JSON object matching the schema."
            )
            messages = [
                {"role": "system", "content": full_system},
                {"role": "user", "content": retry_user},
            ]
            content = self._request(messages)
            try:
                return schema.model_validate_json(content)
            except ValidationError as second_error:
                raise LLMOutputError(
                    f"LLM failed to produce valid {schema.__name__} JSON after one retry. "
                    f"First error: {first_error}. Second error: {second_error}."
                ) from second_error

    def _request(self, messages: list[dict[str, str]]) -> str:
        response = self._client.chat.completions.create(
            model=self._config.model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if content is None:
            raise LLMOutputError("LLM response contained no content.")
        return content
