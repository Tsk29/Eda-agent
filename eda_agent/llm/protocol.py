from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class LLMClient(Protocol):
    """Provider-agnostic interface for structured LLM completions.

    Every implementation must return a validated instance of the requested
    Pydantic schema. Implementations never return free-form text and never
    return a partial or default object when parsing fails -- they raise.
    """

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...
