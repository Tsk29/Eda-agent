from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ExecutionError(BaseModel):
    kind: Literal["timeout", "memory_exceeded", "sql_error", "disallowed", "unknown"]
    message: str


class ExecutionResult(BaseModel):
    ok: bool
    columns: list[str] | None
    rows: list[dict] | None
    error: ExecutionError | None
