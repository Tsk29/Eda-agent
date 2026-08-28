"""Structural interfaces the planner depends on.

The planner is the orchestration core: profile -> LLM -> SQL -> verify.
`eda_agent/llm` and `eda_agent/verifier` are built concurrently by other
agents, so this module never imports their concrete classes. Instead it
declares the shapes it needs as a `Protocol` (for the LLM client) and a
plain `Callable` type alias (for verification). Anything matching these
shapes -- the real implementation or a test fake -- works here without
modification.

`LLMClient.complete` must structurally match `eda_agent.llm.protocol.LLMClient`:
same method name, same signature, same return contract (a validated
instance of the requested Pydantic schema, never free text, never a
partial/default object on parse failure).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel

from eda_agent.schemas import Claim, VerifiedClaim


class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


# Matches the eventual real verifier's `verify_claim` signature, partially
# applied over its `run_sql` dependency so the planner only needs to know
# "give me a claim, get back a verified claim."
VerifyFn = Callable[[Claim], VerifiedClaim]
