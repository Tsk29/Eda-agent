"""Stage 3: the linear agent loop.

profile -> single LLM call -> verify every claim -> return ALL results.

No branching, no retries. This is the simplest form of the verification
loop, per CLAUDE.md's build order: get this correct before adding LangGraph
routing on top of it in `eda_agent.planner.graph`.
"""

from __future__ import annotations

from pydantic import BaseModel

from eda_agent.log import get_logger
from eda_agent.planner.prompts import build_prompt
from eda_agent.planner.protocols import LLMClient, VerifyFn
from eda_agent.profiler.schemas import TableProfile
from eda_agent.schemas import Claim, VerifiedClaim

logger = get_logger(__name__)


class ClaimBatch(BaseModel):
    claims: list[Claim]


def run_linear(
    profile: TableProfile,
    llm: LLMClient,
    verify: VerifyFn,
) -> list[VerifiedClaim]:
    """Profile -> one LLM call -> verify every claim -> return all results.

    Every claim the model produced is verified and returned, including
    rejected ones. Per CLAUDE.md ("Rejected claims are more valuable than
    accepted ones. They are the result"), this function never drops or
    filters a failed `VerifiedClaim` from its output.
    """
    system_prompt, user_prompt = build_prompt(profile)

    batch = llm.complete(system_prompt, user_prompt, ClaimBatch)
    if not isinstance(batch, ClaimBatch):
        # Defensive: a well-behaved LLMClient always returns a validated
        # instance of the requested schema, but Protocol structural typing
        # cannot enforce that at runtime.
        batch = ClaimBatch.model_validate(batch)

    logger.info("run_linear: received %d claim(s) from LLM", len(batch.claims))

    verified_claims: list[VerifiedClaim] = []
    for claim in batch.claims:
        verified_claim = verify(claim)
        if not verified_claim.passed:
            logger.warning("run_linear: claim rejected: %r", claim.text)
        verified_claims.append(verified_claim)

    return verified_claims
