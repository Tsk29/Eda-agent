"""Stage 4: LangGraph routing over the same steps as the linear loop.

Replaces the linear sequence in `eda_agent.planner.linear` with a small
`StateGraph` carrying exactly one conditional edge: after verification, if
any claim in the current batch failed and a retry budget remains, loop back
to claim generation (with verification-failure context appended to the
prompt so the model can try to correct itself); otherwise end.

Nodes:
    generate  -- build/extend the prompt from profile + prior failures,
                 call the LLM once for a `ClaimBatch`, store it on state.
    verify_node -- verify every claim in the current batch, store the
                   `VerifiedClaim`s on state, decrement the retry budget.

Edges:
    START -> generate -> verify_node -> [conditional] -> generate | END

The conditional edge is the entirety of Stage 4's "replace the linear loop
with conditional edges" requirement -- deliberately not elaborate.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from eda_agent.log import get_logger
from eda_agent.planner.linear import ClaimBatch
from eda_agent.planner.prompts import append_retry_context, build_prompt
from eda_agent.planner.protocols import LLMClient, VerifyFn
from eda_agent.profiler.schemas import TableProfile
from eda_agent.schemas import Claim, VerifiedClaim

logger = get_logger(__name__)


class PlannerState(TypedDict):
    profile: TableProfile
    system_prompt: str
    user_prompt: str
    claims: list[Claim]
    verified: list[VerifiedClaim]
    retries_left: int


def _failure_descriptions(verified: list[VerifiedClaim]) -> list[str]:
    return [
        f"{v.claim.text!r} (sql={v.claim.sql!r}, "
        f"stated={v.claim.stated_values!r}, "
        f"recomputed={v.recomputed_values!r})"
        for v in verified
        if not v.passed
    ]


def _make_generate_node(llm: LLMClient):
    def generate(state: PlannerState) -> PlannerState:
        previous_verified = state.get("verified", [])
        is_retry = bool(previous_verified)

        failures = _failure_descriptions(previous_verified)
        user_prompt = append_retry_context(state["user_prompt"], failures)

        batch = llm.complete(state["system_prompt"], user_prompt, ClaimBatch)
        if not isinstance(batch, ClaimBatch):
            batch = ClaimBatch.model_validate(batch)

        logger.info("graph.generate: received %d claim(s) from LLM", len(batch.claims))

        update: PlannerState = {"claims": batch.claims}
        if is_retry:
            # The retry budget is spent exactly when we come back around to
            # generate a corrected batch, not when verification merely
            # observes a failure -- otherwise the budget would be consumed
            # before the retry it is meant to authorize ever happens.
            update["retries_left"] = state["retries_left"] - 1
        return update

    return generate


def _make_verify_node(verify: VerifyFn):
    def verify_node(state: PlannerState) -> PlannerState:
        verified_claims = [verify(claim) for claim in state["claims"]]
        for verified_claim in verified_claims:
            if not verified_claim.passed:
                logger.warning("graph.verify_node: claim rejected: %r", verified_claim.claim.text)
        return {"verified": verified_claims}

    return verify_node


def _should_retry(state: PlannerState) -> str:
    any_failed = any(not v.passed for v in state["verified"])
    if any_failed and state["retries_left"] > 0:
        return "generate"
    return END


def build_graph(llm: LLMClient, verify: VerifyFn):
    """Assemble the Stage 4 `StateGraph`. Exposed for tests/inspection."""
    graph = StateGraph(PlannerState)
    graph.add_node("generate", _make_generate_node(llm))
    graph.add_node("verify_node", _make_verify_node(verify))

    graph.set_entry_point("generate")
    graph.add_edge("generate", "verify_node")
    graph.add_conditional_edges(
        "verify_node",
        _should_retry,
        {"generate": "generate", END: END},
    )

    return graph.compile()


def run_graph(
    profile: TableProfile,
    llm: LLMClient,
    verify: VerifyFn,
    max_retries: int = 1,
) -> list[VerifiedClaim]:
    """Run the Stage 4 graph: generate -> verify -> (retry once | end).

    Returns every `VerifiedClaim` from the final attempt, including any
    that still failed after retries are exhausted -- rejected claims are
    never dropped.
    """
    system_prompt, user_prompt = build_prompt(profile)
    compiled = build_graph(llm, verify)

    initial_state: PlannerState = {
        "profile": profile,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "claims": [],
        "verified": [],
        "retries_left": max_retries,
    }

    final_state = compiled.invoke(initial_state)
    return final_state["verified"]
