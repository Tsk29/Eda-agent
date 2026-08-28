from __future__ import annotations

from pydantic import BaseModel

from eda_agent.planner.graph import run_graph
from eda_agent.planner.linear import ClaimBatch, run_linear
from eda_agent.planner.prompts import append_retry_context, build_prompt
from eda_agent.profiler.schemas import ColumnProfile, TableProfile
from eda_agent.schemas import Claim, VerifiedClaim


def make_profile() -> TableProfile:
    return TableProfile(
        table_name="events",
        row_count=100,
        candidate_primary_keys=["id"],
        columns=[
            ColumnProfile(
                name="id",
                dtype="BIGINT",
                null_count=0,
                null_fraction=0.0,
                n_unique=100,
                quantiles={"min": 1.0, "25%": 25.0, "50%": 50.0, "75%": 75.0, "max": 100.0},
                top_values=None,
                sentinel_candidates=[],
            ),
            ColumnProfile(
                name="amount",
                dtype="DOUBLE",
                null_count=5,
                null_fraction=0.05,
                n_unique=80,
                quantiles={"min": 0.0, "25%": 10.0, "50%": 20.0, "75%": 30.0, "max": 999.0},
                top_values=None,
                sentinel_candidates=["-999"],
            ),
            ColumnProfile(
                name="category",
                dtype="VARCHAR",
                null_count=0,
                null_fraction=0.0,
                n_unique=3,
                quantiles=None,
                top_values=[("A", 50), ("B", 30), ("C", 20)],
                sentinel_candidates=[],
            ),
        ],
    )


def make_claim(text: str) -> Claim:
    return Claim(text=text, sql=f"SELECT 1 AS v -- {text}", stated_values={"v": 1.0})


class FakeLLM:
    """Returns canned ClaimBatches from a queue, one per `.complete()` call."""

    def __init__(self, batches: list[ClaimBatch]) -> None:
        self._batches = list(batches)
        self.calls: list[tuple[str, str, type[BaseModel]]] = []

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls.append((system, user, schema))
        if not self._batches:
            raise AssertionError("FakeLLM.complete called more times than batches provided")
        return self._batches.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)


def fake_verify(claim: Claim) -> VerifiedClaim:
    """Deterministic pass/fail: any claim whose text contains 'BAD' fails."""
    passed = "BAD" not in claim.text
    return VerifiedClaim(
        claim=claim,
        recomputed_values={} if not passed else dict(claim.stated_values),
        passed=passed,
        max_relative_error=0.0 if passed else 1.0,
    )


# --- build_prompt --------------------------------------------------------


def test_build_prompt_contains_only_profile_fields_no_raw_rows() -> None:
    profile = make_profile()
    system, user = build_prompt(profile)

    assert isinstance(system, str) and system
    assert "100" in user  # row_count
    assert "id" in user and "amount" in user and "category" in user
    assert "null_fraction=0.0500" in user
    assert "-999" in user  # sentinel candidate
    assert "A" in user and "50" in user  # top_values


def test_append_retry_context_includes_failures() -> None:
    _, user = build_prompt(make_profile())
    updated = append_retry_context(user, ["claim X failed: sql errored"])
    assert "claim X failed" in updated
    assert user in updated  # original content preserved


def test_append_retry_context_noop_when_no_failures() -> None:
    _, user = build_prompt(make_profile())
    assert append_retry_context(user, []) == user


# --- run_linear ------------------------------------------------------------


def test_run_linear_returns_all_claims_including_failed() -> None:
    profile = make_profile()
    batch = ClaimBatch(
        claims=[
            make_claim("mean of amount is 20"),
            make_claim("BAD claim with wrong number"),
        ]
    )
    llm = FakeLLM([batch])

    result = run_linear(profile, llm, fake_verify)

    assert len(result) == 2
    assert llm.call_count == 1
    passed_texts = {vc.claim.text for vc in result if vc.passed}
    failed_texts = {vc.claim.text for vc in result if not vc.passed}
    assert "mean of amount is 20" in passed_texts
    assert "BAD claim with wrong number" in failed_texts
    # the failed claim must be present in the output, not hidden
    assert any(not vc.passed for vc in result)


def test_run_linear_empty_batch_returns_empty_list() -> None:
    profile = make_profile()
    llm = FakeLLM([ClaimBatch(claims=[])])

    result = run_linear(profile, llm, fake_verify)

    assert result == []
    assert llm.call_count == 1


# --- run_graph ---------------------------------------------------------


def test_run_graph_retries_once_then_terminates() -> None:
    profile = make_profile()
    first_batch = ClaimBatch(claims=[make_claim("BAD first attempt")])
    second_batch = ClaimBatch(claims=[make_claim("good second attempt")])
    llm = FakeLLM([first_batch, second_batch])

    result = run_graph(profile, llm, fake_verify, max_retries=1)

    assert llm.call_count == 2
    assert len(result) == 1
    assert result[0].passed is True
    assert result[0].claim.text == "good second attempt"


def test_run_graph_terminates_immediately_when_all_pass() -> None:
    profile = make_profile()
    batch = ClaimBatch(claims=[make_claim("all good"), make_claim("also good")])
    llm = FakeLLM([batch])

    result = run_graph(profile, llm, fake_verify, max_retries=1)

    assert llm.call_count == 1
    assert len(result) == 2
    assert all(vc.passed for vc in result)


def test_run_graph_still_returns_failed_claims_after_retry_budget_exhausted() -> None:
    profile = make_profile()
    first_batch = ClaimBatch(claims=[make_claim("BAD attempt one")])
    second_batch = ClaimBatch(claims=[make_claim("BAD attempt two")])
    llm = FakeLLM([first_batch, second_batch])

    result = run_graph(profile, llm, fake_verify, max_retries=1)

    assert llm.call_count == 2
    assert len(result) == 1
    assert result[0].passed is False
    assert result[0].claim.text == "BAD attempt two"


def test_run_graph_zero_retries_calls_llm_once_even_on_failure() -> None:
    profile = make_profile()
    batch = ClaimBatch(claims=[make_claim("BAD no retries allowed")])
    llm = FakeLLM([batch])

    result = run_graph(profile, llm, fake_verify, max_retries=0)

    assert llm.call_count == 1
    assert len(result) == 1
    assert result[0].passed is False
