"""Streamlit consumption UI for the verified EDA agent.

This is a thin rendering layer over data that has already been computed,
verified, and stored by the rest of the pipeline (profiler -> planner ->
executor -> verifier -> guardrails -> storage). It performs no analysis of
its own: every number shown here was already checked by the verifier before
it reached Postgres.

Run with:
    streamlit run app/main.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# `streamlit run app/main.py` puts this file's own directory on sys.path[0],
# not the repo root, so `app` and `eda_agent` aren't importable as packages
# without this. Must run before the local/project imports below.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from app.logic import compute_rejection_rate, sort_findings
from eda_agent.storage import get_claims, get_engine, get_findings

st.set_page_config(page_title="Verified EDA Agent", layout="wide")


@st.cache_resource
def _get_cached_engine(dsn: str) -> Any:
    """Cache the SQLAlchemy engine across Streamlit reruns."""
    return get_engine(dsn)


def _render_findings_section(engine: Any, run_id: int) -> None:
    st.subheader("Findings")
    findings = get_findings(engine, run_id)
    if not findings:
        st.info("No findings for this run.")
        return

    ordered = sort_findings(findings)
    table_rows = [
        {
            "kind": f["kind"],
            "severity": f["severity"],
            "effect_size": f["effect_size"],
            "p_value_corrected": f["p_value_corrected"],
            "columns involved": ", ".join(f["columns"]) if f["columns"] else "",
        }
        for f in ordered
    ]
    st.dataframe(pd.DataFrame(table_rows), width="stretch", hide_index=True)


def _render_claim(claim: dict[str, Any]) -> None:
    st.markdown(f"**{claim['text']}**")

    if claim["passed"]:
        st.success("Verified: recomputed values match stated values")
    else:
        st.error("Rejected: recomputed values did not match stated values")

    stated = claim["stated_values"]
    recomputed = claim["recomputed_values"]
    keys = sorted(set(stated) | set(recomputed))
    comparison = pd.DataFrame(
        {
            "stated": [stated.get(k) for k in keys],
            "recomputed": [recomputed.get(k) for k in keys],
        },
        index=keys,
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(comparison, width="stretch")
    with col2:
        st.metric("Max relative error", f"{claim['max_relative_error']:.2e}")

    with st.expander("SQL"):
        st.code(claim["sql"], language="sql")

    st.divider()


def _render_claims_section(claims: list[dict[str, Any]]) -> None:
    st.subheader("Claims")
    if not claims:
        st.info("No claims for this run.")
        return

    for claim in claims:
        _render_claim(claim)


def main() -> None:
    st.title("Verified EDA Agent")

    dsn = os.environ.get("POSTGRES_DSN", "postgresql+psycopg://eda:eda@localhost:5432/eda")
    engine = _get_cached_engine(dsn)

    # NOTE: run_id is hardcoded to 1 for now. Storage does not yet expose a
    # list_runs()-style function, so there is nothing to build a dropdown
    # selector against. Replace this with a real selector once that lands.
    run_id = 1
    st.caption(f"Run ID: {run_id} (hardcoded until a run-listing function exists in storage)")

    claims = get_claims(engine, run_id)
    rejection_rate = compute_rejection_rate(claims)
    st.metric(
        "Verification rejection rate",
        f"{rejection_rate:.1%}" if claims else "N/A",
        help="Fraction of claims whose recomputed values failed to match their stated values.",
    )

    _render_findings_section(engine, run_id)
    _render_claims_section(claims)


if __name__ == "__main__":
    main()
