"""Prompt construction from a `TableProfile`.

Per CLAUDE.md rule 1 ("The LLM never sees raw rows. It sees a compact
profile object only."), everything built here comes from the profiler's
aggregate output -- column names, dtypes, null fractions, quantiles/top
values, sentinel candidates, candidate primary keys, row count -- and
nothing else. No row-level data ever passes through this module.
"""

from __future__ import annotations

from eda_agent.profiler.schemas import ColumnProfile, TableProfile

_SYSTEM_PROMPT = (
    "You are a careful data analyst investigating a tabular dataset. "
    "You are given only a statistical profile of the table -- never raw "
    "rows. For every claim you make, you must provide the exact SQL query "
    "that computes the value(s) you are citing, and the numeric value(s) "
    "your SQL is expected to produce. Every number in your prose text must "
    "appear as a value in `stated_values` and be reproducible by `sql`. "
    "Do not state a number that your SQL does not compute. Every SQL query "
    "must query the exact table name given to you below -- never invent a "
    "placeholder table name. Your SQL's SELECT list must alias every output "
    "column (e.g. `SELECT COUNT(*) AS row_count`), and each key in "
    "`stated_values` must exactly match one of those column aliases -- "
    "never use the numeric value itself, or a description of the value, as "
    "a `stated_values` key. Respond with a batch of claims matching the "
    "requested schema."
)


def _format_column(column: ColumnProfile) -> str:
    lines = [
        f"- {column.name} (dtype={column.dtype})",
        f"    null_count={column.null_count} null_fraction={column.null_fraction:.4f}",
        f"    n_unique={column.n_unique}",
    ]
    if column.quantiles is not None:
        quantiles_str = ", ".join(f"{k}={v}" for k, v in column.quantiles.items())
        lines.append(f"    quantiles: {quantiles_str}")
    if column.top_values is not None:
        top_values_str = ", ".join(f"{v!r}={c}" for v, c in column.top_values)
        lines.append(f"    top_values: {top_values_str}")
    if column.sentinel_candidates:
        lines.append(f"    sentinel_candidates: {column.sentinel_candidates}")
    return "\n".join(lines)


def build_prompt(profile: TableProfile) -> tuple[str, str]:
    """Build (system, user) prompt strings from a compact `TableProfile`.

    Contains only profile fields: row count, candidate primary keys, and
    per-column dtype/null/cardinality/quantile/top-value/sentinel summaries.
    Never includes raw row data.
    """
    lines = [
        f"table_name: {profile.table_name}",
        f"row_count: {profile.row_count}",
        f"candidate_primary_keys: {profile.candidate_primary_keys}",
        "columns:",
    ]
    for column in profile.columns:
        lines.append(_format_column(column))

    user_prompt = "\n".join(lines)
    return _SYSTEM_PROMPT, user_prompt


def append_retry_context(user_prompt: str, failures: list[str]) -> str:
    """Append verification-failure feedback to a user prompt for a retry.

    `failures` is a list of short human-readable descriptions of claims
    that failed verification on the previous attempt (e.g. claim text plus
    why it failed), so the model can try to correct itself instead of
    repeating the same mistake.
    """
    if not failures:
        return user_prompt

    feedback_lines = [
        "The following claims from your previous attempt FAILED independent "
        "verification (the SQL did not reproduce the stated values, or the "
        "SQL itself errored). Do not repeat them. Produce a corrected batch "
        "of claims:",
    ]
    for failure in failures:
        feedback_lines.append(f"- {failure}")

    return user_prompt + "\n\n" + "\n".join(feedback_lines)
