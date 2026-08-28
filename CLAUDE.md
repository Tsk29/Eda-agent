# CLAUDE.md

Project instructions for Claude Code. Read this before writing any code.

---

## What this project is

A verified EDA agent. It profiles a tabular dataset, decides what is worth
investigating, generates SQL to investigate it, **verifies every number it
reports**, applies statistical guardrails, and writes results to Postgres for
consumption in Power BI.

The point of the project is not the agent. The point is the **measurement**:
how often does an LLM state a number that is wrong, and how much does
verification catch? That number is the deliverable.

Runs entirely locally. No cloud deployment.

---

## Non-negotiable design rules

These are the rules that make this project different from the hundreds of
"chat with your CSV" repos. Do not violate them, and push back if I ask you to.

1. **The LLM never sees raw rows.** It sees a compact profile object only.
   All data access is via generated SQL executed by DuckDB.

2. **Every claim carries its evidence.** A claim is `{text, sql, stated_values}`.
   Never emit prose containing a number that is not bound to the SQL that
   produced it.

3. **Verification is mandatory and independent.** Before any claim reaches the
   user, re-execute its SQL and compare recomputed values to stated values.
   Mismatch beyond tolerance → claim is rejected, logged, and never displayed.

4. **Statistics are deterministic and outrank the model.** Multiple-comparison
   correction, leakage detection, sentinel detection, and subgroup-reversal
   checks are plain Python. The LLM cannot override them or opt out.

5. **Generated code is never `exec`'d in-process.** Subprocess only, with
   timeout, memory cap, no network, and an AST-level import allowlist.

6. **Everything is logged, including failures.** Rejected claims are more
   valuable than accepted ones. They are the result.

---

## Repository layout

```
eda_agent/
  profiler/       deterministic dataset profiling (NO LLM)
  planner/        LangGraph graph, LLM prompts, routing logic
  executor/       sandboxed subprocess execution
  verifier/       claim re-execution and comparison
  guardrails/     BH correction, leakage, sentinels, subgroup reversal
  storage/        SQLAlchemy models, Postgres access
  llm/            provider-agnostic client (Ollama / Gemini / Groq)
benchmark/
  datasets/       base public datasets
  planting/       defect injection
  scoring/        recall / false-positive scoring
app/              Streamlit UI
tests/
docker-compose.yml
```

---

## Build order

Do not skip ahead. Each stage must be tested before the next begins.

**Stage 1 — profiler + executor. No LLM at all.**
The profiler takes a table and returns dtypes, row count, per-column null
counts, cardinality, quantiles for numerics, top-k values for categoricals,
candidate primary keys, and detected sentinel values. The executor runs a SQL
string against DuckDB in a subprocess and returns a DataFrame or a structured
error. Both fully unit tested. Nothing else exists yet.

**Stage 2 — benchmark harness. Still no agent.**
Defect injection and scoring built before the thing being scored. Start with
three datasets and three defect types.

**Stage 3 — linear agent loop.**
Profile → single LLM call → SQL → execute → verify → store. No branching yet.
Get the verification loop correct in the simplest possible form.

**Stage 4 — LangGraph routing.**
Replace the linear loop with conditional edges. Only now.

**Stage 5 — guardrails, Postgres schema, Streamlit, Power BI.**

---

## Stack

| Layer | Choice |
|---|---|
| Language | Python 3.11 |
| Analytical engine | DuckDB over Parquet |
| Operational store | Postgres 16 (Docker) |
| ORM | SQLAlchemy 2.x Core |
| Validation | Pydantic v2 |
| LLM serving | Ollama, OpenAI-compatible endpoint |
| Orchestration | LangGraph |
| Statistics | statsmodels, scipy, pandas, numpy |
| UI | Streamlit |
| Testing | pytest |
| Lint/format | ruff |

**Do not add:** Airflow, Kafka, Spark, PyTorch, a vector database, Next.js,
or any auth system. If you think one is needed, say why and wait for me.

---

## LLM client

The provider is a config value, never hardcoded. Ollama exposes an
OpenAI-compatible API, so one interface covers local and hosted models.

```python
class LLMClient(Protocol):
    def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...
```

All LLM output is structured. Parse into a Pydantic model. Never regex over
free text. If parsing fails, retry once with the validation error appended,
then fail loudly.

Default model: `qwen2.5-coder:14b` (fall back to `:7b` if RAM-constrained).

---

## Core schemas

```python
class ColumnProfile(BaseModel):
    name: str
    dtype: str
    null_count: int
    null_fraction: float
    n_unique: int
    quantiles: dict[str, float] | None
    top_values: list[tuple[str, int]] | None
    sentinel_candidates: list[str]

class Claim(BaseModel):
    text: str                    # prose with {placeholders}
    sql: str                     # must reproduce stated_values
    stated_values: dict[str, float]

class VerifiedClaim(BaseModel):
    claim: Claim
    recomputed_values: dict[str, float]
    passed: bool
    max_relative_error: float

class Finding(BaseModel):
    kind: Literal["missingness", "outlier", "leakage", "correlation",
                  "subgroup_reversal", "duplicate_key", "sentinel", "drift"]
    severity: Literal["low", "medium", "high"]
    columns: list[str]
    effect_size: float | None
    p_value: float | None
    p_value_corrected: float | None
    claim: VerifiedClaim
```

---

## Verification rules

- Relative tolerance `1e-6` for floats; exact match for counts.
- A claim with any unverifiable value is rejected wholesale, not partially.
- Rejected claims are written to `claims` with `passed=false` and the delta.
- Never silently repair a claim by substituting the recomputed value. The
  failure is the finding.

---

## Guardrails

Implement as pure functions in `guardrails/`, called after findings are
collected and before storage.

- **Multiple comparisons** — Benjamini–Hochberg across all p-values produced in
  a run. Store both raw and corrected. Report corrected.
- **Leakage** — flag any single feature with near-perfect predictive power on a
  declared target (AUC > 0.95 or correlation > 0.98).
- **Sentinels** — detect `-999`, `9999`, `-1` in non-negative columns, and
  string `"NA"`/`"N/A"`/`""` before any aggregate is computed.
- **Subgroup reversal** — for any reported aggregate relationship, test whether
  its sign flips within major subgroups. Flag as Simpson's paradox risk.

---

## Benchmark

Planted defects with known ground truth. Score recall and false-positive rate.

Defect types to inject: outlier cluster, missingness correlated with target,
Simpson's paradox reversal, leaked column, duplicated IDs, sentinel value.

Scoring: a planted defect counts as detected if a finding of matching `kind`
names the correct column(s). Any finding not corresponding to a planted defect
and not present in the clean baseline counts as a false positive.

Report per model: recall, false-positive rate, mean latency, verification
rejection rate.

---

## Testing

- Profiler and guardrails: deterministic unit tests, full coverage.
- Executor: test timeout, memory cap, and import-allowlist rejection.
- Verifier: fixture claims with known-good and known-bad values.
- LLM calls: mocked in tests. No test hits a live model.
- `pytest` must pass before any commit.

---

## Conventions

- Type hints everywhere. `ruff` clean.
- No bare `except`. Catch specific exceptions.
- No `print`. Use the configured logger.
- SQL in `.sql` files or clearly-named constants, not inline f-strings scattered
  through logic.
- Secrets in `.env`, never committed. `.env.example` stays current.

---

## Commands

```bash
docker compose up -d          # Postgres + Ollama
uv sync                       # or pip install -r requirements.txt
pytest                        # tests
ruff check . && ruff format .  # lint
streamlit run app/main.py     # UI
python -m benchmark.run --model qwen2.5-coder:14b   # benchmark
```

---

## How I want you to work with me

- Plan before coding. Show me the approach for anything non-trivial and wait.
- Write the failing test first, then the implementation.
- One component at a time. Do not scaffold the whole repo in one pass.
- After writing a module, explain the design decisions back to me. I have to be
  able to defend every line of this in an interview — if I cannot explain it,
  it does not belong in the repo.
- If I ask for something that violates the non-negotiable rules above, tell me.
