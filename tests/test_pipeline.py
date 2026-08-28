from __future__ import annotations

import duckdb
import pytest
from pydantic import BaseModel

from eda_agent.pipeline import make_run_sql, run_pipeline
from eda_agent.planner.linear import ClaimBatch
from eda_agent.schemas import Claim


class FakeLLM:
    def __init__(self, batch: ClaimBatch) -> None:
        self._batch = batch
        self.calls = 0

    def complete(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls += 1
        assert schema is ClaimBatch
        return self._batch


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE orders AS SELECT * FROM "
        "(VALUES (1, 100.0), (2, 200.0), (3, 300.0)) AS t(id, amount)"
    )
    con.close()
    return path


def test_make_run_sql_returns_rows_on_success(db_path):
    run_sql = make_run_sql(db_path)
    rows = run_sql("SELECT COUNT(*) AS n FROM orders")
    assert rows == [{"n": 3}]


def test_make_run_sql_raises_on_sql_error(db_path):
    run_sql = make_run_sql(db_path)
    with pytest.raises(RuntimeError):
        run_sql("SELECT * FROM does_not_exist")


def test_run_pipeline_end_to_end_correct_claim_passes(db_path):
    claim = Claim(
        text="There are 3 orders totaling 600.0.",
        sql="SELECT COUNT(*) AS n, SUM(amount) AS total FROM orders",
        stated_values={"n": 3, "total": 600.0},
    )
    llm = FakeLLM(ClaimBatch(claims=[claim]))

    verified = run_pipeline(db_path, "orders", llm=llm, use_graph=False)

    assert len(verified) == 1
    assert verified[0].passed is True
    assert verified[0].recomputed_values == {"n": 3, "total": 600.0}


def test_run_pipeline_end_to_end_wrong_claim_rejected(db_path):
    claim = Claim(
        text="There are 3 orders totaling 999.0.",
        sql="SELECT COUNT(*) AS n, SUM(amount) AS total FROM orders",
        stated_values={"n": 3, "total": 999.0},
    )
    llm = FakeLLM(ClaimBatch(claims=[claim]))

    verified = run_pipeline(db_path, "orders", llm=llm, use_graph=False)

    assert len(verified) == 1
    assert verified[0].passed is False
    assert verified[0].recomputed_values == {"n": 3, "total": 600.0}


def test_run_pipeline_end_to_end_via_graph(db_path):
    claim = Claim(
        text="There are 3 orders totaling 600.0.",
        sql="SELECT COUNT(*) AS n, SUM(amount) AS total FROM orders",
        stated_values={"n": 3, "total": 600.0},
    )
    llm = FakeLLM(ClaimBatch(claims=[claim]))

    verified = run_pipeline(db_path, "orders", llm=llm, use_graph=True, max_retries=1)

    assert len(verified) == 1
    assert verified[0].passed is True
    assert llm.calls == 1
