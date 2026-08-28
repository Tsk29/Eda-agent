import duckdb
import pytest

from eda_agent.profiler import profile_table


@pytest.fixture
def con():
    connection = duckdb.connect(":memory:")
    yield connection
    connection.close()


def test_numeric_column_gets_quantiles_not_top_values(con):
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (1), (2), (3), (4), (5)) AS t(score)")

    profile = profile_table(con, "t")

    col = profile.columns[0]
    assert col.name == "score"
    assert col.quantiles is not None
    assert col.quantiles["min"] == 1.0
    assert col.quantiles["50%"] == 3.0
    assert col.quantiles["max"] == 5.0
    assert col.top_values is None


def test_categorical_column_gets_top_values_not_quantiles(con):
    con.execute(
        "CREATE TABLE t AS SELECT * FROM "
        "(VALUES ('a'), ('a'), ('a'), ('b'), ('c')) AS t(label)"
    )

    profile = profile_table(con, "t")

    col = profile.columns[0]
    assert col.quantiles is None
    assert col.top_values is not None
    assert col.top_values[0] == ("a", 3)


def test_boolean_column_treated_as_categorical(con):
    con.execute("CREATE TABLE t AS SELECT * FROM (VALUES (true), (true), (false)) AS t(flag)")

    profile = profile_table(con, "t")

    col = profile.columns[0]
    assert col.quantiles is None
    assert col.top_values is not None


def test_string_sentinel_tokens_detected(con):
    con.execute(
        "CREATE TABLE t AS SELECT * FROM "
        "(VALUES ('x'), ('y'), ('N/A'), ('  '), ('unknown')) AS t(val)"
    )

    profile = profile_table(con, "t")

    col = profile.columns[0]
    # 'N/A' and whitespace-only both normalize to known null-like tokens.
    # 'unknown' is not in our fixed token list for stage 1.
    assert "N/A" in col.sentinel_candidates
    assert "  " in col.sentinel_candidates
    assert "unknown" not in col.sentinel_candidates


def test_numeric_sentinel_flagged_when_isolated_outlier(con):
    values = [10, 12, 11, 13, 9, 10, 11, -999]
    rows = ", ".join(f"({v})" for v in values)
    con.execute(f"CREATE TABLE t AS SELECT * FROM (VALUES {rows}) AS t(age)")

    profile = profile_table(con, "t")

    col = profile.columns[0]
    assert "-999" in col.sentinel_candidates


def test_numeric_magic_number_not_flagged_when_in_range(con):
    # -1 is a plausible in-range value here (a signed delta column), so it
    # must NOT be flagged even though -1 is in the magic-number list.
    values = [-5, -3, -1, 0, 2, 4, -1, -1]
    rows = ", ".join(f"({v})" for v in values)
    con.execute(f"CREATE TABLE t AS SELECT * FROM (VALUES {rows}) AS t(delta)")

    profile = profile_table(con, "t")

    col = profile.columns[0]
    assert "-1" not in col.sentinel_candidates


def test_candidate_primary_key_detected(con):
    con.execute(
        "CREATE TABLE t AS SELECT * FROM "
        "(VALUES (1, 'a'), (2, 'a'), (3, 'b')) AS t(id, name)"
    )

    profile = profile_table(con, "t")

    assert profile.candidate_primary_keys == ["id"]


def test_column_with_nulls_excluded_from_primary_keys(con):
    con.execute(
        "CREATE TABLE t AS SELECT * FROM "
        "(VALUES (1, 'a'), (2, NULL), (3, 'c')) AS t(id, name)"
    )

    profile = profile_table(con, "t")

    assert profile.candidate_primary_keys == ["id"]


def test_empty_table(con):
    con.execute("CREATE TABLE t (id INTEGER, name VARCHAR)")

    profile = profile_table(con, "t")

    assert profile.row_count == 0
    assert profile.candidate_primary_keys == []
    for col in profile.columns:
        assert col.null_count == 0
        assert col.null_fraction == 0.0
        assert col.n_unique == 0
        assert col.quantiles is None
        assert col.top_values is None
        assert col.sentinel_candidates == []


def test_all_null_column(con):
    con.execute(
        "CREATE TABLE t AS SELECT * FROM "
        "(VALUES (1, NULL), (2, NULL), (3, NULL)) AS t(id, name)"
    )

    profile = profile_table(con, "t")

    name_col = next(c for c in profile.columns if c.name == "name")
    assert name_col.null_count == 3
    assert name_col.null_fraction == 1.0
    assert name_col.n_unique == 0
    assert name_col.quantiles is None
    assert name_col.top_values is None
    assert name_col.sentinel_candidates == []
    assert "name" not in profile.candidate_primary_keys
