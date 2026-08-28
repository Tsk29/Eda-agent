from __future__ import annotations

import duckdb

from eda_agent.profiler.schemas import ColumnProfile, TableProfile

_NUMERIC_TYPE_PREFIXES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "UHUGEINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "REAL",
)

_STRING_TYPE_PREFIXES = ("VARCHAR", "CHAR", "BPCHAR", "TEXT", "STRING")

_STRING_SENTINEL_TOKENS = {"na", "n/a", "null", "none", ""}

_NUMERIC_MAGIC_NUMBERS = (-999, -9999, 9999, -1, 999, -99999, 99999)

_TOP_K = 10
_OUTLIER_IQR_MULTIPLIER = 3.0


def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _is_numeric_type(duckdb_type: str) -> bool:
    base = duckdb_type.upper()
    return any(base.startswith(prefix) for prefix in _NUMERIC_TYPE_PREFIXES)


def _is_string_type(duckdb_type: str) -> bool:
    base = duckdb_type.upper()
    return any(base.startswith(prefix) for prefix in _STRING_TYPE_PREFIXES)


def _is_isolated_outlier(candidate: float, rest: list[float]) -> bool:
    if not rest:
        return False
    sorted_rest = sorted(rest)
    n = len(sorted_rest)
    q1 = sorted_rest[int(0.25 * (n - 1))]
    q3 = sorted_rest[int(0.75 * (n - 1))]
    iqr = q3 - q1
    if iqr > 0:
        lower = q1 - _OUTLIER_IQR_MULTIPLIER * iqr
        upper = q3 + _OUTLIER_IQR_MULTIPLIER * iqr
        return candidate < lower or candidate > upper
    max_abs_rest = max(abs(v) for v in sorted_rest)
    if max_abs_rest == 0:
        return candidate != 0
    return abs(candidate) > 10 * max_abs_rest


def _numeric_sentinel_candidates(
    con: duckdb.DuckDBPyConnection, table: str, column: str
) -> list[str]:
    qcol = _quote_ident(column)
    qtable = _quote_ident(table)
    magic_list = ",".join(str(v) for v in _NUMERIC_MAGIC_NUMBERS)
    present = con.execute(
        f"SELECT DISTINCT {qcol} FROM {qtable} WHERE {qcol} IN ({magic_list})"
    ).fetchall()

    candidates: list[str] = []
    for (value,) in present:
        rest_rows = con.execute(
            f"SELECT {qcol} FROM {qtable} WHERE {qcol} IS NOT NULL AND {qcol} != ?",
            [value],
        ).fetchall()
        rest_values = [float(r[0]) for r in rest_rows]
        if _is_isolated_outlier(float(value), rest_values):
            candidates.append(str(value))
    return candidates


def _string_sentinel_candidates(
    con: duckdb.DuckDBPyConnection, table: str, column: str
) -> list[str]:
    qcol = _quote_ident(column)
    qtable = _quote_ident(table)
    placeholders = ",".join("?" for _ in _STRING_SENTINEL_TOKENS)
    rows = con.execute(
        f"SELECT DISTINCT {qcol} FROM {qtable} "
        f"WHERE {qcol} IS NOT NULL AND lower(trim({qcol})) IN ({placeholders})",
        list(_STRING_SENTINEL_TOKENS),
    ).fetchall()
    return [str(r[0]) for r in rows]


def _quantiles(con: duckdb.DuckDBPyConnection, table: str, column: str) -> dict[str, float]:
    qcol = _quote_ident(column)
    qtable = _quote_ident(table)
    row = con.execute(
        f"SELECT min({qcol}), quantile_cont({qcol}, 0.25), "
        f"quantile_cont({qcol}, 0.5), quantile_cont({qcol}, 0.75), max({qcol}) "
        f"FROM {qtable}"
    ).fetchone()
    return {
        "min": float(row[0]),
        "25%": float(row[1]),
        "50%": float(row[2]),
        "75%": float(row[3]),
        "max": float(row[4]),
    }


def _top_values(
    con: duckdb.DuckDBPyConnection, table: str, column: str
) -> list[tuple[str, int]]:
    qcol = _quote_ident(column)
    qtable = _quote_ident(table)
    rows = con.execute(
        f"SELECT {qcol}, COUNT(*) AS c FROM {qtable} "
        f"WHERE {qcol} IS NOT NULL GROUP BY {qcol} "
        f"ORDER BY c DESC, {qcol} ASC LIMIT {_TOP_K}"
    ).fetchall()
    return [(str(value), count) for value, count in rows]


def profile_table(con: duckdb.DuckDBPyConnection, table: str) -> TableProfile:
    qtable = _quote_ident(table)
    row_count = con.execute(f"SELECT COUNT(*) FROM {qtable}").fetchone()[0]
    schema_rows = con.execute(f"DESCRIBE {qtable}").fetchall()

    columns: list[ColumnProfile] = []
    candidate_primary_keys: list[str] = []

    for column_name, column_type, *_rest in schema_rows:
        qcol = _quote_ident(column_name)

        non_null_count = con.execute(f"SELECT COUNT({qcol}) FROM {qtable}").fetchone()[0]
        null_count = row_count - non_null_count
        null_fraction = (null_count / row_count) if row_count > 0 else 0.0

        n_unique = con.execute(f"SELECT COUNT(DISTINCT {qcol}) FROM {qtable}").fetchone()[0]

        has_data = row_count > 0 and null_count < row_count
        numeric = _is_numeric_type(column_type)
        stringy = _is_string_type(column_type)

        quantiles = _quantiles(con, table, column_name) if (has_data and numeric) else None
        top_values = (
            _top_values(con, table, column_name) if (has_data and not numeric) else None
        )

        if not has_data:
            sentinel_candidates: list[str] = []
        elif numeric:
            sentinel_candidates = _numeric_sentinel_candidates(con, table, column_name)
        elif stringy:
            sentinel_candidates = _string_sentinel_candidates(con, table, column_name)
        else:
            sentinel_candidates = []

        columns.append(
            ColumnProfile(
                name=column_name,
                dtype=column_type,
                null_count=null_count,
                null_fraction=null_fraction,
                n_unique=n_unique,
                quantiles=quantiles,
                top_values=top_values,
                sentinel_candidates=sentinel_candidates,
            )
        )

        if row_count > 0 and null_count == 0 and n_unique == row_count:
            candidate_primary_keys.append(column_name)

    return TableProfile(
        table_name=table,
        row_count=row_count,
        columns=columns,
        candidate_primary_keys=candidate_primary_keys,
    )
