"""Opt-in, source-local quality queries; no imports or execution from Soda.

Catalog scope was researched from https://github.com/sodadata/soda-core and
https://docs.soda.io/soda-documentation/developer-tools/contract-language-reference .
These independently authored native SQL templates are not a Soda runtime or a
contract runner. Install stores FnDef records only; users choose when to run SQL.
Percentages return NULL for an empty denominator. Missing means SQL NULL;
blank strings are counted separately. Duplicate/unique metrics exclude NULL.
"""
from __future__ import annotations

from d8r.sql import FnDef, parse_parameters


def quality_functions(dialect: str) -> tuple[FnDef, ...]:
    """Return native query definitions for PostgreSQL, D1 SQLite or DuckDB.

    No regex/statistical extension is assumed on SQLite. Schema queries use
    information_schema on PostgreSQL/DuckDB and pragma_table_info on SQLite.
    They reflect the current connection's visible catalog, not schema history.
    Freshness requires native timestamps (SQLite date/time-compatible values).
    Predicates are user-authored SQL expressions, not identifier strings.
    """
    if dialect not in {"postgres", "sqlite", "duckdb"}:
        raise ValueError(f"Unsupported quality-template dialect: {dialect}")
    definitions: list[FnDef] = []

    def add(name: str, parameters: str, body: str, description: str) -> None:
        names, kinds = parse_parameters(parameters)
        definitions.append(FnDef("dq_" + name, names, body, description, kinds))

    source = "source:relation"
    column = source + ", field:identifier"
    add("row_count", source, "SELECT COUNT(*) AS value FROM @source", "Number of rows, including NULLs.")
    add("row_count_delta", source + ", reference:relation",
        "SELECT (SELECT COUNT(*) FROM @source) - (SELECT COUNT(*) FROM @reference) AS value",
        "Signed row-count difference: source minus reference.")
    for name, condition in (("missing", "@field IS NULL"), ("present", "@field IS NOT NULL")):
        count = f"COUNT(CASE WHEN {condition} THEN 1 END)"
        add(name + "_count", column, f"SELECT {count} AS value FROM @source", f"{name.title()} (SQL NULL) row count.")
        add(name + "_percent", column, f"SELECT 100.0 * {count} / NULLIF(COUNT(*), 0) AS value FROM @source",
            f"{name.title()} percentage of all rows; NULL for empty input.")
    for name, case in (("valid", "CASE WHEN @condition THEN 1 END"),
                       ("invalid", "CASE WHEN @condition THEN NULL ELSE 1 END")):
        parameters = source + ", condition"
        add(name + "_count", parameters, f"SELECT COUNT({case}) AS value FROM @source",
            f"{name.title()} predicate count; valid means TRUE, invalid means FALSE or NULL.")
        add(name + "_percent", parameters,
            f"SELECT 100.0 * COUNT({case}) / NULLIF(COUNT(*), 0) AS value FROM @source",
            f"{name.title()} predicate percentage; FALSE/NULL are invalid; empty input returns NULL.")
    add("distinct_count", column, "SELECT COUNT(DISTINCT @field) AS value FROM @source",
        "Distinct non-NULL values.")
    duplicates = "COUNT(@field) - COUNT(DISTINCT @field)"
    add("duplicate_count", column, f"SELECT {duplicates} AS value FROM @source",
        "Excess non-NULL duplicate rows: group size minus one for each value.")
    add("duplicate_percent", column,
        f"SELECT 100.0 * ({duplicates}) / NULLIF(COUNT(@field), 0) AS value FROM @source",
        "Excess duplicate rows as a percentage of non-NULL rows.")
    groups = "SELECT COUNT(*) AS n FROM @source WHERE @field IS NOT NULL GROUP BY @field"
    add("duplicate_groups", column, f"SELECT COUNT(*) AS value FROM ({groups} HAVING COUNT(*) > 1) AS groups",
        "Number of repeated non-NULL value groups.")
    add("unique_count", column, f"SELECT COUNT(*) AS value FROM ({groups} HAVING COUNT(*) = 1) AS groups",
        "Non-NULL values occurring exactly once (singleton rows).")
    add("unique_percent", column,
        f"SELECT 100.0 * COUNT(CASE WHEN n = 1 THEN 1 END) / NULLIF(SUM(n), 0) AS value FROM ({groups}) AS groups",
        "Singleton rows as a percentage of non-NULL rows.")
    for name in ("min", "max", "sum", "avg"):
        add(name, column, f"SELECT {name.upper()}(@field) AS value FROM @source",
            f"Native {name.upper()} over non-NULL values; NULL for empty input.")
    add("out_of_range_count", column + ", minimum, maximum",
        "SELECT COUNT(*) AS value FROM @source WHERE @field < @minimum OR @field > @maximum",
        "Non-NULL values outside inclusive bounds; use compatible non-NULL bounds.")
    add("in_range_count", column + ", minimum, maximum",
        "SELECT COUNT(*) AS value FROM @source WHERE @field BETWEEN @minimum AND @maximum",
        "Non-NULL values inside inclusive bounds.")
    for name in ("min", "max", "avg"):
        add("length_" + name, column,
            f"SELECT {name.upper()}(LENGTH(CAST(@field AS TEXT))) AS value FROM @source",
            f"{name.upper()} character length after native text conversion; NULLs excluded.")
    add("blank_count", column,
        "SELECT COUNT(*) AS value FROM @source WHERE TRIM(CAST(@field AS TEXT)) = ''",
        "Empty or space-only strings after text conversion; NULLs excluded.")
    add("like_count", column + ", pattern",
        "SELECT COUNT(*) AS value FROM @source WHERE CAST(@field AS TEXT) LIKE @pattern",
        "Native LIKE pattern matches; case sensitivity follows the engine.")
    add("fk_missing_count", column + ", reference:relation, reference_field:identifier",
        "SELECT COUNT(*) AS value FROM @source AS child WHERE child.@field IS NOT NULL "
        "AND NOT EXISTS (SELECT 1 FROM @reference AS parent WHERE parent.@reference_field = child.@field)",
        "Non-NULL child keys without a parent match; single-column foreign key.")
    freshness = {
        "postgres": "EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(@field)))",
        "duckdb": "EPOCH(CURRENT_TIMESTAMP - MAX(@field))",
        "sqlite": "(JULIANDAY('now') - JULIANDAY(MAX(@field))) * 86400.0",
    }[dialect]
    add("freshness_seconds", column, f"SELECT {freshness} AS value FROM @source",
        "Age of newest timestamp in seconds; NULL without timestamps; future values are negative.")
    if dialect != "sqlite":
        for name in ("stddev_pop", "stddev_samp", "var_pop", "var_samp"):
            add(name, column, f"SELECT {name.upper()}(@field) AS value FROM @source",
                f"Native {name.upper()} over non-NULL numeric values.")
        regex = "CAST(@field AS TEXT) ~ @pattern" if dialect == "postgres" else "REGEXP_MATCHES(CAST(@field AS TEXT), @pattern)"
        add("regex_count", column + ", pattern", f"SELECT COUNT(*) AS value FROM @source WHERE {regex}",
            "Regex search matches; anchor ^...$ for full-string validation; native regex dialect.")
    if dialect == "sqlite":
        metadata = "SELECT name AS column_name, type AS data_type, cid + 1 AS ordinal_position FROM pragma_table_info(@table_name, @schema_name)"
    else:
        metadata = ("SELECT column_name, data_type, ordinal_position FROM information_schema.columns "
                    "WHERE table_schema = @schema_name AND table_name = @table_name AND table_catalog = CURRENT_DATABASE()")
    schema = "schema_name, table_name"
    add("schema_columns", schema, metadata + " ORDER BY ordinal_position",
        "Visible column names, native data types and positions; pass schema/table names as SQL string values.")
    add("schema_column_count", schema, f"SELECT COUNT(*) AS value FROM ({metadata}) AS columns",
        "Visible column count; zero also means absent or inaccessible table.")
    add("column_exists", schema + ", column_name",
        f"SELECT COUNT(*) AS value FROM ({metadata}) AS columns WHERE column_name = @column_name",
        "One when the exact column name is visible, otherwise zero; names are SQL strings.")
    add("column_type_matches", schema + ", column_name, expected_type",
        f"SELECT COUNT(*) AS value FROM ({metadata}) AS columns WHERE column_name = @column_name AND LOWER(data_type) = LOWER(@expected_type)",
        "One when column and native reported type match; case-insensitive type spelling, not type aliases.")
    return tuple(definitions)
