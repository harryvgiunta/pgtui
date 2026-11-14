"""Native behavior tests; PostgreSQL integration is opt-in via a real DSN."""
from __future__ import annotations

import hashlib
import os
import threading

import duckdb
import pytest

from d8r.engine import (
    POSTGRES_VERSION, EngineError, add_parquet_source, add_postgres_source,
    cancel_source, close_source, column_values, execute_sql, load,
    refresh_source_schema, sample_rows, transaction_open,
)
from d8r.sql import relation_name


def test_empty_startup_is_disconnected():
    sources = load()
    assert set(sources) == {"postgres"}
    assert sources["postgres"].kind == "disconnected"
    assert sources["postgres"].con is None
    assert sources["postgres"].datasets == {}
    assert POSTGRES_VERSION == "18.7"
    with pytest.raises(EngineError, match="disconnected"):
        execute_sql(sources["postgres"], "SELECT 1")


def test_parquet_metadata_and_native_query(parquet_source):
    source = parquet_source
    assert source.kind == "parquet"
    assert source.dialect == "duckdb"
    assert source.server_version == duckdb.__version__
    assert set(source.datasets) == {"data"}
    metadata = source.datasets["data"]
    assert metadata["rows"] is None
    assert [column.name for column in metadata["columns"]] == ["id", "label"]
    assert metadata["foreign_keys"] == ()
    result = execute_sql(source, "SELECT id, label FROM data ORDER BY id", max_rows=2)
    assert result["rows"] == [[1, "alpha"], [2, "beta"]]
    assert result["total"] == 3
    assert result["columns"] == ["id", "label"]
    assert result["dtypes"] == ["BIGINT", "VARCHAR"]
    assert result["sql"] == "SELECT id, label FROM data ORDER BY id"
    assert not result["schema_changed"]
    assert result["ms"] >= 0


def test_preview_limit_is_not_query_limit(parquet_source):
    query = "SELECT range AS n FROM range(20000)"
    result = execute_sql(parquet_source, query, max_rows=0)
    assert len(result["rows"]) == 10000
    assert result["total"] == 10001
    assert result["sql"] == query
    explicit = execute_sql(parquet_source, "SELECT * FROM data LIMIT 1", max_rows=50)
    assert len(explicit["rows"]) == explicit["total"] == 1
    count = execute_sql(parquet_source, "SELECT count(*) FROM data", max_rows=1)
    assert count["rows"] == [[3]]


def test_empty_result_retains_types(parquet_source):
    result = execute_sql(parquet_source, "SELECT * FROM data WHERE false")
    assert result["columns"] == ["id", "label"]
    assert result["dtypes"] == ["BIGINT", "VARCHAR"]
    assert result["rows"] == []
    assert result["total"] == 0


@pytest.mark.parametrize("query", ["", "-- only a comment", "SELECT 1; SELECT 2"])
def test_requires_one_statement(parquet_source, query):
    with pytest.raises(EngineError, match="one SQL statement"):
        execute_sql(parquet_source, query)


def test_semicolon_in_literal_is_not_statement_boundary(parquet_source):
    assert execute_sql(parquet_source, "SELECT ';' AS value;")["rows"] == [[";"]]


@pytest.mark.parametrize("limit", [-1, True, 1.5, "10"])
def test_rejects_invalid_fetch_limits(parquet_source, limit):
    with pytest.raises(EngineError, match="Row limit"):
        execute_sql(parquet_source, "SELECT 1", max_rows=limit)


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_rejects_invalid_deadlines(parquet_source, timeout):
    with pytest.raises(EngineError, match="timeout"):
        execute_sql(parquet_source, "SELECT 1", timeout=timeout)


def test_manual_ddl_dml_and_transactions(parquet_source):
    source = parquet_source
    assert execute_sql(source, "CREATE TABLE scratch (id BIGINT)")["schema_changed"]
    execute_sql(source, "INSERT INTO scratch VALUES (1)")
    execute_sql(source, "BEGIN")
    assert transaction_open(source)
    execute_sql(source, "INSERT INTO scratch VALUES (2)")
    with pytest.raises(EngineError, match="open transaction"):
        execute_sql(source, "SELECT * FROM data", readonly=True)
    execute_sql(source, "ROLLBACK")
    assert not transaction_open(source)
    assert execute_sql(source, "SELECT * FROM scratch")["rows"] == [[1]]
    refresh_source_schema(source)
    assert relation_name("main", "scratch") in source.datasets


@pytest.mark.parametrize("query", [
    "DELETE FROM data", "CREATE TABLE forbidden AS SELECT 1", "SELECT * INTO forbidden FROM data",
    "WITH changed AS (DELETE FROM data RETURNING *) SELECT * FROM changed",
    "SET enable_external_access = true", "PRAGMA enable_external_access=true",
    "COPY data TO 'forbidden.parquet'", "ATTACH 'forbidden.db' AS forbidden",
    "INSTALL httpfs", "LOAD httpfs", "PREPARE bad AS SELECT * FROM data", "CALL checkpoint()",
])
def test_parquet_readonly_rejects_mutations(parquet_source, query):
    with pytest.raises(EngineError):
        execute_sql(parquet_source, query, readonly=True)
    assert not transaction_open(parquet_source)
    assert execute_sql(parquet_source, "SELECT count(*) FROM data", readonly=True)["rows"] == [[3]]


def test_parquet_readonly_blocks_side_effects_inside_select(parquet_source):
    execute_sql(parquet_source, "CREATE SEQUENCE protected_sequence START 1")
    with pytest.raises(EngineError):
        execute_sql(parquet_source, "SELECT nextval('protected_sequence')", readonly=True)
    assert execute_sql(parquet_source, "SELECT nextval('protected_sequence')")["rows"] == [[1]]


def test_parquet_external_access_and_original_file_protection(tmp_path):
    original = tmp_path / "original.parquet"
    con = duckdb.connect()
    try:
        con.execute("COPY (SELECT 42 AS id) TO ? (FORMAT PARQUET)", [str(original)])
    finally:
        con.close()
    checksum = hashlib.sha256(original.read_bytes()).digest()
    source = add_parquet_source(original)
    other = tmp_path / "private.txt"
    other.write_text("private-value", encoding="utf-8")
    try:
        for readonly in (False, True):
            with pytest.raises(EngineError):
                execute_sql(source, f"COPY data TO '{original.as_posix()}' (FORMAT PARQUET)", readonly=readonly)
            with pytest.raises(EngineError):
                execute_sql(source, f"SELECT * FROM read_text('{other.as_posix()}')", readonly=readonly)
            with pytest.raises(EngineError):
                execute_sql(source, "SELECT * FROM read_parquet('https://example.com/private.parquet')", readonly=readonly)
        with pytest.raises(EngineError):
            execute_sql(source, "SET lock_configuration=false")
        assert hashlib.sha256(original.read_bytes()).digest() == checksum
        assert execute_sql(source, "SELECT * FROM data", readonly=True)["rows"] == [[42]]
    finally:
        close_source(source)


@pytest.mark.parametrize("path", ["https://host/data.parquet", "file:///tmp/data.parquet", "s3://bucket/data.parquet", "//host/share/data.parquet", "\\\\host\\share\\data.parquet", "*.parquet", "[a].pq", "missing.parquet"])
def test_parquet_rejects_nonlocal_or_missing_paths(path):
    with pytest.raises(EngineError):
        add_parquet_source(path)


def test_parquet_rejects_invalid_file_without_leaking_contents(tmp_path):
    file = tmp_path / "secret.parquet"
    file.write_bytes(b"private-password-not-parquet")
    with pytest.raises(EngineError) as error:
        add_parquet_source(file)
    assert "private-password" not in str(error.value)
    assert str(file) not in str(error.value)


def test_metadata_sampling_quotes_and_validates(parquet_source):
    assert sample_rows(parquet_source, "data", ["label"], 2)["columns"] == ["label"]
    values, truncated = column_values(parquet_source, "data", "id", 2)
    assert len(values) == 2
    assert truncated
    with pytest.raises(EngineError, match="registered table"):
        sample_rows(parquet_source, "data; DROP VIEW data", ["label"], 2)
    with pytest.raises(EngineError, match="registered column"):
        column_values(parquet_source, "data", 'id"; SELECT 1', 2)
    execute_sql(parquet_source, 'CREATE TABLE "a.b" ("odd column" INTEGER)')
    execute_sql(parquet_source, 'INSERT INTO "a.b" VALUES (7)')
    refresh_source_schema(parquet_source)
    assert sample_rows(parquet_source, relation_name("main", "a.b"), ["odd column"], 1)["rows"] == [[7]]


def test_deadline_rolls_back_and_connection_remains_usable(parquet_source):
    with pytest.raises(EngineError, match="deadline"):
        execute_sql(parquet_source, "SELECT sum(sqrt(i::DOUBLE)) FROM range(1000000000000) t(i)", readonly=True, timeout=0.02)
    assert not transaction_open(parquet_source)
    assert execute_sql(parquet_source, "SELECT 42", readonly=True)["rows"] == [[42]]


def test_explicit_cancel_and_close(parquet_source):
    timer = threading.Timer(0.05, cancel_source, args=(parquet_source,))
    timer.start()
    try:
        with pytest.raises(EngineError):
            execute_sql(parquet_source, "SELECT sum(sqrt(i::DOUBLE)) FROM range(1000000000000) t(i)", timeout=3)
    finally:
        timer.cancel()
        timer.join()
    assert execute_sql(parquet_source, "SELECT 1")["rows"] == [[1]]
    close_source(parquet_source)
    close_source(parquet_source)
    assert parquet_source.con is None


@pytest.fixture
def postgres_source():
    dsn = os.environ.get("PGTUI_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set PGTUI_TEST_POSTGRES_DSN to exercise native PostgreSQL")
    import psycopg
    params = psycopg.conninfo.conninfo_to_dict(dsn)
    with psycopg.connect(dsn, autocommit=True) as probe:
        host, port, database, user = probe.info.host, probe.info.port, probe.info.dbname, probe.info.user
    source = add_postgres_source("test-postgres", host=host, port=port, database=database, user=user,
                                 password=params.get("password", os.environ.get("PGPASSWORD", "")),
                                 sslmode=params.get("sslmode", "prefer"))
    try:
        yield source
    finally:
        close_source(source)


def test_postgres_native_dml_ddl_and_server_function(postgres_source):
    source = postgres_source
    assert source.server_version
    assert source.default_schema == "public"
    execute_sql(source, "CREATE TEMP TABLE native_items(id INTEGER PRIMARY KEY, label TEXT)")
    assert execute_sql(source, "INSERT INTO native_items VALUES (1,'alpha'),(2,'beta')")["total"] == 2
    execute_sql(source, "BEGIN")
    execute_sql(source, "DELETE FROM native_items WHERE id=1")
    assert transaction_open(source)
    execute_sql(source, "ROLLBACK")
    assert not transaction_open(source)
    query = "SELECT id FROM native_items ORDER BY id"
    assert execute_sql(source, query, max_rows=1)["rows"] == [[1]]
    assert execute_sql(source, query, max_rows=1)["total"] == 2
    execute_sql(source, "CREATE FUNCTION pg_temp.double_it(n INTEGER) RETURNS INTEGER LANGUAGE SQL AS $$ SELECT n * 2 $$")
    assert execute_sql(source, "SELECT pg_temp.double_it(4)")["rows"] == [[8]]


def test_postgres_catalog_metadata_and_composite_fk(postgres_source):
    source = postgres_source
    execute_sql(source, "CREATE TEMP TABLE native_parent(a INTEGER, b INTEGER, PRIMARY KEY (a,b))")
    execute_sql(source, "CREATE TEMP TABLE native_child(x INTEGER, y INTEGER, FOREIGN KEY (x,y) REFERENCES native_parent(a,b))")
    execute_sql(source, "COMMENT ON COLUMN native_child.x IS 'paired x'")
    refresh_source_schema(source)
    child = next(value for key, value in source.datasets.items() if key.endswith('."native_child"'))
    assert child["rows"] is None
    assert child["columns"][0].doc == "paired x"
    assert child["foreign_keys"][0].columns == ("x", "y")
    assert child["foreign_keys"][0].target_columns == ("a", "b")
    assert child["foreign_keys"][0].target_table.endswith('."native_parent"')


@pytest.mark.parametrize("query", [
    "INSERT INTO native_guard VALUES (2)",
    "WITH changed AS (DELETE FROM native_guard RETURNING *) SELECT * FROM changed",
    "SELECT * INTO TEMP native_stolen FROM native_guard",
    "SELECT pg_catalog.lo_export(1, '/tmp/forbidden')",
    'SELECT pg_catalog."lo_export"(1, \'/tmp/forbidden\')',
    "SELECT pg_advisory_lock(1)",
    'SELECT "pg_read_file"(\'/tmp/private\')',
    "SELECT dblink_connect('host=example.com')",
    "SELECT set_config('statement_timeout','0',false)",
])
def test_postgres_readonly_mutations_and_unsafe_functions(postgres_source, query):
    source = postgres_source
    execute_sql(source, "CREATE TEMP TABLE native_guard(id INTEGER)")
    execute_sql(source, "INSERT INTO native_guard VALUES (1)")
    with pytest.raises(EngineError):
        execute_sql(source, query, readonly=True)
    assert execute_sql(source, "SELECT * FROM native_guard", readonly=True)["rows"] == [[1]]
    assert not transaction_open(source)


def test_postgres_readonly_refuses_user_transaction(postgres_source):
    execute_sql(postgres_source, "BEGIN")
    with pytest.raises(EngineError, match="open transaction"):
        execute_sql(postgres_source, "SELECT 1", readonly=True)
    assert transaction_open(postgres_source)
    execute_sql(postgres_source, "ROLLBACK")


def test_postgres_error_sanitization_deadline_and_recovery(postgres_source):
    with pytest.raises(EngineError) as error:
        execute_sql(postgres_source, "SELECT 'private-password-value'::INTEGER")
    assert "private-password-value" not in str(error.value)
    assert "SQLSTATE 22P02" in str(error.value)
    with pytest.raises(EngineError, match="deadline|cancelled"):
        execute_sql(postgres_source, "SELECT pg_sleep(3)", readonly=True, timeout=0.02)
    assert not transaction_open(postgres_source)
    assert execute_sql(postgres_source, "SELECT 42", readonly=True)["rows"] == [[42]]


@pytest.mark.parametrize("state,hint", [
    ("42601", "commas"), ("42702", "alias.column"), ("42P09", "distinct alias"),
    ("42P01", "schema qualification"), ("42703", "column spelling"),
    ("42803", "every nonaggregate SELECT expression"), ("42883", "argument count"),
    ("42725", "explicit argument casts"), ("42804", "target column types"),
    ("42846", "supported conversion"), ("22P02", "input format"),
])
def test_postgres_structured_error_hints_never_expose_driver_fields(state, hint):
    import psycopg
    from psycopg.pq import DiagnosticField
    from d8r.engine.datasources import _safe_error

    secret = b"private-password-value SELECT 'private-literal'"
    exc = psycopg.errors.lookup(state)(secret.decode(), info={
        DiagnosticField.SQLSTATE: state.encode(),
        DiagnosticField.STATEMENT_POSITION: b"17",
        DiagnosticField.MESSAGE_PRIMARY: secret,
        DiagnosticField.MESSAGE_DETAIL: secret,
        DiagnosticField.MESSAGE_HINT: secret,
        DiagnosticField.CONTEXT: secret,
        DiagnosticField.INTERNAL_QUERY: secret,
        DiagnosticField.COLUMN_NAME: secret,
        DiagnosticField.TABLE_NAME: secret,
    })
    message = str(_safe_error(exc))
    assert f"SQLSTATE {state}" in message
    assert "at character 17" in message
    assert hint in message
    assert "private-" not in message
    if state == "42803":
        assert "* or alias.*" in message


@pytest.mark.parametrize("position", [b"0", b"-1", b"2147483648", b"9" * 1000,
                                      b"17 private-password", b"1\n2"])
def test_postgres_error_positions_require_bounded_positive_diagnostics(position):
    import psycopg
    from psycopg.pq import DiagnosticField
    from d8r.engine.datasources import _safe_error

    exc = psycopg.errors.SyntaxError("secret", info={DiagnosticField.STATEMENT_POSITION: position})
    message = str(_safe_error(exc))
    assert "at character" not in message
    assert "secret" not in message


def test_arbitrary_errors_cannot_impersonate_native_diagnostics():
    from d8r.engine.datasources import _safe_error

    class UntrustedError(Exception):
        @property
        def sqlstate(self):
            raise AssertionError("Arbitrary diagnostics must not be inspected")

        def __str__(self):
            raise AssertionError("Arbitrary exceptions must not be stringified")

    assert str(_safe_error(UntrustedError())) == "SQL execution failed"


@pytest.mark.parametrize("query,hint", [
    ("SELECT FROM data", "commas"),
    ("SELECT id FROM data AS a JOIN data AS b ON a.id = b.id", "alias.column"),
    ("SELECT label, count(*) FROM data", "every nonaggregate SELECT expression"),
    ("SELECT * FROM data GROUP BY id", "* or alias.*"),
    ('SELECT "private-password-value" FROM data', "column spelling"),
    ('SELECT * FROM "private-password-value"', "schema qualification"),
    ("SELECT private_missing_function(id) FROM data", "argument count"),
    ("SELECT CAST('private-password-value' AS INTEGER)", "input format"),
])
def test_duckdb_native_errors_offer_safe_hints(parquet_source, query, hint):
    with pytest.raises(EngineError) as error:
        execute_sql(parquet_source, query)
    assert hint in str(error.value)
    assert "private-" not in str(error.value)
    assert "private_missing_function" not in str(error.value)
    assert execute_sql(parquet_source, "SELECT 42")["rows"] == [[42]]


@pytest.mark.parametrize("query,state,hint", [
    ("SELECT FROM", "42601", "commas"),
    ("SELECT id FROM (VALUES (1)) a(id) CROSS JOIN (VALUES (2)) b(id)", "42702", "alias.column"),
    ("SELECT id, count(*) FROM (VALUES (1)) a(id)", "42803", "every nonaggregate SELECT expression"),
    ("SELECT a.*, count(*) FROM (VALUES (1, 2)) a(id, value)", "42803", "* or alias.*"),
    ('SELECT "private-password-value"', "42703", "column spelling"),
    ('SELECT * FROM "private-password-value"', "42P01", "schema qualification"),
    ("SELECT pg_catalog.length(1)", "42883", "argument count"),
])
def test_postgres_native_errors_offer_safe_hints(postgres_source, query, state, hint):
    with pytest.raises(EngineError) as error:
        execute_sql(postgres_source, query)
    message = str(error.value)
    assert f"SQLSTATE {state}" in message
    assert hint in message
    assert "private-" not in message
    assert execute_sql(postgres_source, "SELECT 42")["rows"] == [[42]]


@pytest.mark.parametrize("sql", ["SELECT 'private-password-value", "SELECT 1 /* private-password-value"])
def test_unterminated_sql_offers_safe_local_hint(parquet_source, sql):
    with pytest.raises(EngineError) as error:
        execute_sql(parquet_source, sql)
    assert "Hint: Check matching quotes" in str(error.value)
    assert "private-password-value" not in str(error.value)
