"""Native routine discovery without binding or evaluating user routines."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import os

import pytest

from d8r.engine import execute_sql, refresh_source_schema, transaction_open
from d8r.engine import datasources
from d8r.engine.datasources import DataSource, EngineError
from d8r.sql import ColumnDef, RoutineDef, SchemaContext, relation_name, relation_parts
from d8r.tui import session as session_module
from d8r.tui.session import Session


def named(source, name):
    return [routine for routine in source.routines
            if relation_parts(routine.name, dialect=source.dialect)[-1] == name]


def test_routine_metadata_is_frozen_and_schema_positional_arguments_stay_compatible():
    parameters = ["value INTEGER"]
    types = ["INTEGER"]
    columns = [ColumnDef("answer", "INTEGER")]
    routine = RoutineDef("main.answers", parameters, "TABLE", "docs", "table", False, types, columns)
    routines = [routine]
    snapshot = SchemaContext((), (), "public", routines)
    parameters.clear()
    types.clear()
    columns.clear()
    routines.clear()
    assert routine.parameters == ("value INTEGER",)
    assert routine.parameter_types == ("INTEGER",)
    assert routine.columns == (ColumnDef("answer", "INTEGER"),)
    assert snapshot.routines == (routine,)
    assert snapshot.default_schema == "public"
    assert SchemaContext((), (), "public").routines == ()
    with pytest.raises(FrozenInstanceError):
        routine.name = "changed"


def test_duckdb_native_overloads_types_and_descriptions(parquet_source):
    source = parquet_source
    actual = source.con.execute("""
        SELECT parameter_types, return_type, description
        FROM duckdb_functions()
        WHERE schema_name = 'main' AND function_name = 'abs' AND function_type = 'scalar'
    """).fetchall()
    discovered = named(source, "abs")
    assert len(discovered) == len(actual) > 1
    assert {(routine.parameter_types, routine.return_type, routine.doc) for routine in discovered} == {
        (tuple(types), result or "", doc or "") for types, result, doc in actual
    }
    assert all(routine.kind == "scalar" for routine in discovered)
    assert all(routine.name == relation_name("main", "abs") for routine in discovered)
    assert any(routine.doc for routine in discovered)
    assert any(routine.kind == "aggregate" for routine in named(source, "sum"))
    assert any(routine.kind == "table" for routine in named(source, "range"))
    assert any(routine.variadic for routine in named(source, "concat"))


def test_duckdb_user_macros_are_discovered_without_evaluation(parquet_source):
    source = parquet_source
    source.con.execute("CREATE MACRO completion_scalar(value) AS error('must not execute')")
    source.con.execute("CREATE MACRO completion_table(value) AS TABLE SELECT error('must not execute') AS output")
    refresh_source_schema(source)
    scalar, = named(source, "completion_scalar")
    table, = named(source, "completion_table")
    assert scalar.kind == "scalar"
    assert scalar.parameters[0].startswith("value")
    assert scalar.parameter_types == ("",)
    assert table.kind == "table"
    # Native catalog does not expose macro/table result columns. Do not guess.
    assert table.columns == ()
    assert scalar.columns == ()


def test_duckdb_refresh_keeps_user_transaction_and_uncommitted_catalog(parquet_source):
    source = parquet_source
    execute_sql(source, "BEGIN")
    try:
        execute_sql(source, "CREATE MACRO completion_uncommitted(value) AS value + 1")
        refresh_source_schema(source)
        assert named(source, "completion_uncommitted")
        assert transaction_open(source)
    finally:
        execute_sql(source, "ROLLBACK")
    refresh_source_schema(source)
    assert not named(source, "completion_uncommitted")


def test_refresh_failure_preserves_both_snapshots_and_redacts_error(parquet_source, monkeypatch):
    source = parquet_source
    previous_tables, previous_routines = source.datasets, source.routines
    source.con.execute("CREATE TEMP TABLE not_published (id INTEGER)")

    def fail(_):
        raise RuntimeError("private password and SQL body")

    monkeypatch.setattr(datasources, "_duckdb_routines", fail)
    with pytest.raises(EngineError, match="^Schema discovery failed$"):
        refresh_source_schema(source)
    assert source.datasets is previous_tables
    assert source.routines is previous_routines


def test_session_handoff_is_offline_and_retains_old_routine_snapshots(parquet_source, monkeypatch):
    session = Session({parquet_source.id: parquet_source})
    previous = session.schema
    replacement = RoutineDef(relation_name("main", "replacement"), ("value INTEGER",))
    parquet_source.routines = (replacement,)

    def no_io(*args, **kwargs):
        raise AssertionError("Session snapshot handoff must not request metadata")

    class ExecutionLock:
        def __enter__(self):
            raise AssertionError("Editor snapshot must never wait for a running query")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(session_module, "refresh_source_schema", no_io)
    with monkeypatch.context() as guarded:
        guarded.setattr(parquet_source, "_lock", ExecutionLock())
        session.refresh_schema()
    assert session.schema.routines == (replacement,)
    assert previous.routines and replacement not in previous.routines
    disconnected = DataSource("offline", "Offline", "", "disconnected", "postgres")
    session.sources[disconnected.id] = disconnected
    session.active_id = disconnected.id
    session.refresh_schema()
    assert session.schema.routines == ()


def test_d1_and_disconnected_sources_publish_no_server_routines(d1_source):
    d1_source.routines = (RoutineDef("stale"),)
    refresh_source_schema(d1_source)
    assert d1_source.routines == ()
    source = DataSource("offline", "Offline", "", "disconnected", "postgres",
                        routines=(RoutineDef("stale"),))
    refresh_source_schema(source)
    assert source.routines == ()
    assert source.datasets == {}


@pytest.fixture
def postgres_routine_source():
    dsn = os.environ.get("PGTUI_TEST_POSTGRES_DSN")
    if not dsn:
        pytest.skip("Set PGTUI_TEST_POSTGRES_DSN for native PostgreSQL routine discovery")
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as con:
        con.execute("CREATE TEMP TABLE completion_result (id INTEGER, label TEXT)")
        source = DataSource("routine-postgres", "PostgreSQL", "", "postgres-live", "postgres",
                            con=con, default_schema="public")
        yield source


def test_postgres_accessible_native_overloads_out_columns_and_docs(postgres_routine_source):
    source = postgres_routine_source
    source.con.execute("""
        CREATE FUNCTION pg_temp.completion_overload(value INTEGER) RETURNS INTEGER
        LANGUAGE SQL AS $$ SELECT value + 1 $$;
        CREATE FUNCTION pg_temp.completion_overload(value TEXT) RETURNS TEXT
        LANGUAGE SQL AS $$ SELECT value || '!' $$;
        COMMENT ON FUNCTION pg_temp.completion_overload(INTEGER) IS 'Integer overload';
        CREATE FUNCTION pg_temp.completion_outputs(value INTEGER)
        RETURNS TABLE(answer INTEGER, label TEXT)
        LANGUAGE SQL AS $$ SELECT value, 'label'::TEXT $$;
        CREATE FUNCTION pg_temp.completion_composite()
        RETURNS SETOF pg_temp.completion_result
        LANGUAGE SQL AS $$ SELECT * FROM pg_temp.completion_result $$;
        CREATE FUNCTION pg_temp.completion_variadic(VARIADIC items INTEGER[])
        RETURNS INTEGER LANGUAGE SQL AS $$ SELECT cardinality(items) $$;
        CREATE PROCEDURE pg_temp.completion_procedure(IN value INTEGER)
        LANGUAGE SQL AS $$ SELECT value $$
    """)
    refresh_source_schema(source)
    overloads = named(source, "completion_overload")
    assert {routine.parameter_types for routine in overloads} == {("integer",), ("text",)}
    integer = next(routine for routine in overloads if routine.parameter_types == ("integer",))
    assert integer.parameters == ('"value" integer',)
    assert integer.return_type == "integer"
    assert integer.doc == "Integer overload"
    outputs, = named(source, "completion_outputs")
    assert outputs.kind == "table"
    assert outputs.parameter_types == ("integer",)
    assert outputs.columns == (ColumnDef("answer", "integer"), ColumnDef("label", "text"))
    composite, = named(source, "completion_composite")
    assert composite.columns == (ColumnDef("id", "integer"), ColumnDef("label", "text"))
    variadic, = named(source, "completion_variadic")
    assert variadic.variadic and variadic.parameter_types == ("integer[]",)
    assert variadic.parameters[0].startswith("VARIADIC ")
    procedure, = named(source, "completion_procedure")
    assert procedure.kind == "procedure" and procedure.return_type == ""
    assert any(routine.kind == "aggregate" for routine in named(source, "sum"))
    assert any(routine.kind == "window" for routine in named(source, "row_number"))
    assert not transaction_open(source)


def test_postgres_discovery_never_executes_user_functions(postgres_routine_source):
    source = postgres_routine_source
    source.con.execute("""
        CREATE FUNCTION pg_temp.completion_unsafe() RETURNS SETOF INTEGER
        LANGUAGE SQL AS $$ SELECT 1 / 0 $$
    """)
    refresh_source_schema(source)
    routine, = named(source, "completion_unsafe")
    assert routine.kind == "table"
    assert routine.columns == ()


def test_postgres_catalog_failure_preserves_user_transaction(postgres_routine_source, monkeypatch):
    source = postgres_routine_source
    refresh_source_schema(source)
    previous_tables, previous_routines = source.datasets, source.routines
    source.con.execute("BEGIN")
    try:
        source.con.execute("INSERT INTO pg_temp.completion_result VALUES (1, 'retained')")
        monkeypatch.setattr(datasources, "_PG_ROUTINES", "SELECT 1 / 0")
        with pytest.raises(EngineError, match="Schema discovery failed"):
            refresh_source_schema(source)
        assert transaction_open(source)
        assert source.con.execute("SELECT label FROM pg_temp.completion_result").fetchall() == [("retained",)]
        assert source.datasets is previous_tables
        assert source.routines is previous_routines
    finally:
        source.con.execute("ROLLBACK")


def test_postgres_declared_enums_complete_without_reading_rows(postgres_routine_source):
    from d8r.completion.engine import complete
    source = postgres_routine_source
    source.con.execute("""
        CREATE TYPE pg_temp.completion_state AS ENUM ('ready', 'can''t');
        CREATE DOMAIN pg_temp.completion_state_domain AS pg_temp.completion_state;
        CREATE TEMP TABLE completion_enums (
            state pg_temp.completion_state,
            domain_state pg_temp.completion_state_domain)
    """)
    refresh_source_schema(source)
    schema = SchemaContext(tables=tuple(session_module.tables_of(source)), routines=source.routines)
    table = schema.table_by_name("completion_enums")
    assert [column.values for column in table.columns] == [("ready", "can't"), ("ready", "can't")]
    for name in ("state", "domain_state"):
        document = f"SELECT * FROM completion_enums WHERE {name} = 'ca"
        result = complete(document, len(document), schema, dialect="postgres")
        assert [item.insert for item in result.items] == ["'can''t'"]
    assert source.datasets[table.name]["rows"] is None
