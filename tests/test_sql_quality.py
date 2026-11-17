"""Typed macro safety and independently authored native quality SQL contracts."""
from __future__ import annotations

import asyncio
import json
import sqlite3

import duckdb
import pytest

from d8r.ai.context import AIContext
from d8r.quality import quality_functions
from d8r.sql import (FnDef, SqlError, expand_functions, format_parameters, function_query,
                     parse_parameters, validate_function)
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


@pytest.fixture
def database(tmp_path):
    connection = duckdb.connect(str(tmp_path / "quality.duckdb"))
    connection.execute("CREATE TABLE data(id INTEGER, label VARCHAR, updated TIMESTAMP)")
    connection.execute("INSERT INTO data VALUES (1, 'alpha', CURRENT_TIMESTAMP), (1, ' ', CURRENT_TIMESTAMP), "
                       "(2, NULL, NULL), (NULL, 'beta', CURRENT_TIMESTAMP), (9, 'beta', CURRENT_TIMESTAMP)")
    connection.execute("CREATE TABLE parent AS SELECT 1 AS id UNION ALL SELECT 2")
    try:
        yield connection
    finally:
        connection.close()


def test_parameter_declarations_keep_legacy_names_and_types_separate():
    assert parse_parameters("x, y:expression") == (("x", "y"), ())
    names, kinds = parse_parameters("source:relation, field:identifier, minimum")
    fn = FnDef("project", names, "SELECT @field FROM @source WHERE @field > @minimum", param_types=kinds)
    assert fn.params == ("source", "field", "minimum")
    assert kinds == ("relation", "identifier", "expression")
    assert format_parameters(fn) == "source:relation, field:identifier, minimum"
    assert FnDef("legacy", ["x"], "SELECT @x").param_types == ()
    validate_function(fn)


@pytest.mark.parametrize("text", ["x:int", "x:relation:identifier", "x,", "@x", "x,x", "x=1", ":identifier"])
def test_bad_parameter_declarations_are_rejected(text):
    with pytest.raises(SqlError):
        parse_parameters(text)


@pytest.mark.parametrize("kinds", [("relation", "identifier"), ("sql",)])
def test_parameter_type_arity_and_names_are_validated(kinds):
    with pytest.raises(SqlError):
        validate_function(FnDef("bad", ("x",), "SELECT @x", param_types=kinds))


@pytest.mark.parametrize("dialect", ["postgres", "sqlite", "duckdb"])
@pytest.mark.parametrize("argument", ["data; DROP TABLE data", "data WHERE TRUE", "data JOIN parent ON TRUE",
                                      "(SELECT * FROM data)", "data()", "'data'", "data -- comment", "data/*x*/",
                                      '"unterminated', '"bad\x00name"', "a.b.c.d"])
def test_relation_arguments_cannot_smuggle_sql(dialect, argument):
    fn = FnDef("rows", ("source",), "SELECT * FROM @source", param_types=("relation",))
    with pytest.raises(SqlError):
        function_query(fn, argument, dialect=dialect)


@pytest.mark.parametrize("argument", ["a.b", "COUNT(*)", "id + 1", "id DESC", "*", "id AS other", "id, label", "'id'"])
def test_identifier_arguments_are_one_name_not_expressions(argument):
    fn = FnDef("column_value", ("field",), "SELECT @field FROM data", param_types=("identifier",))
    with pytest.raises(SqlError):
        function_query(fn, argument, dialect="duckdb")


def test_quoted_names_split_schema_and_expression_precedence_execute(database):
    database.execute('CREATE SCHEMA "odd schema"')
    database.execute('CREATE TABLE "odd schema"."semi;table" ("a""b" INTEGER)')
    database.execute('INSERT INTO "odd schema"."semi;table" VALUES (4)')
    fn = FnDef("project", ("schema", "table", "field", "factor"),
               "SELECT @field * @factor AS value FROM @schema.@table",
               param_types=("identifier", "identifier", "identifier", "expression"))
    sql = function_query(fn, '\"odd schema\", "semi;table", "a""b", 1 + 2', dialect="duckdb")
    assert database.execute(sql).fetchone() == (12,)
    assert '"odd schema"."semi;table"' in sql
    assert '"a""b"' in sql
    assert database.execute("SELECT COUNT(*) FROM data").fetchone() == (5,)


def test_typed_arguments_forward_through_nested_functions(database):
    leaf = FnDef("leaf", ("source", "field", "offset"), "SELECT @field + @offset AS value FROM @source",
                 param_types=("relation", "identifier", "expression"))
    outer = FnDef("outer_rows", leaf.params, "SELECT * FROM leaf(@source, @field, @offset * 2)",
                  param_types=leaf.param_types)
    validate_function(outer, [leaf], dialect="duckdb")
    sql = function_query(outer, "main.parent, id, 1 + 2", [leaf], dialect="duckdb")
    assert database.execute(sql + " ORDER BY value").fetchall() == [(7,), (8,)]
    recursive = FnDef("cycle", ("source",), "SELECT * FROM cycle(@source)", param_types=("relation",))
    with pytest.raises(SqlError, match="cycle"):
        validate_function(recursive, dialect="duckdb")


def test_legacy_expression_slots_and_query_only_restrictions_remain():
    for body in ("SELECT * FROM @source", "SELECT @source.id", "DELETE FROM @source"):
        with pytest.raises(SqlError):
            validate_function(FnDef("bad", ("source",), body))
    with pytest.raises(SqlError):
        validate_function(FnDef("bad", ("source",), "DELETE FROM @source", param_types=("relation",)))


@pytest.mark.parametrize("dialect", ["postgres", "sqlite", "duckdb"])
def test_every_native_quality_definition_validates_offline(dialect):
    functions = quality_functions(dialect)
    assert len({fn.name.casefold() for fn in functions}) == len(functions)
    for fn in functions:
        validate_function(fn, functions, dialect=dialect)
    assert ("dq_regex_count" in {fn.name for fn in functions}) == (dialect != "sqlite")


QUALITY_CASES = [
    ("row_count", "data", 5), ("row_count_delta", "data, parent", 3),
    ("missing_count", "data, id", 1), ("missing_percent", "data, id", 20),
    ("present_count", "data, id", 4), ("present_percent", "data, id", 80),
    ("valid_count", "data, id >= 2", 2), ("invalid_count", "data, id >= 2", 3),
    ("valid_percent", "data, id >= 2", 40), ("invalid_percent", "data, id >= 2", 60),
    ("distinct_count", "data, id", 3), ("duplicate_count", "data, id", 1),
    ("duplicate_percent", "data, id", 25), ("duplicate_groups", "data, id", 1),
    ("unique_count", "data, id", 2), ("unique_percent", "data, id", 50),
    ("min", "data, id", 1), ("max", "data, id", 9), ("sum", "data, id", 13),
    ("avg", "data, id", 3.25), ("out_of_range_count", "data, id, 1, 2", 1),
    ("in_range_count", "data, id, 1, 2", 3), ("blank_count", "data, label", 1),
    ("length_min", "data, label", 1), ("length_max", "data, label", 5),
    ("length_avg", "data, label", 3.5), ("like_count", "data, label, 'b%'", 2),
    ("fk_missing_count", "data, id, parent, id", 1),
    ("schema_column_count", "'main', 'data'", 3), ("column_exists", "'main', 'data', 'id'", 1),
    ("column_exists", "'main', 'data', 'absent'", 0),
    ("column_type_matches", "'main', 'data', 'id', 'INTEGER'", 1),
]


@pytest.mark.parametrize("name, arguments, expected", QUALITY_CASES)
def test_quality_metrics_execute_in_native_duckdb(database, name, arguments, expected):
    registry = {fn.name: fn for fn in quality_functions("duckdb")}
    sql = function_query(registry["dq_" + name], arguments, registry, dialect="duckdb")
    assert database.execute(sql).fetchone() == (expected,)


def test_statistical_regex_freshness_and_schema_native_results(database):
    registry = {fn.name: fn for fn in quality_functions("duckdb")}
    for operation in ("stddev_pop", "stddev_samp", "var_pop", "var_samp"):
        sql = function_query(registry["dq_" + operation], "data, id", registry, dialect="duckdb")
        assert database.execute(sql).fetchone() == database.execute(f"SELECT {operation}(id) FROM data").fetchone()
    sql = function_query(registry["dq_regex_count"], "data, label, '^b'", registry, dialect="duckdb")
    assert database.execute(sql).fetchone() == (2,)
    sql = function_query(registry["dq_freshness_seconds"], "data, updated", registry, dialect="duckdb")
    assert 0 <= database.execute(sql).fetchone()[0] < 60
    sql = function_query(registry["dq_schema_columns"], "'main', 'data'", registry, dialect="duckdb")
    assert database.execute(sql).fetchall() == [("id", "INTEGER", 1), ("label", "VARCHAR", 2), ("updated", "TIMESTAMP", 3)]


def test_empty_and_all_null_quality_denominators(database):
    database.execute("CREATE TABLE empty AS SELECT * FROM data WHERE FALSE")
    registry = {fn.name: fn for fn in quality_functions("duckdb")}
    for name in ("missing_percent", "duplicate_percent", "unique_percent", "avg", "min"):
        assert database.execute(function_query(registry["dq_" + name], "empty, id", registry, dialect="duckdb")).fetchone() == (None,)
    for name in ("row_count", "missing_count", "duplicate_count", "unique_count"):
        arguments = "empty" if name == "row_count" else "empty, id"
        assert database.execute(function_query(registry["dq_" + name], arguments, registry, dialect="duckdb")).fetchone() == (0,)
    database.execute("INSERT INTO empty VALUES (NULL, NULL, NULL)")
    assert database.execute(function_query(registry["dq_duplicate_percent"], "empty, id", registry, dialect="duckdb")).fetchone() == (None,)


def test_all_sqlite_templates_execute_without_extensions(tmp_path):
    connection = sqlite3.connect(tmp_path / "quality.sqlite")
    try:
        connection.executescript("CREATE TABLE data(id INTEGER, label TEXT, updated TEXT);"
                                 "INSERT INTO data VALUES (1,'a',datetime('now')),(NULL,NULL,NULL);"
                                 "CREATE TABLE parent(id INTEGER); INSERT INTO parent VALUES (1);")
        values = {"source": "data", "field": "id", "reference": "parent", "reference_field": "id",
                  "condition": "id > 0", "minimum": "0", "maximum": "2", "pattern": "'1%'",
                  "schema_name": "'main'", "table_name": "'data'", "column_name": "'id'", "expected_type": "'INTEGER'"}
        functions = quality_functions("sqlite")
        for fn in functions:
            arguments = dict(values, field="updated") if fn.name == "dq_freshness_seconds" else values
            sql = function_query(fn, ", ".join(arguments[param] for param in fn.params), functions, dialect="sqlite")
            result = connection.execute(sql).fetchone()
            assert result is not None, fn.name
            if fn.name in {"dq_column_exists", "dq_column_type_matches", "dq_fk_missing_count"}:
                assert result == ((0,) if fn.name == "dq_fk_missing_count" else (1,))
    finally:
        connection.close()


@pytest.fixture
def postgres_quality(postgres_source):
    connection = postgres_source.con
    connection.execute("CREATE TEMP TABLE data(id integer, label text, updated timestamp)")
    connection.execute("INSERT INTO data VALUES (1, 'alpha', CURRENT_TIMESTAMP), (1, ' ', CURRENT_TIMESTAMP), "
                       "(2, NULL, NULL), (NULL, 'beta', CURRENT_TIMESTAMP), (9, 'beta', CURRENT_TIMESTAMP)")
    connection.execute("CREATE TEMP TABLE parent AS SELECT 1 AS id UNION ALL SELECT 2")
    namespace = connection.execute("SELECT pg_my_temp_schema()::regnamespace::text").fetchone()[0]
    session = Session({postgres_source.id: postgres_source})
    session.install_quality_functions()
    connection.execute("BEGIN")
    try:
        yield session, namespace
    finally:
        connection.execute("ROLLBACK")


@pytest.mark.parametrize("name, arguments, expected", QUALITY_CASES)
def test_quality_metrics_execute_in_native_postgres(postgres_quality, name, arguments, expected):
    session, namespace = postgres_quality
    arguments = arguments.replace("'main'", "'" + namespace + "'")
    outcome = session.fn_preview("dq_" + name, arguments)
    assert outcome.ok, outcome.error
    assert outcome.rows == [[expected]]
    assert not session.history


def test_postgres_statistical_regex_freshness_and_schema_results(postgres_quality):
    session, namespace = postgres_quality
    for operation in ("stddev_pop", "stddev_samp", "var_pop", "var_samp"):
        expected = session.source.con.execute(f"SELECT {operation}(id) FROM data").fetchone()[0]
        outcome = session.fn_preview("dq_" + operation, "data, id")
        assert outcome.ok, outcome.error
        assert outcome.rows == [[expected]]
    regex = session.fn_preview("dq_regex_count", "data, label, '^b'")
    assert regex.ok, regex.error
    assert regex.rows == [[2]]
    expected = session.source.con.execute("SELECT EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - MAX(updated))) FROM data").fetchone()[0]
    freshness = session.fn_preview("dq_freshness_seconds", "data, updated")
    assert freshness.ok, freshness.error
    assert freshness.rows == [[expected]]
    columns = session.fn_preview("dq_schema_columns", "'" + namespace + "', 'data'")
    assert columns.ok, columns.error
    assert columns.rows == [["id", "integer", 1], ["label", "text", 2],
                            ["updated", "timestamp without time zone", 3]]


def test_quality_install_is_opt_in_atomic_source_local_and_persistent(parquet_source, d1_source, tmp_path, monkeypatch):
    import d8r.tui.session as session_module

    def forbidden(*args, **kwargs):
        pytest.fail("Installing definitions must not execute SQL")

    monkeypatch.setattr(session_module, "execute_sql", forbidden)
    path = tmp_path / "quality-state"
    session = Session({parquet_source.id: parquet_source, d1_source.id: d1_source}, data_dir=path)
    assert session.functions_for(parquet_source.id) == {}
    assert session.functions_for(d1_source.id) == {}
    with session.target_source(parquet_source.id):
        count = session.install_quality_functions()
        assert count == len(quality_functions("duckdb"))
        before = dict(session.fns)
        assert session.install_quality_functions() == 0
        assert session.fns == before
    assert not session.functions_for(d1_source.id)
    restored = Session({parquet_source.id: parquet_source, d1_source.id: d1_source}, data_dir=path)
    assert restored.functions_for(parquet_source.id) == before
    assert not session.history_for(parquet_source.id)
    with session.target_source(d1_source.id):
        session.save_fn("DQ_ROW_COUNT", "", "SELECT 123", "user definition")
        before_collision = dict(session.fns)
        with pytest.raises(ValueError):
            session.install_quality_functions()
        assert session.fns == before_collision


def test_typed_ai_proposals_and_baselines_survive_storage(parquet_source, tmp_path):
    session = Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "typed-ai")
    original = session.save_fn("project", "source:relation, field:identifier", "SELECT @field FROM @source", "")
    context = AIContext(session, parquet_source.id, "SELECT 1", save_guard=lambda: None)
    metadata = {"kind": "function", "original_name": "project", "name": "project", "description": "typed",
                "parameters": ["source:relation", "field:identifier"], "arguments": "data, id"}
    answer = "```sql\nSELECT @field FROM @source\n```\n```json\n" + json.dumps(metadata) + "\n```"
    proposal = context.read_proposal(answer)
    assert proposal.function.param_types == ("relation", "identifier")
    assert context.validate_replacement(proposal) is None
    tool_args = {key: metadata[key] for key in ("original_name", "name", "description", "parameters")}
    tool_args["body"] = proposal.body
    result = json.loads(asyncio.run(context.call_tool("save_function", tool_args)))
    assert result["saved"] is True
    assert session.fns["project"].param_types == original.param_types
    stale = AIContext(session, parquet_source.id, "SELECT 1")
    session.save_fn("project", "source:relation, field", "SELECT @field FROM @source", "changed types")
    assert "changed" in stale.replacement_guard(proposal)
