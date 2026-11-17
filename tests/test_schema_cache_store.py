"""Durable catalog storage, isolated from runtime rows, connections and value pools."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from d8r import schema_cache, storage
from d8r.engine import execute_sql, refresh_source_schema
from d8r.schema_cache import SchemaCacheStore, SchemaSnapshot, capture_schema
from d8r.sql import ColumnDef, ForeignKey, RoutineDef, TableDef, relation_name


STAMP = "2026-11-14T12:00:00+00:00"


def catalog_snapshot():
    return SchemaSnapshot(
        "postgres", "18.7", "public",
        (TableDef('public."Orders"', "Order documentation", (
            ColumnDef("tenant", "integer", "Tenant documentation"),
            ColumnDef("id", "bigint"),
            ColumnDef("state", "public.order_state_domain", "Catalog enum", ("new", "done", "")),
        ), (ForeignKey(("tenant", "id"), "public.customers", ("tenant_id", "id")),)),),
        (RoutineDef('public."Order output"', ('"tenant" integer', 'VARIADIC text[]'),
                    "TABLE(id bigint, state text)", "Native routine documentation", "table", True,
                    ("integer", "text[]"), (ColumnDef("id", "bigint"), ColumnDef("state", "text"))),
         RoutineDef("pg_catalog.count", ("*",), "bigint", "Aggregate docs", "aggregate")),
        STAMP,
    )


def test_snapshot_roundtrip_preserves_native_metadata_and_engine_shape(tmp_path):
    snapshot = catalog_snapshot()
    store = SchemaCacheStore(tmp_path)
    store.save_snapshot("postgres:catalog-target", snapshot)
    before = store.path.read_bytes()
    restored = SchemaCacheStore(tmp_path)
    assert restored.error == ""
    assert restored.snapshots == {"postgres:catalog-target": snapshot}
    assert restored.path.read_bytes() == before
    datasets = restored.snapshots["postgres:catalog-target"].datasets()
    table = snapshot.tables[0]
    assert datasets == {table.name: {"columns": table.columns, "doc": table.doc,
                                    "foreign_keys": table.foreign_keys, "rows": None, "temp": False}}
    datasets[table.name]["rows"] = [["runtime only"]]
    assert snapshot.datasets()[table.name]["rows"] is None
    assert restored.snapshots["postgres:catalog-target"] is restored.snapshots["postgres:catalog-target"]
    with pytest.raises(FrozenInstanceError):
        snapshot.dialect = "sqlite"


def test_snapshot_owns_immutable_collections():
    tables, routines = list(catalog_snapshot().tables), list(catalog_snapshot().routines)
    snapshot = SchemaSnapshot("postgres", "18.7", "public", tables, routines, STAMP)
    tables.clear()
    routines.clear()
    assert snapshot == catalog_snapshot()


def test_missing_cache_load_never_creates_directory(tmp_path):
    directory = tmp_path / "missing"
    store = SchemaCacheStore(directory)
    assert store.snapshots == {}
    assert store.error == ""
    assert not directory.exists()


def test_parquet_capture_omits_ephemeral_tables_macros_rows_and_values(parquet_source, tmp_path):
    source = parquet_source
    source.con.execute("CREATE TABLE ephemeral_table AS SELECT 42 AS secret")
    source.con.execute("CREATE TEMP TABLE ephemeral_temp AS SELECT 43 AS secret")
    source.con.execute("CREATE MACRO ephemeral_macro(value) AS value + 1")
    refresh_source_schema(source)
    assert len(source.datasets) > 1 and source.routines
    columns = source.datasets["data"]["columns"]
    source.datasets["data"]["columns"] = tuple(
        ColumnDef(column.name, column.type, column.doc, ("private-distinct-value",)) for column in columns
    )
    source.datasets["data"]["rows"] = [["private-sampled-row"]]
    source.postgres["password"] = "private-password"
    source.d1["api_token"] = "private-api-token"
    snapshot = capture_schema(source)
    assert snapshot is not None
    assert [table.name for table in snapshot.tables] == ["data"]
    assert snapshot.tables[0].columns == columns
    assert snapshot.routines == ()
    assert source.datasets["data"]["rows"] == [["private-sampled-row"]]
    store = SchemaCacheStore(tmp_path / "cache")
    store.save_snapshot(source.connection_key, replace(snapshot, captured_at=STAMP))
    text = store.path.read_text(encoding="utf-8")
    for forbidden in ("ephemeral", "private-", '"rows"', '"temp"', '"password"', '"api_token"', '"fns"'):
        assert forbidden not in text
    restored = SchemaCacheStore(tmp_path / "cache").snapshots[source.connection_key]
    assert restored.datasets()["data"]["rows"] is None
    assert restored.captured_at == STAMP


def test_capture_skips_uncommitted_catalog_and_disconnected_source(parquet_source):
    source = parquet_source
    execute_sql(source, "BEGIN")
    try:
        execute_sql(source, "CREATE TABLE uncommitted AS SELECT 1 AS id")
        refresh_source_schema(source)
        assert capture_schema(source) is None
    finally:
        execute_sql(source, "ROLLBACK")
    refresh_source_schema(source)
    assert capture_schema(source) is not None
    connection = source.con
    source.con = None
    try:
        assert capture_schema(source) is None
    finally:
        source.con = connection


def test_d1_capture_is_catalog_only_without_io_or_execution_lock(d1_source, d1_harness, monkeypatch):
    source = d1_source
    request_count = len(d1_harness.requests)
    dataset = relation_name("records")
    source.datasets[dataset]["rows"] = [["private-row"]]
    column = source.datasets[dataset]["columns"][0]
    source.datasets[dataset]["columns"] = (ColumnDef(column.name, column.type, values=("private-value",)),)

    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("capture acquired execution lock")

        def __exit__(self, *args):
            return False

    def no_file_io(*args, **kwargs):
        raise AssertionError("capture touched the filesystem")

    with monkeypatch.context() as guarded:
        guarded.setattr(source, "_lock", ForbiddenLock())
        guarded.setattr(Path, "open", no_file_io)
        snapshot = capture_schema(source)
    assert len(d1_harness.requests) == request_count
    assert snapshot is not None and snapshot.dialect == "sqlite"
    assert snapshot.routines == ()
    assert snapshot.tables[0].columns == (ColumnDef(column.name, column.type),)
    assert all(entry["rows"] is None for entry in snapshot.datasets().values())
    assert snapshot.captured_at.endswith("+00:00")


def test_postgres_catalog_filtering_preserves_labels_signatures_and_fk_pairs(d1_source):
    # Exercise pure catalog capture using the real isolated fixture connection;
    # PostgreSQL catalog records need no PostgreSQL server or driver operations.
    source = d1_source
    snapshot = catalog_snapshot()
    source.dialect = "postgres"
    source.server_version = snapshot.server_version
    source.default_schema = snapshot.default_schema
    source.datasets = snapshot.datasets()
    source.routines = (*snapshot.routines, RoutineDef('"pg_temp_12"."temporary function"'))
    source.datasets['"pg_temp_12"."temporary table"'] = {
        "columns": (ColumnDef("id", "integer"),), "temp": True,
    }
    source.datasets["public.temporary_flag"] = {"columns": (), "temp": True}
    source.datasets["pg_temp_13.namespace_only"] = {"columns": (), "temp": False}
    entry = source.datasets[snapshot.tables[0].name]
    entry["foreign_keys"] += (
        ForeignKey(("id",), 'pg_temp_12."temporary table"', ("id",)),
        ForeignKey(("id",), "public.temporary_flag", ("id",)),
        ForeignKey(("id",), "pg_temp_99.not_in_catalog", ("id",)),
    )
    captured = capture_schema(source)
    assert captured is not None
    assert replace(captured, captured_at=STAMP) == snapshot
    assert len(source.datasets) == 4
    assert len(source.routines) == 3


def test_capture_retains_only_string_catalog_labels(d1_source):
    d1_source.dialect = "postgres"
    d1_source.datasets[relation_name("records")]["columns"] = (
        ColumnDef("enum_column", "public.enum_type", values=("a", "b")),
        ColumnDef("non_catalog_values", "integer", values=(1, 2)),
    )
    snapshot = capture_schema(d1_source)
    assert snapshot is not None
    assert snapshot.tables[0].columns[0].values == ("a", "b")
    assert snapshot.tables[0].columns[1].values is None


@pytest.mark.parametrize("mutation", [
    lambda doc: doc.update(version=True),
    lambda doc: doc.update(version=99),
    lambda doc: doc.update(password="not-catalog"),
    lambda doc: doc.update(snapshots=[]),
    lambda doc: doc["snapshots"]["target"].update(dialect="unknown"),
    lambda doc: doc["snapshots"]["target"].update(captured_at="not-a-date"),
    lambda doc: doc["snapshots"]["target"].update(captured_at="2026-11-14T12:00:00"),
    lambda doc: doc["snapshots"]["target"].update(captured_at="2026-11-14T12:00:00+01:00"),
    lambda doc: doc["snapshots"]["target"]["tables"][0].update(rows=[["private"]]),
    lambda doc: doc["snapshots"]["target"]["tables"][0].update(name="pg_temp_1.bad"),
    lambda doc: doc["snapshots"]["target"]["tables"][0]["columns"][2].update(values=[1]),
    lambda doc: doc["snapshots"]["target"]["tables"][0]["foreign_keys"][0].update(target_columns=[]),
    lambda doc: doc["snapshots"]["target"]["routines"][0].update(variadic="yes"),
    lambda doc: doc["snapshots"]["target"]["routines"][0].update(kind="template"),
    lambda doc: doc["snapshots"]["target"]["routines"][0].update(body="SELECT private"),
    lambda doc: doc["snapshots"]["target"]["routines"][0].update(name="pg_temp_2.bad"),
])
def test_invalid_documents_block_load_and_save_without_replacing_file(tmp_path, mutation):
    store = SchemaCacheStore(tmp_path)
    store.save_snapshot("target", catalog_snapshot())
    document = json.loads(store.path.read_text(encoding="utf-8"))
    mutation(document)
    store.path.write_text(json.dumps(document), encoding="utf-8")
    original = store.path.read_bytes()
    invalid = SchemaCacheStore(tmp_path)
    assert invalid.error
    assert invalid.snapshots == {}
    with pytest.raises(ValueError, match="Local saves are disabled"):
        invalid.save_snapshot("replacement", catalog_snapshot())
    assert store.path.read_bytes() == original


@pytest.mark.parametrize("raw", [b"{broken", b"\xff", b'{"version":1,"version":1,"snapshots":{}}',
                                 b'{"version":1,"snapshots":NaN}'])
def test_corrupt_json_is_never_repaired_or_overwritten(tmp_path, raw):
    path = tmp_path / "schema_cache.json"
    path.write_bytes(raw)
    store = SchemaCacheStore(tmp_path)
    assert store.error and store.snapshots == {}
    with pytest.raises(ValueError):
        store.save_snapshot("target", catalog_snapshot())
    assert path.read_bytes() == raw


def test_stale_writer_preserves_latest_file_and_its_own_loaded_snapshot(tmp_path):
    first = SchemaCacheStore(tmp_path)
    first.save_snapshot("target", catalog_snapshot())
    stale = SchemaCacheStore(tmp_path)
    original_snapshot = stale.snapshots["target"]
    latest = replace(catalog_snapshot(), server_version="19.0")
    first.save_snapshot("target", latest)
    original_bytes = first.path.read_bytes()
    with pytest.raises(ValueError, match="changed outside"):
        stale.save_snapshot("target", replace(catalog_snapshot(), server_version="20.0"))
    assert stale.snapshots["target"] is original_snapshot
    assert first.path.read_bytes() == original_bytes
    assert SchemaCacheStore(tmp_path).snapshots["target"] == latest


def test_failed_replace_preserves_disk_document_and_decoded_snapshot(tmp_path, monkeypatch):
    store = SchemaCacheStore(tmp_path)
    snapshot = catalog_snapshot()
    store.save_snapshot("target", snapshot)
    original, document = store.path.read_bytes(), store.document

    def fail_replace(*args):
        raise OSError("private failure details")

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(ValueError, match="Nothing was changed") as caught:
        store.save_snapshot("target", replace(snapshot, server_version="19.0"))
    assert "private failure" not in str(caught.value)
    assert store.path.read_bytes() == original
    assert store.document is document
    assert store.snapshots["target"] is snapshot
    assert not store.path.with_suffix(".lock").exists()
    assert not list(tmp_path.glob(".schema_cache-*.tmp"))


def test_saved_sources_are_isolated_and_decoded_once(tmp_path, monkeypatch):
    store = SchemaCacheStore(tmp_path)
    first, second = catalog_snapshot(), replace(catalog_snapshot(), default_schema="other")
    store.save_snapshot("first", first)
    store.save_snapshot("second", second)
    loaded = SchemaCacheStore(tmp_path)

    def no_decode(*args):
        raise AssertionError("an already loaded snapshot was decoded again")

    monkeypatch.setattr(schema_cache, "_decode", no_decode)
    assert loaded.snapshots == {"first": first, "second": second}
    loaded.save_snapshot("first", replace(first, server_version="19.0"))
    assert loaded.snapshots["second"] == second
    assert loaded.snapshots["first"].server_version == "19.0"


def test_closed_connection_does_not_replace_successful_catalog(d1_source, monkeypatch):
    assert capture_schema(d1_source) is not None
    monkeypatch.setattr(d1_source.con, "closed", True, raising=False)
    assert capture_schema(d1_source) is None


def test_invalid_snapshot_save_preserves_previous_disk_and_memory(tmp_path):
    store = SchemaCacheStore(tmp_path)
    snapshot = catalog_snapshot()
    store.save_snapshot("target", snapshot)
    original = store.path.read_bytes()
    invalid = replace(snapshot, tables=(TableDef("public.invalid", columns=(
        ColumnDef("id", "integer", values=(42,)),
    )),))
    with pytest.raises(ValueError, match="enum labels"):
        store.save_snapshot("target", invalid)
    assert store.path.read_bytes() == original
    assert store.snapshots["target"] is snapshot
    assert SchemaCacheStore(tmp_path).snapshots["target"] == snapshot
