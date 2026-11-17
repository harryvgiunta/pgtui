"""Source-isolated offline metadata and memory-only value-cache lifecycle."""
from __future__ import annotations

from dataclasses import replace
from uuid import uuid4

import pytest

from d8r.completion.engine import complete
from d8r.engine import execute_sql
from d8r.schema_cache import SchemaCacheStore
from d8r.sql import relation_name
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


def remember_workspace(session):
    session.save_workspace(document="SELECT 1", source=session.source_key(), dialect=session.dialect)


@pytest.mark.parametrize("fixture", ["parquet_source", "d1_source"])
def test_source_metadata_and_functions_restore_without_runtime_values(request, fixture, tmp_path, monkeypatch):
    source = request.getfixturevalue(fixture)
    home = tmp_path / "state"
    original = Session({source.id: source}, data_dir=home)
    table = original.schema.tables[0]
    original.save_fn("known_rows", "", f"SELECT id, label FROM {table.name}", "Known table")
    values = original.distinct_values(source, table.name, "label")
    assert "alpha" in values[0]
    remember_workspace(original)
    key = original.source_key()
    cache_path = home / "schema_cache.json"
    assert "alpha" not in cache_path.read_text(encoding="utf-8")
    saved = cache_path.read_bytes()

    def forbidden(*args, **kwargs):
        raise AssertionError("Restoring or consuming cached metadata must not perform database or file writes")

    import d8r.tui.session as session_module
    monkeypatch.setattr(session_module, "refresh_source_schema", forbidden)
    monkeypatch.setattr(session_module, "column_values", forbidden)
    restored = Session(data_dir=home)
    assert restored.source_key() == key
    assert not restored.source_connected()
    assert restored.schema.table_by_name(table.name).columns == table.columns
    assert set(restored.fns) == {"known_rows"}
    assert restored.schema_cache_info()
    assert restored.cached_values(restored.source, table.name, "label") is None
    monkeypatch.setattr(restored._schema_cache, "save_snapshot", forbidden)
    with restored.target_source():
        offers = complete(f"SELECT d.la FROM {table.name} d", len("SELECT d.la"), restored.schema, dialect=restored.dialect)
    assert any(item.insert == "label" for item in offers.items)
    assert restored.compile("SELECT * FROM known_rows()").ok
    assert cache_path.read_bytes() == saved


@pytest.mark.parametrize("fixture", ["parquet_source", "d1_source"])
def test_cached_values_survive_offline_source_copy_but_not_reconnect(request, fixture, monkeypatch):
    source = request.getfixturevalue(fixture)
    session = Session({source.id: source})
    table = session.schema.tables[0].name
    cached = session.distinct_values(source, table, "label")
    offline = replace(source, con=None, datasets={}, routines=())
    session.sources[offline.id] = offline
    session.refresh_schema()
    assert session.cached_values(offline, table, "label") == cached

    def forbidden(*args, **kwargs):
        raise AssertionError("Existing offline values must not query the database")

    import d8r.tui.session as session_module
    monkeypatch.setattr(session_module, "column_values", forbidden)
    assert session.distinct_values(offline, table, "label") == cached
    sql = f"SELECT d.label FROM {table} d WHERE d.label = 'a"
    offers = complete(sql, len(sql), session.schema, dialect=session.dialect,
                      values=lambda dataset, column: session.cached_values(offline, dataset, column))
    assert any(item.insert == "'alpha'" for item in offers.items)
    other_identity = replace(offline, connection_key="another-database-identity")
    assert session.cached_values(other_identity, table, "label") is None
    session.register(source)
    assert session.source_connected()
    assert session.cached_values(source, table, "label") is None
    assert not session.schema_cache_info()


def test_explicit_clear_revokes_values_after_offline_handoff(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    session.distinct_values(parquet_source, "data", "label")
    offline = replace(parquet_source, con=None)
    session.sources[offline.id] = offline
    session.clear_value_cache(offline.id)
    assert session.cached_values(offline, "data", "label") is None


def test_failed_metadata_save_keeps_successful_ddl_and_latest_in_memory_catalog(d1_source, tmp_path, monkeypatch):
    session = Session({d1_source.id: d1_source}, data_dir=tmp_path / "state")
    path = session._schema_cache.path
    previous = path.read_bytes()

    def unavailable(*args):
        raise ValueError("Cache is read-only")

    monkeypatch.setattr(session._schema_cache, "save_snapshot", unavailable)
    outcome = session.run("CREATE TABLE new_catalog_entry(id INTEGER)")
    assert outcome.ok and not outcome.error
    assert session.metadata_cache_warning in outcome.status
    assert path.read_bytes() == previous
    offline = replace(d1_source, con=None, datasets={}, routines=())
    session.sources[offline.id] = offline
    session.refresh_schema()
    assert session.schema.table_by_name("new_catalog_entry") is not None
    assert session.history[0].doc == "CREATE TABLE new_catalog_entry(id INTEGER)"


def test_live_empty_catalog_replaces_old_metadata_and_late_source_cannot_republish(d1_source, tmp_path):
    session = Session({d1_source.id: d1_source}, data_dir=tmp_path / "state")
    key = session.source_key()
    outcome = session.run("DROP TABLE records", record=False)
    assert outcome.ok
    assert SchemaCacheStore(tmp_path / "state").snapshots[key].tables == ()
    stale = replace(d1_source, datasets={relation_name("stale"): {"columns": (), "rows": None}})
    before = session._schema_cache.path.read_bytes()
    session.cache_source_schema(stale)
    assert session._schema_cache.path.read_bytes() == before
    offline = replace(d1_source, con=None, datasets={})
    session.sources[offline.id] = offline
    session.refresh_schema()
    assert session.schema.tables == ()
    assert session.schema_cache_info()


def test_snapshot_handoff_performs_no_cache_writes_or_execution_lock_wait(parquet_source, monkeypatch):
    session = Session({parquet_source.id: parquet_source})

    class BusyLock:
        def __enter__(self):
            raise AssertionError("Completion may not wait for execution")

        def __exit__(self, *args):
            return False

    def forbidden(*args, **kwargs):
        raise AssertionError("Completion may not write its metadata cache")

    with monkeypatch.context() as guarded:
        guarded.setattr(parquet_source, "_lock", BusyLock())
        guarded.setattr(session._schema_cache, "save_snapshot", forbidden)
        offline = replace(parquet_source, con=None)
        session.sources[offline.id] = offline
        with session.target_source():
            first = session.schema
        with session.target_source():
            assert session.schema is first
        offline.datasets = {}
        with session.target_source():
            assert session.schema.table_by_name("data") is not None


def test_postgres_transactions_temp_objects_and_new_session_cache(postgres_source, tmp_path):
    namespace = "cache_" + uuid4().hex
    execute_sql(postgres_source, f'CREATE SCHEMA "{namespace}"')
    table = relation_name(namespace, "dataset_symbol")
    try:
        session = Session({postgres_source.id: postgres_source}, data_dir=tmp_path / "state")
        key = session.source_key()
        remember_workspace(session)
        before = session._schema_cache.path.read_bytes()
        assert session.run("BEGIN", record=False).ok
        assert session.run(f"CREATE TABLE {table}(id INTEGER)", record=False).ok
        assert session._schema_cache.path.read_bytes() == before
        assert session.run("COMMIT", record=False).ok
        assert session.run("CREATE TEMP TABLE only_this_session(id INTEGER)", record=False).ok
        assert session.run("CREATE FUNCTION pg_temp.only_this_session() RETURNS INTEGER LANGUAGE SQL AS 'SELECT 1'", record=False).ok
        restored = Session(data_dir=tmp_path / "state")
        assert restored.source_key() == key
        assert restored.schema.table_by_name(table) is not None
        assert restored.schema.table_by_name("only_this_session") is None
        assert not any("only_this_session" in routine.name for routine in restored.schema.routines)
        candidates = complete("data", 4, restored.schema, dialect="postgres").items
        assert any(item.table == table and item.insert.startswith("SELECT  FROM ") for item in candidates)
        assert not restored.source_connected()
        assert session.run(f"DROP TABLE {table}", record=False).ok
        final = Session(data_dir=tmp_path / "state")
        assert final.schema.table_by_name(table) is None
    finally:
        execute_sql(postgres_source, f'DROP SCHEMA "{namespace}" CASCADE')
