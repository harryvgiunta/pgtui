"""Native SQL session behavior over generated local Parquet and restored targets."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from threading import Event

import pytest

from d8r.completion.engine import complete
from d8r.ai.client import AIConfig
from d8r.engine import EngineError, add_parquet_source, close_source, execute_sql
from d8r.sql import SQLDocument
from d8r.tui.connection import build_connection, build_saved_source, saved_source_label
from d8r.tui.session import RunOutcome, Session


def session_for(source):
    return Session({source.id: source})


def test_empty_startup_is_blank_disconnected_postgres():
    session = Session()
    assert session.source.kind == "disconnected"
    assert session.dialect == "postgres"
    assert session.workspace["document"] == ""
    assert session.history == []
    assert not session.schema.tables
    assert not session.source_connected()
    assert session.run("SELECT 1").error
    assert session.history == []


def test_parse_and_compile_are_offline_native_sql():
    session = Session()
    document = session.parse("BEGIN; SELECT ';' AS value; ROLLBACK;")
    assert isinstance(document, SQLDocument)
    assert len(document.statements) == 3
    outcome = session.compile(document.sql)
    assert isinstance(outcome, RunOutcome)
    assert outcome.ok
    assert outcome.sql == document.sql
    assert "server syntax/schema not validated" in outcome.status
    assert not session.history
    # A lexical check must not pretend to validate PostgreSQL grammar offline.
    assert session.compile("SELECT FROM missing_relation").ok


def test_function_expansion_and_preview_preserve_authored_history(parquet_source):
    session = session_for(parquet_source)
    session.save_fn("after_id", "minimum", "SELECT * FROM data WHERE id > @minimum", "Filtered rows")
    sql = "SELECT * FROM after_id(1) ORDER BY id"
    compiled = session.compile(sql)
    assert compiled.ok and "@minimum" not in compiled.sql
    assert session.run(sql).rows == [[2, "beta"], [3, None]]
    assert session.history[0].doc == sql
    before = list(session.history)
    preview = session.fn_preview("after_id", "2")
    assert preview.ok and preview.rows == [[3, None]]
    assert session.history == before
    assert session.fn_preview("not_saved").error


def test_function_validation_persists_only_valid_definitions(parquet_source):
    session = session_for(parquet_source)
    with pytest.raises(ValueError):
        session.save_fn("broken", "x,x", "SELECT @x", "")
    assert "broken" not in session.fns
    session.save_fn("labels", "", "SELECT label FROM data", "Labels")
    restored = Session()
    restored.restore_source(session.source_key())
    assert restored.fns["labels"].body == "SELECT label FROM data"
    restored.delete_fn("labels")
    assert "labels" not in restored.fns


def test_default_rows_is_fetch_cap_not_query_limit(parquet_source):
    session = session_for(parquet_source)
    session.update_settings(default_rows=1)
    sql = "SELECT id FROM data ORDER BY id LIMIT 3"
    outcome = session.run(sql)
    assert outcome.ok and outcome.rows == [[1]]
    assert outcome.sql == sql
    assert outcome.capped and outcome.total == 2
    assert "more rows available" in outcome.status
    assert "of 3" not in outcome.status
    aggregate = session.run("SELECT count(*) FROM data")
    assert aggregate.rows == [[3]]
    session.update_settings(default_rows=0)
    assert session.run(sql).rows == [[1], [2], [3]]


def test_multiple_native_statements_and_transactions(parquet_source):
    session = session_for(parquet_source)
    outcomes = session.run_all(
        "CREATE TEMP TABLE scratch AS SELECT id FROM data; "
        "BEGIN; INSERT INTO scratch VALUES (4); SELECT count(*) FROM scratch; "
        "ROLLBACK; SELECT count(*) FROM scratch"
    )
    assert len(outcomes) == 6
    assert all(outcome.ok for outcome in outcomes)
    assert outcomes[0].schema_changed
    assert outcomes[3].rows == [[4]]
    assert outcomes[5].rows == [[3]]
    assert not session.tx_state().open
    assert len(session.history) == 6
    # Session state follows the actual engine even if another caller opens it.
    execute_sql(parquet_source, "BEGIN")
    assert session.tx_state().open
    assert session.run("ROLLBACK").ok
    assert not session.tx_state().open


def test_native_error_stops_later_statements_and_records_successes_only(parquet_source):
    session = session_for(parquet_source)
    outcomes = session.run_all("SELECT 1; SELECT * FROM absent_table; SELECT 3")
    assert len(outcomes) == 2
    assert outcomes[0].ok and outcomes[1].error
    assert [entry.doc for entry in session.history] == ["SELECT 1"]


def test_lexical_error_prevents_all_execution(parquet_source):
    session = session_for(parquet_source)
    outcomes = session.run_all("CREATE TEMP TABLE must_not_exist AS SELECT 1; SELECT 'unterminated")
    assert len(outcomes) == 1 and outcomes[0].error
    assert not session.history
    assert session.run("SELECT * FROM must_not_exist").error


def test_empty_run_does_not_record_history(parquet_source):
    session = session_for(parquet_source)
    assert session.run("-- comment only").status == "nothing to run"
    assert not session.history


def test_restore_parquet_requires_explicit_reconnect_and_keeps_workspace(parquet_source):
    session = session_for(parquet_source)
    session.run("SELECT id FROM data ORDER BY id")
    session.save_fn("all_data", "", "SELECT * FROM data", "")
    session.save_workspace(document="SELECT label FROM data", source=session.source_key(),
                           dialect="duckdb", cursor=[0, 7])
    key = session.source_key()
    restored = Session()
    assert restored.source_key() == key
    assert restored.source.kind == "disconnected"
    assert restored.dialect == "duckdb"
    assert restored.workspace["document"] == "SELECT label FROM data"
    assert restored.history[0].doc == "SELECT id FROM data ORDER BY id"
    assert restored.compile("SELECT * FROM all_data()").ok
    assert restored.run("SELECT * FROM all_data()").error
    connected = build_saved_source(restored, profile=restored.saved_source_profile(key))
    try:
        restored.register(connected)
        assert restored.source_key() == key
        assert len(restored.history) == 1
        assert restored.run("SELECT count(*) FROM all_data()").rows == [[3]]
    finally:
        close_source(connected)


def test_functions_and_history_are_target_scoped(parquet_source):
    session = session_for(parquet_source)
    session.save_fn("local_rows", "", "SELECT * FROM data", "")
    session.run("SELECT 1")
    original = session.source_key()
    other = session.restore_source("saved-postgres:other", activate=False)
    with pytest.raises(RuntimeError):
        with session.target_source(other):
            assert not session.fns and not session.history
            session.save_fn("remote_rows", "", "SELECT 9", "")
            raise RuntimeError("leave scope")
    assert session.source_key() == original
    assert set(session.fns) == {"local_rows"}
    assert len(session.history) == 1
    assert set(session.functions_for(other)) == {"remote_rows"}
    assert session.history_for(other) == []


def test_value_cache_is_typed_bounded_and_invalidated(parquet_source):
    session = session_for(parquet_source)
    session.update_settings(value_cache_enabled=True, value_cache_limit=2)
    values, capped = session.distinct_values(parquet_source, "data", "id")
    assert values == [1, 2] and capped
    assert session.cached_value_columns == 1
    assert session.cached_values(parquet_source, "data", "id") == ([1, 2], True)
    epoch = session.value_cache_epoch
    session.clear_value_cache()
    assert session.cached_values(parquet_source, "data", "id") is None
    with pytest.raises(EngineError):
        session.distinct_values(parquet_source, "data", "id", epoch=epoch)
    session.distinct_values(parquet_source, "data", "id")
    session.run("SELECT 1")
    assert session.cached_value_columns == 0


def test_native_completion_uses_cached_metadata(parquet_source):
    session = session_for(parquet_source)
    document = "SELECT d.la FROM data d"
    result = complete(document, len("SELECT d.la"), session.schema, dialect=session.dialect)
    assert any(item.label == "d.label" for item in result.items)
    unknown = "SELECT missing.id FROM data d"
    assert not complete(unknown, len("SELECT missing.id"), session.schema, dialect=session.dialect).items
    # A disconnected copy retains completion metadata but cannot execute.
    offline = replace(parquet_source, con=None)
    session.sources[offline.id] = offline
    session.refresh_schema()
    assert any(item.label == "d.label" for item in complete(
        document, len("SELECT d.la"), session.schema, dialect=session.dialect).items)
    assert session.compile("SELECT id FROM data").ok


def test_dialect_cannot_be_switched_independently(parquet_source):
    session = session_for(parquet_source)
    with pytest.raises(ValueError, match="intrinsic"):
        session.update_settings(dialect="postgres")
    session.update_settings(dialect="duckdb")
    assert "dialect" not in session._settings.document
    assert "source_dialects" not in session._settings.document


def postgres_profile(password="session-secret"):
    return {"host": "localhost", "port": "5432", "database": "work", "user": "tester",
            "password": password, "schema": "public", "sslmode": "prefer", "display": "Work"}


def test_postgres_remember_off_never_persists_through_function_save(credential_backend):
    session = Session()
    session.remember_postgres(postgres_profile(), persist=False)
    assert session.postgres_profiles[0]["password"] == "session-secret"
    session.save_fn("one", "", "SELECT 1", "")
    session.update_settings(default_rows=25)
    assert Session().postgres_profiles == []
    assert not credential_backend.values
    assert "session-secret" not in session.storage_path.read_text(encoding="utf-8")


def test_session_only_override_preserves_remembered_credentials():
    session = Session()
    session.remember_postgres(postgres_profile("remembered"), persist=True)
    session.remember_postgres(postgres_profile("ephemeral"), persist=False)
    session.save_fn("one", "", "SELECT 1", "")
    assert session.postgres_profiles[0]["password"] == "ephemeral"
    assert Session().postgres_profiles[0]["password"] == "remembered"


def test_ai_remember_off_stays_session_only(credential_backend):
    session = Session()
    config = AIConfig(base_url="https://example.com/v1", model="example", api_key="temporary-ai-secret")
    session.update_settings(ai=config, persist_ai=False)
    session.update_settings(default_rows=25)
    session.save_fn("one", "", "SELECT 1", "")
    assert session.ai_config.api_key == "temporary-ai-secret"
    assert Session().ai_config.api_key == ""
    assert not credential_backend.values
    assert "temporary-ai-secret" not in session.settings_path.read_text(encoding="utf-8")


def test_connection_labels_and_errors_do_not_leak_driver_secrets():
    profile = {**postgres_profile(), "kind": "postgres-live", "display": "session-secret work"}
    assert "session-secret" not in saved_source_label(profile=profile)
    assert "session-secret" not in Session().execution_error(RuntimeError("session-secret"))
    with pytest.raises(ValueError):
        build_saved_source(Session(), profile={"kind": "sqlite"})


def test_cancelled_connection_disposes_late_native_result(parquet_source, monkeypatch):
    from d8r.tui import connection

    started, finish, disposed = Event(), Event(), Event()
    release = connection.release_source

    def close_and_signal(source):
        release(source)
        disposed.set()

    def build():
        started.set()
        finish.wait(5)
        return add_parquet_source(parquet_source.connection_key)

    monkeypatch.setattr(connection, "release_source", close_and_signal)
    async def scenario():
        pending = asyncio.create_task(build_connection(build))
        assert await asyncio.to_thread(started.wait, 5)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        finish.set()
        assert await asyncio.to_thread(disposed.wait, 5)

    asyncio.run(scenario())


def test_function_preview_target_does_not_switch_workspace(parquet_source):
    session = session_for(parquet_source)
    session.save_fn("local_rows", "", "SELECT * FROM data", "")
    other = session.restore_source("saved-postgres:another")
    schema = session.schema
    outcome = session.fn_preview("local_rows", source_id=parquet_source.id)
    assert outcome.ok and len(outcome.rows) == 3
    assert session.active_id == other
    assert session.schema is schema
    assert not session.history_for(parquet_source.id)
    assert not session.history


def test_schema_refresh_tracks_native_ddl_and_rollback(parquet_source):
    session = session_for(parquet_source)
    original = session.schema
    assert session.run("BEGIN").ok
    created = session.run("CREATE TEMP TABLE transient_rows AS SELECT id FROM data")
    assert created.ok and created.schema_changed
    assert session.schema is not original
    table = session.schema.table_by_name("transient_rows")
    assert table is not None
    assert [column.name for column in table.columns] == ["id"]
    query = "SELECT t.i FROM transient_rows t"
    assert any(item.label == "t.id" for item in complete(
        query, len("SELECT t.i"), session.schema, dialect=session.dialect).items)
    tables = complete("SELECT * FROM tran", len("SELECT * FROM tran"), session.schema, dialect=session.dialect)
    assert any(item.table == table.name for item in tables.items)
    rolled_back = session.run("ROLLBACK")
    assert rolled_back.ok and rolled_back.schema_changed
    assert session.schema.table_by_name("transient_rows") is None
    assert not complete(query, len("SELECT t.i"), session.schema, dialect=session.dialect).items
    tables = complete("SELECT * FROM tran", len("SELECT * FROM tran"), session.schema, dialect=session.dialect)
    assert not any(item.table == table.name for item in tables.items)


def test_metadata_failure_warns_without_relabeling_executed_ddl(parquet_source, monkeypatch):
    from d8r.tui import session as session_module

    session = session_for(parquet_source)

    def unavailable_metadata(source):
        raise EngineError("Metadata unavailable.")

    monkeypatch.setattr(session_module, "refresh_source_schema", unavailable_metadata)
    sql = "CREATE TEMP TABLE executed_rows AS SELECT id FROM data"
    outcome = session.run(sql)
    assert outcome.ok and not outcome.error
    assert "schema not refreshed: Metadata unavailable." in outcome.status
    assert session.history[0].doc == sql
    assert session.run("SELECT count(*) FROM executed_rows").rows == [[3]]
