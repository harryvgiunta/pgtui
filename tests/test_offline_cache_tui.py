"""Pilot regressions for durable, explicitly stale offline metadata snapshots."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from functools import wraps
from pathlib import Path
from uuid import uuid4

import pytest
from textual.widgets import Static, Tree

from d8r.ai import sql_completion
from d8r.ai.client import AIConfig
from d8r.ai.context import AIContext
from d8r.engine import close_source, execute_sql, refresh_source_schema
from d8r.schema_cache import SchemaCacheStore
from d8r.sql import relation_name
from d8r.tui.app import D8RApp
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


async def load_editor(app, pilot, document, cursor=None):
    app.editor.load_text(document)
    app.editor.move_cursor(cursor or (0, len(document)))
    app.editor.focus()
    await pilot.pause()


def save_draft(session, document="", cursor=None):
    session.save_workspace(document=document, source=session.source_key(),
                           dialect=session.source.dialect, cursor=list(cursor or (0, len(document))))


@async_test
async def test_restart_restores_metadata_without_sql_or_provider_requests(d1_source, d1_harness, monkeypatch):
    provider_requests = []

    async def provider(*args):
        provider_requests.append(args)
        return ()

    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    previous = Session({d1_source.id: d1_source})
    previous.update_settings(
        value_cache_enabled=True, ai_completion=True,
        ai=AIConfig("https://provider.invalid/v1", "test", "isolated-test-key"), remember_ai_key=True,
    )
    dataset = relation_name("records")
    previous.distinct_values(d1_source, dataset, "label")
    document = "SELECT r.la FROM records r"
    save_draft(previous, document, (0, 11))
    key = previous.source_key()
    close_source(d1_source)
    requests = len(d1_harness.requests)

    session = Session()
    assert session.source_key() == key
    assert session.ai_completion and session.ai_config.api_key
    assert not session.source_connected()
    assert session.cached_value_columns == 0
    assert session.cached_values(session.source, dataset, "label") is None
    cache_info = session.schema_cache_info()
    assert "may be stale" in cache_info
    assert session.source.datasets[dataset]["rows"] is None
    context = AIContext(session, session.active_id, document)
    snapshot = json.loads(context.system_prompt().split("Editor snapshot (JSON data):\n", 1)[1])
    assert snapshot["connected"] is False
    assert snapshot["metadata_cache_info"] == cache_info
    schema = json.loads(await context.call_tool("schema", {}))
    assert schema["metadata_cache_info"] == cache_info
    assert schema["connected"] is False
    assert any(table["name"] == dataset for table in schema["tables"])

    request = sql_completion.CompletionRequest(
        document=document, cursor=11, start=9, end=11, dialect=session.source.dialect,
        server_version=session.source.server_version, schema=session.schema,
        metadata_cache_info=cache_info,
    )
    packet = sql_completion._request_data(request)
    assert packet["metadata_cache_info"] == cache_info
    assert any(table["name"] == dataset for table in packet["tables"])
    private = sql_completion._request_data(replace(request, allow_schema=False))
    assert "tables" not in private and "metadata_cache_info" not in private

    app = D8RApp(session)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause(0.85)
        assert app.editor.text == document
        assert cache_info in str(app.query_one("#source-version", Static).content)
        tree = app.query_one("#schema-tree", Tree)
        table = next(node for node in tree.root.children if node.data["name"] == dataset)
        table.expand()
        await pilot.pause()
        assert {node.data["name"] for node in table.children} == {"id", "label"}
        label = next(node for node in table.children if node.data["name"] == "label")
        label.expand()
        await pilot.pause()
        assert not label.is_expanded
        assert "Reconnect" in str(app.query_one("#status", Static).content)
        await pilot.press("left", "right", "ctrl+space")
        await pilot.pause(0.85)
        assert app.palette.is_open and app.palette.visible
        assert app.palette.details.display
        assert cache_info in app.palette.details.content.plain
        entry = next(i for i, entry in enumerate(app.palette.view.entries)
                     if entry.kind == "column" and entry.insert == "label")
        app.palette.highlighted = entry
        await pilot.press("tab")
        assert app.editor.text == "SELECT r.label FROM records r"
        assert app.editor.cursor_location == (0, 14)
        assert app._main_outcome is None and not session.history
        assert len(d1_harness.requests) == requests
        assert not provider_requests


@async_test
async def test_failed_reconnect_keeps_cached_relation_popup(parquet_source):
    previous = Session({parquet_source.id: parquet_source})
    save_draft(previous)
    key = previous.source_key()
    close_source(parquet_source)
    # Exercise the real saved-source reconnect failure, not a replacement Session.
    Path(key).unlink()
    session = Session()
    assert session.source_key() == key and not session.source_connected()
    app = D8RApp(session)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not session.source_connected()
        assert "existing .parquet" in str(app.query_one("#status", Static).content)
        assert session.schema.table_by_name("data") is not None
        prefix = "SELECT * FROM "
        await load_editor(app, pilot, prefix)
        await pilot.press("d", "a")
        await pilot.pause()
        assert app.palette.is_open and app.palette.visible
        assert app.palette.region.width > 0 and app.palette.region.height > 0
        assert session.schema_cache_info() in app.palette.details.content.plain
        index = next(i for i, entry in enumerate(app.palette.view.entries)
                     if entry.kind == "table" and entry.table == "data")
        steps = (index - app.palette.highlighted) % app.palette.option_count
        await pilot.press(*(["down"] * steps), "enter")
        assert app.editor.text == prefix + "data AS d"
        assert app.editor.cursor_location == (0, len(prefix + "data AS d"))
        app.palette.close()
        await pilot.press("ctrl+z")
        assert app.editor.text == prefix + "da"
        assert not session.history
        # A second explicit failed reconnect must not erase the hydrated catalog.
        assert await pilot.click("#reconnect-source")
        await app.workers.wait_for_complete()
        await pilot.pause()
        app.editor.focus()
        await pilot.press("ctrl+space")
        assert any(entry.table == "data" for entry in app.palette.view.entries)
        assert "may be stale" in str(app.query_one("#source-version", Static).content)


@async_test
async def test_offline_explorer_reuses_values_but_cache_miss_requires_reconnect(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    session.update_settings(value_cache_enabled=True)
    dataset = relation_name("records")
    values, capped = session.distinct_values(d1_source, dataset, "label")
    assert "alpha" in values and not capped
    close_source(d1_source)
    requests = len(d1_harness.requests)
    app = D8RApp(session)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        tree = app.query_one("#schema-tree", Tree)
        table = next(node for node in tree.root.children if node.data["name"] == dataset)
        table.expand()
        await pilot.pause()
        assert {node.data["name"] for node in table.children} == {"id", "label"}
        label = next(node for node in table.children if node.data["name"] == "label")
        label.expand()
        await pilot.pause()
        assert label.is_expanded
        assert "'alpha'" in {node.label.plain for node in label.children}
        assert "'beta'" in {node.label.plain for node in label.children}
        identifier = next(node for node in table.children if node.data["name"] == "id")
        identifier.expand()
        await pilot.pause()
        assert not identifier.is_expanded
        assert "Reconnect" in str(app.query_one("#status", Static).content)
        await load_editor(app, pilot, "SELECT * FROM records WHERE label = 'al")
        await pilot.press("ctrl+space")
        assert any(entry.insert == "'alpha'" for entry in app.palette.view.entries)
        assert session.schema_cache_info() in app.palette.details.content.plain
        assert len(d1_harness.requests) == requests


@async_test
async def test_register_cache_write_warning_does_not_fail_connection(d1_source, tmp_path, monkeypatch):
    home = tmp_path / "cache-warning"
    session = Session(data_dir=home)
    cache_path = SchemaCacheStore(home).path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{broken concurrent cache", encoding="utf-8")
    app = D8RApp(session)
    notifications = []
    notify = app.notify

    def capture_notification(message, **kwargs):
        notifications.append((message, kwargs))
        return notify(message, **kwargs)

    monkeypatch.setattr(app, "notify", capture_notification)
    async with app.run_test(size=(160, 50)) as pilot:
        await pilot.pause()
        app._source_added(d1_source)
        await pilot.pause()
        assert session.source is d1_source and session.source_connected()
        assert session.metadata_cache_warning
        assert any(message == session.metadata_cache_warning and options["severity"] == "warning"
                   for message, options in notifications)
        assert "added" in str(app.query_one("#status", Static).content)
        assert not session.schema_cache_info()
        await load_editor(app, pilot, "SELECT r.la FROM records r", (0, 11))
        await pilot.press("ctrl+space")
        assert any(entry.insert == "label" for entry in app.palette.view.entries)
        assert cache_path.read_text(encoding="utf-8") == "{broken concurrent cache"


@async_test
async def test_postgres_cached_enum_and_column_documentation_survive_restart(postgres_source):
    namespace = "offline_" + uuid4().hex
    relation = relation_name(namespace, "records")
    execute_sql(postgres_source, f'CREATE SCHEMA "{namespace}"')
    try:
        execute_sql(postgres_source, f'CREATE TYPE "{namespace}".state AS ENUM (\'ready\', \'can\'\'t\')')
        execute_sql(postgres_source, f'CREATE TABLE {relation} (status "{namespace}".state)')
        execute_sql(postgres_source, f"COMMENT ON COLUMN {relation}.status IS 'Declared workflow state'")
        refresh_source_schema(postgres_source)
        previous = Session({postgres_source.id: postgres_source})
        save_draft(previous)
        session = Session()
        assert session.source_key() == previous.source_key()
        assert not session.source_connected()
        column = session.schema.table_by_name(relation).columns[0]
        assert column.values == ("ready", "can't")
        app = D8RApp(session)
        async with app.run_test(size=(160, 50)) as pilot:
            await load_editor(app, pilot, f"SELECT r.st FROM {relation} r", (0, 11))
            await pilot.press("ctrl+space")
            app.palette.highlighted = next(i for i, entry in enumerate(app.palette.view.entries)
                                           if entry.insert == "status")
            await pilot.pause()
            assert "Declared workflow state" in app.palette.details.content.plain
            assert session.schema_cache_info() in app.palette.details.content.plain
            document = f"SELECT * FROM {relation} WHERE status = 'ca"
            await load_editor(app, pilot, document)
            await pilot.press("ctrl+space")
            assert app.palette.is_open and app.palette.visible
            index = next(i for i, entry in enumerate(app.palette.view.entries) if entry.insert == "'can''t'")
            app.palette.highlighted = index
            await pilot.pause()
            assert "Declared workflow state" in app.palette.details.content.plain
            assert session.schema_cache_info() in app.palette.details.content.plain
            await pilot.press("tab")
            assert app.editor.text == document[:-3] + "'can''t'"
            assert session.cached_value_columns == 0
            assert not session.history
    finally:
        execute_sql(postgres_source, f'DROP SCHEMA "{namespace}" CASCADE')


@pytest.mark.parametrize("prefix", ["", "SELECT 1; "])
@async_test
async def test_postgres_offline_automatic_table_first_popup(postgres_source, prefix):
    namespace = "offline_" + uuid4().hex
    relation = relation_name(namespace, "records")
    execute_sql(postgres_source, f'CREATE SCHEMA "{namespace}"')
    try:
        execute_sql(postgres_source, f"CREATE TABLE {relation} (id integer)")
        refresh_source_schema(postgres_source)
        previous = Session({postgres_source.id: postgres_source})
        save_draft(previous)
        session = Session()
        assert session.source_key() == previous.source_key()
        assert not session.source_connected()
        assert session.saved_source_profile(session.active_id) is None
        app = D8RApp(session)
        async with app.run_test(size=(160, 50)) as pilot:
            await load_editor(app, pilot, prefix)
            await pilot.press("o", "f", "f")
            await pilot.pause()
            assert app.palette.is_open and app.palette.visible
            assert app.palette.region.width > 0 and app.palette.region.height > 0
            assert session.schema_cache_info() in app.palette.details.content.plain
            index = next(i for i, entry in enumerate(app.palette.view.entries)
                         if entry.kind == "table" and entry.table == relation)
            steps = (index - app.palette.highlighted) % app.palette.option_count
            await pilot.press(*(["down"] * steps), "enter")
            assert app.editor.text == prefix + f"SELECT  FROM {relation} AS r"
            assert app.editor.cursor_location == (0, len(prefix + "SELECT "))
            app.palette.close()
            await pilot.press("ctrl+z")
            assert app.editor.text == prefix + "off"
            assert not session.history and not session.source_connected()
    finally:
        execute_sql(postgres_source, f'DROP SCHEMA "{namespace}" CASCADE')
