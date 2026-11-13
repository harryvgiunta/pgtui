"""Focused pilot coverage of the native SQL workspace and source form."""
from __future__ import annotations

import asyncio
from functools import wraps
from textual.widgets import DataTable, Input, Select, Static, TextArea

from d8r.tui.add_source import AddSourceModal
from d8r.tui.app import D8RApp
from d8r.tui.palette import view_for
from d8r.tui.session import Session
from d8r.tui.settings import MENUS


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


@async_test
async def test_blank_disconnected_startup_and_offline_sql_preview():
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(130, 42)) as pilot:
        await pilot.pause()
        assert app.title == "PGTUI"
        assert app.editor.text == ""
        assert not session.source_connected()
        assert not session.source.datasets
        assert not session.history
        assert not list(app.query("#dialect-select"))
        assert "dialects" not in MENUS
        assert "18.7" in str(app.query_one("#source-version", Static).content)
        app.editor.load_text("SELECT server_owned_syntax FROM absent_table")
        await pilot.press("ctrl+k")
        await pilot.pause()
        assert app.query_one("#sql-text", TextArea).text == app.editor.text
        assert not session.history
        assert not session.source_connected()


def test_completion_uses_cached_metadata_and_native_keywords(parquet_source, monkeypatch):
    session = Session({parquet_source.id: parquet_source})

    def forbidden(*args, **kwargs):
        raise AssertionError("Completion must never fetch rows")

    monkeypatch.setattr(session, "distinct_values", forbidden)
    select = view_for(session, "SEL", "SEL", 3)
    assert select is not None and "SELECT" in select.labels
    table = view_for(session, "SELECT * FROM da", "SELECT * FROM da", 16)
    assert table is not None and "data" in table.labels
    column = view_for(session, "SELECT la", "SELECT la", 9)
    assert column is not None and "label" in column.labels
    param = view_for(session, "SELECT @mi", "SELECT @mi", 10, parameters=("min_id",))
    assert param is not None and param.labels == ["@min_id"]
    assert view_for(session, "SELECT 'la", "SELECT 'la", 10) is None
    assert view_for(session, "-- SELECT la", "-- SELECT la", 12) is None
    actions = view_for(session, "\\", "\\", 1)
    assert actions is not None and all(entry.action for entry in actions.entries)
    assert not any(entry.label in {"\\from", "\\select", "\\where"} for entry in actions.entries)


@async_test
async def test_sql_completion_run_results_and_multiple_tabs(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        app.editor.load_text("SEL")
        app.editor.move_cursor((0, 3))
        app.editor.focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.editor.text == "SELECT"
        sql = "SELECT id, label FROM data ORDER BY id; SELECT count(*) AS n FROM data;"
        app.editor.load_text(sql)
        app.editor.focus()
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._main_outcome is not None and app._main_outcome.ok
        assert app._main_outcome.rows == [[1, "alpha"], [2, "beta"], [3, None]]
        assert app.query_one("#results-table", DataTable).row_count == 3
        assert len(app._data_tabs) == 1
        assert session.history
        assert app.editor.text == sql
        assert "DuckDB" in str(app.query_one("#source-version", Static).content)


@async_test
async def test_sql_function_save_and_preview(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        app.action_fn()
        await pilot.pause()
        form = app.function_editor
        form.query_one("#fn-name", Input).value = "rows_above"
        form.query_one("#fn-params", Input).value = "min_id"
        form.query_one("#fn-args", Input).value = "1"
        form.query_one("#fn-body", TextArea).load_text("SELECT id, label FROM data WHERE id > @min_id ORDER BY id")
        form.query_one("#fn-body", TextArea).focus()
        await pilot.pause()
        await pilot.press("ctrl+r")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.functions_for(parquet_source.id)["rows_above"].params == ("min_id",)
        assert form.preview_grid.row_count == 2
        assert not session.history


@async_test
async def test_postgres_form_remember_and_cancel_are_local(credential_backend):
    from textual.widgets import Checkbox

    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(130, 42)) as pilot:
        app.action_add_source()
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#source-backend", Select).value == "postgres"
        assert not modal.query_one("#d1-fields").display
        assert not list(modal.query("#backend-fields"))
        password = modal.query_one("#postgres-password", Input)
        assert password.password
        password.value = "session-only-secret"
        remember = modal.query_one("#remember-credentials", Checkbox)
        assert remember.value
        remember.value = False
        await pilot.pause()
        assert modal._fields()[-1] == "session"
        await pilot.press("escape")
        await pilot.pause()
        assert not credential_backend.values
        assert not session.postgres_profiles
        assert not session.source_connected()


@async_test
async def test_parquet_test_cancel_saves_nothing(parquet_source):
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(130, 42)) as pilot:
        app.action_add_source()
        await pilot.pause()
        modal = app.screen
        assert isinstance(modal, AddSourceModal)
        modal.query_one("#source-backend", Select).value = "parquet"
        modal.query_one("#parquet-path", Input).value = parquet_source.connection_key
        await pilot.pause()
        modal.action_test()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert modal._built is not None
        assert not session.postgres_profiles
        assert list(session.sources) == ["postgres"]
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not modal
        assert list(session.sources) == ["postgres"]
        assert not session.source_connected()


@async_test
async def test_parquet_test_then_add_transfers_connection(parquet_source):
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(130, 42)) as pilot:
        app.action_add_source()
        await pilot.pause()
        modal = app.screen
        modal.query_one("#source-backend", Select).value = "parquet"
        modal.query_one("#parquet-path", Input).value = parquet_source.connection_key
        await pilot.pause()
        modal.action_test()
        await app.workers.wait_for_complete()
        await pilot.pause()
        tested = modal._built[1]
        modal.action_add()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.source is tested
        assert session.source_connected()
        assert session.source.dialect == "duckdb"
        assert app.editor.text == ""
        assert not session.history
        assert session.saved_source_profile(tested.id)["kind"] == "parquet"


@async_test
async def test_saved_parquet_startup_reconnect_preserves_draft(parquet_source):
    previous = Session({parquet_source.id: parquet_source})
    draft = "SELECT id, label FROM data WHERE id > 1 ORDER BY id;"
    previous.update_settings(source=parquet_source.id)
    previous.save_workspace(document=draft, source=previous.source_key(), dialect="duckdb", cursor=[0, 7])
    session = Session()
    assert not session.source_connected()
    assert session.saved_source_profile(session.active_id)["kind"] == "parquet"
    app = D8RApp(session)
    async with app.run_test(size=(130, 42)) as pilot:
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert session.source_connected()
        assert session.source.connection_key == parquet_source.connection_key
        assert app.editor.text == draft
        assert app.editor.cursor_location == (0, 7)
        assert not session.history
        assert app._main_outcome is None
        assert app.query_one("#results-table", DataTable).row_count == 0
