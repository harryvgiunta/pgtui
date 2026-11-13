"""Pilot acceptance coverage of native Cloudflare D1 connection ownership."""
from __future__ import annotations

import asyncio
import json
from functools import wraps
from threading import Event

import pytest
from textual.widgets import Checkbox, DataTable, Input, Select, Static

from d8r.tui.add_source import AddSourceModal
from d8r.tui.app import D8RApp
from d8r.tui.palette import view_for
from d8r.tui.session import Session
from tests.d1_support import ACCOUNT_ID, API_TOKEN, DATABASE_ID, DATABASE_NAME


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


async def open_d1(app, pilot, *, remember=True):
    app.action_add_source()
    await pilot.pause()
    modal = app.screen
    assert isinstance(modal, AddSourceModal)
    modal.query_one("#source-backend", Select).value = "d1"
    modal.query_one("#d1-account", Input).value = ACCOUNT_ID
    modal.query_one("#d1-database", Input).value = DATABASE_NAME
    modal.query_one("#d1-token", Input).value = API_TOKEN
    modal.query_one("#remember-credentials", Checkbox).value = remember
    await pilot.pause()
    return modal


async def finish_connection(app, pilot):
    await app.workers.wait_for_complete()
    await pilot.pause()


@async_test
async def test_d1_test_cancel_masks_token_and_saves_nothing(d1_harness, credential_backend):
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        modal = await open_d1(app, pilot)
        assert modal.query_one("#d1-fields").display
        assert not modal.query_one("#postgres-fields").display
        assert not modal.query_one("#parquet-fields").display
        assert modal.query_one("#d1-token", Input).password
        modal.action_test()
        await finish_connection(app, pilot)
        assert modal._built is not None
        source = modal._built[1]
        assert source.d1["database"] == DATABASE_ID
        assert API_TOKEN not in str(modal.query_one("#add-source-message", Static).content)
        assert not session.d1_profiles
        assert not credential_backend.values
        await pilot.press("escape")
        await pilot.pause()
        assert source.con is None
        assert app.screen is not modal
        assert not session.source_connected()
        assert not session.d1_profiles
        assert not credential_backend.values


@pytest.mark.parametrize("remember", [True, False])
@async_test
async def test_d1_add_reuses_exact_test_and_remembers_only_resolved_uuid(
    d1_harness, credential_backend, remember,
):
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        modal = await open_d1(app, pilot, remember=remember)
        modal.action_test()
        await finish_connection(app, pilot)
        assert modal._built is not None
        tested = modal._built[1]
        requests = len(d1_harness.requests)
        modal.action_add()
        await finish_connection(app, pilot)
        assert session.source is tested
        assert session.source_connected()
        assert len(d1_harness.requests) == requests
        assert session.d1_profiles[0]["database"] == DATABASE_ID
        assert session.d1_profiles[0]["account_id"] == ACCOUNT_ID
        assert bool(credential_backend.values) == remember
        if session.storage_path.exists():
            assert API_TOKEN not in session.storage_path.read_text()
        restarted = Session()
        assert bool(restarted.d1_profiles) == remember
        if remember:
            disk = json.loads(session.storage_path.read_text())
            assert disk["d1_profiles"][0]["database"] == DATABASE_ID
            assert "api_token" not in disk["d1_profiles"][0]
            assert disk["d1_profiles"][0]["credential_ref"]
        assert app.editor.text == ""
        assert not session.history


@pytest.mark.parametrize("field,value", [
    ("d1-account", "b" * 32),
    ("d1-database", DATABASE_ID),
    ("d1-token", API_TOKEN + "-changed"),
    ("display-name", "renamed"),
    ("d1-database", DATABASE_NAME + " "),
    ("remember-credentials", False),
    ("source-backend", "postgres"),
])
@async_test
async def test_d1_test_cache_invalidates_on_every_identity_change(d1_harness, field, value):
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        modal = await open_d1(app, pilot)
        modal.action_test()
        await finish_connection(app, pilot)
        tested = modal._built[1]
        widget = modal.query_one(f"#{field}")
        widget.value = value
        await pilot.pause()
        assert modal._built is None
        assert tested.con is None
        assert not session.d1_profiles
        await pilot.press("escape")


@async_test
async def test_d1_cancel_disposes_late_builder_without_saving(d1_harness, credential_backend, monkeypatch):
    from d8r.tui import connection

    session = Session()
    app = D8RApp(session)
    started, unblock, released = Event(), Event(), Event()
    build = session.build_d1_source
    release = connection.release_source
    built = []

    def delayed_build(**kwargs):
        started.set()
        if not unblock.wait(10):
            raise RuntimeError("Test builder was not released")
        source = build(**kwargs)
        built.append(source)
        return source

    def capture_release(source):
        release(source)
        released.set()

    monkeypatch.setattr(session, "build_d1_source", delayed_build)
    monkeypatch.setattr(connection, "release_source", capture_release)
    async with app.run_test(size=(140, 48)) as pilot:
        modal = await open_d1(app, pilot)
        modal.action_add()
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.press("escape")
            await pilot.pause()
        finally:
            unblock.set()
        assert await asyncio.to_thread(released.wait, 5)
        assert built and built[0].con is None
        assert not session.source_connected()
        assert not session.d1_profiles
        assert not credential_backend.values


@async_test
async def test_d1_connection_errors_never_render_or_log_token(d1_harness, monkeypatch, caplog):
    session = Session()
    app = D8RApp(session)

    def fail(**kwargs):
        raise RuntimeError(f"Authorization: Bearer {API_TOKEN}")

    monkeypatch.setattr(session, "build_d1_source", fail)
    async with app.run_test(size=(140, 48)) as pilot:
        modal = await open_d1(app, pilot)
        modal.action_test()
        await finish_connection(app, pilot)
        message = str(modal.query_one("#add-source-message", Static).content)
        assert "Cloudflare D1" in message
        assert API_TOKEN not in message
        assert API_TOKEN not in caplog.text
        assert modal._built is None
        await pilot.press("escape")


@async_test
async def test_d1_saved_form_prefills_masked_target_and_missing_token_requires_reentry(d1_harness):
    session = Session()
    session.remember_d1({"account_id": ACCOUNT_ID, "database": DATABASE_ID,
                         "api_token": API_TOKEN, "display": "Saved D1"})
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        modal = await open_d1(app, pilot)
        modal.query_one("#d1-profile", Select).value = 0
        await pilot.pause()
        assert modal.query_one("#d1-database", Input).value == DATABASE_ID
        assert modal.query_one("#d1-token", Input).value == API_TOKEN
        assert modal.query_one("#d1-token", Input).password
        await pilot.press("escape")
        await pilot.pause()
        profile = {"kind": "d1-live", "account_id": ACCOUNT_ID, "database": DATABASE_ID,
                   "display": "Saved D1", "credentials_missing": True}
        app.push_screen(AddSourceModal(session, profile=profile))
        await pilot.pause()
        modal = app.screen
        assert modal.query_one("#source-backend", Select).value == "d1"
        assert modal.query_one("#d1-token", Input).value == ""
        requests = len(d1_harness.requests)
        modal.action_test()
        await finish_connection(app, pilot)
        assert len(d1_harness.requests) == requests
        assert modal._built is None
        modal.query_one("#d1-token", Input).value = API_TOKEN
        await pilot.pause()
        modal.action_test()
        await finish_connection(app, pilot)
        assert modal._built is not None
        await pilot.press("escape")


@async_test
async def test_d1_editor_completion_highlighting_and_native_results(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.pause()
        assert "Cloudflare D1 · SQLite" in str(app.query_one("#source-version", Static).content)
        assert app.editor.dialect == "sqlite"
        requests = len(d1_harness.requests)
        for quote in ("`", "["):
            doc = "SELECT " + quote + "label"
            quoted = view_for(session, doc, doc, len(doc))
            assert quoted is not None
            close_quote = "]" if quote == "[" else quote
            assert any(entry.insert == quote + "label" + close_quote for entry in quoted.entries)
        doc = "SELECT r.la FROM [records] AS r"
        offered = view_for(session, doc, doc, len("SELECT r.la"))
        assert offered is not None and "r.label" in offered.labels
        app.editor.load_text("SEL")
        app.editor.move_cursor((0, 3))
        app.editor.focus()
        await pilot.pause()
        await pilot.press("tab")
        assert app.editor.text == "SELECT"
        app.editor.load_text("SELECT [label], `id` FROM [records]")
        app.editor.get_line(0)
        assert app.editor._syntax_key[1] == "sqlite"
        assert len(d1_harness.requests) == requests
        sql = "SELECT [id], `label` FROM [records] ORDER BY [id];"
        app.editor.load_text(sql)
        app.editor.focus()
        await pilot.press("f5")
        await finish_connection(app, pilot)
        assert app._main_outcome is not None and app._main_outcome.ok
        assert app._main_outcome.rows == [[1, "alpha"], [2, "beta"], [3, None]]
        assert app.query_one("#results-table", DataTable).row_count == 3
        assert app.editor.text == sql
        assert session.history[-1].dialect == "sqlite"


@async_test
async def test_d1_saved_startup_reconnect_restores_draft_without_running(d1_source, d1_harness):
    previous = Session({d1_source.id: d1_source})
    previous.remember_d1({"account_id": ACCOUNT_ID, "database": DATABASE_ID,
                          "api_token": API_TOKEN, "display": "Saved D1"})
    draft = "SELECT [label] FROM [records] WHERE [id] > 1"
    previous.update_settings(source=d1_source.id)
    previous.save_workspace(document=draft, source=previous.source_key(), dialect="sqlite", cursor=[0, 7])
    session = Session()
    assert not session.source_connected()
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.pause()
        await finish_connection(app, pilot)
        assert session.source_connected()
        assert session.source.kind == "d1-live"
        assert app.editor.dialect == "sqlite"
        assert app.editor.text == draft
        assert app.editor.cursor_location == (0, 7)
        assert not session.history
        assert app._main_outcome is None
        assert app.query_one("#results-table", DataTable).row_count == 0


@async_test
async def test_d1_missing_token_preserves_display_and_reconnect_type(d1_harness):
    session = Session()
    key = f"d1:{ACCOUNT_ID}:{DATABASE_ID}"
    session.restore_source(key)
    app = D8RApp(session)
    async with app.run_test(size=(140, 48)) as pilot:
        await pilot.pause()
        assert "Cloudflare D1" in str(app.query_one("#source-version", Static).content)
        assert not session.source_connected() and not d1_harness.requests
        app.reconnect_source(session.active_id, app._source_added)
        await pilot.pause()
        assert app.screen.query_one("#source-backend", Select).value == "d1"
        assert app.screen.query_one("#d1-account", Input).value == ACCOUNT_ID
        assert app.screen.query_one("#d1-database", Input).value == DATABASE_ID
        assert app.screen.query_one("#d1-token", Input).value == ""
        await pilot.press("escape")
