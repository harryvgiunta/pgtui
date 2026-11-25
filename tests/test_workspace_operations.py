"""Workspace regressions at the draft/connection/driver boundaries."""
from __future__ import annotations

import asyncio
import os
from threading import Event

import psycopg
import pytest
from textual.widgets import Button, Input, OptionList, Static, TextArea, Tree

from d8r.ai.context import AIProposal, AIQuery
from d8r.engine import close_source
from d8r.tui.app import D8RApp
from d8r.tui.session import Session
from tests.test_d1_session import postgres_profile
from tests.test_native_engine import postgres_source
from tests.test_native_tui import async_test


@async_test
async def test_pages_keep_creation_order_across_open_search_proposal_and_restore(parquet_source, tmp_path):
    session = Session({parquet_source.id: parquet_source}, data_dir=tmp_path)
    app = D8RApp(session)
    async with app.run_test(size=(120, 40)) as pilot:
        listing = app.query_one("#pages-list", OptionList)

        def visible():
            return [option.id for option in listing.options]

        first = app._document_identity
        app.editor.load_text("SELECT 'match first'")
        app.action_new_query()
        middle = app._document_identity
        app.editor.load_text("SELECT 'middle'")
        app.action_new_query()
        last = app._document_identity
        app.editor.load_text("SELECT 'match last'")
        app.open_page(middle)
        assert visible() == [first, middle, last]
        search = app.query_one("#page-search", Input)
        search.value = "match"
        await pilot.pause()
        assert visible() == [first, last]
        search.value = ""
        app._apply_ai_document(AIProposal("", queries=(AIQuery("A", "SELECT 1"), AIQuery("B", "SELECT 2"), AIQuery("C", "SELECT 3"))))
        await pilot.pause()
        pages = session.workspace["pages"]
        assert visible() == [page["id"] for page in pages]
        assert visible()[:3] == [first, middle, last]
        assert [page["title"] for page in pages[3:]] == ["B", "C"]
        assert pages[1]["title"] == "A"
        ids = visible()
        app.open_page(first)
        assert visible() == ids
    assert [page["id"] for page in Session(data_dir=tmp_path).workspace["pages"]] == ids


@async_test
async def test_copy_live_page_to_connected_and_saved_workspaces(parquet_source, d1_source, tmp_path):
    sources = {s.id: s for s in (parquet_source, d1_source)}
    session = Session(sources, data_dir=tmp_path)
    prod = {**postgres_profile(), "display": "PROD"}
    session.remember_postgres(prod)
    prod_key = session._postgres_profile_id(prod)
    stage_key = session.source_key(d1_source.id)
    app = D8RApp(session)
    app._WORKSPACE_SAVE_INTERVAL = 60  # Copy before the idle save of SQL and Page Details.
    async with app.run_test(size=(110, 40)) as pilot:
        await pilot.pause()
        assert app.select_source(d1_source.id)
        old_stage = app._document_identity
        app.editor.load_text("SELECT 'previous draft'")
        assert app.select_source(parquet_source.id)
        dev = app._document_identity
        sql = "-- Keep the entire page\nSELECT 42 AS answer"
        app.editor.load_text(sql)
        app.editor.move_cursor((1, 8))
        await pilot.pause()  # The toolbar relayouts for the restored source.
        assert await pilot.click("#page-details-toggle")
        app.query_one("#page-name", Input).value = "Answer"
        app.query_one("#page-description", TextArea).load_text("Deployment check")

        assert not session.history
        button = app.query_one("#copy-page-to-ws", Button)
        for key in (stage_key, prod_key):
            if button.has_class("-active"):
                await pilot.pause(button.active_effect_duration)
            assert await pilot.click(button)
            options = app.screen.query_one("#copy-page-targets", OptionList)
            assert {option.id for option in options.options} == {stage_key, prod_key}
            options.highlighted = options.get_option_index(key)
            await pilot.press("enter")
            assert session.source is parquet_source and app._document_identity == dev
            assert not session.history

        assert set(session.sources) == set(sources)  # PROD wasn't even registered to copy to it.
        pages = session.workspace["pages"]
        copies = [page for page in pages if page["source"] in {stage_key, prod_key} and page["id"] != old_stage]
        assert {page["source"]: page["dialect"] for page in copies} == {stage_key: "sqlite", prod_key: "postgres"}
        assert len({dev, old_stage, *(page["id"] for page in copies)}) == 4
        for page in copies:
            assert (page["document"], page["title"], page["description"], page["cursor"]) == (
                sql, "Answer", "Deployment check", [1, 8])
        assert app.select_source(d1_source.id)
        assert app._document_identity == old_stage and app.editor.text == "SELECT 'previous draft'"
        stage_copy = next(page for page in copies if page["source"] == stage_key)
        listing = app.query_one("#pages-list", OptionList)
        listing.highlighted = listing.get_option_index(stage_copy["id"])
        app.open_page(stage_copy["id"])
        assert app.editor.text == sql
        app.editor.load_text("SELECT 'edited in STAGE'")

    pages = Session(sources, data_dir=tmp_path).workspace["pages"]
    assert next(page for page in pages if page["id"] == dev)["document"] == sql
    assert next(page for page in pages if page["source"] == prod_key)["document"] == sql


@async_test
async def test_refresh_button_discovers_external_changes_without_running_draft(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(120, 40)) as pilot:
        app.editor.load_text("DROP TABLE external_change")
        parquet_source.con.execute("CREATE TABLE external_change (new_column INTEGER)")
        assert not any("external_change" in table for table in parquet_source.datasets)
        await pilot.click("#refresh-schema")
        await app.workers.wait_for_complete()
        await pilot.pause()
        tree = app.query_one("#schema-tree", Tree)
        node = next(node for node in tree.root.children if "external_change" in node.data["name"])
        assert session.schema.table_by_name(node.data["name"]).columns[0].name == "new_column"
        assert app.editor.text == "DROP TABLE external_change" and not session.history


@async_test
async def test_cancel_query_keeps_lane_and_does_not_dispatch_remaining_write(d1_source, d1_harness):
    started, release = Event(), Event()

    def hold_query():
        started.set()
        if not release.wait(10):
            raise RuntimeError("test query not released")
        return 42

    con = d1_harness.connection
    con.create_function("hold_query", 0, hold_query)
    con.execute("CREATE TABLE effects(n INTEGER)")
    session = Session({d1_source.id: d1_source})
    app = D8RApp(session)
    async with app.run_test(size=(120, 40)) as pilot:
        app.editor.load_text("INSERT INTO effects VALUES (1); SELECT hold_query(); INSERT INTO effects VALUES (2)")
        await pilot.press("ctrl+shift+f5")
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.click("#cancel-query")
            assert session.busy and app.query_one("#cancel-query").disabled
            app.action_run()  # Still owned by the interrupted driver call.
        finally:
            release.set()
        await app.workers.wait_for_complete()
        assert not session.busy and not app.query_one("#cancel-query").display
        assert con.execute("SELECT n FROM effects").fetchall() == [(1,)]
        app.editor.load_text("SELECT 42")
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        assert app._main_outcome.rows == [[42]]


@pytest.mark.parametrize("cancel", [False, True])
@async_test
async def test_submission_joins_pending_connection_and_can_be_cancelled(parquet_source, monkeypatch, cancel):
    previous = Session({parquet_source.id: parquet_source})
    previous.update_settings(source=parquet_source.id)
    previous.save_workspace(document="SELECT 11", source=previous.source_key(), dialect="duckdb")
    session = Session()
    started, unblock, discarded = Event(), Event(), Event()
    build = session.build_parquet_source
    attempts = []

    def delayed(*args):
        attempts.append(args)
        started.set()
        if not unblock.wait(10):
            raise RuntimeError("test builder not released")
        return build(*args)

    def discard(source):
        close_source(source)
        discarded.set()

    monkeypatch.setattr(session, "build_parquet_source", delayed)
    monkeypatch.setattr("d8r.tui.connection.release_source", discard)
    app = D8RApp(session)
    async with app.run_test(size=(120, 40)) as pilot:
        try:
            assert await asyncio.to_thread(started.wait, 5)
            await pilot.press("f5")
            app.editor.load_text("SELECT 99")
            if cancel:
                await pilot.press("f8")
        finally:
            unblock.set()
        if cancel:
            assert await asyncio.to_thread(discarded.wait, 5)
            assert not session.source_connected() and not session.history
            assert app._main_outcome is None
        else:
            async with asyncio.timeout(10):
                while app._main_outcome is None:
                    await pilot.pause(0.02)
            assert app._main_outcome.rows == [[11]] and len(session.history) == 1
            close_source(session.source)
        assert len(attempts) == 1 and app.editor.text == "SELECT 99"


@async_test
async def test_one_run_after_postgres_idle_timeout_uses_new_connection(postgres_source):
    session = Session({postgres_source.id: postgres_source})
    params = psycopg.conninfo.conninfo_to_dict(os.environ["PGTUI_TEST_POSTGRES_DSN"])
    session.remember_postgres({**postgres_source.postgres, "display": "Idle target",
                               "password": params.get("password", os.environ.get("PGPASSWORD", ""))}, persist=False)
    app = D8RApp(session)
    async with app.run_test(size=(120, 40)) as pilot:
        old_pid = postgres_source.con.info.backend_pid
        app.editor.load_text("SELECT pg_backend_pid()")
        postgres_source.con.execute("SET idle_session_timeout = '100ms'")
        await pilot.pause(0.25)
        await pilot.press("f5")
        try:
            async with asyncio.timeout(10):
                while app._main_outcome is None:
                    await pilot.pause(0.02)
            assert app._main_outcome.ok and app._main_outcome.rows != [[old_pid]]
            assert app._main_outcome.rows == [[session.source.con.info.backend_pid]]
            assert len(session.history) == 1 and session.history[0].doc == app.editor.text
            assert "disconnected" not in str(app.query_one("#source-version", Static).content)
        finally:
            if session.source is not postgres_source:
                close_source(session.source)
