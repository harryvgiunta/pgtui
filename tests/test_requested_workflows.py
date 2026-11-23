"""Requested keyboard flows and source-bound persistence regressions."""
from __future__ import annotations

import asyncio
from functools import wraps

import pytest
from textual.widgets import Input, OptionList, TabbedContent

from d8r.tui.app import D8RApp
from d8r.tui.session import Session


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


async def load_sql(app, pilot, text, offset=None):
    app.editor.load_text(text)
    app.editor.move_cursor(app.editor.document.get_location_from_index(len(text) if offset is None else offset))
    app.editor.focus()
    await pilot.pause()


@async_test
async def test_alt_arrows_move_query_cursor_without_editing(parquet_source):
    app = D8RApp(Session({parquet_source.id: parquet_source}))
    document = "\n".join(["SELECT label FROM data"] * 100)
    async with app.run_test(size=(130, 42)) as pilot:
        await load_sql(app, pilot, document, 12)
        await pilot.press("ctrl+space", "alt+left")
        assert app.editor.cursor_location == (0, 7)
        await pilot.press("alt+right")
        assert app.editor.cursor_location == (0, 12)
        await pilot.press("escape", "alt+down")
        assert app.editor.cursor_location[0] == app.editor.content_size.height
        await pilot.press("alt+up", "ctrl+a")
        assert app.editor.selected_text == document and app.editor.has_focus


@pytest.mark.parametrize("tab,control", [
    ("tab-schema", "#schema-tree"), ("tab-schema", "#explorer-close"),
    ("tab-pages", "#page-search"), ("tab-pages", "#pages-list"),
])
@async_test
async def test_explorer_escape_hides_pane_and_restores_query_caret(parquet_source, tab, control):
    app = D8RApp(Session({parquet_source.id: parquet_source}))
    async with app.run_test(size=(130, 42)) as pilot:
        await load_sql(app, pilot, "SELECT 42", len("SELECT "))
        caret = app.editor.cursor_location
        app.query_one("#explorer-tabs", TabbedContent).active = tab
        await pilot.pause()
        target = app.query_one(control)
        target.focus()
        await pilot.pause()
        assert target.has_focus

        await pilot.press("escape")
        assert not app.query_one("#explorer-slot").display
        assert not app.query_one("#explorer-splitter").display
        assert app.editor.has_focus and app.editor.cursor_location == caret
        assert app.editor.text == "SELECT 42"

        app.action_toggle_schema()
        await pilot.press("escape")  # Escape in the editor doesn't close the reopened explorer.
        assert app.query_one("#explorer-slot").display and app.editor.has_focus


@async_test
async def test_pages_tab_up_search_and_functions_escape():
    app = D8RApp(Session())
    async with app.run_test(size=(130, 42)) as pilot:
        await pilot.press("ctrl+n")
        tabs = app.query_one("#explorer-tabs", TabbedContent)
        pages = tabs.get_tab("tab-pages")
        search = app.query_one("#page-search", Input)
        listing = app.query_one("#pages-list", OptionList)
        # The padding and the gap after the button keep their resting background.
        def backgrounds():
            y = pages.region.y + 1
            return [app.screen.get_style_at(x, y).bgcolor for x in (pages.region.x + 1, pages.region.right)]

        resting = backgrounds()
        await pilot.click(pages)
        assert backgrounds() == resting
        await pilot.press("up")
        assert search.has_focus
        await pilot.press("down", "home", "down")
        assert listing.has_focus and listing.highlighted == 1
        await pilot.press("up")
        assert listing.has_focus and listing.highlighted == 0
        await pilot.press("up", *"missing", "down")
        assert listing.has_focus and listing.option_count == 0
        await pilot.press("up")
        assert search.has_focus and search.value == "missing"

        await pilot.click(tabs.get_tab("tab-functions"))
        await pilot.press("escape")
        assert not app.query_one("#explorer-slot").display
        assert tabs.active == "tab-pages" and app.editor.has_focus
        assert search.value == "missing"


@async_test
async def test_inline_functions_search_inserts_at_original_caret(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    session.save_fn("rows_above", "minimum", "SELECT * FROM data WHERE id > @minimum", "Filter rows")
    session.save_fn("all_rows", "", "SELECT * FROM data", "All records")
    app = D8RApp(session)
    text = "SELECT * FROM \\functions WHERE true"
    async with app.run_test(size=(150, 48)) as pilot:
        await load_sql(app, pilot, text, len("SELECT * FROM \\functions"))
        await pilot.press("ctrl+space", "enter")
        await pilot.pause()
        assert app.query_one("#explorer-tabs", TabbedContent).active == "tab-functions"
        search = app.query_one("#fn-search", Input)
        assert app.focused is search and not app.function_mode
        await pilot.press(*"rows_above", "down", "enter")
        await pilot.pause()
        expected = "SELECT * FROM rows_above() WHERE true"
        assert app.editor.text == expected
        assert app.editor.document.get_index_from_location(app.editor.cursor_location) == expected.index(")")
        assert app.focused is app.editor
        assert not session.history
        app.palette.close()
        await pilot.press("ctrl+z")
        assert app.editor.text == "SELECT * FROM  WHERE true"


@async_test
async def test_inline_ai_preserves_prefix_suffix_and_never_executes(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        text = "SELECT id, \\AI label FROM data"
        await load_sql(app, pilot, text, len("SELECT id, \\AI"))
        await pilot.press("ctrl+space", "enter")
        await pilot.pause()
        assert app.editor.text == "SELECT id,  label FROM data"
        assert app.ai_panel.display and app.ai_panel.has_focus_within
        assert not session.history
        assert all(chat.status != "working" for chat in app.ai_chats.chats.values())


@pytest.mark.parametrize("source_fixture,table", [("parquet_source", "data"), ("d1_source", "records")])
@async_test
async def test_where_values_are_bounded_on_demand_and_undoable(request, source_fixture, table, monkeypatch):
    source = request.getfixturevalue(source_fixture)
    session = Session({source.id: source})
    dataset = session.schema.tables[0].name
    session.update_settings(value_cache_limit=2)
    reads = []
    original = session.distinct_values

    def tracked(source, dataset, column, **kwargs):
        reads.append((dataset, column))
        return original(source, dataset, column, **kwargs)

    monkeypatch.setattr(session, "distinct_values", tracked)
    app = D8RApp(session)
    sql = f"SELECT r.label FROM {table} AS r WHERE r.label = "
    async with app.run_test(size=(150, 48)) as pilot:
        await load_sql(app, pilot, sql)
        await pilot.press("left", "right")
        await pilot.pause()
        assert not reads
        await pilot.press("ctrl+space")
        await app.workers.wait_for_complete()
        await pilot.pause()
        cached = session.cached_values(source, dataset, "label")
        assert cached == ([None, "alpha"] if source.dialect == "sqlite" else ["alpha", "beta"], True)
        assert reads == [(dataset, "label")]
        view = app.palette.view
        index = next(i for i, entry in enumerate(view.entries) if entry.kind == "value" and entry.insert == "'alpha'")
        app.palette.highlighted = index
        await pilot.press("enter")
        assert app.editor.text == sql + "'alpha'"
        app.palette.close()
        await pilot.press("ctrl+z")
        assert app.editor.text == sql
        assert not session.history


@async_test
async def test_where_real_edit_fetches_without_explorer(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        text = "SELECT * FROM data d WHERE d.label = "
        await load_sql(app, pilot, text)
        assert session.cached_values(parquet_source, "data", "label") is None
        await pilot.press("'", "a")
        await pilot.pause()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert any(entry.kind == "value" and entry.insert == "'alpha'" for entry in app.palette.view.entries)
        assert app.editor.text == text + "'a"
        assert not session.history


def test_typed_functions_persist_and_quality_install_is_atomic(parquet_source, tmp_path, monkeypatch):
    from d8r.quality import quality_functions

    session = Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state")
    saved = session.save_fn("project_field", "table:relation, field:identifier", "SELECT @field FROM @table", "Projection")
    assert saved.param_types == ("relation", "identifier")
    restored = Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "state")
    assert restored.fns["project_field"] == saved
    assert restored.run("SELECT * FROM project_field(data, label)", record=False).rows == [["alpha"], ["beta"], [None]]
    library = quality_functions("duckdb")
    restored.save_fn(library[-1].name.upper(), "", "SELECT 99", "Do not overwrite")
    before = dict(restored.fns)
    disk = restored.storage_path.read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        restored.install_quality_functions()
    assert restored.fns == before and restored.storage_path.read_bytes() == disk
    restored.delete_fn(library[-1].name.upper())
    count = restored.install_quality_functions()
    assert count == len(library)
    assert restored.install_quality_functions() == 0
    assert not restored.history


@async_test
async def test_project_home_is_source_scoped_and_restores(parquet_source, d1_source, tmp_path):
    sources = {parquet_source.id: parquet_source, d1_source.id: d1_source}
    session = Session(sources, data_dir=tmp_path / "state")
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        assert app.set_dbt_enabled(True)
        assert app.set_dbt_project_dir(str(tmp_path / "project-one"))
        app.select_source(d1_source.id)
        await pilot.pause()
        assert not session.dbt_enabled
        assert session.dbt_project_dir == ""
        assert app.set_dbt_enabled(True)
        assert app.set_dbt_project_dir(str(tmp_path / "project-two"))
        assert app.set_dbt_enabled(False)
        app.select_source(parquet_source.id)
        await pilot.pause()
        assert session.dbt_project_dir == str(tmp_path / "project-one")
        assert session.dbt_enabled
    restored = Session(sources, data_dir=tmp_path / "state")
    assert restored.dbt_enabled
    assert restored.dbt_project_dir == str(tmp_path / "project-one")
    restored.set_active(d1_source.id)
    assert restored.dbt_project_dir == str(tmp_path / "project-two")
    assert not restored.dbt_enabled


@pytest.mark.parametrize("dismiss", [False, True])
@async_test
async def test_distinct_read_uses_current_literal_but_respects_escape(parquet_source, monkeypatch, dismiss):
    from threading import Event

    session = Session({parquet_source.id: parquet_source})
    started, release = Event(), Event()
    original = session.distinct_values

    def held(*args, **kwargs):
        started.set()
        if not release.wait(5):
            raise AssertionError("The test must release the real native value read")
        return original(*args, **kwargs)

    monkeypatch.setattr(session, "distinct_values", held)
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        await load_sql(app, pilot, "SELECT * FROM data d WHERE d.label = ")
        try:
            await pilot.press("ctrl+space")
            assert await asyncio.to_thread(started.wait, 2)
            assert session.busy
            if dismiss:
                await pilot.press("escape")
            await pilot.press("'", "b")
        finally:
            release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert not session.busy
        assert app.editor.text.endswith("= 'b")
        if dismiss:
            assert app.palette.view is None and not app.palette.details.display
        else:
            values = [entry.insert for entry in app.palette.view.entries if entry.kind == "value"]
            assert values == ["'beta'"]


@async_test
async def test_conversation_new_key_does_not_create_query_page(parquet_source):
    from d8r.tui.ai import ConversationTabs

    app = D8RApp(Session({parquet_source.id: parquet_source}))
    async with app.run_test(size=(150, 48)) as pilot:
        await load_sql(app, pilot, "SELECT 17")
        identity = app._document_identity
        app.action_ai()
        await pilot.pause()
        await pilot.click("#conversation-new")
        await pilot.pause()
        original = app.ai_panel.chat.id
        app.action_conversations()
        await pilot.pause()
        app.query_one(ConversationTabs).focus()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert app.ai_panel.chat.id != original
        assert original in app.ai_chats.chats
        assert app.editor.text == "SELECT 17" and app._document_identity == identity
        app.action_conversations()
        await pilot.press("escape")
        assert not app.query_one("#assistant-slot").display and app.editor.has_focus
        assert original in app.ai_chats.chats
