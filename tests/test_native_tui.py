"""Focused pilot coverage of the native SQL workspace and source form."""
from __future__ import annotations

import asyncio
from functools import wraps
from textual.widgets import Button, DataTable, Input, Select, Static, Tab, TabbedContent, TextArea

from d8r.tui.add_source import AddSourceModal
from d8r.tui.app import D8RApp
from d8r.tui.palette import view_for
from d8r.tui.results import ExportModal, ResultsTable
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
        sql = "SELECT id, label FROM data ORDER BY id; SELECT count(*) AS n FROM data; SELECT 42 AS answer;"
        app.editor.load_text(sql)
        app.editor.focus()
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._main_outcome is not None and app._main_outcome.ok
        assert app._main_outcome.rows == [[1, "alpha"], [2, "beta"], [3, None]]
        assert app.query_one("#results-table", DataTable).row_count == 3
        tabs = app.query_one("#result-tabs", TabbedContent)
        buttons = list(tabs.query(Tab))
        assert [tab.label_text for tab in buttons] == ["Results 1", "Results 2", "Results 3", "SQL", "History"]
        assert tabs.active == "tab-results"
        for tab, rows in zip(buttons[1:3], [((3,),), ((42,),)]):
            await pilot.click(tab)
            assert tabs.get_pane(tabs.active).query_one(ResultsTable).raw_rows == rows
        assert session.history
        assert app.editor.text == sql
        assert "DuckDB" in str(app.query_one("#source-version", Static).content)
        footer = str(app.query_one("#status", Static).content)
        assert "3 results" in footer and "3 rows" not in footer


@async_test
async def test_results_drag_bar_operates_on_current_tab_and_preserves_pins(parquet_source):
    app = D8RApp(Session({parquet_source.id: parquet_source}))
    async with app.run_test(size=(120, 45)) as pilot:
        tabs = app.query_one("#result-tabs", TabbedContent)
        app.editor.load_text("SELECT 11 AS n; SELECT 22 AS n")
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        await pilot.click("#results-pin")  # Keep a snapshot of Results 1.
        pinned = tabs.active
        assert tabs.get_tab(pinned).label_text == "Results 3 ◆"
        assert app.query_one("#results-pin", Button).label.plain == "Unpin"
        # Even with both sidebars open, [x] and Close are separate hit targets.
        await pilot.resize_terminal(80, 45)
        app.action_ai()
        await pilot.pause()
        handle = app.query_one("#editor-splitter")
        for control in app.query("#output-controls Button"):
            assert handle.region.contains_region(control.region)
            assert app.screen.get_widget_at(*control.region.offset)[0] is control
        await pilot.press("escape")
        await pilot.resize_terminal(120, 45)
        second = next(tab for tab in tabs.query(Tab) if tab.label_text == "Results 2")
        await pilot.click(second)
        title = app.query_one("#output-controls .pane-title")
        for control in [title, *app.query("#output-controls Button")]:
            assert control.region.height == 1 and control.region.y == handle.region.y
        initial = app.session.pane_sizes["editor"]
        await pilot.click("#results-copy")
        assert app.clipboard.splitlines() == ["n", "22"]
        await pilot.click("#results-export")
        assert isinstance(app.screen, ExportModal) and app.screen.snapshot.rows == ((22,),)
        await pilot.press("escape")
        assert app.session.pane_sizes["editor"] == initial
        x, y = title.region.x + 1, title.region.y + 2
        await pilot.mouse_down(title, offset=(1, 0))
        await pilot.mouse_up(offset=(x, y))
        assert app.session.pane_sizes["editor"] == initial + 2

        app.editor.load_text("SELECT 33 AS n; SELECT 44 AS n")
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert tabs.active == "tab-results"
        assert [tab.label_text for tab in tabs.query(Tab)][:3] == ["Results 1", "Results 2 ◆", "Results 3"]
        await pilot.click(tabs.get_tab(pinned))
        assert tabs.get_pane(pinned).query_one(ResultsTable).raw_rows == ((11,),)
        await pilot.click("#results-pin")
        assert tabs.get_tab(pinned).label_text == "Results 2"
        await pilot.click("#results-close")
        assert [tab.label_text for tab in tabs.query(Tab)] == ["Results 1", "Results 2", "SQL", "History"]
        await pilot.click(tabs.get_tab("tab-sql"))
        assert not app.query_one("#results-actions").display
        await pilot.click("#output-close")
        assert not handle.display and not app.query_one("#output-slot").display


@async_test
async def test_paged_d1_read_navigates_grid_not_editor(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    session.update_settings(default_rows=1)
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        sql = "SELECT id FROM records ORDER BY id LIMIT 2 OFFSET 1"
        app.editor.load_text(sql)
        await pilot.press("ctrl+enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        grid = app.query_one("#results-table", ResultsTable)
        assert grid.raw_rows == ((2,),) and app.query_one("#result-prev", Button).disabled
        assert str(app.query_one("#result-page-number", Static).content) == "Page 1"
        app.editor.load_text("DELETE FROM records")
        await pilot.click("#result-next")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert grid.raw_rows == ((3,),) and grid.page_offset == 1
        assert app.query_one("#result-next", Button).disabled
        assert str(app.query_one("#result-page-number", Static).content) == "Page 2"
        grid.action_export()
        assert isinstance(app.screen, ExportModal) and app.screen.snapshot.rows == ((3,),)
        await pilot.press("escape")
        grid.focus()
        await pilot.press("ctrl+left")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert grid.raw_rows == ((2,),) and grid.page_offset == 0
        assert [entry.doc for entry in session.history] == [sql]
        assert app.editor.text == "DELETE FROM records"
        assert d1_harness.connection.execute("SELECT count(*) FROM records").fetchone() == (3,)

        # The same pager remains visible, with both boundaries disabled, for a
        # result that fits. A stale Next must not fetch another page or script.
        app.editor.load_text("SELECT id FROM records WHERE id=1")
        await pilot.click("#run-query")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert grid.raw_rows == ((1,),) and app.query_one("#result-pager").display
        assert app.query_one("#result-prev", Button).disabled and app.query_one("#result-next", Button).disabled

        # The existing read-only shortcut stays read-only, even for a write.
        app.editor.load_text("DELETE FROM records")
        await pilot.press("shift+f5")
        await app.workers.wait_for_complete()
        assert app._main_outcome.error and not app.query_one("#result-pager").display
        assert d1_harness.connection.execute("SELECT count(*) FROM records").fetchone() == (3,)


@async_test
async def test_zero_row_summary_appears_only_in_results(parquet_source):
    app = D8RApp(Session({parquet_source.id: parquet_source}))
    async with app.run_test(size=(140, 45)) as pilot:
        app.editor.load_text("SELECT id FROM data WHERE false")
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        summary = str(app.query_one("#results-status", Static).content)
        assert "0 rows" in summary and " ms" in summary
        footer = str(app.query_one("#status", Static).content)
        assert "0 rows" not in footer and " ms" not in footer
        assert "running" not in footer.lower()


@async_test
async def test_sql_function_save_and_preview(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        app.action_query_to_fn()
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


@async_test
async def test_source_form_full_string_typing_and_control_focus(credential_backend):
    from textual.widgets import Checkbox

    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(140, 60)) as pilot:
        await pilot.press("ctrl+o")
        await pilot.pause()
        modal = app.screen
        for name, value in (("postgres-host", "db.example.test"),
                            ("postgres-database", "analytics"),
                            ("postgres-user", "readonly"),
                            ("postgres-password", "session-only-secret"),
                            ("display-name", "Primary database")):
            field = modal.query_one(f"#{name}", Input)
            field.value = ""
            field.focus()
            await pilot.pause()
            await pilot.press(*value)
            assert field.value == value
            assert modal.focused is field
        host = modal.query_one("#postgres-host", Input)
        host.focus()
        await pilot.press("tab")
        assert modal.focused is modal.query_one("#postgres-port", Input)
        sslmode = modal.query_one("#postgres-sslmode", Select)
        sslmode.focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert sslmode.value == "require"
        assert modal.focused is sslmode
        remember = modal.query_one("#remember-credentials", Checkbox)
        remember.focus()
        await pilot.press("space")
        assert not remember.value
        assert modal.focused is remember
        backend = modal.query_one("#source-backend", Select)
        backend.focus()
        await pilot.press("enter", "end", "enter")
        await pilot.pause()
        assert backend.value == "d1"
        assert modal.focused is backend
        token = modal.query_one("#d1-token", Input)
        token.focus()
        await pilot.press(*"session-token-only")
        assert token.value == "session-token-only"
        assert modal.focused is token
        await pilot.press("escape")
        await pilot.pause()
        assert app.screen is not modal
        assert token.value == ""
        assert not credential_backend.values
        assert not session.source_connected()


@async_test
async def test_source_edit_cancels_inflight_builder_and_restores_field(parquet_source, monkeypatch):
    from threading import Event
    from d8r.tui import connection

    session = Session()
    app = D8RApp(session)
    started, unblock, released = Event(), Event(), Event()
    build, release = session.build_parquet_source, connection.release_source
    built = []

    def delayed_build(*args):
        started.set()
        if not unblock.wait(10):
            raise RuntimeError("Test builder was not released")
        source = build(*args)
        built.append(source)
        return source

    def capture_release(source):
        release(source)
        released.set()

    monkeypatch.setattr(session, "build_parquet_source", delayed_build)
    monkeypatch.setattr(connection, "release_source", capture_release)
    async with app.run_test(size=(140, 48)) as pilot:
        app.action_add_source()
        await pilot.pause()
        modal = app.screen
        modal.query_one("#source-backend", Select).value = "parquet"
        field = modal.query_one("#parquet-path", Input)
        field.value = parquet_source.connection_key
        await pilot.pause()
        await pilot.click(field)
        original, selection = field.value, field.selection
        modal.action_test()
        try:
            assert await asyncio.to_thread(started.wait, 5)
            field.value += ".changed"
            await pilot.pause()
            assert not modal._busy
            assert modal.focused is field
            assert field.selection == selection
            await pilot.press(*"-still-editing")
            start, end = sorted(selection)
            assert field.value == original[:start] + "-still-editing" + original[end:] + ".changed"
        finally:
            unblock.set()
        assert await asyncio.to_thread(released.wait, 5)
        assert built[0].con is None
        assert modal._built is None
        assert not session.source_connected()
        await pilot.press("escape")


@async_test
async def test_new_query_shortcut_preserves_query_function_and_chat_drafts():
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT current_date")
        first_id = app._document_identity
        await pilot.press("ctrl+n")
        await pilot.pause()
        second_id = app._document_identity
        assert second_id != first_id
        assert app.editor.text == ""
        assert app.screen.focused is app.editor
        assert next(page for page in session.workspace["pages"] if page["id"] == first_id)["document"] == "SELECT current_date"
        app.action_add_source()
        await pilot.pause()
        modal = app.screen
        await pilot.press("ctrl+n")
        assert app.screen is modal
        assert app._document_identity == second_id
        await pilot.press("escape")
        await pilot.pause()
        app.editor.load_text("SELECT current_time")
        app.action_ai()
        await pilot.pause()
        composer = app.ai_panel.query_one("#ai-input", TextArea)
        composer.load_text("Keep this unsent question")
        composer.focus()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert app.editor.text == ""
        assert any(chat.input == "Keep this unsent question" for chat in app.ai_chats.chats.values())
        app.action_query_to_fn()
        await pilot.pause()
        app.function_editor.query_one("#fn-name", Input).value = "unsaved_function"
        app.function_editor.query_one("#fn-body", TextArea).load_text("SELECT 42")
        app.function_editor.query_one("#fn-name", Input).focus()
        await pilot.press("ctrl+n")
        await pilot.pause()
        assert not app.function_mode
        assert app.editor.text == ""
        assert session.workspace["function_draft"]["name"] == "unsaved_function"
        assert session.workspace["function_draft"]["body"] == "SELECT 42"
        assert not session.history


@async_test
async def test_bound_button_hover_displays_real_shortcuts():
    from textual.widgets import Button, Tooltip

    app = D8RApp(Session())
    app.TOOLTIP_DELAY = 0.01
    async with app.run_test(size=(160, 60), tooltips=True) as pilot:
        await pilot.pause()
        await pilot.hover("#query-new")
        await pilot.pause(0.1)
        tooltip = app.screen.query_one(Tooltip)
        assert tooltip.display
        assert "Ctrl+N" in str(tooltip.content)
        app.action_query_to_fn()
        await pilot.pause()
        await pilot.hover("#fn-preview")
        await pilot.pause(0.1)
        assert tooltip.display
        assert "Ctrl+R" in str(tooltip.content)
        assert "F5" in str(tooltip.content)
        app.action_ai()
        await pilot.pause()
        await pilot.hover("#ai-send")
        await pilot.pause(0.1)
        assert tooltip.display
        assert "Ctrl+Enter" in str(tooltip.content)
        assert app.query_one("#ai-cancel", Button).tooltip is None
        app.action_add_source()
        await pilot.pause()
        cancel = app.screen.query_one("#cancel", Button)
        cancel.scroll_visible(immediate=True)
        await pilot.pause()
        await pilot.hover(cancel)
        await pilot.pause(0.1)
        tooltip = app.screen.query_one(Tooltip)
        assert tooltip.display
        assert "Escape" in str(tooltip.content)
        await pilot.press("escape")
