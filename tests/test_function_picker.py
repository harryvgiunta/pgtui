"""Function picker keyboard insertion, explicit management and draft isolation."""
from __future__ import annotations

import asyncio
from functools import wraps

import pytest
from textual.widgets import Button, DataTable, Input, OptionList, TextArea

from d8r.tui.app import D8RApp
from d8r.tui.fn import FnExplorer, FunctionPicker
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


def library(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    session.save_fn("rows_above", "minimum", "SELECT * FROM data WHERE id > @minimum", "Filter rows")
    session.save_fn("all_rows", "", "SELECT * FROM data", "Everything")
    session.save_fn("project", "source:relation, field:identifier", "SELECT @field FROM @source", "Column projection")
    return session


@async_test
async def test_search_down_enter_inserts_at_captured_caret_and_undo(parquet_source):
    session = library(parquet_source)
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        original = "SELECT * FROM  WHERE true"
        app.editor.load_text(original)
        app.editor.move_cursor((0, 14))
        app.editor.focus()
        app.action_fn()
        await pilot.pause()
        assert not app.function_mode
        search = app.query_one("#fn-search", Input)
        assert search.has_focus
        search.value = "filter minimum"
        await pilot.pause()
        listing = app.query_one("#fn-list", OptionList)
        assert listing.option_count == 1
        assert listing.get_option_at_index(0).id == "rows_above"
        await pilot.press("down", "enter")
        await pilot.pause()
        assert app.editor.text == "SELECT * FROM rows_above() WHERE true"
        assert app.editor.cursor_location == (0, 25)
        assert app.editor.has_focus
        assert not session.history
        app.editor.action_undo()
        assert app.editor.text == original



@pytest.mark.parametrize("selection", ["keyboard", "mouse"])
@async_test
async def test_inline_fn_first_selection_survives_startup_reconnect(parquet_source, monkeypatch, selection):
    from threading import Event
    from d8r.engine import close_source

    previous = library(parquet_source)
    previous.update_settings(source=parquet_source.id)
    original = "SELECT * FROM  WHERE true"
    previous.save_workspace(document=original, source=previous.source_key(), dialect="duckdb", cursor=[0, 14])
    session = Session()
    started, unblock = Event(), Event()
    build = session.build_parquet_source

    def delayed_build(*args):
        started.set()
        if not unblock.wait(10):
            raise RuntimeError("Test connection was not released")
        return build(*args)

    monkeypatch.setattr(session, "build_parquet_source", delayed_build)
    app = D8RApp(session)
    try:
        async with app.run_test(size=(150, 48)) as pilot:
            assert await asyncio.to_thread(started.wait, 5)
            assert not session.source_connected()
            await pilot.press("backslash", "f", "n", "enter")
            search = app.query_one("#fn-search", Input)
            assert search.has_focus
            search.value = "rows_above"
            await pilot.pause()
            unblock.set()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert session.source_connected()
            assert session.source_key() == previous.source_key()
            if selection == "keyboard":
                await pilot.press("down", "enter")
            else:
                await pilot.click("#fn-list", offset=(2, 0))
            expected = "SELECT * FROM rows_above() WHERE true"
            assert app.editor.text == expected
            assert app.editor.cursor_location == (0, expected.index(")"))
            assert app.editor.has_focus
            assert not session.history and app._main_outcome is None
            app.palette.close()
            await pilot.press("ctrl+z")
            assert app.editor.text == original
    finally:
        unblock.set()
        for source in session.sources.values():
            close_source(source)

@async_test
async def test_no_argument_call_caret_finishes_after_parentheses(parquet_source):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 48)) as pilot:
        app.editor.load_text("SELECT * FROM ")
        app.editor.move_cursor((0, 14))
        app.action_fn()
        await pilot.pause()
        app.query_one("#fn-search", Input).value = "everything"
        await pilot.pause()
        await pilot.press("down", "enter")
        assert app.editor.text == "SELECT * FROM all_rows()"
        assert app.editor.cursor_location == (0, 24)


@async_test
async def test_filter_searches_typed_signature_and_empty_results_are_safe(parquet_source):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 48)) as pilot:
        app.action_fn()
        await pilot.pause()
        search = app.query_one("#fn-search", Input)
        search.value = "RELATION projection"
        await pilot.pause()
        listing = app.query_one("#fn-list", OptionList)
        assert listing.option_count == 1
        assert listing.get_option_at_index(0).id == "project"
        search.value = "no such saved definition"
        await pilot.pause()
        assert listing.option_count == 0
        assert app.query_one("#fn-edit-selected", Button).disabled
        await pilot.press("down", "enter")
        assert app.editor.text == ""
        assert not app.function_mode


@pytest.mark.parametrize("chat_picker", [False, True])
@async_test
async def test_up_from_first_function_or_empty_list_returns_to_search(parquet_source, chat_picker):
    session = library(parquet_source)
    app = D8RApp(session)
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT 99")
        if chat_picker:
            app.action_ai()
            await pilot.pause()
            await pilot.press("backslash", "f", "n")
        else:
            app.action_fn()
        await pilot.pause()
        search = app.screen.query_one("#fn-search", Input)
        listing = app.screen.query_one("#fn-list", OptionList)
        search.value = "rows"
        await pilot.pause()
        await pilot.press("down")
        assert listing.has_focus and listing.highlighted == 0
        await pilot.press("down")
        assert listing.has_focus and listing.highlighted == 1
        await pilot.press("up")
        assert listing.has_focus and listing.highlighted == 0
        await pilot.press("up")
        assert search.has_focus and search.value == "rows"
        assert listing.highlighted == 0
        await pilot.press("ctrl+shift+a", *"project")
        await pilot.pause()
        assert search.value == "project"
        assert listing.get_option_at_index(0).id == "project"
        await pilot.press("down", "up")
        assert search.has_focus
        search.value = "no matching function"
        await pilot.pause()
        await pilot.press("down")
        assert listing.has_focus and listing.highlighted is None
        await pilot.press("up")
        assert search.has_focus and search.value == "no matching function"
        assert app.editor.text == "SELECT 99" and not session.history
        if chat_picker:
            assert app.ai_panel.chat.input == "\\fn" and not app.ai_panel.chat.messages
        await pilot.press("escape")
        await pilot.pause()
        assert app.editor.has_focus


@async_test
async def test_edit_new_and_query_conversion_are_explicit(parquet_source):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 48)) as pilot:
        app.editor.load_text("SELECT 42 AS answer")
        app.action_fn()
        await pilot.pause()
        app.query_one("#fn-search", Input).value = "project"
        await pilot.pause()
        app.query_one("#fn-edit-selected", Button).press()
        await pilot.pause()
        assert app.function_mode
        form = app.function_editor
        assert form.selected == "project"
        assert form.query_one("#fn-params", Input).value == "source:relation, field:identifier"
        assert form._parameter_names() == ["source", "field"]
        assert form._ai_target().parameters == ("source:relation", "field:identifier")
        before = form._ai_target()
        form.query_one("#fn-params", Input).value = "source:relation, field"
        assert form._ai_target() != before
        app.query_one("#fn-new", Button).press()
        await pilot.pause()
        assert form.query_one("#fn-name", Input).value == ""
        assert form.query_one("#fn-body", TextArea).text == ""
        app.action_workspace()
        app.action_fn()
        await pilot.pause()
        app.query_one("#query-to-fn", Button).press()
        await pilot.pause()
        assert app.function_mode
        assert form.query_one("#fn-body", TextArea).text == "SELECT 42 AS answer"
        assert app.editor.text == "SELECT 42 AS answer"
        assert not app.session.history


@async_test
async def test_picker_inserts_into_function_body_without_touching_query(parquet_source):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 48)) as pilot:
        app.editor.load_text("SELECT 99")
        app._open_fn(restore_draft=False)
        await pilot.pause()
        body = app.function_editor.query_one("#fn-body", TextArea)
        body.load_text("SELECT * FROM ")
        body.move_cursor((0, 14))
        body.focus()
        app.action_fn()
        await pilot.pause()
        app.query_one("#fn-search", Input).value = "rows_above"
        await pilot.pause()
        await pilot.press("down", "enter")
        assert body.text == "SELECT * FROM rows_above()"
        assert body.cursor_location == (0, 25)
        assert app.editor.text == "SELECT 99"
        body.action_undo()
        assert body.text == "SELECT * FROM "


@pytest.mark.parametrize("change", ["text", "reload", "caret", "page", "source"])
@async_test
async def test_stale_picker_cannot_modify_replaced_editor(parquet_source, d1_source, change):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 48)) as pilot:
        app.editor.load_text("SELECT * FROM ")
        app.editor.move_cursor((0, 14))
        app.action_fn()
        await pilot.pause()
        app.query_one("#fn-search", Input).value = "all_rows"
        await pilot.pause()
        if change == "text":
            app.editor.load_text("SELECT * FROM  WHERE true")
            app.editor.move_cursor((0, 14))
        elif change == "reload":
            app.editor.load_text(app.editor.text)
            app.editor.move_cursor((0, 14))
        elif change == "caret":
            app.editor.move_cursor((0, 0))
        elif change == "page":
            app._load_document(app.editor.text)
            app.editor.move_cursor((0, 14))
        else:
            app.session.register(d1_source)
            app.session.save_fn("all_rows", "", "SELECT * FROM records", "Everything")
        unchanged = app.editor.text
        app.query_one(FnExplorer).focus_search()
        await pilot.press("down", "enter", "enter")
        assert app.editor.text == unchanged
        assert not app.session.history


@async_test
async def test_quality_button_stores_only_and_collision_is_atomic(parquet_source, monkeypatch):
    import d8r.tui.session as session_module

    def forbidden(*args, **kwargs):
        raise AssertionError("Quality installation must not run SQL")

    monkeypatch.setattr(session_module, "execute_sql", forbidden)
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        assert not session.fns
        app.action_fn()
        await pilot.pause()
        app.query_one("#fn-install-quality", Button).press()
        await pilot.pause()
        assert "dq_row_count" in session.fns
        assert session.fns["dq_row_count"].param_types == ("relation",)
        installed = dict(session.fns)
        app.query_one("#fn-install-quality", Button).press()
        await pilot.pause()
        assert session.fns == installed
        assert app.editor.text == ""
        assert not session.history


@pytest.mark.parametrize("source_fixture", ["parquet_source", "d1_source", "postgres_source"])
@async_test
async def test_quality_picker_starts_a_runnable_query(request, source_fixture):
    source = request.getfixturevalue(source_fixture)
    table = "data" if source.dialect == "duckdb" else "records"
    if source.dialect == "postgres":
        source.con.execute("CREATE TEMP TABLE quality_picker_actor(id integer)")
        source.con.execute("INSERT INTO quality_picker_actor VALUES (1), (2), (3)")
        table = "pg_temp.quality_picker_actor"
    session = Session({source.id: source})
    session.install_quality_functions()
    app = D8RApp(session)
    async with app.run_test(size=(150, 48)) as pilot:
        app.action_fn()
        await pilot.pause()
        app.query_one("#fn-search", Input).value = "dq_row_count"
        await pilot.pause()
        await pilot.press("down", "enter")
        assert app.editor.text == "SELECT * FROM dq_row_count();"
        assert app.editor.cursor_location == (0, len("SELECT * FROM dq_row_count("))
        assert not session.history
        app.editor.action_undo()
        assert app.editor.text == ""
        app.editor.action_redo()
        app.editor.move_cursor((0, len("SELECT * FROM dq_row_count(")))
        app.editor.insert(table)
        await pilot.press("f5")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app._main_outcome.ok, app._main_outcome.error
        assert app._main_outcome.rows == [[3]]
        assert [str(cell) for cell in app.query_one("#results-table", DataTable).get_row_at(0)] == ["3"]


@pytest.mark.parametrize("function_mode,name,search", [
    (False, "rows_above", "filter minimum"),
    (True, "all_rows", "everything"),
])
@async_test
async def test_chat_function_command_replaces_only_command_and_preserves_editors(
    parquet_source, monkeypatch, function_mode, name, search,
):
    session = library(parquet_source)
    app = D8RApp(session)

    def forbidden(*args, **kwargs):
        pytest.fail("Choosing a chat function must not send a request or execute SQL")

    monkeypatch.setattr(app.ai_chats, "start", forbidden)
    monkeypatch.setattr("d8r.tui.session.execute_sql", forbidden)
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT 99")
        if function_mode:
            app._open_fn(focus="project", restore_draft=False)
            await pilot.pause()
        body = app.function_editor.query_one("#fn-body", TextArea).text
        functions = dict(session.fns)
        app.action_ai()
        await pilot.pause()
        panel = app.ai_panel
        composer = panel.query_one("#ai-input", TextArea)
        composer.load_text("Please explain  on this source.")
        composer.move_cursor((0, len("Please explain ")))
        composer.focus()
        await pilot.press("backslash", "f", "n")
        await pilot.pause()
        assert isinstance(app.screen, FunctionPicker)
        assert app.screen.query_one("#fn-search", Input).has_focus
        assert not app.screen.query("#fn-install-quality, #fn-edit-selected, #fn-new")
        app.screen.query_one("#fn-search", Input).value = search
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        expected = f"Please explain {name}() on this source."
        assert composer.text == expected
        assert composer.has_focus
        assert composer.cursor_location == (0, len("Please explain ") + len(name) + (1 if session.fns[name].params else 2))
        assert panel.chat.input == expected
        assert session.load_chats()[panel.chat.id]["input"] == expected
        assert app.editor.text == "SELECT 99"
        assert app.function_editor.query_one("#fn-body", TextArea).text == body
        assert session.fns == functions
        assert not session.history and not panel.chat.messages
        composer.action_undo()
        assert composer.text == "Please explain \\fn on this source."
        await pilot.pause()
        assert not isinstance(app.screen, FunctionPicker)


@pytest.mark.parametrize("key", ["enter", "ctrl+enter"])
@pytest.mark.parametrize("function_mode", [False, True])
@async_test
async def test_restored_chat_command_requires_explicit_activation_and_can_cancel(parquet_source, monkeypatch, key, function_mode):
    app = D8RApp(library(parquet_source))
    monkeypatch.setattr(app.ai_chats, "start", lambda *args: pytest.fail("The local command must not send"))
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT 99")
        if function_mode:
            app._open_fn(focus="rows_above", restore_draft=False)
            await pilot.pause()
        app.action_ai()
        await pilot.pause()
        composer = app.ai_panel.query_one("#ai-input", TextArea)
        composer.load_text("\\fn")
        composer.move_cursor((0, 3))
        await pilot.pause()
        assert not isinstance(app.screen, FunctionPicker)
        await pilot.press(key)
        await pilot.pause()
        assert isinstance(app.screen, FunctionPicker)
        app.screen.query_one("#fn-search", Input).value = "no matching function"
        await pilot.pause()
        assert app.screen.query_one("#fn-list", OptionList).option_count == 0
        assert app.screen.query_one("#fn-picker-empty").display
        chat = app.ai_panel.chat
        app.screen.query_one("#fn-search" if key == "enter" else "#fn-list").focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.editor.has_focus and not app.function_mode
        assert app.editor.text == "SELECT 99"
        assert chat.input == "\\fn"
        assert app.session.load_chats()[chat.id]["input"] == "\\fn"
        assert not chat.messages and not app.session.history


@pytest.mark.parametrize("change", ["chat", "source", "reload", "definition", "hide"])
@async_test
async def test_chat_function_selection_rejects_changed_target(parquet_source, d1_source, change):
    session = library(parquet_source)
    session.register(d1_source, activate=False)
    with session.target_source(d1_source.id):
        session.save_fn("other_source_only", "", "SELECT 1", "Another library")
    app = D8RApp(session)
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT 99")
        app.action_ai()
        await pilot.pause()
        panel = app.ai_panel
        composer = panel.query_one("#ai-input", TextArea)
        await pilot.press("backslash", "f", "n")
        await pilot.pause()
        picker = app.screen
        assert isinstance(picker, FunctionPicker)
        listing = picker.query_one("#fn-list", OptionList)
        assert {listing.get_option_at_index(i).id for i in range(listing.option_count)} == {
            "rows_above", "all_rows", "project",
        }
        if change == "chat":
            panel.action_clear()
        elif change == "source":
            session.register(d1_source)
            panel.target_changed()
        elif change == "reload":
            composer.load_text("\\fn")
            composer.move_cursor((0, 3))
        elif change == "definition":
            session.save_fn("rows_above", "minimum", "SELECT * FROM data WHERE id >= @minimum", "Changed")
        else:
            panel.action_close()
        expected = composer.text
        picker.query_one("#fn-search", Input).value = "rows_above"
        picker.query_one("#fn-search", Input).focus()
        await pilot.pause()
        await pilot.press("down", "enter")
        await pilot.pause()
        assert composer.text == expected
        assert app.editor.text == "SELECT 99"
        assert not session.history


@pytest.mark.parametrize("prefix", ["Use C:\\", "Use \\"])
@async_test
async def test_chat_function_command_does_not_hijack_paths_or_escaped_text(parquet_source, prefix):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 55)) as pilot:
        app.action_ai()
        await pilot.pause()
        composer = app.ai_panel.query_one("#ai-input", TextArea)
        composer.load_text(prefix)
        composer.move_cursor((0, len(prefix)))
        if prefix == "Use \\":
            await pilot.press("backslash")
        await pilot.press("f", "n")
        await pilot.pause()
        assert not isinstance(app.screen, FunctionPicker)
        assert composer.text.endswith("\\fn")
        assert not app.ai_panel.chat.messages


@pytest.mark.parametrize("focus", [
    "#fn-search", "#fn-list", "#fn-name", "#fn-args", "#fn-body", "#ai-input", "#ai-send",
])
@async_test
async def test_escape_returns_to_query_without_losing_function_or_chat_drafts(parquet_source, monkeypatch, focus):
    session = library(parquet_source)
    functions = dict(session.fns)
    app = D8RApp(session)

    def forbidden(*args, **kwargs):
        pytest.fail("Escape must not send a request or execute SQL")

    monkeypatch.setattr(app.ai_chats, "start", forbidden)
    monkeypatch.setattr("d8r.tui.session.execute_sql", forbidden)
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT 99")
        page_id = app._document_identity
        app._open_fn(focus="rows_above", restore_draft=False)
        await pilot.pause()
        draft = "SELECT * FROM data WHERE id >= @minimum"
        app.function_editor.query_one("#fn-body", TextArea).load_text(draft)
        chat = None
        if focus in {"#fn-search", "#fn-list"}:
            app.action_fn()
        elif focus.startswith("#ai-"):
            app.action_ai()
            await pilot.pause()
            app.ai_panel.query_one("#ai-input", TextArea).load_text("Keep this unsent chat draft")
            await pilot.pause()
            chat = app.ai_panel.chat
        await pilot.pause()
        app.query_one(focus).focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert app.editor.has_focus and not app.function_mode
        assert app.editor.text == "SELECT 99" and app._document_identity == page_id
        assert session.workspace["function_draft"]["body"] == draft
        assert session.fns == functions
        assert not session.history
        assert app.query_one("#explorer-slot").display == (focus not in {"#fn-search", "#fn-list"})
        if focus.startswith("#fn-"):
            assert app.query_one("#explorer-tabs").active != "tab-functions"
        if chat is not None:
            assert not app.query_one("#assistant-slot").display
            assert chat.input == "Keep this unsent chat draft"
            assert session.load_chats()[chat.id]["input"] == chat.input
            assert not chat.messages


@async_test
async def test_chat_escape_returns_focus_even_when_preferences_cannot_save(parquet_source, monkeypatch):
    app = D8RApp(library(parquet_source))
    async with app.run_test(size=(150, 55)) as pilot:
        app.editor.load_text("SELECT 99")
        app.action_ai()
        await pilot.pause()
        app.ai_panel.query_one("#ai-input", TextArea).load_text("Keep this unsent chat draft")
        await pilot.pause()
        chat = app.ai_panel.chat

        def unavailable(**changes):
            raise ValueError("Preference storage unavailable")

        monkeypatch.setattr(app.session, "update_settings", unavailable)
        await pilot.press("escape")
        await pilot.pause()
        assert app.editor.has_focus
        assert app.ai_panel.display
        assert chat.input == "Keep this unsent chat draft"
        assert app.editor.text == "SELECT 99"
        assert not chat.messages and not app.session.history
