"""Real Textual/Pilot completion coverage; providers are deterministic async seams."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps

import pytest
from textual.widgets import Input, Static, Tree

from d8r.ai import sql_completion
from d8r.ai.client import AIConfig
from d8r.ai.sql_completion import AICompletion
from d8r.tui.app import D8RApp
from d8r.tui.palette import CommandPalette
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


def configured(source, **settings):
    session = Session({source.id: source})
    session.update_settings(ai=AIConfig("https://provider.invalid/v1", "test", "private-test-key"),
                            persist_ai=False, **settings)
    return session


async def load_editor(app, pilot, document, cursor):
    app.editor.load_text(document)
    app.editor.move_cursor(cursor)
    app.editor.focus()
    await pilot.pause()


@pytest.mark.parametrize("document,cursor,insert,expected", [
    ("SELECT\n d.laxbel FROM data d WHERE id > 0", (1, 5), "label",
     "SELECT\n d.label FROM data d WHERE id > 0"),
    ('SELECT\n d."laxbel" FROM data d', (1, 6), '"label"',
     'SELECT\n d."label" FROM data d'),
    ("SELECT low('X') FROM data", (0, 10), "lower",
     "SELECT lower('X') FROM data"),
    ("SELECT * FROM da AS d", (0, 16), "data", "SELECT * FROM data AS d"),
])
@async_test
async def test_absolute_spans_suffixes_and_single_undo(parquet_source, document, cursor, insert, expected):
    app = D8RApp(configured(parquet_source))
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, cursor)
        await pilot.press("ctrl+space")
        view = app.palette.view
        assert view is not None
        index = next(i for i, entry in enumerate(view.entries)
                     if entry.insert == insert or (entry.kind == "function" and entry.label.casefold() == insert))
        app.palette.highlighted = index
        await pilot.press("tab")
        assert app.editor.text == expected
        app.palette.close()
        await pilot.press("ctrl+z")
        assert app.editor.text == document
        assert not app.session.history


@async_test
async def test_local_only_signature_and_stable_highlight(parquet_source, monkeypatch):
    requests = []

    async def provider(*args):
        requests.append(args)
        return ()

    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    app = D8RApp(configured(parquet_source))
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, "SELECT l FROM data", (0, 8))
        await pilot.press("a")
        await pilot.pause(0.85)
        assert not requests  # Real typing remains local while automatic AI is off.
        await pilot.press("ctrl+space")
        assert app.palette.view is not None
        app.palette.highlighted = min(1, len(app.palette.view.entries) - 1)
        chosen = app.palette.view.entries[app.palette.highlighted].identity
        app.palette.sync(manual=True)
        assert app.palette.view.entries[app.palette.highlighted].identity == chosen
        await load_editor(app, pilot, "SELECT coalesce(1, ", (0, 19))
        await pilot.press("ctrl+space")
        assert app.palette.view is not None and app.palette.view.signature is not None
        assert app.palette.view.signature.active_parameter == 1
        assert "coalesce" in app.palette.details.content.plain.casefold()
        await pilot.press("escape", "left", "right")
        assert not app.palette.is_open
        assert not requests
        app.update_settings(intellisense=False)
        await pilot.press("ctrl+space", "ctrl+shift+space")
        assert not app.palette.is_open and not requests


@async_test
async def test_ai_multiline_preview_manual_acceptance_and_permissions(parquet_source, monkeypatch):
    requests = []
    ready = asyncio.Event()

    async def provider(config, request):
        requests.append(request)
        ready.set()
        return (AICompletion("label,\n  id", "Two cached columns"),)

    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    session = configured(parquet_source, ai_allow_schema_refreshes=False, ai_auto_accept=True)
    app = D8RApp(session)
    original = "SELECT la FROM data"
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, original, (0, 9))
        await pilot.press("f4")
        await asyncio.wait_for(ready.wait(), 2)
        await pilot.pause()
        assert len(requests) == 1 and requests[0].allow_schema is False
        assert requests[0].dialect == "duckdb"
        assert (requests[0].start, requests[0].end) == (7, 9)
        assert app.editor.text == original
        assert app.palette.view.phase == "ai"
        assert "label,\n  id" in app.palette.details.content.plain
        assert app.palette.details.region.bottom <= app.editor.content_region.bottom
        await pilot.press("tab")
        assert app.editor.text == "SELECT label,\n  id FROM data"
        assert app.editor.cursor_location == (1, 4)
        await pilot.press("ctrl+z")
        assert app.editor.text == original
        assert not session.history and not session.fns


@async_test
async def test_automatic_ai_only_after_real_typing_idle(parquet_source, monkeypatch):
    requests = []
    ready = asyncio.Event()

    async def provider(config, request):
        requests.append(request)
        ready.set()
        return (AICompletion("label"),)

    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    previous = configured(parquet_source, ai_completion=True)
    previous.save_workspace(document="SELECT la FROM data", source=previous.source_key(),
                            dialect="duckdb", cursor=[0, 9])
    session = Session({parquet_source.id: parquet_source})
    assert session.ai_completion
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.pause(0.85)
        assert app.editor.text == "SELECT la FROM data"
        assert not requests  # Stored nonempty startup draft is not typing.
        app._restore_page({"id": "restored-offline-page", "document": "SELECT l FROM data",
                           "cursor": [0, 8], "source": session.source_key(), "dialect": "duckdb"})
        app.editor.focus()
        await pilot.pause()
        await pilot.press("left", "right")
        await pilot.pause(0.85)
        assert not requests
        await pilot.press("a")
        await pilot.pause(0.1)
        assert not requests
        await asyncio.wait_for(ready.wait(), 3)
        await pilot.pause()
        assert len(requests) == 1
        assert app.editor.text == "SELECT la FROM data"
        assert app.palette.view.phase == "ai"
        await pilot.press("escape")
        await pilot.press("b")
        await pilot.pause(0.85)
        assert len(requests) == 1  # Escape stays dismissed while typing.
        assert not app.palette.is_open


@pytest.mark.parametrize("change", ["escape", "text", "caret", "source", "schema", "functions",
                                    "provider", "page", "mode", "off", "automatic-off", "blur"])
@async_test
async def test_late_ai_response_is_never_published(parquet_source, monkeypatch, change):
    started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def provider(config, request):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            # Simulate a transport that races cancellation and still completes.
            await release.wait()
        completed.set()
        return (AICompletion("label", "obsolete provider response"),)

    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    session = configured(parquet_source, ai_completion=change == "automatic-off")
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, "SELECT la FROM data", (0, 9))
        await pilot.press("ctrl+shift+space")
        await asyncio.wait_for(started.wait(), 2)
        if change == "escape":
            await pilot.press("escape")
        elif change == "text":
            await pilot.press("x", "ctrl+z")
        elif change == "caret":
            await pilot.press("left")
        elif change == "source":
            session.sources[session.active_id] = replace(parquet_source)
        elif change == "schema":
            session.refresh_schema()
        elif change == "functions":
            session.save_fn("local_rows", "", "SELECT id FROM data", "")
        elif change == "provider":
            session.ai_config = replace(session.ai_config, model="other-provider-model")
        elif change == "page":
            app._document_identity = "different-page-with-identical-text"
        elif change == "mode":
            app.action_fn()
        elif change == "off":
            app.update_settings(intellisense=False)
        elif change == "automatic-off":
            app.update_settings(ai_completion=False)
        elif change == "blur":
            app.query_one("#results-table").focus()
        expected = app.editor.text
        await pilot.pause(0.15)
        release.set()
        await asyncio.wait_for(completed.wait(), 2)
        await pilot.pause()
        assert app.editor.text == expected
        assert app.palette.view is None or app.palette.view.phase != "ai"
        assert "obsolete provider response" not in app.palette._detail_content().plain
        assert not session.history


@async_test
async def test_function_parameters_invalidate_ai_and_keep_query_isolated(parquet_source, d1_source, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    requests = []

    async def provider(config, request):
        requests.append(request)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            await release.wait()
        return (AICompletion("@minimum"),)

    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    session = configured(parquet_source)
    session.register(d1_source, activate=False)
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, "SELECT id FROM data", (0, 7))
        app.action_fn()
        await pilot.pause()
        form = app.function_editor
        form._select_target(d1_source.id)
        form.query_one("#fn-params", Input).value = "minimum"
        body = form.query_one("#fn-body")
        body.load_text("SELECT @mi")
        body.move_cursor((0, 10))
        body.focus()
        await pilot.pause()
        await pilot.press("ctrl+shift+space")
        await asyncio.wait_for(started.wait(), 2)
        assert requests[0].parameters == ("minimum",)
        assert requests[0].dialect == "sqlite"
        form.query_one("#fn-params", Input).value = "other"
        await pilot.pause()
        release.set()
        await pilot.pause()
        palette = form.query_one(CommandPalette)
        assert palette.view is None or palette.view.phase != "ai"
        assert body.text == "SELECT @mi"
        assert app.editor.text == "SELECT id FROM data"
        assert not session.functions_for(d1_source.id)


@async_test
async def test_provider_errors_are_sanitized_and_exit_cancels(parquet_source, monkeypatch):
    async def failing(config, request):
        raise RuntimeError("secret-api-key https://user:password@provider.invalid/private")

    monkeypatch.setattr(sql_completion, "complete_sql", failing)
    app = D8RApp(configured(parquet_source))
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, "SELECT la FROM data", (0, 9))
        await pilot.press("ctrl+shift+space")
        await pilot.pause()
        text = app.palette.details.content.plain
        assert "unavailable" in text
        assert "secret-api-key" not in text and "password" not in text
        started, cancelled = asyncio.Event(), asyncio.Event()

        async def pending(config, request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        monkeypatch.setattr(sql_completion, "complete_sql", pending)
        await pilot.press("ctrl+shift+space")
        await asyncio.wait_for(started.wait(), 2)
        app.exit()
        await asyncio.wait_for(cancelled.wait(), 2)


@async_test
async def test_cached_local_completion_remains_available_while_query_busy(parquet_source, monkeypatch):
    session = configured(parquet_source)

    def forbidden(*args, **kwargs):
        raise AssertionError("Completion must not acquire a database connection")

    monkeypatch.setattr(session, "distinct_values", forbidden)
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        session.busy = "Query running"
        await load_editor(app, pilot, "SEL", (0, 3))
        await pilot.press("ctrl+space")
        assert "SELECT" in app.palette.view.labels
        await load_editor(app, pilot, "SELECT d.la FROM data d", (0, 11))
        await pilot.press("ctrl+space")
        assert any(entry.insert == "label" for entry in app.palette.view.entries)
        assert app.palette.is_open
        assert not session.history
        session.busy = ""


@pytest.mark.parametrize("document", [
    "SELECT sum(id) FROM data",
    "SELECT suum(id) FROM data",
    "SELECT suum /*a /*nested*/ b*/ (id) FROM data",
])
@async_test
async def test_accepting_existing_call_enters_arguments_without_duplication(parquet_source, document):
    app = D8RApp(configured(parquet_source))
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, 9))
        await pilot.press("ctrl+space")
        app.palette.highlighted = next(i for i, entry in enumerate(app.palette.view.entries)
                                       if entry.kind == "function" and entry.label.casefold() == "sum")
        await pilot.press("tab")
        expected = document.replace("suum", "sum", 1)
        assert app.editor.text == expected
        assert app.editor.cursor_location == (0, expected.index("(id)") + 1)
        if expected != document:
            app.palette.close()
            await pilot.press("ctrl+z")
            assert app.editor.text == document
        assert not app.session.history


@async_test
async def test_long_ai_preview_scrolls_without_focus_or_hidden_acceptance(parquet_source, monkeypatch):
    insertion = "label\n" + "\n".join(f"-- explanatory line {index}" for index in range(18)) + "\nAS final_probe"
    async def provider(*args):
        return (AICompletion(insertion, "Multiline SQL continuation"),)
    monkeypatch.setattr(sql_completion, "complete_sql", provider)
    app = D8RApp(configured(parquet_source))
    document = "SELECT la FROM data\n" + "\n".join("-- remaining document" for _ in range(50))
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, 9))
        await pilot.press("f4")
        await pilot.pause()
        details = app.palette.details
        assert app.palette.view.phase == "ai" and details.display
        for _ in range(12):
            await pilot.press("alt+down")
        await pilot.pause()
        assert "final_probe" in app.export_screenshot()
        assert app.focused is app.editor and app.editor.text == document
        app.editor.scroll_to(y=30, animate=False)
        await pilot.pause()
        assert not details.display
        await pilot.press("tab")
        assert app.editor.text == document


def tree_nodes(node):
    for child in node.children:
        yield child
        yield from tree_nodes(child)


async def select_tree_node(app, pilot, node):
    parent = node.parent
    while parent is not None:
        parent.expand()
        parent = parent.parent
    await pilot.pause()
    tree = app.query_one("#schema-tree", Tree)
    tree.focus()
    tree.move_cursor(node)
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()


@pytest.mark.parametrize("source_fixture", ["parquet_source", "d1_source"])
@async_test
async def test_explorer_table_acceptance_uses_identity_and_native_alias(request, source_fixture):
    from d8r.sql import relation_parts
    source = request.getfixturevalue(source_fixture)
    session = Session({source.id: source})
    app = D8RApp(session)
    table = session.schema.tables[0]
    name = relation_parts(table.name, dialect=source.dialect)[-1]
    document = "SELECT * FROM " + name[:2]
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, len(document)))
        tree = app.query_one("#schema-tree", Tree)
        node = next(node for node in tree_nodes(tree.root) if isinstance(node.data, dict)
                    and node.data.get("kind") == "dataset" and node.data.get("name") == table.name)
        await select_tree_node(app, pilot, node)
        assert " AS " in app.editor.text
        assert app.focused is app.editor and not session.history
        result = session.run(app.editor.text, record=False)
        assert result.ok and result.rows == [[1, "alpha"], [2, "beta"], [3, None]]


@async_test
async def test_explorer_column_uses_caret_statement_not_prior_alias(parquet_source):
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    document = "SELECT a.id FROM data a;\nSELECT  FROM data b"
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (1, len("SELECT ")))
        tree = app.query_one("#schema-tree", Tree)
        next(node for node in tree_nodes(tree.root) if isinstance(node.data, dict)
             and node.data.get("kind") == "dataset").expand()
        await pilot.pause()
        node = next(node for node in tree_nodes(tree.root) if isinstance(node.data, dict)
                    and node.data.get("kind") == "column" and node.data.get("name") == "label")
        await select_tree_node(app, pilot, node)
        assert app.editor.text == 'SELECT a.id FROM data a;\nSELECT b."label" FROM data b'
        results = session.run_all(app.editor.text, record=False)
        assert all(result.ok for result in results)
        assert results[-1].rows == [["alpha"], ["beta"], [None]]


@async_test
async def test_explorer_qualified_postgres_table_keeps_schema_prefix(postgres_source):
    from d8r.engine import execute_sql, refresh_source_schema
    from d8r.sql import relation_parts, quote_identifier
    source = postgres_source
    execute_sql(source, "CREATE TEMP TABLE explorer_probe(id INTEGER)")
    execute_sql(source, "INSERT INTO explorer_probe VALUES (42)")
    refresh_source_schema(source)
    session = Session({source.id: source})
    table = session.schema.table_by_name("explorer_probe")
    namespace = relation_parts(table.name)[0]
    document = "SELECT * FROM " + quote_identifier(namespace) + ".expl"
    app = D8RApp(session)
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, len(document)))
        tree = app.query_one("#schema-tree", Tree)
        node = next(node for node in tree_nodes(tree.root) if isinstance(node.data, dict)
                    and node.data.get("kind") == "dataset" and node.data.get("name") == table.name)
        await select_tree_node(app, pilot, node)
        result = session.run(app.editor.text, record=False)
        assert result.ok and result.rows == [[42]]


@async_test
async def test_typed_join_keyword_and_default_alias_acceptance(parquet_source):
    app = D8RApp(configured(parquet_source))
    document = "SELECT * FROM data d "
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, len(document)))
        await pilot.press("i", "n", "n", "tab", "d", "a", "tab")
        assert app.editor.text == document + "INNER JOIN data AS d2"
        app.editor.insert(" ON d.id = d2.id")
        result = app.session.run(app.editor.text, record=False)
        assert result.ok and len(result.rows) == 3


@async_test
async def test_alias_prefix_offers_all_fields_and_undoes_once(parquet_source):
    app = D8RApp(configured(parquet_source))
    document = "SELECT c FROM data c"
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, len("SELECT c")))
        await pilot.press("ctrl+space")
        entries = app.palette.view.entries
        assert {"c.id", "c.label"} <= {entry.insert for entry in entries}
        app.palette.highlighted = next(i for i, entry in enumerate(entries) if entry.insert == "c.label")
        await pilot.press("tab")
        assert app.editor.text == "SELECT c.label FROM data c"
        assert app.session.run(app.editor.text, record=False).rows == [["alpha"], ["beta"], [None]]
        await pilot.press("ctrl+z")
        assert app.editor.text == document


@async_test
async def test_grouping_hint_and_full_qualified_star_group_acceptance(parquet_source):
    app = D8RApp(configured(parquet_source))
    document = "SELECT d.*, count(*) AS n FROM data d"
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, document, (0, len(document)))
        await pilot.press("ctrl+enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "GROUP BY" in str(app.query_one("#results-error", Static).content)
        app.editor.insert(" GROUP BY ")
        await pilot.press("ctrl+space", "tab", "ctrl+enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert "GROUP BY d.id, d.label" in app.editor.text
        assert app._main_outcome.ok
        assert sorted(app._main_outcome.rows, key=lambda row: row[0]) == [
            [1, "alpha", 1], [2, "beta", 1], [3, None, 1]]


@async_test
async def test_sql_colors_refresh_on_edit_undo_redo_and_load(parquet_source):
    app = D8RApp(configured(parquet_source))
    async with app.run_test(size=(140, 45)) as pilot:
        await load_editor(app, pilot, "SELECT 1", (0, 6))

        def color():
            return app.editor.get_line(0).get_style_at_offset(app.console, 1)

        keyword = color()
        app.editor.history.checkpoint()
        app.editor.replace("custom", (0, 0), (0, 6))
        app.editor.history.checkpoint()
        identifier = color()
        assert identifier != keyword
        app.editor.undo()
        assert color() == keyword
        app.editor.redo()
        assert color() == identifier
        app.editor.load_text("SELECT 2")
        assert color() == keyword
