"""Inline folds change visual rows, never the source SQL under selection or Run."""
import asyncio

import pytest
from textual.widgets import Input

from d8r.sql import lex_sql
from d8r.tui.app import D8RApp
from d8r.tui.query_editor import _table_offsets
from d8r.tui.query_folds import sections
from d8r.tui.session import Session


@pytest.mark.parametrize("dialect, sql, names", [
    ("postgres", '''WITH RECURSIVE "Rows"(v) AS NOT MATERIALIZED (
        SELECT t.id FROM "Odd Schema" /* FROM ignored */ .\n"Table\nName" AS t
        JOIN public.events e ON e.id = t.id, log l
        WHERE extract(day FROM t.stamp) > 1 AND t.msg = $$JOIN shadow$$
    ) SELECT r.v, 'FROM nope' AS data FROM "Rows" r
      JOIN generate_series(1,2) AS series(n) ON r.v = series.n''',
     ['"Rows"', '"Odd Schema"', '"Table\nName"', 'public', 'events', 'log', '"Rows"']),
    ("sqlite", 'INSERT OR IGNORE INTO [A T](col) SELECT x.col FROM `other` x; '
               'UPDATE [A T] SET col=2; CREATE TABLE IF NOT EXISTS q(col REFERENCES [A T](col));',
     ['[A T]', '`other`', '[A T]', 'q', '[A T]']),
    ("duckdb", 'FROM data AS d SELECT d.id AS data; UPDATE main.data d SET id=s.id FROM stage s',
     ['data', 'main', 'data', 'stage']),
    ("postgres", 'SELECT (a IS DISTINCT FROM b), substring(a FROM b) FROM src AS alias '
                 'JOIN (SELECT c FROM nested_src) AS derived ON alias.a=derived.c',
     ['src', 'nested_src']),
])
def test_table_color_uses_only_relation_positions(dialect, sql, names):
    tokens = lex_sql(sql, dialect=dialect)
    marked = _table_offsets(tokens, dialect)
    assert [t.text for t in tokens if t.start in marked] == names


def test_statement_ranges_use_native_boundaries_and_not_matching_comment_text():
    trigger = "CREATE TRIGGER x AFTER INSERT ON a BEGIN\n SELECT ';'; SELECT 2;\nEND;"
    sql = f"/* SELECT 42 */;  SELECT 42;\r\n{trigger} SELECT 42;"
    regions = sections(sql, "sqlite")
    assert [sql[r.start:r.end] for r in regions] == ["SELECT 42;", trigger, "SELECT 42;"]
    opaque = "SELECT $$ (SELECT\n1); WITH q AS (\nSELECT 1) $$; SELECT '(SELECT\n2)';"
    assert len(sections(opaque, "postgres")) == 2


def test_gutter_folds_keep_mouse_selection_and_run_in_the_editor(parquet_source):
    async def run():
        session = Session({parquet_source.id: parquet_source})
        session.update_settings(intellisense=False)
        app = D8RApp(session)
        async with app.run_test(size=(140, 45)) as pilot:
            editor = app.editor
            sql = ("WITH alpha AS (\r\n SELECT 1 AS id\r\n), beta AS (\r\n"
                   " SELECT * FROM (SELECT\r\n  2 AS id\r\n ) s\r\n)\r\n"
                   "SELECT id FROM alpha;\r\nSELECT 33;")
            editor.load_text(sql)
            editor.focus()
            await pilot.pause()

            def lines():
                return [editor.render_line(y).text for y in range(editor.virtual_size.height)]

            async def click_arrow(row):
                y = editor.wrapped_document.location_to_offset((row, 0)).y - editor.scroll_offset.y
                assert await pilot.click(editor, offset=(editor.gutter.left + editor.gutter_width - 2, editor.gutter.top + y))

            assert sum(line.count("▾") for line in lines()) == 3
            assert len(lines()) == 9
            await click_arrow(0)
            assert lines()[0].count("▸") == 1 and len(lines()) == 8
            assert "SELECT 1 AS id" not in "\n".join(lines())
            await click_arrow(3)  # A nested arrow below a hidden body.
            visible = "\n".join(lines())
            assert "2 AS id" not in visible and "), beta AS (" in visible and " ) s" in visible
            assert editor.has_focus and editor.text == sql

            editor.move_cursor((0, 1))
            await pilot.press("down")
            assert editor.cursor_location == (2, 1)

            y = editor.gutter.top + editor.wrapped_document.location_to_offset((8, 0)).y - editor.scroll_offset.y
            x = editor.gutter.left + editor.gutter_width
            start, end = (x, y), (x + len("SELECT 33;"), y)
            await pilot.mouse_down(editor, offset=start)
            await pilot.hover(editor, offset=end)
            await pilot.mouse_up(editor, offset=end)
            assert editor.selected_text == "SELECT 33;"
            await pilot.press("ctrl+enter")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app._main_outcome.ok and app._main_outcome.rows == [[33]]
            assert editor.text == sql and editor.selected_text == "SELECT 33;"

            assert await pilot.click("#query-sections")
            assert editor.has_focus and editor.selected_text == "SELECT 33;" and len(lines()) == 9
            await pilot.press("f7")
            assert editor.has_focus and editor.selected_text == "SELECT 33;"
            assert len(lines()) == 4 and "SELECT id FROM alpha;" in "\n".join(lines())
            editor.move_cursor((0, 0))
            await pilot.press("ctrl+f")
            app.query_one("#find-query", Input).value = "2 AS id"
            await pilot.pause()
            assert editor.selected_text == "2 AS id"
            assert "2 AS id" in "\n".join(lines()) and "SELECT 1 AS id" not in "\n".join(lines())

    asyncio.run(run())


@pytest.mark.parametrize("soft_wrap", [False, True])
def test_fold_coordinates_survive_reflow_edits_and_undo(soft_wrap):
    async def run():
        session = Session()
        session.update_settings(intellisense=False)
        app = D8RApp(session)
        async with app.run_test(size=(140, 45)) as pilot:
            editor = app.editor
            editor.soft_wrap = soft_wrap
            sql = "SELECT\n    '" + "long text " * 20 + "';\nSELECT 42;"
            editor.load_text(sql)
            editor.focus()
            await pilot.pause()
            await pilot.press("shift+f7")
            assert editor.virtual_size.height == 2
            await pilot.resize_terminal(100, 35)
            assert editor.virtual_size.height == 2
            await pilot.press("down", "end")
            assert editor.cursor_location == (2, len("SELECT 42;"))
            editor.insert("-- comment\n", (0, 0))
            assert editor.virtual_size.height == 3 and editor.cursor_location[0] == 3
            await pilot.press("ctrl+z")
            assert editor.text == sql and editor.virtual_size.height == 2
            await pilot.press("ctrl+a")
            assert editor.selected_text == sql  # Selection includes the folded body.
            editor.load_text(sql)  # A new page must not inherit stale fold offsets.
            assert editor.virtual_size.height > 2

            editor.load_text('SELECT a.id FROM "G".\r\n"two\r\nlines" AS a')

            def ink(row, column):
                return editor.get_line(row).get_style_at_offset(app.console, column).color

            table_ink = ink(0, len('SELECT a.id FROM '))
            assert table_ink is not None and table_ink == ink(1, 2) == ink(2, 2)
            assert table_ink != ink(0, 0) and table_ink != ink(0, 9) and table_ink != ink(2, 10)

    asyncio.run(run())
