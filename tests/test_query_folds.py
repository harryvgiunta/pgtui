"""The fold projection never becomes an editable or executable document."""
import asyncio

import pytest
from textual.widgets import OptionList

from d8r.sql import lex_sql
from d8r.tui.app import D8RApp
from d8r.tui.query_editor import _table_offsets
from d8r.tui.query_folds import QueryFoldView, sections


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


def test_folds_hide_expand_navigate_and_never_replace_sql_or_selection():
    async def run():
        app = D8RApp()
        async with app.run_test(size=(140, 45)) as pilot:
            editor = app.editor
            view = editor.parent.query_one(QueryFoldView)
            options = view.query_one(OptionList)
            editor.load_text('SELECT a.id FROM "G".\r\n"two\r\nlines" AS a')
            def ink(row, column):
                return editor.get_line(row).get_style_at_offset(app.console, column).color

            table_ink = ink(0, len('SELECT a.id FROM '))
            assert table_ink is not None and table_ink == ink(1, 2) == ink(2, 2)
            assert table_ink != ink(0, 0) and table_ink != ink(0, 9) and table_ink != ink(2, 10)
            sql = ("WITH alpha AS (\n SELECT 'secret alpha'\n), beta AS (\n"
                   " SELECT * FROM (SELECT\n  'secret nested') s\n) SELECT * FROM alpha;\nSELECT 2;")
            editor.load_text(sql)
            editor.move_cursor((1, 2))
            editor.move_cursor((1, 9), select=True)
            selected = editor.selected_text
            selection = editor.selection
            editor.action_toggle_folds()
            await pilot.pause()

            def labels():
                return [str(o.prompt) for o in options.options]

            assert options.has_focus and not editor.display
            assert not any('secret alpha' in label or 'secret nested' in label for label in labels())
            assert editor.text == sql and editor.selected_text == selected and selected
            assert 'CTE alpha' in labels()[options.highlighted]
            await pilot.press('enter')
            assert any('secret alpha' in label for label in labels())
            assert not any('secret nested' in label for label in labels())
            await pilot.click(view.query_one('.fold-toggle'))
            assert not any('secret alpha' in label for label in labels())
            await pilot.press('escape')
            assert editor.has_focus and editor.selection == selection and editor.text == sql

            editor.action_toggle_folds()
            await pilot.pause()
            options.highlighted = next(i for i, label in enumerate(labels()) if 'CTE beta' in label)
            await pilot.press('enter')
            subquery = next(i for i, label in enumerate(labels()) if 'Subquery' in label)
            await pilot.press(*(['down'] * (subquery - options.highlighted)))
            assert 'Subquery' in labels()[options.highlighted]
            await pilot.press('enter')
            assert any('secret nested' in label for label in labels())
            await pilot.press('e')
            assert editor.has_focus and editor.cursor_location == (3, len(' SELECT * FROM '))
            assert editor.text == sql

            # Formatting, replacing SQL programmatically, and retargeting even
            # to identical SQL must never leave stale rows or offsets visible.
            editor.history.checkpoint()
            editor.action_toggle_folds()
            editor.insert(' /* format */', (0, 0))
            assert editor.display and not view.display
            editor.action_toggle_folds()
            editor.undo()
            assert editor.display and editor.text == sql
            editor.action_toggle_folds()
            editor.load_text(sql)
            assert not view.display and editor.display
            editor.action_toggle_folds()
            editor.reveal_for_edit()
            editor.move_cursor((1, 1))
            assert editor.display and editor.cursor_location == (1, 1)

    asyncio.run(run())
