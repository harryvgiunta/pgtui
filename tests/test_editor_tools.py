"""SQL layout invariants and find navigation through real Textual editors."""
import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Input, Static, TextArea
from textual.widgets.text_area import Selection

from d8r.sql import lex_sql
from d8r.tui.editor_tools import FindBar, format_editor
from d8r.tui.query_editor import QueryTextArea
from d8r.tui.sql_layout import format_sql


def test_readable_cte_join_and_predicates_without_changing_spelling():
    sql = ("with a(i) AS MATERIALIZED (select i from t), b as (select i from a) "
           "select a.i from a left outer join b on a.i = b.i and b.i > 0 "
           "where a.i between 1 and 9 and a.i is distinct from 3;")
    expected = """with a(i) AS MATERIALIZED (
    select i
    from t
),
b as (
    select i
    from a
)
select a.i
from a
left outer join b
    on a.i = b.i
    and b.i > 0
where a.i between 1 and 9
    and a.i is distinct from 3;"""
    assert format_sql(sql) == expected
    assert format_sql(expected) == expected
    assert format_sql("WITH u AS (SELECT\nid,\nname FROM\nusers) SELECT * FROM u") == (
        "WITH u AS (\n    SELECT\n        id,\n        name\n    FROM\n        users\n)\nSELECT *\nFROM u"
    )


@pytest.mark.parametrize(("dialect", "sql"), [
    ("postgres", "/* SELECT x FROM y */; SELECT x FROM y; "
     "WITH q AS (SELECT $tag$ select ' from (\n) $tag$, E'a\\' from', @name::text "
     "/* outer /* from */ inner */ FROM t -- keep )\r\nWHERE t.x <@ '{2}') SELECT * FROM q;"),
    ("sqlite", "WITH x AS(SELECT [a from b], `order by`, '@p AND', :from, $where, @and "
     "FROM `left join` -- comment\rnot terminated by CR\nWHERE [k] != 0) SELECT * FROM x;"),
    ("duckdb", "WITH a AS (FROM read_parquet('FROM --.parquet') SELECT {'and': [1, 2]}) "
     "SELECT COLUMNS('.*') FROM a ASOF LEFT JOIN b ON a.t >= b.t AND b.i = 5;"),
])
def test_lexical_invariants(dialect, sql):
    def concrete(text):
        return [(token.kind, token.text) for token in lex_sql(text, dialect=dialect)
                if token.kind != "whitespace"]

    formatted = format_sql(sql, dialect)
    assert formatted != sql
    assert concrete(formatted) == concrete(sql)  # Includes comments and their exact contents.
    assert format_sql(formatted, dialect) == formatted


def test_only_real_clause_boundaries_and_keep_newline_literal_concatenation():
    sql = ("select 'a'\r\n  'b', extract(year from t.x), case when a and b then 1 end "
           "from t where t.i between (1 + 1) and 8 and t.p")
    assert format_sql(sql) == ("select 'a'\r\n  'b', extract(year from t.x), case when a and b then 1 end"
                               "\r\nfrom t\r\nwhere t.i between (1 + 1) and 8\r\n    and t.p")


@pytest.mark.parametrize(("dialect", "sql"), [
    ("postgres", "SELECT 'unterminated FROM foo"),
    ("postgres", "WITH a AS (SELECT foo FROM bar"),
    ("postgres", "CREATE FUNCTION f() RETURNS int LANGUAGE SQL BEGIN ATOMIC\n"
     "SELECT a FROM t; SELECT b FROM u; END;"),
    ("postgres", "COPY t FROM STDIN;\nselect a from raw;\n\\.\n"),
    ("postgres", "SELECT 'a' /* unterminated FROM t"),
    ("postgres", r"SELECT 'a\' FROM t"),
    ("sqlite", "CREATE TRIGGER z AFTER UPDATE ON t BEGIN SELECT a FROM x; SELECT b FROM y; END;"),
    ("sqlite", "SELECT $ns::value(select a from t) FROM records"),
    ("unknown", "SELECT * FROM t"),
])
def test_uncertain_native_constructs_are_not_rewritten(dialect, sql):
    assert format_sql(sql, dialect) == sql


class Workspace(App):
    """No Session or source; format and find work offline on plain and SQL TextAreas."""

    CSS = "TextArea { height: 1fr; }"
    BINDINGS = [("ctrl+f", "find", "Find")]

    def compose(self) -> ComposeResult:
        yield TextArea(id="query")
        yield QueryTextArea(id="body")
        yield Input(id="other")
        yield FindBar()

    def action_find(self):
        self.query_one(FindBar).open(self.query_one("#query", TextArea))


def test_format_selected_span_is_one_undoable_edit_on_both_editors():
    async def exercise():
        app = Workspace()
        async with app.run_test() as pilot:
            for editor in app.query(TextArea):
                # A selection entirely inside a dollar body must not turn that body into SQL.
                original = "select $$select i from t$$;\nselect α from data where true;\nselect β from data;"
                editor.load_text(original)
                editor.selection = Selection((0, 9), (0, 24))
                assert not format_editor(editor, "postgres")
                editor.selection = Selection((1, len("select α from data where true;")), (1, 0))
                assert format_editor(editor, "postgres")
                assert editor.text == original.replace("α from data where", "α\nfrom data\nwhere")
                assert editor.selected_text == "select α\nfrom data\nwhere true;"
                assert editor.selection.end == (1, 0)  # Reverse selections retain their direction.
                assert not format_editor(editor, "postgres")
                editor.action_undo()
                assert editor.text == original
                editor.move_cursor((2, len("select β from data")))
                assert format_editor(editor, "postgres")
                assert editor.selection.is_empty
                assert editor.document.get_line(editor.cursor_location[0]) == "from data;"
                assert editor.cursor_location[1] == len("from data")
                editor.action_undo()
                assert editor.text == original
                editor.load_text("select * from ")
                editor.insert("tbl", editor.document.end)
                assert format_editor(editor, "postgres")
                editor.action_undo()
                assert editor.text == "select * from tbl"
                editor.action_undo()
                assert editor.text == "select * from "
                editor.read_only = True
                assert not format_editor(editor, "postgres")
            await pilot.pause()

    asyncio.run(exercise())


def test_find_literal_wrap_scroll_reveal_focus_and_retarget():
    async def exercise():
        app = Workspace()
        async with app.run_test(size=(80, 25)) as pilot:
            bar = app.query_one(FindBar)
            query, body = app.query(TextArea)
            sql = "α.foo\n" + "x\n" * 50 + "β.foo\n"
            query.load_text(sql)
            query.focus()
            await pilot.pause()
            await pilot.press("ctrl+f")
            field = bar.query_one(Input)
            assert app.focused is field
            await pilot.press(".", "f", "o", "o")
            assert query.selected_text == ".foo" and query.selection.start == (0, 1)
            await pilot.press("enter")
            assert query.selection.start == (51, 1) and query.scroll_y > 0
            await pilot.press("enter")
            assert query.selection.start == (0, 1)
            assert str(bar.query_one("#find-status", Static).content) == "Wrapped"
            await pilot.press("shift+enter")
            assert query.selection.start == (51, 1)
            field.value = "missing"
            await pilot.pause()
            assert str(bar.query_one(Static).content) == "No match"
            await pilot.press("escape")
            assert app.focused is query and not bar.display and query.text == sql

            other = app.query_one("#other", Input)
            other.focus()
            await pilot.pause()
            assert not bar.open(query) and app.focused is other
            await app.push_screen(ModalScreen())
            assert not bar.open(query, from_toolbar=True)
            app.pop_screen()
            await pilot.pause()

            seen = []
            body.load_text("select 1\nfrom a;")
            assert bar.open(body, from_toolbar=True, reveal=lambda e, a, b: seen.append((e.selected_text, a, b)))
            field.value = "from"
            await pilot.pause()
            assert seen[-1] == ("", (1, 0), (1, 4)) and body.selected_text == "from"
            bar.close()
            assert app.focused is body
            assert query.selected_text == ".foo" and body.text == "select 1\nfrom a;"

    asyncio.run(exercise())
