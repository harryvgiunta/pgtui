"""Offline diagnostics and authoring on actual Parquet and D1 workspace sources."""
from __future__ import annotations

import asyncio

import pytest
from textual.widgets import OptionList, Tree

from d8r.dbt_analysis import analyze_file
from d8r.dbt_column_rename import plan_column_rename
from d8r.dbt_files import DbtFiles
from d8r.dbt_index import DbtIndex
from d8r.dbt_rename import apply_rename
from d8r.engine import refresh_source_schema
from d8r.sql import ColumnDef, RoutineDef, SchemaContext, TableDef
from d8r.tui.app import D8RApp
from d8r.tui.dbt_authoring import DbtProblems
from d8r.tui.dbt_model_completion import complete_dbt
from d8r.tui.session import Session
from tests.test_dbt_exchange import project, put, source
from tests.test_dbt_workspace import open_home


def lint(text, dialect, schema=SchemaContext()):
    return analyze_file(None, 'model.sql', text, dialect, schema)


@pytest.mark.parametrize('dialect', ['postgres', 'duckdb', 'sqlite'])
def test_join_warnings_distinguish_mismatches_and_missing_evidence(dialect):
    prefix = "WITH numbers AS (SELECT 1 AS k, 'label' AS label), strings AS (SELECT 'x' AS k) "
    for condition in ['ON (a.k) = b.k', 'USING (k)']:
        sql = prefix + f'SELECT a.k FROM numbers a JOIN strings b {condition}'
        warning, = lint(sql, dialect).diagnostics
        assert warning.severity == 'warning'
        assert 'JOIN keys have different known type families' in warning.message
        assert warning.start == sql.rindex('k')
        assert sql[warning.start:warning.end] == 'k'
    for join in ['strings b ON CAST(a.k AS TEXT) = b.k', 'strings b ON a.k = a.label',
                 'unindexed b ON a.k = k']:
        sql = prefix + f'SELECT a.k FROM numbers a JOIN {join}'
        assert not lint(sql, dialect).diagnostics
    for join in ['unindexed b ON a.k = b.k', 'unindexed b USING (k)']:
        sql = prefix + f'SELECT a.k FROM numbers a JOIN {join}'
        warning, = lint(sql, dialect).diagnostics
        assert warning.start == sql.rindex('k')
        assert 'Cannot verify this JOIN key type offline' in warning.message
        assert 'integer' in warning.message.lower()
    sql = prefix + 'SELECT a.k FROM numbers a JOIN unindexed b ON a.k = CAST(b.k AS TEXT)'
    warning, = lint(sql, dialect).diagnostics
    assert 'different known type families' in warning.message


def test_join_key_can_be_bound_but_untyped_in_a_partial_dbt_project(tmp_path):
    home = project(tmp_path)
    put(home, 'models/left_input.sql', 'select 1 as id')
    put(home, 'models/right_input.sql', "select some_udf() as id")
    sql = "SELECT a.id FROM {{ ref('left_input') }} a JOIN {{ ref('right_input') }} b ON b.id = a.id"
    path = 'models/report.sql'
    put(home, path, sql)
    index = DbtIndex(DbtFiles(home), {})
    warning, = analyze_file(index, path, sql, 'postgres', SchemaContext()).diagnostics
    assert sql[warning.start - 2:warning.end] == 'b.id'
    assert 'Cannot verify this JOIN key type offline' in warning.message


@pytest.mark.parametrize(('dialect', 'valid'), [
    ('postgres', "SELECT '2' + 1, ' Ye ' AND true"),
    ('duckdb', "SELECT 'yes' AND true, 2 AND true"),
    ('sqlite', "SELECT '2.5e1' + 1, '  +.3' AND true"),
])
def test_literal_warnings_respect_coercion_and_authored_spans(dialect, valid):
    assert not lint(valid, dialect).diagnostics
    sql = "{{ config(materialized='view') -}}\r\nSELECT 'broken' + 1, NOT ('bogus')"
    warnings = lint(sql, dialect).diagnostics
    assert [(sql[w.start:w.end], w.severity) for w in warnings] == [
        ("'broken'", 'warning'), ("'bogus'", 'warning'),
    ]
    if dialect == 'sqlite':
        assert 'zero' in warnings[0].message and 'false' in warnings[1].message


def test_schema_warnings_skip_dialect_specific_name_resolution():
    sqlite = SchemaContext(tables=(TableDef('records', columns=(ColumnDef('id', 'INTEGER'),)),))
    assert not lint('SELECT rowid, [ID], "a string", max([ID]) FROM records', 'sqlite', sqlite).diagnostics
    assert not lint("WITH a AS (SELECT 1 AS id), b AS (SELECT {'p': 2} AS a) "
                    "SELECT a.p AS prop, prop FROM a, b ORDER BY ALL", 'duckdb').diagnostics
    assert not lint('SELECT r.ctid FROM records r', 'postgres', sqlite).diagnostics


@pytest.mark.parametrize(('dialect', 'sql', 'call'), [
    ('postgres', "SELECT replace(concat('a,b', 'c'), 'b')", 'replace'),
    ('duckdb', 'SELECT now(1)', 'now'),
    ('sqlite', 'SELECT coalesce(1), random()', 'coalesce'),
])
def test_arity_warnings_count_authored_arguments(dialect, sql, call):
    warning, = lint(sql, dialect).diagnostics
    assert sql[warning.start:warning.end] == call
    assert 'arguments; got' in warning.message


def test_cached_overload_arities_cover_defaults_variadics_and_qualification():
    schema = SchemaContext(routines=(
        RoutineDef('ext.format', ('a text', 'b text'), min_args=2),
        RoutineDef('ext.format', ('a text', 'b text', 'c text', 'd text'), min_args=3),
        RoutineDef('other.format', ('a text',), min_args=1),
        RoutineDef('pg_catalog.concat', ('VARIADIC any',), variadic=True, min_args=1),
        RoutineDef('ext."With.Dot"', ('x int',), min_args=1),
    ))
    valid = "SELECT format('a'), ext.format('a','b'), ext.format('a','b','c'), concat('x','y','z'), no_catalog(1)"
    assert not lint(valid, 'postgres', schema).diagnostics
    sql = "SELECT ext.format('a'), ext.format('a','b','c','d','e'), ext.\"With.Dot\"(), ext.\"with.dot\"()"
    warnings = lint(sql, 'postgres', schema).diagnostics
    assert [sql[w.start:w.end] for w in warnings] == ['format', 'format', '"With.Dot"']
    assert [w.message.rsplit('accepts ', 1)[1] for w in warnings] == ['1 argument.', '5 arguments.', '0 arguments.']


def test_duckdb_authored_calls_use_discovered_macros_and_overloads(parquet_source):
    parquet_source.con.execute("""CREATE SCHEMA ext;
        CREATE MACRO ext.native_demo(x, y := 1) AS error('must not execute')""")
    refresh_source_schema(parquet_source)
    schema = SchemaContext(routines=parquet_source.routines)
    valid = "SELECT ext.native_demo(1), ext.native_demo(1, y := 2), format('{}', 3), concat('a', 'b')"
    assert not lint(valid, 'duckdb', schema).diagnostics
    sql = 'SELECT ext.native_demo(1, 2, 3)'
    warning, = lint(sql, 'duckdb', schema).diagnostics
    assert sql[warning.start:warning.end] == 'native_demo'
    assert 'No cached overload' in warning.message


def test_sqlite_grouping_warns_about_row_choice_not_validity():
    prefix = "WITH rows AS (SELECT 1 AS id, 'a' AS label, 3 AS amount) "
    for projection in ['label, sum(amount)', 'label, min(amount), max(amount)']:
        sql = prefix + f'SELECT {projection} FROM rows GROUP BY id'
        warning, = lint(sql, 'sqlite').diagnostics
        assert sql[warning.start:warning.end] == 'label'
        assert warning.severity == 'warning' and 'any row in the group' in warning.message
    for select in ['SELECT label, max(amount) FROM rows',
                   'SELECT label, sum(amount) AS n FROM rows GROUP BY 1 HAVING n > 0',
                   'SELECT label, sum(amount) OVER () FROM rows']:
        assert not lint(prefix + select, 'sqlite').diagnostics


def crowded_sql():
    arithmetic = ',\n'.join(f"'bad {i}' + 1 AS c{i}" for i in range(105))
    return (f'SELECT\n{arithmetic};\n'
            "WITH src AS (SELECT 'x' AS label) SELECT label, count(*) FROM src;\n")


def test_crowded_file_keeps_late_grouping_warnings():
    sql = crowded_sql()
    diagnostics = lint(sql, 'sqlite').diagnostics
    assert [sql[w.start:w.end] for w in diagnostics] == [*(f"'bad {i}'" for i in range(105)), 'label']
    assert 'row in the group' in diagnostics[-1].message


async def feedback_ready(app, pilot):
    await pilot.pause(0.4)
    await app.workers.wait_for_complete()
    await pilot.pause()
    assert app.dbt_authoring.current


def test_typing_source_arguments_shows_project_names_without_accepting_macro(tmp_path, parquet_source):
    home = project(tmp_path)
    source(home)  # landing.orders
    put(home, 'models/other.yml', 'sources: [{name: other, tables: [{name: oranges}]}]\n')
    head, tail = 'select * from ', ") }} as s\nwhere s.status = 'active'"
    path = 'models/report.sql'
    put(home, path, head + tail)
    app = D8RApp(Session({parquet_source.id: parquet_source}, data_dir=tmp_path / 'workspace'))

    async def scenario():
        async with app.run_test(size=(140, 50)) as pilot:
            await open_home(app, pilot, home)
            app.open_dbt_file(str(home), path)
            await feedback_ready(app, pilot)
            app.editor.move_cursor((0, len(head)))
            await pilot.press(*"{{ source ('la")
            await pilot.pause(0.05)
            assert app.palette.suggestions_visible and app.palette.view.labels == ['landing']
            await pilot.press(*"nding', 'or")  # Type the source name too, without accepting it.
            await pilot.pause(0.05)
            assert app.palette.suggestions_visible and app.palette.view.labels == ['orders']
            typed = app.editor.text
            await pilot.press('tab')
            assert app.editor.text == head + "{{ source ('landing', 'orders'" + tail
            app.editor.action_undo()
            assert app.editor.text == typed

    asyncio.run(scenario())


@pytest.mark.parametrize('source_fixture, fields', [
    ('parquet_source', ('id', 'label')),  # Cached main.data; no columns in sources.yml.
    ('d1_source', ('order_id', 'amount')),  # Declared dbt fields, not in the native catalog.
])
def test_jinja_left_join_completes_each_alias_before_where(tmp_path, request, source_fixture, fields):
    native = request.getfixturevalue(source_fixture)
    d1 = request.getfixturevalue('d1_harness') if source_fixture == 'd1_source' else None
    home = project(tmp_path)
    definition = ('schema: main, tables: [{name: orders, identifier: data}]' if not d1 else
                  'schema: warehouse, tables: [{name: orders, columns: [{name: order_id}, {name: amount}]}]')
    put(home, 'models/sources.yml', 'sources: [{name: landing, ' + definition + '}]\n')
    field = fields[0]
    source_call = "{{ source('landing','orders') }}"
    head = f'WITH joined AS (\r\nSELECT x.{field}\r\nFROM {source_call} x\r\nLEFT JOIN {source_call} y\r\nON '
    tail = f'\r\nWHERE x.{field} > 0\r\n) SELECT * FROM joined'
    app = D8RApp(Session({native.id: native}, data_dir=tmp_path / 'workspace'))

    async def scenario():
        async with app.run_test(size=(140, 50)) as pilot:
            await open_home(app, pilot, home)
            before = list(d1.requests) if d1 else None
            editor = app.editor
            editor.load_text(head + tail)
            editor.move_cursor(editor.document.get_location_from_index(len(head)))
            editor.focus()
            await feedback_ready(app, pilot)
            for alias, prefix in [('x', 'x.'), ('y', ' = y.')]:
                await pilot.press(*prefix)
                await pilot.pause(0.05)
                palette = app.palette
                assert editor.has_focus and palette.suggestions_visible
                assert {e.label for e in palette.view.entries if e.kind == 'column'} == {f'{alias}.{f}' for f in fields}
                if alias == 'x':
                    await feedback_ready(app, pilot)
                    dot = len(head) + 1
                    assert 'column' in editor.dbt_detail(dot).lower()
                    assert not editor.dbt_detail(editor.text.index('WHERE'))
                palette.highlighted = palette.view.labels.index(f'{alias}.{field}')
                await pilot.press('tab')
            assert editor.text == head + f'x.{field} = y.{field}' + tail
            await feedback_ready(app, pilot)
            assert not [d for d in app.query_one(DbtProblems).diagnostics if d.severity == 'error']
            assert not app.session.history
            if d1:
                assert d1.requests == before

    asyncio.run(scenario())


@pytest.mark.parametrize(('marked', 'label', 'expected'), [
    ('''{{ source ("la|) }} join {{ ref("model") }}''', 'landing',
     '''{{ source ("landing", ) }} join {{ ref("model") }}'''),
    ("{{ source('landing', 'or|\nwhere status = 'active'", 'orders',
     "{{ source('landing', 'orders')\nwhere status = 'active'"),
    ("{{ source('landing', 'or|; -- finish here", 'orders',
     "{{ source('landing', 'orders'); -- finish here"),
    ("{{ source('la|', 'old_table') }}", 'landing', "{{ source('landing', 'old_table') }}"),
])
def test_source_completion_edits_only_the_authored_slot(tmp_path, marked, label, expected):
    home = project(tmp_path)
    source(home)
    put(home, 'models/other.yml', 'sources: [{name: other, tables: [{name: old_table}]}]\n')
    index = DbtIndex(DbtFiles(home), {})
    before, after = marked.split('|')
    text = before + after
    result = complete_dbt(index, text, len(before), manual=False)
    item, = result.items
    assert item.label == label
    assert text[:result.context.start] + item.insert + text[result.context.end:] == expected


def test_later_diagnostics_are_reachable_in_lineage_and_problems(tmp_path, d1_source):
    sql = crowded_sql()
    home = project(tmp_path)
    path = 'models/crowded.sql'
    put(home, path, sql)
    app = D8RApp(Session({d1_source.id: d1_source}, data_dir=tmp_path / 'workspace'))

    async def scenario():
        async with app.run_test(size=(145, 55)) as pilot:
            await open_home(app, pilot, home)
            app.open_dbt_file(str(home), path)
            expected = app.editor.document.get_location_from_index(sql.rindex('label'))
            app.action_dbt_lineage()
            await feedback_ready(app, pilot)
            tree = app.query_one('#dbt-lineage-tree', Tree)
            group = next(node for node in tree.root.children if node.label.plain.startswith('File diagnostics'))
            group.expand()
            last_page = group.children[-1]
            last_page.expand()
            await pilot.pause()
            warning = last_page.children[-1]
            assert 'any row in the group' in warning.label.plain
            tree.select_node(warning)
            await pilot.pause()
            assert app.editor.cursor_location == expected

            app.action_dbt_problems()
            await feedback_ready(app, pilot)
            listing = app.query_one('#dbt-problems-list', OptionList)
            listing.highlighted = listing.option_count - 1
            assert listing.get_option_at_index(listing.highlighted).id == 'next'
            await pilot.press('enter')
            assert listing.get_option_at_index(0).id == 'previous'
            listing.highlighted = listing.option_count - 1
            assert 'any row in the group' in str(listing.get_option_at_index(listing.highlighted).prompt)
            app.editor.move_cursor((0, 0))
            await pilot.press('enter')
            assert app.editor.cursor_location == expected

    asyncio.run(scenario())


@pytest.mark.parametrize(('source_fixture', 'relation', 'id_type'), [
    ('parquet_source', 'data', 'bigint'), ('d1_source', 'records', 'integer'),
])
def test_native_workspace_lineage_diagnostics_and_rename(tmp_path, request, source_fixture, relation, id_type):
    native = request.getfixturevalue(source_fixture)
    d1 = request.getfixturevalue('d1_harness') if source_fixture == 'd1_source' else None
    home = project(tmp_path)
    base = f'SELECT t.id AS "KeyID" FROM {relation} t\n'
    base_path, report_path = 'models/base.sql', 'models/report.sql'
    base_file = put(home, base_path, base)
    put(home, 'models/labels.sql', f'SELECT t.label AS "KeyID" FROM {relation} t\n')
    report = """{{ config(materialized='view') }}
SELECT a."KEYID" AS record_id, a.missing AS missing,
       replace('x', 'y') AS bad_arity, 'text' AS declared_integer
FROM {{ ref('base') }} a JOIN {{ ref('labels') }} b ON a."KEYID" = b."KEYID"
"""
    report_file = put(home, report_path, report)
    put(home, 'models/properties.yml', '''models:
  - name: report
    columns: [{name: declared_integer, data_type: INTEGER}]
''')
    orphan_file = put(home, 'models/orphans.yml', 'models: [{name: missing_one}, {name: missing_two}]\n')
    app = D8RApp(Session({native.id: native}, data_dir=tmp_path / 'workspace'))

    async def scenario():
        async with app.run_test(size=(145, 55)) as pilot:
            await pilot.pause()
            if d1:
                before = list(d1.requests)
            await open_home(app, pilot, home)
            app.open_dbt_file(str(home), report_path)
            app.action_dbt_lineage()
            await feedback_ready(app, pilot)

            tree = app.query_one('#dbt-lineage-tree', Tree)
            graph = {node.label.plain.split(' (')[0]: node for node in tree.root.children}
            notes = [node.label.plain for node in graph['Project index notes'].children]
            assert {note.split("model '")[1].split("'")[0] for note in notes} == {'missing_one', 'missing_two'}
            diagnostics = graph['File diagnostics']
            assert [report[n.data.start:n.data.end] for n in diagnostics.children] == [
                'missing', 'replace', 'declared_integer', '"KEYID"',
            ]
            resource = next(n for n in tree.root.children if n.data and n.data[0] == report_path)
            columns = next(n for n in resource.children if n.label.plain == 'Columns')
            record_id = next(n for n in columns.children if n.label.plain == f'record_id · {id_type}'
                             or n.label.plain.lower() == f'record_id · {id_type}')
            origin, = record_id.children[0].children
            assert origin.data[0] == base_path and origin.label.plain.endswith('.KeyID')

            diagnostics.expand()
            await pilot.pause()
            join = diagnostics.children[-1]
            tree.select_node(join)
            await pilot.pause()
            assert app.editor.cursor_location == app.editor.document.get_location_from_index(join.data.start)

            app.action_dbt_problems()
            await feedback_ready(app, pilot)
            listing = app.query_one('#dbt-problems-list', OptionList)
            prompts = [str(listing.get_option_at_index(i).prompt) for i in range(listing.option_count)]
            assert prompts == [n.label.plain for n in diagnostics.children] + [f'dbt index  {note}' for note in notes]
            if d1:
                assert d1.requests == before

    asyncio.run(scenario())
    orphan_file.unlink()  # Refactoring requires a clean index, unlike advisory lineage.
    files = DbtFiles(home)
    plan = plan_column_rename(files, base_path, base.index('KeyID'), 'RenamedKey', dialect=native.dialect)
    apply_rename(files, plan)
    assert base_file.read_text() == base.replace('"KeyID"', '"RenamedKey"')
    assert report_file.read_text() == report.replace('a."KEYID"', 'a."RenamedKey"')
