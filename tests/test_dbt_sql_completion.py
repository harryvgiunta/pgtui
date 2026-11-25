"""Boundary cases for projecting draft dbt SQL into native editor scopes."""
from __future__ import annotations

import pytest

from d8r.dbt_files import DbtFiles
from d8r.dbt_index import DbtIndex
from d8r.sql import ColumnDef, SchemaContext, TableDef
from d8r.tui.dbt_completion import complete_dbt_sql
from tests.test_dbt_exchange import project, put


@pytest.fixture
def index(tmp_path):
    root = project(tmp_path)
    put(root, 'models/sources.yml', '''sources:
  - name: logical
    schema: main
    tables: [{name: orders, identifier: physical}]
  - name: archive
    schema: elsewhere
    tables: [{name: orders, identifier: physical}]
''')
    path = put(root, 'models/model.sql', 'select 1 as model_key')
    snapshot = DbtIndex(DbtFiles(root), {}, include_artifact=False)
    path.unlink()  # The editor projection may use only the indexed version.
    return snapshot


def at(index, sql, schema, dialect='postgres'):
    before, after = sql.split('|')
    return complete_dbt_sql(index, before + after, len(before), schema, dialect=dialect, manual=True)


def table(name, *columns):
    return TableDef(name, columns=tuple(ColumnDef(col, 'integer') for col in columns))


@pytest.mark.parametrize(('dialect', 'default'), [('duckdb', 'main'), ('d1', None)])
def test_default_namespace_is_not_a_bare_name_fallback(index, dialect, default):
    sql = "select * from {{ source('logical', 'orders') }} x where x.|"
    base = table('"physical"', 'id')
    foreign = table('other.physical', 'unrelated')
    def columns(query, tables):
        schema = SchemaContext(tables, default_schema=default)
        return [item.label for item in at(index, query, schema, dialect).items if item.kind == 'column']

    assert columns(sql, (base, foreign)) == ['x.id']
    assert not columns(sql.replace('logical', 'archive'), (base,))
    assert not columns(sql, (foreign,))
    assert not columns(sql, (base, table('main.physical', 'ambiguous')))


def test_nested_scope_and_quoted_mid_token_edit_keep_the_authored_suffix(index):
    sql = ('WITH q AS (SELECT s."Native ID" FROM {{\r\n source(\'logical\', \'orders\')\r\n}} s)\r\n'
           'SELECT * FROM (SELECT * FROM q) AS "with spaces"\r\n'
           'JOIN {{ ref(\'demo\', \'model\') }} y ON "with spaces"."Na|stale"\r\nWHERE y.model_key > 0')
    schema = SchemaContext((table('main.physical', 'Native ID'), table('logical.orders', 'decoy')))
    result = at(index, sql, schema)
    item, = [item for item in result.items if item.kind == 'column']
    assert item.label == '"with spaces".Native ID'
    doc = sql.replace('|', '')
    assert doc[result.context.start:result.context.end] == '"Nastale"'
    assert doc[:result.context.start] + item.insert + doc[result.context.end:] == sql.replace('"Na|stale"', '"Native ID"')
    result = at(index, sql.replace('"with spaces"."Na|stale"', 'y.|'), schema)
    assert [item.label for item in result.items if item.kind == 'column'] == ['y.model_key']


@pytest.mark.parametrize('relation', ["ref('missing')", "ref('other', 'model')", "ref('model', v=2)",
                                      "ref(var('model'))", "source('logical', var('orders'))"])
def test_unresolved_relations_never_inherit_a_neighbor_or_pool(index, relation):
    sql = "select * from {{ ref('model') }} x join {{ " + relation + ' }} y on y.|'
    schema = SchemaContext((table('model', 'model_key'), table('y', 'wrong')))
    result = at(index, sql, schema)
    assert not [item for item in result.items if item.kind == 'column']


def test_partial_projection_cannot_rewrite_star_through_a_cte(index):
    sql = "with q as (select * from {{ ref('model') }}) select q.*| from q"
    # Explicitly projected names are usable hints, not proof that the upstream
    # draft, its dependencies, and its effective materialization are complete.
    result = at(index, sql, SchemaContext())
    assert not result.items
