"""Preview must select the authored CTE, not a different compiled query."""
import pytest

from d8r.dbt import DbtError
from d8r.dbt_preview import dbt_preview_query


def test_preview_keeps_compiled_dependencies_and_quoted_cte_identity():
    source = ('{{ config(materialized="view") }}\n'
              'WITH "Recent Orders" (id) AS (SELECT * FROM {{ ref("orders") }}),\n'
              'filtered AS (SELECT id FROM "Recent Orders" WHERE id > 3)\n'
              'SELECT * FROM filtered;')
    compiled = ('WITH __dbt__cte__orders AS (SELECT 4),\n'
                '"Recent Orders" (id) AS (SELECT * FROM __dbt__cte__orders),\n'
                'filtered AS (SELECT id FROM "Recent Orders" WHERE id > 3)\n'
                'SELECT * FROM filtered;')
    query, label = dbt_preview_query(compiled, "postgres", source=source, caret=source.index('ref('))
    assert query == compiled[:compiled.rindex("SELECT")] + 'SELECT * FROM "Recent Orders"'
    assert label == 'CTE "Recent Orders"'


def test_preview_refuses_nested_or_missing_cte_binding():
    source = "WITH outer_cte AS (WITH inner_cte AS (SELECT 1) SELECT * FROM inner_cte) SELECT * FROM outer_cte"
    with pytest.raises(DbtError, match="top-level CTE"):
        dbt_preview_query(source, "postgres", source=source, caret=source.index("inner_cte"))
    with pytest.raises(DbtError, match="cannot be matched"):
        dbt_preview_query("SELECT 1", "postgres", source=source, caret=source.index("outer_cte"))


@pytest.mark.parametrize("compiled", [
    "WITH changed AS (DELETE FROM t RETURNING id) SELECT * FROM changed",
    "SELECT 1; DROP TABLE t",
])
def test_preview_refuses_mutating_ctes_and_extra_statements(compiled):
    with pytest.raises(DbtError, match="requires"):
        dbt_preview_query(compiled, "postgres", source=compiled, caret=None)
