"""Executed previews provide completion samples, never column inspection results."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Event

import pytest

from d8r.completion.engine import complete
from d8r.sql import relation_name
from d8r.tui import session as session_module
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


@pytest.fixture(params=["parquet_source", "d1_source", "postgres_source"])
def result_table(request):
    source = request.getfixturevalue(request.param)
    session = Session({source.id: source})
    if source.dialect == "postgres":
        assert session.run("CREATE TEMP TABLE observed(id int, label text)", record=False).ok
        assert session.run("INSERT INTO observed VALUES (1,'alpha'),(2,'beta'),(3,NULL)", record=False).ok
        return session, source, session.schema.table_by_name("observed").name
    return session, source, next(iter(source.datasets))

def test_executed_alias_seeds_a_later_cte_predicate(result_table, monkeypatch):
    session, source, dataset = result_table
    outcome = session.run(f"SELECT label AS s FROM {dataset} WHERE id = 2", record=False)
    assert outcome.ok and outcome.rows == [["beta"]]

    def forbidden(*args, **kwargs):
        pytest.fail("A cached completion must not inspect the column")

    monkeypatch.setattr(session_module, "column_values", forbidden)
    draft = f"WITH q(s) AS (SELECT label FROM {dataset}) SELECT * FROM q x WHERE x.s = 'b"
    offers = complete(draft, len(draft), session.schema, dialect=source.dialect,
                      values=lambda table, column: session.cached_values(source, table, column))
    assert [item.insert for item in offers.items if item.kind == "value"] == ["'beta'"]
    assert session.cached_values(source, dataset, "s") is None
    assert session.cached_values(replace(source, connection_key="other"), dataset, "label") is None


@pytest.mark.parametrize("projection", ["upper(label) AS label", "CAST(id AS boolean) AS id"])
def test_transformed_projection_does_not_seed_base_column(result_table, projection):
    session, source, dataset = result_table
    assert session.run(f"SELECT {projection} FROM {dataset}", record=False).ok
    assert session.cached_values(source, dataset, "label") is None
    assert session.cached_values(source, dataset, "id") is None


def test_mixed_origin_cte_does_not_seed_first_branch(result_table):
    session, source, dataset = result_table
    sql = (f"WITH q AS (SELECT label FROM {dataset} UNION ALL SELECT 'not-in-table') "
           "SELECT label FROM q")
    assert session.run(sql, record=False).ok
    assert session.cached_values(source, dataset, "label") is None


def test_unknown_scalar_slot_does_not_shift_neighbors(result_table):
    session, source, dataset = result_table
    outcome = session.run(f"SELECT count(*), id FROM {dataset} GROUP BY id ORDER BY id", record=False)
    assert outcome.ok and outcome.rows == [[1, 1], [1, 2], [1, 3]]
    assert session.cached_values(source, dataset, "id") == ([1, 2, 3], False)
    assert session.cached_value_columns == 1


def test_using_star_does_not_attribute_a_same_width_mismatch(result_table):
    session, source, dataset = result_table
    sql = f"SELECT *, 7 FROM {dataset} a JOIN {dataset} b USING (id) ORDER BY id"
    outcome = session.run(sql, record=False)
    assert outcome.ok and outcome.rows[0] == [1, "alpha", "alpha", 7]
    observed = session.cached_values(source, dataset, "label")
    assert observed is None or observed == (["alpha", "beta", None], False)


def test_samples_are_first_seen_typed_and_bounded(result_table):
    session, source, dataset = result_table
    session.update_settings(default_rows=2, value_cache_limit=2)
    outcome = session.run(f"SELECT *, label AS again FROM {dataset} ORDER BY id DESC", record=False)
    assert outcome.ok and outcome.capped
    assert session.cached_values(source, dataset, "id") == ([3, 2], True)
    assert session.cached_values(source, dataset, "label") == ([None, "beta"], True)

    assert session.run(f"SELECT label FROM {dataset} WHERE id = 1", record=False).ok
    assert session.cached_values(source, dataset, "label") == (["alpha"], False)
    assert session.cached_values(source, dataset, "id") == ([3, 2], True)
    assert not session.run(f"SELECT absent FROM {dataset}", record=False).ok
    assert session.cached_values(source, dataset, "label") == (["alpha"], False)

    session.update_settings(default_rows=10, value_cache_limit=1)
    outcome = session.run(f"SELECT label FROM {dataset} WHERE id < 3 ORDER BY id DESC", record=False)
    assert outcome.ok and not outcome.capped
    assert session.cached_values(source, dataset, "label") == (["beta"], True)


def test_filtered_sample_does_not_replace_explicit_inspection(result_table):
    session, source, dataset = result_table
    assert session.run(f"SELECT label FROM {dataset} WHERE id = 2", record=False).ok
    assert session.cached_values(source, dataset, "label") == (["beta"], False)
    values, capped = session.distinct_values(source, dataset, "label")
    assert set(values) == {"alpha", "beta", None} and not capped
    assert session.run(f"SELECT label FROM {dataset} WHERE id = 1", record=False).ok
    assert session.cached_values(source, dataset, "label")[0][0] == "alpha"
    assert set(session.cached_values(source, dataset, "label")[0]) == set(values)


def test_clear_revokes_result_from_in_flight_run(parquet_source, monkeypatch):
    source = parquet_source
    session = Session({source.id: source})
    finished, publish = Event(), Event()
    execute = session_module.execute_sql

    def hold_result(*args, **kwargs):
        result = execute(*args, **kwargs)
        finished.set()
        assert publish.wait(5)
        return result

    monkeypatch.setattr(session_module, "execute_sql", hold_result)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(session.run, "SELECT label FROM data", record=False)
        try:
            assert finished.wait(5)
            session.clear_value_cache()
        finally:
            publish.set()
        assert future.result(timeout=5).ok
    assert session.cached_values(source, "data", "label") is None


def test_autocommit_returning_observes_inserts_but_not_deletes(result_table):
    session, source, dataset = result_table
    if source.dialect == "duckdb":
        assert session.run("CREATE TABLE observed(id int, label text)", record=False).ok
        dataset = relation_name("main", "observed")
    outcome = session.run(f"INSERT INTO {dataset} VALUES (4, 'new') RETURNING label AS s", record=False)
    assert outcome.ok and outcome.rows == [["new"]]
    assert session.cached_values(source, dataset, "label") == (["new"], False)
    outcome = session.run(f"DELETE FROM {dataset} WHERE id = 4 RETURNING label", record=False)
    assert outcome.ok and outcome.rows == [["new"]]
    assert session.cached_values(source, dataset, "label") is None


def test_result_samples_do_not_escape_rollback(postgres_source):
    source = postgres_source
    session = Session({source.id: source})
    assert session.run("CREATE TEMP TABLE observed(label text)", record=False).ok
    dataset = session.schema.table_by_name("observed").name
    assert session.run("BEGIN", record=False).ok
    try:
        inserted = session.run(f"INSERT INTO {dataset} VALUES ('private') RETURNING label", record=False)
        assert inserted.ok and inserted.rows == [["private"]]
        assert session.cached_values(source, dataset, "label") is None
        selected = session.run(f"SELECT label FROM {dataset}", record=False)
        assert selected.ok and selected.rows == [["private"]]
        assert session.cached_values(source, dataset, "label") is None
    finally:
        assert session.run("ROLLBACK", record=False).ok
    assert session.cached_values(source, dataset, "label") is None
