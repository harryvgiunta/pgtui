"""Offline checks for the post-execution projection proof."""

import pytest

from d8r.completion.lineage import projection_origins
from d8r.sql import ColumnDef


def _datasets(*ids):
    return {dataset: {"columns": (ColumnDef("ID", "INTEGER"), ColumnDef("label", "TEXT"))}
            for dataset in ids}


@pytest.mark.parametrize(("dialect", "dataset", "table"), [
    ("d1", '"Records"', 'main."records"'),
    ("duckdb", "data", "MAIN.data"),
])
def test_aliases_and_scalar_slots(dialect, dataset, table):
    datasets = _datasets(dataset)
    assert projection_origins(
        f"SELECT t.id first_id, upper(t.label), CAST(id AS INT) AS copied, t.label AS final FROM {table} AS t",
        dialect, datasets, ("first_id", "upper(t.label)", "copied", "final"),
    ) == ((dataset, "ID"), None, None, (dataset, "label"))
    assert projection_origins(
        f"SELECT count(*), id FROM {table} GROUP BY id", dialect, datasets, ("count(*)", "ID"),
    ) == (None, (dataset, "ID"))
    assert projection_origins(f"SELECT * FROM {table}", dialect, datasets, ("ID", "label")) == (
        (dataset, "ID"), (dataset, "label"),
    )


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_fixed_width_cte_and_derived_renames(dialect):
    sql = """WITH c(renamed, unknown) AS (SELECT label, upper(label) FROM records)
             SELECT d.x AS final, d.y
             FROM (SELECT renamed AS x, unknown AS y FROM c) d"""
    assert projection_origins(sql, dialect, _datasets('"records"'), ("final", "y")) == (
        ('"records"', "label"), None,
    )


def test_duckdb_derived_column_alias_list():
    assert projection_origins(
        "SELECT d.x, d.y FROM (SELECT label, CAST(id AS INT) FROM data) AS d(x, y)",
        "duckdb", _datasets("data"), ("x", "y"),
    ) == (("data", "label"), None)


@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_insert_returning_only_attributes_target_columns(dialect):
    assert projection_origins(
        "INSERT INTO records(label) VALUES ('one'), ('two') RETURNING ID AS new_id, upper(label), label;",
        dialect, _datasets('"records"'), ("new_id", "upper(label)", "label"),
    ) == (('"records"', "ID"), None, ('"records"', "label"))



@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_join_and_fixed_projection_star(dialect):
    datasets = _datasets('"records"', '"other"')
    assert projection_origins(
        "SELECT r.label AS left_label, o.label AS right_label "
        "FROM records r JOIN other o ON NOT (r.id <> o.id)",
        dialect, datasets, ("left_label", "right_label"),
    ) == (('"records"', "label"), ('"other"', "label"))
    assert projection_origins(
        "WITH c AS (SELECT label AS name, upper(label) AS display FROM records) SELECT c.*, 0 FROM c",
        dialect, datasets, ("name", "display", "0"),
    ) == (('"records"', "label"), None, None)

@pytest.mark.parametrize("dialect", ["sqlite", "duckdb"])
def test_same_width_does_not_make_an_unproven_shape_safe(dialect):
    datasets = _datasets('"records"', '"other"')
    for sql, headers in (
        ("SELECT upper(label) AS label FROM records", ("label",)),
        ("SELECT id FROM records UNION ALL SELECT id FROM other", ("ID",)),
        ("SELECT * FROM records NATURAL JOIN other", ("ID", "label")),
        ("SELECT * FROM records JOIN other USING (id, label)", ("ID", "label")),
        ("SELECT * FROM records", ("label", "ID")),
        ("WITH c(x, y) AS (SELECT * FROM records) SELECT x, y FROM c", ("x", "y")),
        ("SELECT d.x FROM (SELECT *, id AS x FROM records) d", ("x",)),
        ("SELECT r.id FROM records r CROSS JOIN json_each('[1]') d", ("ID",)),
    ):
        assert projection_origins(sql, dialect, datasets, headers) == (None,) * len(headers)


def test_duckdb_expansions_do_not_get_treated_as_scalars():
    for sql, headers in (
        ("SELECT COLUMNS('^label$'), id FROM data", ("label", "ID")),
        ("SELECT * EXCLUDE (id), id FROM data", ("label", "ID")),
        ("SELECT unnest({'x': id}), id FROM data", ("x", "ID")),
        ("SELECT v.*, id FROM (SELECT id, {'x': label} AS v FROM data) d", ("x", "ID")),
        ("SELECT t.x FROM data AS t(x, y)", ("x",)),
    ):
        assert projection_origins(sql, "duckdb", _datasets("data"), headers) == (None,) * len(headers)


def test_sqlite_forward_cte_name_cannot_fall_through_to_catalog():
    assert projection_origins(
        "WITH c AS (SELECT id FROM records), records AS (SELECT id FROM other) SELECT id FROM c",
        "sqlite", _datasets('"records"', '"other"'), ("ID",),
    ) == (None,)


def test_sqlite_blob_literal_cannot_be_mistaken_for_an_implicit_alias():
    datasets = {'"records"': {"columns": (ColumnDef("x", "TEXT"),)}}
    assert projection_origins(
        "WITH c(name) AS (SELECT x'61' FROM records) SELECT name FROM c", "sqlite", datasets, ("name",),
    ) == (None,)


def test_sqlite_and_duckdb_identifier_folding_is_ascii_only():
    for dialect in ("sqlite", "duckdb"):
        datasets = {'"records"': {"columns": (ColumnDef("Ü", "TEXT"), ColumnDef("ü", "TEXT"))}}
        assert projection_origins(
            'SELECT t."ü" AS seen FROM records AS t', dialect, datasets, ("seen",),
        ) == (('"records"', "ü"),)
