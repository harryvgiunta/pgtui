"""PostgreSQL catalog edges: conflict arbiters and overload identity."""
from __future__ import annotations

import pytest

from d8r.completion.engine import complete
from d8r.engine import EngineError, postgres_definition, refresh_source_schema, transaction_open
from d8r.sql import IndexDef, SchemaContext, TableDef
from tests.test_routine_metadata import named, postgres_routine_source


def test_conflict_completion_excludes_indexes_that_cannot_arbitrate_the_bare_keys():
    schema = SchemaContext((TableDef('"public"."Target"', indexes=(
        IndexDef('"public"."standalone"', "btree", ("Tenant ID", "id"), ("payload",), unique=True),
        IndexDef('"public"."partial"', "btree", ("state",), unique=True, partial=True),
        IndexDef('"public"."expression"', "btree", (None,), unique=True),
        IndexDef('"public"."invalid"', "btree", ("id",), unique=True, valid=False),
        IndexDef('"public"."deferred"', "btree", ("slug",), unique=True, immediate=False),
        IndexDef('"public"."overlapping"', "btree", ("slug",), unique=True),
        IndexDef('"public"."nonunique"', "btree", ("payload",)),
    )), TableDef('"other"."Target"', indexes=(
        IndexDef('"other"."unique_other"', "btree", ("wrong",), unique=True),
    ))), default_schema="public")
    sql = 'WITH "Target" AS (SELECT 1) INSERT INTO "Target" SELECT 1 ON CONFLICT '
    assert [item.insert for item in complete(sql, len(sql), schema).items] == ['("Tenant ID", id)']
    sql += '("Ten") DO NOTHING'
    caret = sql.index('Ten') + len('Ten')
    result = complete(sql, caret, schema)
    item, = result.items
    assert sql[:result.context.start] + item.insert + sql[result.context.end:] == (
        'WITH "Target" AS (SELECT 1) INSERT INTO "Target" SELECT 1 ON CONFLICT ("Tenant ID", id) DO NOTHING'
    )


def test_definition_uses_pinned_overload_and_leaves_user_transaction_open(postgres_routine_source):
    source = postgres_routine_source
    source.con.execute("""
        CREATE FUNCTION pg_temp.definition_target(integer) RETURNS integer
        LANGUAGE SQL AS $$ SELECT 1 / 0 $$;
        CREATE FUNCTION pg_temp.definition_target(text) RETURNS text
        LANGUAGE SQL AS $$ SELECT 'another overload'::text $$
    """)
    refresh_source_schema(source)
    target = next(r for r in named(source, "definition_target") if r.parameter_types == ("integer",))
    source.con.execute("BEGIN")
    try:
        source.con.execute("INSERT INTO pg_temp.completion_result VALUES (1, 'keep')")
        assert "1 / 0" in postgres_definition(source, "routine", target.name, oid=target.oid)
        source.con.execute("DROP FUNCTION pg_temp.definition_target(integer)")
        with pytest.raises(EngineError, match="no longer visible"):
            postgres_definition(source, "routine", target.name, oid=target.oid)
        assert transaction_open(source)
        assert source.con.execute("SELECT label FROM pg_temp.completion_result").fetchone() == ("keep",)
    finally:
        source.con.execute("ROLLBACK")
