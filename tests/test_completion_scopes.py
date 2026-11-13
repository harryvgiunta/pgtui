"""Behavioral contracts for completion-only scope and lineage discovery."""
import pytest

from d8r.completion.context import analyze
from d8r.sql import ColumnDef, FnDef, RoutineDef, SchemaContext, TableDef


@pytest.fixture
def schema():
    return SchemaContext(
        tables=(
            TableDef("users", columns=(ColumnDef("id", "integer"), ColumnDef("name", "text"), ColumnDef("team_id", "integer"))),
            TableDef("orders", columns=(ColumnDef("id", "bigint"), ColumnDef("user_id", "integer"), ColumnDef("total", "numeric"))),
            TableDef("teams", columns=(ColumnDef("id", "integer"), ColumnDef("title", "text"))),
            TableDef('"Odd Table"', columns=(ColumnDef("Full Name", "text"), ColumnDef("MixedCase", "integer"))),
        ),
        fns=(
            FnDef("recent", ("minimum",), "SELECT id, name AS display_name FROM users WHERE id > @minimum"),
            FnDef("project", ("value",), "SELECT @value AS projected"),
            FnDef("nested", ("arg",), "SELECT * FROM project(@arg)"),
            FnDef("loop", (), "SELECT * FROM loop()"),
        ),
        routines=(RoutineDef("native_rows", ("count integer",), kind="table", columns=(ColumnDef("item", "integer"),)),),
    )


def at(sql, schema, **kwargs):
    offset = sql.index("|")
    return analyze(sql.replace("|", "", 1), offset, schema, **kwargs)


def refs(context):
    return {relation.reference: relation for relation in context.relations}


def fields(relation):
    return {column.name: column for column in relation.fields}


def test_active_statement_and_future_from(schema):
    context = at("SELECT * FROM orders old; SELECT u.| FROM users u; SELECT * FROM teams later", schema)
    assert set(refs(context)) == {"u"}
    assert context.qualifier == "u"
    assert context.clause == "select"
    assert context.statement_start == len("SELECT * FROM orders old;")
    assert context.statement_end == len("SELECT * FROM orders old; SELECT u. FROM users u")


def test_union_branches_do_not_share_aliases(schema):
    context = at("SELECT a.id FROM users a UNION ALL SELECT b.| FROM orders b", schema)
    assert set(refs(context)) == {"b"}


def test_scalar_subquery_correlates_and_aliases_shadow(schema):
    context = at("SELECT (SELECT | FROM orders o WHERE o.user_id = u.id) FROM users u", schema)
    assert set(refs(context)) == {"o", "u"}
    assert refs(context)["u"].outer
    shadowed = at("SELECT (SELECT u.| FROM orders u) FROM users u", schema)
    assert set(refs(shadowed)) == {"u"}
    assert refs(shadowed)["u"].table == "orders"
    assert not refs(shadowed)["u"].outer


def test_sibling_subqueries_are_isolated(schema):
    context = at("SELECT (SELECT id FROM users hidden), (SELECT o.| FROM orders o)", schema)
    assert set(refs(context)) == {"o"}


@pytest.mark.parametrize("lateral,expected", [("", {"o"}), ("LATERAL ", {"u", "o"})])
def test_derived_correlation_requires_lateral(schema, lateral, expected):
    context = at(f"SELECT * FROM users u, {lateral}(SELECT | FROM orders o) d", schema)
    assert set(refs(context)) == expected
    if lateral:
        assert refs(context)["u"].outer


def test_lateral_can_only_see_preceding_relations(schema):
    context = at("SELECT * FROM users u JOIN LATERAL (SELECT |) d ON true JOIN orders later ON true", schema)
    assert set(refs(context)) == {"u"}


def test_join_on_cannot_see_future_join(schema):
    context = at("SELECT * FROM users u JOIN orders o ON | JOIN teams t ON true", schema)
    assert set(refs(context)) == {"u", "o"}
    assert context.join_target.reference == "o"
    assert [r.reference for r in context.join_left] == ["u"]


def test_correlated_subquery_in_on_cannot_see_future_join(schema):
    context = at("SELECT * FROM users u JOIN orders o ON EXISTS (SELECT |) JOIN teams later ON true", schema)
    assert set(refs(context)) == {"u", "o"}
    assert all(r.outer for r in context.relations)


def test_cte_explicit_names_preserve_physical_lineage(schema):
    context = at("WITH people(user_key, display_name) AS (SELECT id, name FROM users) SELECT p.| FROM people p", schema)
    outputs = fields(refs(context)["p"])
    assert set(outputs) == {"user_key", "display_name"}
    assert outputs["display_name"].table == "users"
    assert outputs["display_name"].column == "name"
    assert outputs["user_key"].type == "integer"


def test_nested_cte_shadows_only_its_own_scope(schema):
    context = at("WITH q AS (SELECT name FROM users) SELECT (WITH q AS (SELECT total FROM orders) SELECT q.| FROM q)", schema)
    assert set(fields(refs(context)["q"])) == {"total"}
    outer = at("WITH q AS (SELECT name FROM users) SELECT q.|, (WITH q AS (SELECT total FROM orders) SELECT * FROM q) FROM q", schema)
    assert set(fields(refs(outer)["q"])) == {"name"}


def test_cte_cannot_see_forward_or_nonrecursive_self(schema):
    context = at("WITH first AS (SELECT |), later AS (SELECT id FROM users) SELECT * FROM later", schema)
    assert context.ctes == ()
    assert context.relations == ()


def test_recursive_cte_explicit_columns_available_in_body(schema):
    context = at("WITH RECURSIVE walk(node) AS (SELECT w.| FROM walk w) SELECT * FROM walk", schema)
    assert set(fields(refs(context)["w"])) == {"node"}


def test_derived_star_and_alias_type_lineage(schema):
    context = at("SELECT d.| FROM (SELECT u.*, u.name AS display_name, 1 AS literal, CAST(u.id AS bigint) AS key FROM users u) d", schema)
    derived = fields(refs(context)["d"])
    assert set(derived) == {"id", "name", "team_id", "display_name", "literal", "key"}
    assert derived["display_name"].column == "name"
    assert derived["key"].type == "bigint"
    assert derived["literal"].type == "integer"


def test_unknown_expression_does_not_invent_output_name(schema):
    context = at("SELECT d.| FROM (SELECT mystery(id), id + 1, mystery(id) AS deliberate FROM users) d", schema)
    assert set(fields(refs(context)["d"])) == {"deliberate"}


def test_derived_explicit_column_list(schema):
    context = at("SELECT d.| FROM (SELECT id, name FROM users) AS d(user_key, label)", schema)
    assert set(fields(refs(context)["d"])) == {"user_key", "label"}
    assert fields(refs(context)["d"])["label"].column == "name"


def test_saved_and_native_table_function_outputs(schema):
    local = at("SELECT r.| FROM recent(10) r", schema)
    assert set(fields(refs(local)["r"])) == {"id", "display_name"}
    assert fields(refs(local)["r"])["display_name"].column == "name"
    native = at("SELECT n.| FROM native_rows(3) n", schema)
    assert fields(refs(native)["n"])["item"].type == "integer"


def test_parameterized_saved_function_derives_actual_argument_type(schema):
    context = at("SELECT p.| FROM users u, project(u.name) p", schema)
    projected = fields(refs(context)["p"])["projected"]
    assert (projected.type, projected.table, projected.column) == ("text", "users", "name")
    numeric = at("SELECT p.| FROM project(123) p", schema)
    assert fields(refs(numeric)["p"])["projected"].type == "integer"
    unknown = at("SELECT p.| FROM project(@unknown) p", schema)
    assert fields(refs(unknown)["p"])["projected"].type == ""


def test_nested_parameterized_saved_function_preserves_binding(schema):
    context = at("SELECT n.| FROM users u, nested(u.name) n", schema)
    projected = fields(refs(context)["n"])["projected"]
    assert (projected.type, projected.column) == ("text", "name")


def test_recursive_saved_function_is_bounded_without_fake_fields(schema):
    context = at("SELECT x.| FROM loop() x", schema)
    assert refs(context)["x"].fields == ()


@pytest.mark.parametrize("dialect,reference", [("postgres", '"Odd Table"'), ("sqlite", "`Odd Table`"), ("sqlite", "[Odd Table]")])
def test_native_quoted_relation_styles(schema, dialect, reference):
    context = at(f"SELECT t.| FROM {reference} t", schema, dialect=dialect)
    assert set(fields(refs(context)["t"])) == {"Full Name", "MixedCase"}


@pytest.mark.parametrize("sql,quote,prefix,original", [
    ('SELECT u."na|me" FROM users u', '"', "na", '"name"'),
    ('SELECT u.`na|me` FROM users u', '`', "na", '`name`'),
    ('SELECT u.[na|me] FROM users u', '[', "na", '[name]'),
])
def test_identifier_prefix_and_full_replacement_span(schema, sql, quote, prefix, original):
    context = at(sql, schema, dialect="sqlite")
    document = sql.replace("|", "")
    assert context.token == prefix
    assert context.quote == quote
    assert document[context.start:context.end] == original
    assert context.qualifier == "u"
    assert not context.blocked


def test_incomplete_quote_and_unrelated_suffix_are_tolerated(schema):
    context = at('SELECT u.| FROM users u; SELECT "unfinished', schema)
    assert set(refs(context)) == {"u"}
    assert not context.blocked
    unfinished = at('SELECT * FROM users u WHERE u."na|', schema)
    assert unfinished.token == "na"
    assert unfinished.quote == '"'
    assert not unfinished.blocked


def test_postgres_case_sensitive_quoted_column_lineage(schema):
    good = at('SELECT t."MixedCase" AS key, | FROM "Odd Table" t', schema)
    assert good.outputs[0].column == "MixedCase"
    bad = at('SELECT t.mixedcase AS key, | FROM "Odd Table" t', schema)
    assert bad.outputs[0].column is None


def test_qualified_star_span_is_only_star(schema):
    sql = "SELECT u.*| FROM users u"
    context = at(sql, schema)
    assert context.token == "*"
    assert context.qualifier == "u"
    assert sql[context.start:context.end] == "*"


@pytest.mark.parametrize("sql,name,argument", [
    ("SELECT coalesce(u.name, |) FROM users u", "coalesce", 1),
    ("SELECT coalesce(substr(u.name, 1, |), '') FROM users u", "substr", 2),
    ("SELECT coalesce(substr(u.name, 1, 2), |) FROM users u", "coalesce", 1),
    ("SELECT pg_catalog.concat('a,b', |)", "pg_catalog.concat", 1),
    ("SELECT CAST(id AS |) FROM users", "CAST", 1),
    ("SELECT CAST(| AS integer) FROM users", "CAST", 0),
    ("SELECT EXTRACT(year FROM |) FROM users", "EXTRACT", 1),
    ("SELECT EXTRACT(| FROM created_at) FROM users", "EXTRACT", 0),
])
def test_current_nested_argument(schema, sql, name, argument):
    context = at(sql, schema)
    assert context.call.name == name
    assert context.call.argument == argument


@pytest.mark.parametrize("sql,clause", [
    ("INSERT INTO users (id, |) VALUES (1, 'x')", "insert_columns"),
    ("INSERT INTO users VALUES (|)", "values"),
    ("UPDATE users SET | WHERE id = 1", "update_set"),
    ("SELECT * FROM users GROUP BY |", "group"),
    ("SELECT * FROM users ORDER BY |", "order"),
    ("SELECT * FROM users HAVING |", "having"),
    ("SELECT sum(id) OVER (PARTITION BY |) FROM users", "window"),
    ("SELECT CAST(id AS |) FROM users", "type"),
    ("SELECT id::| FROM users", "type"),
])
def test_native_clause_context(schema, sql, clause):
    assert at(sql, schema).clause == clause


def test_insert_and_update_used_columns(schema):
    insert = at("INSERT INTO users (id, |) VALUES (1, 'x')", schema)
    assert insert.used_columns == ("id",)
    update = at("UPDATE users SET name = 'x', | FROM orders o WHERE users.id = o.user_id", schema)
    assert update.used_columns == ("name",)
    assert set(refs(update)) == {"users"}


def test_using_exposes_join_pair_and_used_columns(schema):
    context = at("SELECT * FROM users u JOIN orders o USING (id, |)", schema)
    assert context.clause == "using"
    assert context.used_columns == ("id",)
    assert context.join_target.reference == "o"
    assert context.join_left[0].reference == "u"
    assert context.call is None


def test_predicate_value_lineage_and_string_replacement(schema):
    sql = "WITH p AS (SELECT name AS label FROM users) SELECT * FROM p WHERE p.label = 'O''Br|ien'"
    context = at(sql, schema)
    assert context.value_field.table == "users"
    assert context.value_field.column == "name"
    assert context.literal
    assert context.value_prefix == "O'Br"
    assert sql.replace("|", "")[context.start:context.end] == "'O''Brien'"
    assert not context.blocked


def test_arbitrary_strings_and_comments_block_generic_completion(schema):
    assert at("SELECT 'not a column|", schema).blocked
    assert at("SELECT * FROM users -- hidden|", schema).blocked
    assert at("SELECT /* unfinished|", schema).blocked
    assert at("SELECT $$private body|", schema).blocked


def test_long_input_is_bounded(schema):
    context = analyze("SELECT " + " " * 200_001, 7, schema)
    assert context.blocked
    assert context.relations == ()


def test_join_target_only_after_complete_relation_token(schema):
    typed = at("SELECT * FROM users u JOIN ord|ers o", schema)
    assert typed.join_target is None
    complete = at("SELECT * FROM users u JOIN orders o |", schema)
    assert complete.join_target.reference == "o"
    assert [r.reference for r in complete.join_left] == ["u"]


def test_insert_select_does_not_expose_write_target(schema):
    context = at("INSERT INTO users (id) SELECT o.| FROM orders o", schema)
    assert context.clause == "select"
    assert set(refs(context)) == {"o"}


def test_in_list_values_keep_lineage_after_comma(schema):
    context = at("SELECT * FROM users WHERE name IN ('first', 'se|')", schema)
    assert context.value_field.column == "name"
    assert context.value_prefix == "se"
    assert not context.blocked


def test_case_insensitive_sqlite_aliases(schema):
    context = at('SELECT "U".name AS label, | FROM users u', schema, dialect="sqlite")
    assert context.outputs[0].column == "name"


@pytest.mark.parametrize("dialect,clause", [("postgres", "call"), ("duckdb", "call"), ("sqlite", "statement")])
def test_call_clause_is_native_only(schema, dialect, clause):
    context = at("CALL ref|", schema, dialect=dialect)
    assert context.clause == clause


def test_call_signature_tracks_procedure_arguments(schema):
    context = at("CALL public.refresh_cache('users', |)", schema)
    assert context.call.name == "public.refresh_cache"
    assert context.call.argument == 1


@pytest.mark.parametrize("dialect,opener,closer", [("postgres", '"', '"'), ("sqlite", "`", "`"), ("sqlite", "[", "]")])
@pytest.mark.parametrize("stale_tail", ["", "al_stale"])
def test_unclosed_identifier_recovers_suffix_and_accepts_only_identifier(schema, dialect, opener, closer, stale_tail):
    from d8r.completion.engine import complete

    marked = f"SELECT o.{opener}tot|{stale_tail} FROM orders o"
    offset = marked.index("|")
    document = marked.replace("|", "")
    result = complete(document, offset, schema, dialect=dialect)
    context = result.context
    assert context.qualifier == "o"
    assert context.token == "tot"
    assert context.quote == opener
    assert refs(context)["o"].table == "orders"
    assert document[context.start:context.end] == opener + "tot" + stale_tail
    assert document[context.end:] == " FROM orders o"
    suggestion = next(item for item in result.items if item.label == "o.total")
    start = context.start if suggestion.start is None else suggestion.start
    end = context.end if suggestion.end is None else suggestion.end
    accepted = document[:start] + suggestion.insert + document[end:]
    assert accepted == f"SELECT o.{opener}total{closer} FROM orders o"


@pytest.mark.parametrize("dialect,opener,closer", [("postgres", '"', '"'), ("sqlite", "`", "`"), ("sqlite", "[", "]")])
@pytest.mark.parametrize("separator", [" ", "\n"])
def test_closed_identifiers_containing_clauses_are_never_recovered(schema, dialect, opener, closer, separator):
    identifier = opener + "total" + separator + "FROM orders o" + closer
    document = "SELECT " + identifier
    offset = len("SELECT " + opener + "tot")
    context = analyze(document, offset, schema, dialect=dialect)
    assert document[context.start:context.end] == identifier
    assert context.relations == ()


@pytest.mark.parametrize("dialect,opener", [("postgres", '"'), ("sqlite", "`"), ("sqlite", "[")])
def test_unclosed_multiline_identifier_is_not_reinterpreted_as_sql(schema, dialect, opener):
    marked = f"SELECT o.{opener}tot|al\nFROM orders o"
    context = at(marked, schema, dialect=dialect)
    document = marked.replace("|", "")
    assert context.end == len(document)
    assert context.relations == ()


def test_recovery_ignores_unfinished_identifier_away_from_caret(schema):
    context = at('SELECT o.| FROM orders o; SELECT "tot FROM users u', schema)
    assert set(refs(context)) == {"o"}
    assert context.quote == ""
