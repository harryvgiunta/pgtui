"""Candidate behavior on immutable metadata; no database, network, or UI needed."""
from dataclasses import replace
from decimal import Decimal

import pytest

from d8r.completion import engine
from d8r.completion.catalog import functions_for, keywords_for, types_for
from d8r.completion.models import Call, Context, Field, Relation
from d8r.sql import ColumnDef, FnDef, ForeignKey, RoutineDef, SchemaContext, TableDef


@pytest.fixture
def schema():
    return SchemaContext((
        TableDef("users", columns=(ColumnDef("id", "integer"), ColumnDef("display_name", "text"),
                                   ColumnDef("email", "text"), ColumnDef("active", "boolean"))),
        TableDef("orders", columns=(ColumnDef("id", "integer"), ColumnDef("user_id", "integer"),
                                    ColumnDef("total", "numeric")),
                 foreign_keys=(ForeignKey(("user_id",), "users", ("id",)),)),
        TableDef("audit_log", columns=(ColumnDef("secret_payload", "text"),)),
    ))


def relation(schema, table, alias):
    definition = schema.table_by_name(table)
    return Relation(table, alias, tuple(Field(column.name, column.type, table, column.name,
                                             column.doc, column.values) for column in definition.columns),
                    table, definition.doc, definition.foreign_keys)


def candidates(monkeypatch, context, schema, document="", **options):
    # Scope extraction has its own suite. These tests pin candidate contracts to
    # immutable Context snapshots, including scopes impossible to get from SQL
    # until the user finishes a partially typed statement.
    monkeypatch.setattr(engine, "analyze", lambda *args, **kwargs: context)
    return engine.complete(document, context.end, schema, **options)


def by_label(result, label):
    return next(item for item in result.items if item.label == label)


def test_qualified_component_keeps_prefix_and_replaces_suffix(monkeypatch, schema):
    context = Context(9, 21, "dis", "u", clause="select", relations=(relation(schema, "users", "u"),))
    result = candidates(monkeypatch, context, schema, "SELECT u.display_name FROM users u")
    item = by_label(result, "u.display_name")
    assert item.insert == "display_name"
    assert item.start is None and item.end is None
    assert result.context.start == 9 and result.context.end == 21
    assert all(item.kind == "column" for item in result.items)


def test_unknown_qualifier_never_falls_back_to_global_columns(monkeypatch, schema):
    context = Context(15, 15, qualifier="missing", clause="select", relations=(relation(schema, "users", "u"),))
    assert not candidates(monkeypatch, context, schema).items


def test_visible_scope_excludes_other_catalog_columns_and_tables(monkeypatch, schema):
    context = Context(7, 7, clause="select", relations=(relation(schema, "users", "u"),))
    result = candidates(monkeypatch, context, schema)
    assert not any(item.label in {"secret_payload", "orders", "audit_log"} for item in result.items)
    assert by_label(result, "display_name").kind == "column"


def test_ambiguous_columns_always_offer_explicit_relations(monkeypatch, schema):
    context = Context(7, 9, "id", clause="select",
                      relations=(relation(schema, "users", "u"), relation(schema, "orders", "o")))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "u.id").insert == "u.id"
    assert by_label(result, "o.id").insert == "o.id"
    assert not any(item.label == "id" for item in result.items)


def test_derived_fields_not_physical_parent_columns(monkeypatch, schema):
    derived = Relation("derived", "d", (Field("gross", "numeric", "orders", "total"),))
    context = Context(9, 9, qualifier="d", clause="select", relations=(derived,))
    result = candidates(monkeypatch, context, schema)
    assert [(item.label, item.insert) for item in result.items] == [("d.gross", "gross")]


def test_global_columns_remain_useful_before_a_source(monkeypatch, schema):
    result = candidates(monkeypatch, Context(7, 10, "dis", clause="select"), schema)
    assert "users" in by_label(result, "display_name").detail


@pytest.mark.parametrize("token,name", [("dspnam", "display_name"), ("dn", "display_name"),
                                         ("dispaly", "display_name"), ("name", "display_name")])
def test_fuzzy_typo_and_abbreviation_matching(monkeypatch, schema, token, name):
    context = Context(7, 7 + len(token), token, clause="select", relations=(relation(schema, "users", "u"),))
    assert by_label(candidates(monkeypatch, context, schema), name)


def test_exact_and_scope_match_precede_history(monkeypatch, schema):
    inner = Relation("inner", "i", (Field("name", "text", "users", "name"), Field("names", "text")))
    outer = Relation("outer", "o", (Field("name_old", "text", "orders", "name_old"),), outer=True)
    context = Context(7, 11, "name", clause="select", relations=(inner, outer))
    usage = {("column", "orders", "name_old"): 0, ("column", "users", "name"): 99}
    items = candidates(monkeypatch, context, schema, usage=usage).items
    assert [item.label for item in items[:3]] == ["name", "names", "name_old"]


def test_successful_source_usage_breaks_equal_match_ties(monkeypatch, schema):
    fields = Relation("r", "r", (Field("amount", "numeric", "orders", "amount"),
                                    Field("active", "boolean", "users", "active")))
    context = Context(7, 8, "a", clause="select", relations=(fields,))
    first = candidates(monkeypatch, context, schema, usage={("column", "orders", "amount"): 0})
    second = candidates(monkeypatch, context, schema, usage={("column", "users", "active"): 0})
    assert first.items[0].label == "amount"
    assert second.items[0].label == "active"


def test_results_bounded_with_large_scoped_snapshot(monkeypatch):
    scope = Relation("wide", "w", tuple(Field(f"column_{index:05}") for index in range(3000)))
    result = candidates(monkeypatch, Context(7, 8, "c", clause="select", relations=(scope,)), SchemaContext())
    assert len(result.items) == 80
    assert result.items[0].label == "column_00000"


def test_relation_positions_exclude_scalar_functions_and_columns(monkeypatch, schema):
    result = candidates(monkeypatch, Context(14, 14, clause="from"), schema)
    assert not any(item.kind == "column" or item.label in {"lower", "sum"} for item in result.items)
    assert by_label(result, "users").kind == "table"
    assert by_label(result, "generate_series").kind == "function"


def test_schema_qualification_replaces_component_only(monkeypatch):
    schema = SchemaContext((TableDef('"Sales Data"."Order Items"'), TableDef("public.other")))
    context = Context(25, 29, "Ord", '"Sales Data"', quote='"', clause="from")
    item = by_label(candidates(monkeypatch, context, schema), "Order Items")
    assert item.insert == '"Order Items"'
    assert not any(item.label == "other" for item in candidates(monkeypatch, context, schema).items)


@pytest.mark.parametrize("quote,expected", [('"', '"select"'), ("`", "`select`"), ("[", "[select]")])
def test_identifier_quote_style_preserved(monkeypatch, quote, expected):
    scope = Relation("r", "r", (Field("select"),))
    context = Context(7, 10, "sel", quote=quote, clause="select", relations=(scope,))
    assert by_label(candidates(monkeypatch, context, SchemaContext()), "select").insert == expected


def test_alias_insertions_avoid_collisions_and_existing_suffix(monkeypatch, schema):
    context = Context(14, 16, "us", clause="from", identifiers=("u", "u2"))
    result = candidates(monkeypatch, context, schema, "SELECT * FROM us")
    assert by_label(result, "users AS u3").insert == "users AS u3"
    result = candidates(monkeypatch, context, schema, "SELECT * FROM us AS chosen")
    assert not any(" AS " in item.label for item in result.items)


def test_cte_relation_candidates(monkeypatch, schema):
    cte = Relation("recent", "recent", (Field("total", "numeric"),))
    result = candidates(monkeypatch, Context(14, 16, "re", clause="from", ctes=(cte,)), schema)
    assert by_label(result, "recent").kind == "cte"


@pytest.mark.parametrize("reverse", [False, True])
def test_composite_foreign_key_join_both_directions(monkeypatch, reverse):
    left = Relation("parent", "p", (Field("tenant_id"), Field("id")), "parent")
    right = Relation("child", "c", (Field("tenant_id"), Field("parent_id")), "child",
                     foreign_keys=(ForeignKey(("tenant_id", "parent_id"), "parent", ("tenant_id", "id")),))
    target, previous = (left, right) if reverse else (right, left)
    context = Context(50, 50, clause="on", relations=(previous, target),
                      join_target=target, join_left=(previous,))
    joins = [item for item in candidates(monkeypatch, context, SchemaContext()).items if item.kind == "join"]
    assert len(joins) == 1
    assert joins[0].insert == "c.tenant_id = p.tenant_id AND c.parent_id = p.id"


def test_using_offers_only_shared_unused_columns(monkeypatch, schema):
    left, right = relation(schema, "users", "u"), relation(schema, "orders", "o")
    context = Context(50, 50, clause="using", relations=(left, right), join_target=right, join_left=(left,))
    assert [item.label for item in candidates(monkeypatch, context, schema).items] == ["id"]
    assert not candidates(monkeypatch, replace(context, used_columns=("id",)), schema).items


def test_qualified_star_expands_ordered_visible_fields_and_whole_span(monkeypatch, schema):
    document = "SELECT u.* FROM users u"
    context = Context(9, 10, "*", "u", clause="select", relations=(relation(schema, "users", "u"),))
    item = candidates(monkeypatch, context, schema, document).items[0]
    assert item.start == 7 and item.end == 10
    assert item.insert == "u.id, u.display_name, u.email, u.active"
    assert document[:item.start] + item.insert + document[item.end:] == (
        "SELECT u.id, u.display_name, u.email, u.active FROM users u")


def test_bare_star_qualifies_ambiguous_fields_and_excludes_outer(monkeypatch, schema):
    context = Context(7, 8, "*", clause="select", relations=(relation(schema, "users", "u"),
                      relation(schema, "orders", "o"), Relation("outer", "x", (Field("hidden"),), outer=True)))
    item = candidates(monkeypatch, context, schema, "SELECT *").items[0]
    assert "u.id" in item.insert and "o.id" in item.insert
    assert "hidden" not in item.insert


def test_insert_missing_columns_and_update_assignments(monkeypatch, schema):
    context = Context(19, 19, clause="insert_columns", relations=(relation(schema, "users", "users"),),
                      used_columns=("id", "email"))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "All missing columns").insert == "display_name, active"
    assert not any(item.label in {"id", "email"} for item in result.items)
    result = candidates(monkeypatch, replace(context, clause="update_set", token="dis"), schema)
    assert by_label(result, "display_name =").insert == "display_name = "


@pytest.mark.parametrize("clause", ["group", "order"])
def test_group_order_output_aliases(monkeypatch, schema, clause):
    context = Context(50, 53, "gro", clause=clause, relations=(relation(schema, "orders", "o"),),
                      outputs=(Field("gross", "numeric"),))
    assert by_label(candidates(monkeypatch, context, schema), "gross").kind == "alias"


def test_routine_overload_signature_and_argument_type_priority(monkeypatch, schema):
    routines = (RoutineDef("app.score", ("value text",), "integer", "Text score", parameter_types=("text",)),
                RoutineDef("app.score", ("value text", "weight numeric"), "numeric", "Weighted score",
                           parameter_types=("text", "numeric")))
    schema = replace(schema, routines=routines)
    scope = Relation("r", "r", (Field("aaa_text", "text"), Field("zzz_amount", "numeric")))
    context = Context(20, 20, clause="select", relations=(scope,), call=Call("app.score", 1))
    result = candidates(monkeypatch, context, schema)
    assert result.signature.active_parameter == 1
    assert result.signature.return_type == "numeric"
    assert result.signature.documentation == "Weighted score"
    assert result.items[0].label == "zzz_amount"
    assert by_label(result, "aaa_text")  # type relevance is not a validity filter


def test_variadic_active_argument_stays_on_variadic_parameter(monkeypatch):
    schema = SchemaContext(routines=(RoutineDef("many", ("value text",), "text", variadic=True,
                                              parameter_types=("text",)),))
    result = candidates(monkeypatch, Context(10, 10, clause="select", call=Call("many", 4)), schema)
    assert result.signature.active_parameter == 0


def test_signature_survives_no_matching_candidates(monkeypatch, schema):
    context = Context(13, 20, "zzzzzzz", clause="select", call=Call("lower", 0))
    result = candidates(monkeypatch, context, schema)
    assert not result.items
    assert result.signature.active_parameter == 0


def test_existing_call_parenthesis_not_duplicated(monkeypatch, schema):
    context = Context(7, 12, "low", clause="select")
    item = by_label(candidates(monkeypatch, context, schema, "SELECT lower(name)"), "lower")
    assert item.insert == "lower(" and item.cursor_back == 0
    assert item.end == len("SELECT lower(")
    assert "SELECT lower(name)"[:7] + item.insert + "SELECT lower(name)"[item.end:] == "SELECT lower(name)"


def test_saved_relation_function_uses_native_call_and_signature(monkeypatch):
    schema = SchemaContext(fns=(FnDef("recent", ("days",), "SELECT 1", "Recent rows"),))
    item = by_label(candidates(monkeypatch, Context(14, 16, "re", clause="from"), schema), "recent")
    assert item.insert == "recent()" and item.cursor_back == 1
    signature = candidates(monkeypatch, Context(20, 20, clause="from", call=Call("recent", 0)), schema).signature
    assert signature.parameters == ("days",)


def test_native_catalog_dialect_separation():
    postgres = {routine.name for routine in functions_for("postgres")}
    sqlite = {routine.name for routine in functions_for("d1")}
    duckdb = {routine.name for routine in functions_for("duckdb")}
    assert "jsonb_build_object" in postgres and "jsonb_build_object" not in sqlite | duckdb
    assert "julianday" in sqlite and "julianday" not in postgres | duckdb
    assert "read_parquet" in duckdb and "read_parquet" not in postgres | sqlite
    assert "QUALIFY" in keywords_for("duckdb") and "QUALIFY" not in keywords_for("postgres")
    assert "PRAGMA" in keywords_for("d1") and "PRAGMA" not in keywords_for("postgres")
    assert "jsonb" in types_for("postgres") and "jsonb" not in types_for("duckdb")


def test_cast_context_offers_native_types_not_columns(monkeypatch, schema):
    result = candidates(monkeypatch, Context(15, 18, "jso", clause="type"), schema)
    assert by_label(result, "jsonb").kind == "type"
    assert all(item.kind == "type" for item in result.items)
    result = candidates(monkeypatch, Context(15, 18, "jso", clause="type"), schema, dialect="d1")
    assert not result.items


def test_predicate_values_are_cache_only_and_safely_quoted(monkeypatch, schema):
    accesses = []

    def cache(table, column):
        accesses.append((table, column))
        return (["O'Reilly", "Other"], False)

    field = Field("display_name", "text", "users", "display_name")
    context = Context(30, 33, "O", clause="where", literal=True, blocked=True,
                      value_field=field, value_prefix="O")
    result = candidates(monkeypatch, context, schema, values=cache)
    assert accesses == [("users", "display_name")]
    assert by_label(result, "'O''Reilly'").insert == "'O''Reilly'"
    assert all(item.kind == "value" for item in result.items)


def test_enum_values_null_booleans_numbers_and_unsafe_objects(monkeypatch):
    field = Field("status", values=(None, True, 12, Decimal("2.50"), float("nan"), "bad\x00text", object()))
    context = Context(30, 30, clause="where", value_field=field)
    result = candidates(monkeypatch, context, SchemaContext())
    literals = {item.insert for item in result.items if item.kind == "value"}
    assert literals == {"NULL", "TRUE", "12", "2.50"}


@pytest.mark.parametrize("dialect,expected", [("postgres", "E'C:\\\\tmp'"),
                                               ("d1", "'C:\\tmp'"), ("duckdb", "'C:\\tmp'")])
def test_value_backslashes_remain_literal(monkeypatch, dialect, expected):
    context = Context(10, 10, clause="where", literal=True, value_field=Field("path", values=("C:\\tmp",)))
    result = candidates(monkeypatch, context, SchemaContext(), dialect=dialect)
    assert result.items[0].insert == expected


def test_blocked_comments_never_consult_cache(monkeypatch, schema):
    def forbidden(*args):
        raise AssertionError("Comment completion must not read values")

    assert not candidates(monkeypatch, Context(10, 10, blocked=True), schema, values=forbidden).items


def test_context_snippets_and_parameters(monkeypatch, schema):
    result = candidates(monkeypatch, Context(7, 10, "cas", clause="select"), schema)
    item = by_label(result, "CASE WHEN")
    assert item.insert[:len(item.insert) - item.cursor_back] == "CASE WHEN "
    result = candidates(monkeypatch, Context(7, 10, "@mi", clause="select"), schema,
                        parameters=("minimum", "maximum"))
    assert [(item.label, item.insert) for item in result.items] == [("@minimum", "@minimum")]


def test_end_to_end_mid_token_and_source_scope(schema):
    document = "SELECT u.display_name FROM users u"
    result = engine.complete(document, len("SELECT u.dis"), schema)
    item = by_label(result, "u.display_name")
    start = result.context.start if item.start is None else item.start
    end = result.context.end if item.end is None else item.end
    assert document[:start] + item.insert + document[end:] == document
    assert not any("secret_payload" in item.label for item in result.items)


def test_end_to_end_cte_output_and_native_function_help(schema):
    document = "WITH q AS (SELECT total AS gross FROM orders) SELECT q.gr FROM q"
    result = engine.complete(document, document.index("q.gr") + 4, schema)
    assert by_label(result, "q.gross").insert == "gross"
    result = engine.complete("SELECT lower(", len("SELECT lower("), schema)
    assert result.signature and result.signature.active_parameter == 0


def test_discovered_catalog_controls_function_availability(monkeypatch):
    schema = SchemaContext(routines=(RoutineDef("app.available", ("value text",), "text"),))
    result = candidates(monkeypatch, Context(7, 7, clause="select"), schema)
    assert by_label(result, "app.available").kind == "function"
    assert not any(item.kind == "function" and item.label == "lower" for item in result.items)


def test_source_function_usage_uses_native_unqualified_history_key(monkeypatch):
    schema = SchemaContext(routines=(RoutineDef("pg_catalog.alpha"), RoutineDef("pg_catalog.another")))
    result = candidates(monkeypatch, Context(7, 8, "a", clause="select"), schema,
                        usage={("function", "", "another"): 0})
    assert result.items[0].label == "another"


def test_call_and_alias_suffixes_ignore_comments(monkeypatch, schema):
    context = Context(7, 12, "low", clause="select")
    result = candidates(monkeypatch, context, schema, "SELECT lower /* nested /* note */ */ (name)")
    item = by_label(result, "lower")
    assert item.insert == "lower /* nested /* note */ */ ("
    assert item.end == len("SELECT lower /* nested /* note */ */ (")
    assert item.cursor_back == 0
    context = Context(14, 16, "us", clause="from")
    result = candidates(monkeypatch, context, schema, "SELECT * FROM us /* note */ AS u")
    assert not any(" AS " in item.label for item in result.items)


def test_quoted_identifier_has_no_bare_keywords(monkeypatch, schema):
    context = Context(7, 10, "sel", quote='"', clause="select",
                      relations=(Relation("r", "r", (Field("select"),)),))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "select").insert == '"select"'
    assert not any(item.kind in {"keyword", "snippet"} for item in result.items)


def test_fk_join_on_after_target_and_projection_list(monkeypatch, schema):
    users, orders = relation(schema, "users", "u"), relation(schema, "orders", "o")
    context = Context(40, 40, clause="join", relations=(users, orders), join_target=orders, join_left=(users,))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "ON o.user_id = u.id").insert == "ON o.user_id = u.id"
    context = Context(7, 7, clause="select", relations=(users,))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "u.* columns").insert == "u.id, u.display_name, u.email, u.active"


def test_native_expression_help_survives_populated_routine_catalog(monkeypatch):
    schema = SchemaContext(routines=(RoutineDef("pg_catalog.lower", ("text text",), "text"),))
    context = Context(7, 10, "coa", clause="select")
    assert by_label(candidates(monkeypatch, context, schema), "coalesce").insert == "coalesce()"
    context = Context(20, 20, clause="type", call=Call("cast", 1))
    result = candidates(monkeypatch, context, schema)
    assert result.signature.active_parameter == 1
    context = Context(15, 15, clause="select", call=Call("extract", 0))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "YEAR").insert == "YEAR"


def test_extract_uses_native_syntax_not_function_commas(monkeypatch, schema):
    result = candidates(monkeypatch, Context(7, 10, "ext", clause="select"), schema)
    item = by_label(result, "EXTRACT")
    assert item.insert == "EXTRACT( FROM )"
    assert item.insert[:len(item.insert) - item.cursor_back] == "EXTRACT("
    result = candidates(monkeypatch, Context(7, 10, "ext", clause="select"), schema, dialect="d1")
    assert not any(item.label == "EXTRACT" for item in result.items)


def test_native_procedures_are_only_offered_after_call(monkeypatch):
    schema = SchemaContext(routines=(RoutineDef("refresh_data", ("tenant integer",), kind="procedure"),
                                    RoutineDef("lookup", ("key integer",), "text")))
    result = candidates(monkeypatch, Context(5, 5, clause="call"), schema)
    assert [item.label for item in result.items] == ["refresh_data"]
    result = candidates(monkeypatch, Context(7, 7, clause="select"), schema)
    assert not any(item.label == "refresh_data" for item in result.items)
    result = candidates(monkeypatch, Context(18, 18, clause="call", call=Call("refresh_data", 0)), schema)
    assert by_label(result, "lookup").insert == "lookup()"
    assert result.signature.parameters == ("tenant integer",)


def test_table_function_arguments_offer_expressions_not_relations(monkeypatch, schema):
    context = Context(30, 30, clause="from", call=Call("generate_series", 0))
    result = candidates(monkeypatch, context, schema)
    assert by_label(result, "lower").kind == "function"
    assert not any(item.kind == "table" for item in result.items)
    assert not any(item.label == "GROUP BY" for item in result.items)


def test_interval_is_not_mistaken_for_integer_type(monkeypatch):
    schema = SchemaContext(routines=(RoutineDef("elapsed", ("duration interval",), "numeric",
                                              parameter_types=("interval",)),))
    fields = Relation("r", "r", (Field("aaa_number", "integer"), Field("zzz_duration", "interval")))
    context = Context(15, 15, clause="select", relations=(fields,), call=Call("elapsed", 0))
    assert candidates(monkeypatch, context, schema).items[0].label == "zzz_duration"


def test_prefix_matches_suppress_typo_noise(monkeypatch, schema):
    result = candidates(monkeypatch, Context(0, 3, "SEL", clause="statement"), schema)
    assert result.items[0].label == "SELECT"
    assert by_label(result, "SELECT FROM").kind == "snippet"
    assert not any(item.label == "DELETE" for item in result.items)
    result = candidates(monkeypatch, Context(7, 10, "sub", clause="select"), schema)
    assert by_label(result, "substr")
    assert not any(item.label.casefold() == "sum" for item in result.items)
    result = candidates(monkeypatch, Context(7, 10, "@mi", clause="select"), schema,
                        parameters=("minimum", "maximum"))
    assert [item.label for item in result.items] == ["@minimum"]


def test_typo_matches_remain_available_without_stronger_candidates(monkeypatch, schema):
    context = Context(7, 14, "dispaly", clause="select", relations=(relation(schema, "users", "u"),))
    assert by_label(candidates(monkeypatch, context, schema), "display_name")


def test_empty_statement_only_offers_statement_syntax(monkeypatch, schema):
    result = candidates(monkeypatch, Context(0, 0, clause="statement"), schema)
    assert by_label(result, "SELECT").kind == "keyword"
    assert all(item.kind in {"keyword", "snippet"} for item in result.items)
    assert not any(item.label in {"display_name", "lower", "COUNT(*)", "CAST", "CASE WHEN"}
                   for item in result.items)


@pytest.mark.parametrize("namespace,dialect", [("pg_catalog", "postgres"), ("main", "duckdb")])
def test_implicit_native_routines_have_bare_labels_with_provenance(monkeypatch, namespace, dialect):
    full_name = f'"{namespace}"."lower"'
    schema = SchemaContext(default_schema=namespace,
                           routines=(RoutineDef(full_name, ("text text",), "text"),))
    result = candidates(monkeypatch, Context(7, 10, "low", clause="select"), schema, dialect=dialect)
    item = by_label(result, "lower")
    assert item.insert == "lower()"
    assert full_name in item.detail


def test_ambiguous_native_routine_names_stay_qualified(monkeypatch):
    schema = SchemaContext(default_schema="public", routines=(RoutineDef("pg_catalog.lower"),
                                                             RoutineDef("public.lower")))
    result = candidates(monkeypatch, Context(7, 10, "low", clause="select"), schema)
    assert by_label(result, "pg_catalog.lower").insert == "pg_catalog.lower()"
    assert by_label(result, "public.lower").insert == "public.lower()"
    assert not any(item.label == "lower" for item in result.items)


def test_native_relation_call_cannot_accidentally_select_saved_template(monkeypatch):
    schema = SchemaContext(default_schema="public", fns=(FnDef("report", (), "SELECT 1"),),
                           routines=(RoutineDef("public.report", kind="table"),))
    result = candidates(monkeypatch, Context(14, 17, "rep", clause="from"), schema)
    assert by_label(result, "public.report").insert == "public.report()"
    assert by_label(result, "report").insert == "report()"


@pytest.mark.parametrize("document", [
    "SELECT count(*) FROM users", "SELECT id * 2 FROM users",
    "SELECT * EXCLUDE (id) FROM users", "SELECT u.* REPLACE (0 AS id) FROM users u",
])
def test_nonprojection_stars_do_not_offer_invalid_column_expansion(schema, document):
    result = engine.complete(document, document.index("*") + 1, schema, dialect="duckdb")
    assert not any(item.kind == "expansion" for item in result.items)


def test_distinct_on_projection_star_keeps_expansion(schema):
    document = "SELECT DISTINCT ON (id) * FROM users"
    result = engine.complete(document, document.index("*") + 1, schema, dialect="postgres")
    item = next(item for item in result.items if item.kind == "expansion")
    assert "display_name" in item.insert and "email" in item.insert


@pytest.mark.parametrize("operator", ["IS DISTINCT FROM", "IS NOT DISTINCT FROM"])
def test_null_safe_comparisons_keep_expression_scope(schema, operator):
    document = f"SELECT * FROM users u WHERE u.email {operator} u.dis"
    result = engine.complete(document, len(document), schema, dialect="postgres")
    assert result.context.clause == "where"
    assert any(item.label == "u.display_name" for item in result.items)
    assert not any(item.kind == "table" for item in result.items)
