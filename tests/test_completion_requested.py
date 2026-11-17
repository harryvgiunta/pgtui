"""Consumer regressions for inline commands, relation edits and cached predicates."""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from d8r.completion.context import analyze, command_span
from d8r.completion.engine import complete, function_insertion
from d8r.sql import ColumnDef, FnDef, SchemaContext, TableDef
from d8r.tui.palette import CommandPalette, view_for


@pytest.fixture
def schema():
    return SchemaContext(tables=(
        TableDef("public.dataset_symbol", columns=(
            ColumnDef("id", "integer"), ColumnDef("display_name", "text"),
            ColumnDef("select", "text"))),
        TableDef("public.other", columns=(ColumnDef("id", "integer"),)),
    ))


@pytest.fixture
def typed_schema(schema):
    return SchemaContext(tables=schema.tables, default_schema="public", fns=(
        FnDef("project_field", ("source", "field"), "SELECT @field AS value FROM @source",
              param_types=("relation", "identifier")),
    ))


def completion(marked, schema, **options):
    offset = marked.index("|")
    document = marked.replace("|", "", 1)
    return document, complete(document, offset, schema, **options)


def accepted(document, result, item):
    start = result.context.start if item.start is None else item.start
    end = result.context.end if item.end is None else item.end
    text = document[:start] + item.insert + document[end:]
    return text, start + len(item.insert) - item.cursor_back


def table_item(result):
    return next(item for item in result.items if item.table == "public.dataset_symbol" and item.kind == "table")


@pytest.mark.parametrize("marked,prefix", [
    ("dataset_sy|mbol; SELECT 2", ""),
    ("SELECT 1;\n dataset_sy|mbol; SELECT 2", "SELECT 1;\n "),
    ("pub|; SELECT 2", ""),
    ("public|; SELECT 2", ""),
    ("public.|; SELECT 2", ""),
    ('/* lead */ public."dataset_sy|mbol"; SELECT 2', "/* lead */ "),
])
def test_postgres_fresh_relation_becomes_select_with_projection_caret(schema, marked, prefix):
    document, result = completion(marked, schema)
    text, caret = accepted(document, result, table_item(result))
    assert text == prefix + 'SELECT  FROM "public"."dataset_symbol" AS ds; SELECT 2'
    assert text[:caret] == prefix + "SELECT "
    assert text[caret:].startswith(' FROM "public"."dataset_symbol" AS ds')


def test_fresh_relation_preserves_existing_alias(schema):
    document, result = completion("dataset_sy|mbol AS chosen WHERE chosen.id = 1", schema)
    text, caret = accepted(document, result, table_item(result))
    assert text == 'SELECT  FROM "public"."dataset_symbol" AS chosen WHERE chosen.id = 1'
    assert caret == len("SELECT ")


@pytest.mark.parametrize("dialect", ["d1", "duckdb"])
def test_fresh_select_shortcut_is_postgres_only(schema, dialect):
    _, result = completion("dataset_sy|", schema, dialect=dialect)
    assert not any(item.kind == "table" for item in result.items)


@pytest.mark.parametrize("dialect", ["postgres", "duckdb", "sqlite"])
@pytest.mark.parametrize("marked,expected", [
    ("|", "SELECT * FROM rows_above(|);"),
    ("|;", "SELECT * FROM rows_above(|);"),
    ("| -- keep comment", "SELECT * FROM rows_above(|); -- keep comment"),
    ("SELECT 1;\n-- next query\n  |; SELECT 2", "SELECT 1;\n-- next query\n  SELECT * FROM rows_above(|); SELECT 2"),
    ("SELECT * FROM | WHERE true", "SELECT * FROM rows_above(|) WHERE true"),
    ("SELECT * FROM data JOIN | ON true", "SELECT * FROM data JOIN rows_above(|) ON true"),
    ("SELECT * FROM wrong| /* arguments */ (2) AS r", "SELECT * FROM rows_above /* arguments */ (|2) AS r"),
])
def test_picked_local_function_is_a_native_relation_edit(schema, dialect, marked, expected):
    offset = marked.index("|")
    document = marked.replace("|", "", 1)
    fn = FnDef("rows_above", ("minimum",), "SELECT * FROM data WHERE id > @minimum")
    edit = function_insertion(document, offset, fn, schema, dialect=dialect)
    assert edit is not None
    text = document[:edit.start] + edit.insert + document[edit.end:]
    caret = edit.start + len(edit.insert) - edit.cursor_back
    assert text[:caret] + "|" + text[caret:] == expected


@pytest.mark.parametrize("marked", [
    "SELECT |", "SELECT * FROM rows_above(|)", "SELECT * FROM data |",
    "SELECT * FROM public.|", "SELECT '|", "-- comment |", "UPDATE |",
])
def test_picker_refuses_non_relation_slots(schema, marked):
    fn = FnDef("all_rows", body="SELECT * FROM data")
    assert function_insertion(marked.replace("|", "", 1), marked.index("|"), fn, schema) is None


def test_picked_zero_argument_function_lands_after_the_call(schema):
    fn = FnDef("all_rows", body="SELECT * FROM data")
    edit = function_insertion("", 0, fn, schema)
    assert edit.insert == "SELECT * FROM all_rows();"
    assert edit.cursor_back == 0


@pytest.mark.parametrize("prefix", ["INSERT INTO", "UPDATE", "DELETE FROM", "CREATE TABLE", "ALTER TABLE", "DROP TABLE"])
def test_write_target_suggestions_never_receive_select_or_alias(schema, prefix):
    _, result = completion(prefix + " dataset_sy|", schema)
    assert table_item(result).insert == "public.dataset_symbol"


def test_alias_collisions_include_future_visible_sources(schema):
    _, result = completion("SELECT * FROM public.other ds JOIN dataset_sy| JOIN public.other ds2 ON true", schema)
    assert table_item(result).insert == "public.dataset_symbol AS ds3"


@pytest.mark.parametrize("marked,expected", [
    ("SELECT dis|play_name FROM public.dataset_symbol ds", "SELECT ds.display_name FROM public.dataset_symbol ds"),
    ('SELECT "sel|ect" FROM public.dataset_symbol ds', 'SELECT ds."select" FROM public.dataset_symbol ds'),
    ("SELECT * FROM public.dataset_symbol ds WHERE dis|play_name = 'x'", "SELECT * FROM public.dataset_symbol ds WHERE ds.display_name = 'x'"),
    ("UPDATE public.dataset_symbol ds SET display_name = dis|play_name WHERE id = 1", "UPDATE public.dataset_symbol ds SET display_name = ds.display_name WHERE id = 1"),
])
def test_unique_columns_are_qualified_without_losing_suffix(schema, marked, expected):
    document, result = completion(marked, schema)
    item = next(item for item in result.items if item.kind == "column")
    assert accepted(document, result, item)[0] == expected


@pytest.mark.parametrize("marked", [
    "INSERT INTO public.dataset_symbol (dis|) VALUES ('x')",
    "UPDATE public.dataset_symbol ds SET dis| = 'x'",
    "SELECT * FROM public.dataset_symbol ds JOIN public.dataset_symbol other USING (dis|)",
])
def test_target_grammar_keeps_bare_columns(schema, marked):
    _, result = completion(marked, schema)
    item = next(item for item in result.items if item.kind == "column")
    assert item.insert == "display_name"


@pytest.mark.parametrize("operator", ["=", "<>", "LIKE", "ILIKE", "IS DISTINCT FROM", "IS NOT DISTINCT FROM", "NOT IN ("])
def test_where_literal_uses_physical_cached_lineage(schema, operator):
    calls = []

    def cached(table, column):
        calls.append((table, column))
        return (["O'Reilly", "Other"], False)

    suffix = ")" if operator.endswith("(") else ""
    document, result = completion("WITH c AS (SELECT display_name AS label FROM public.dataset_symbol) "
                                  f"SELECT * FROM c WHERE label {operator} 'O|ld'{suffix} ORDER BY label", schema,
                                  values=cached)
    item = next(item for item in result.items if item.label == "'O''Reilly'")
    assert calls == [("public.dataset_symbol", "display_name")]
    assert "'O''Reilly'" + suffix + " ORDER BY label" in accepted(document, result, item)[0]
    assert all(item.kind == "value" for item in result.items)


@pytest.mark.parametrize("tail", ["id = 1|0", "id BETWEEN 0 AND 1|0", "id IN (0, 1|0)"])
def test_numeric_predicates_replace_entire_rhs_token(schema, tail):
    document, result = completion("SELECT * FROM public.dataset_symbol WHERE " + tail, schema,
                                  values=lambda table, column: ([12, 15], False))
    item = next(item for item in result.items if item.kind == "value" and item.insert == "12")
    assert result.context.value_field.column == "id"
    assert accepted(document, result, item)[0].endswith("12)" if tail.endswith(")") else "12")


def test_nested_call_in_membership_does_not_borrow_outer_value_pool(schema):
    marked = "SELECT * FROM public.dataset_symbol WHERE display_name IN (concat('x', |))"
    _, result = completion(marked, schema)
    assert result.context.value_field is None
    assert result.context.call.name == "concat"


@pytest.mark.parametrize("command,action", [("AI", "ai"), ("aI", "ai"), ("FUNCTIONS", "fn"), ("functions", "fn"), ("dbt", "dbt")])
def test_inline_commands_replace_only_the_command_span(schema, command, action):
    document = "SELECT ds.id, \\" + command + " FROM public.dataset_symbol ds"
    offset = document.index(command) + len(command)
    session = SimpleNamespace(source=SimpleNamespace(dialect="postgres"), schema=schema)
    view = view_for(session, document, document, offset)
    assert view.phase == "command"
    assert any(item.action == action for item in view.entries)
    assert document[:view.start] + document[view.end:] == "SELECT ds.id,  FROM public.dataset_symbol ds"
    assert view.start == len("SELECT ds.id, ")


def test_partial_command_replaces_its_word_suffix_not_sql(schema):
    document = "SELECT 1; \\FUNCTIONS FROM saved"
    offset = document.index("FUNCTIONS") + 3
    session = SimpleNamespace(source=SimpleNamespace(dialect="postgres"), schema=schema)
    view = view_for(session, document, document, offset)
    assert view.entries[0].action == "fn"
    assert document[view.start:view.end] == "\\FUNCTIONS"
    assert document[view.end:] == " FROM saved"


@pytest.mark.parametrize("marked", [
    "SELECT '\\AI|'", "SELECT E'\\\\AI|'", "SELECT $$\\AI|$$",
    "SELECT $body$\\functions|$body$", "SELECT /* \\AI| */ 1",
    "SELECT -- \\functions|\n1", 'SELECT "\\AI|"',
    "SELECT /* outer /* nested */ \\AI| */ 1", "SELECT 'unfinished \\AI|",
])
def test_commands_never_hijack_opaque_sql(marked):
    offset = marked.index("|")
    assert command_span(marked.replace("|", ""), offset) is None


def test_empty_uncached_predicate_retains_request_metadata(schema):
    session = SimpleNamespace(source=SimpleNamespace(dialect="postgres"), schema=schema, history=(),
                              completion_usage=lambda: {}, cached_values=lambda *args: None,
                              schema_cache_info=lambda: "")
    document = "SELECT * FROM public.dataset_symbol WHERE display_name = 'zz"
    view = view_for(session, document, document, len(document))
    assert view.entries == []
    assert view.value_field.table == "public.dataset_symbol"
    assert view.value_field.column == "display_name"


def test_opaque_predicate_does_not_expose_value_request(schema):
    session = SimpleNamespace(source=SimpleNamespace(dialect="postgres"), schema=schema, history=(),
                              completion_usage=lambda: {}, cached_values=lambda *args: None,
                              schema_cache_info=lambda: "")
    document = "SELECT * FROM public.dataset_symbol WHERE display_name = /* private"
    view = view_for(session, document, document, len(document))
    assert view is None or view.value_field is None


def test_typed_local_relation_argument_offers_tables_and_schema_without_aliases(schema):
    fn = FnDef("scan_rows", ("source",), "SELECT * FROM @source", param_types=("relation",))
    schema = SchemaContext(tables=schema.tables, fns=(fn,))
    _, result = completion("SELECT * FROM scan_rows(dataset_sy|)", schema)
    assert result.signature.parameters == ("source:relation",)
    assert table_item(result).insert == "public.dataset_symbol"
    _, result = completion("SELECT * FROM scan_rows(pub|)", schema)
    assert any(item.kind == "schema" and item.insert == "public." for item in result.items)


@pytest.mark.parametrize("dialect", ["postgres", "duckdb", "sqlite", "d1"])
@pytest.mark.parametrize("marked", [
    "SELECT * FROM project_field(public.dataset_symbol, |);",
    "SELECT * FROM project_field(dataset_symbol, |",
    "SELECT * FROM project_field(public /* , */ .dataset_symbol, |)",
    'SELECT * FROM project_field("public"."dataset_symbol", |)',
    "SELECT * FROM public.other o, project_field(public.dataset_symbol, |) p",
])
def test_identifier_arguments_offer_input_table_columns(typed_schema, dialect, marked):
    document, result = completion(marked, typed_schema, dialect=dialect)
    assert {item.label for item in result.items} == {"id", "display_name", "select"}
    assert all(item.kind == "column" for item in result.items)
    assert result.signature.active_parameter == 1
    item = next(item for item in result.items if item.label == "select")
    assert item.insert == '"select"'
    assert accepted(document, result, item)[0] == marked.replace("|", '"select"')


@pytest.mark.parametrize("dialect,opener,closer", [
    ("postgres", '"', '"'), ("duckdb", '"', '"'),
    ("sqlite", "`", "`"), ("d1", "[", "]"),
])
def test_identifier_arguments_preserve_quotes_and_replace_stale_suffix(typed_schema, dialect, opener, closer):
    marked = f"SELECT * FROM project_field(public.dataset_symbol, {opener}dis|play_stale{closer}) AS result;"
    document, result = completion(marked, typed_schema, dialect=dialect)
    item = next(item for item in result.items if item.label == "display_name")
    assert accepted(document, result, item)[0] == (
        f"SELECT * FROM project_field(public.dataset_symbol, {opener}display_name{closer}) AS result;")


def test_identifier_arguments_use_cte_projection_and_shadowing(typed_schema):
    marked = ("WITH dataset_symbol AS (SELECT display_name AS renamed FROM public.dataset_symbol) "
              "SELECT * FROM project_field(dataset_symbol, |)")
    document, result = completion(marked, typed_schema)
    assert {item.label for item in result.items} == {"renamed"}
    assert accepted(document, result, result.items[0])[0] == marked.replace("|", "renamed")
    _, result = completion(marked.replace("project_field(dataset_symbol", "project_field(public.dataset_symbol"), typed_schema)
    assert {item.label for item in result.items} == {"id", "display_name", "select"}


@pytest.mark.parametrize("source", ["missing", "", "other_alias", "'public.dataset_symbol'",
                                    "coalesce('a,b', 'c')", "(SELECT * FROM public.dataset_symbol)"])
def test_unresolved_relation_arguments_never_borrow_output_or_outer_columns(typed_schema, source):
    _, result = completion(f"SELECT * FROM public.other other_alias, project_field({source}, |)", typed_schema)
    assert result.items == ()
    assert result.signature.active_parameter == 1


def test_unknown_cte_and_ambiguous_table_arguments_stay_unknown(typed_schema):
    _, result = completion("WITH dataset_symbol AS (SELECT * FROM missing) "
                           "SELECT * FROM project_field(dataset_symbol, |)", typed_schema)
    assert result.items == ()
    schema = SchemaContext(tables=typed_schema.tables + (
        TableDef("other.dataset_symbol", columns=(ColumnDef("hidden", "text"),)),
    ), fns=typed_schema.fns)
    _, result = completion("SELECT * FROM project_field(dataset_symbol, |)", schema)
    assert result.items == ()


def test_identifier_arguments_are_isolated_to_the_innermost_call(typed_schema):
    _, result = completion("SELECT * FROM project_field(public.dataset_symbol, "
                           "(SELECT * FROM project_field(public.other, |)))", typed_schema)
    assert {item.label for item in result.items} == {"id"}
    assert result.items[0].insert == "id"


def test_multiple_relation_inputs_keep_bare_column_arguments(typed_schema):
    fn = FnDef("compare_fields", ("left_source", "right_source", "field"),
               "SELECT @field AS value FROM @left_source UNION ALL SELECT @field FROM @right_source",
               param_types=("relation", "relation", "identifier"))
    schema = SchemaContext(tables=typed_schema.tables + (
        TableDef("other_table", columns=(ColumnDef("id", "integer"), ColumnDef("other_name", "text"))),
    ), fns=(fn,))
    _, result = completion("SELECT * FROM compare_fields(public.dataset_symbol, other_table, |)", schema)
    assert {item.insert for item in result.items} == {"id", "display_name", '"select"', "other_name"}


@pytest.mark.parametrize("field", ["public.|", "display|.name"])
def test_identifier_arguments_never_offer_qualified_column_edits(typed_schema, field):
    _, result = completion(f"SELECT * FROM project_field(public.dataset_symbol, {field})", typed_schema)
    assert result.items == ()


def test_fresh_statement_scope_records_only_relation_span(schema):
    document = 'SELECT 1; /* lead */ public."dataset_symbol"'
    context = analyze(document, len(document), schema)
    assert context.statement_relation_start == document.index("public")
    target = "CREATE TABLE public.dataset_symbol"
    assert analyze(target, len(target), schema).statement_relation_start is None


def value_publication(schema, *, changed_index=None, eligible=True):
    source = SimpleNamespace(dialect="postgres")
    prefix = "SELECT * FROM public.dataset_symbol WHERE display_name = '"
    original, updated = prefix + "O", prefix + "Ot"
    snapshot = (original, (0, len(original)), source, id(source), schema, id(schema),
                (), (), None, "page", "draft", False, True, False, False, 1, 0, 1, 0)
    current = list(snapshot)
    current[0], current[1], current[15], current[17] = updated, (0, len(updated)), 2, 2
    if changed_index is not None:
        current[changed_index] = "changed"
    published = []
    palette = SimpleNamespace(
        _eligible=lambda: eligible, _snapshot=lambda: tuple(current),
        session=SimpleNamespace(target_source=lambda source_id: nullcontext(), schema=schema),
        source_id=None, editor=SimpleNamespace(text=updated, cursor_location=(0, len(updated))),
        _parameters=lambda: (), _values_snapshot=snapshot, sync=lambda: published.append(True))
    return palette, snapshot, source, published


def test_value_publication_survives_typing_same_physical_predicate(schema):
    palette, snapshot, source, published = value_publication(schema)
    assert CommandPalette.values_ready(palette, snapshot, source, "public.dataset_symbol", "display_name")
    assert published == [True]
    assert palette._values_snapshot is None


@pytest.mark.parametrize("index", [3, 5, 9, 10, 11, 12, 16, 18])
def test_value_publication_rejects_identity_load_and_dismissal_changes(schema, index):
    palette, snapshot, source, published = value_publication(schema, changed_index=index)
    assert not CommandPalette.values_ready(palette, snapshot, source, "public.dataset_symbol", "display_name")
    assert published == []


def test_value_publication_rejects_blur_and_different_column(schema):
    palette, snapshot, source, published = value_publication(schema, eligible=False)
    assert not CommandPalette.values_ready(palette, snapshot, source, "public.dataset_symbol", "display_name")
    palette._eligible = lambda: True
    assert not CommandPalette.values_ready(palette, snapshot, source, "public.dataset_symbol", "id")
    assert published == []
