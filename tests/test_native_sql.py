"""Native SQL text contracts; database grammar is intentionally not duplicated."""
from dataclasses import FrozenInstanceError
import re

import pytest

from d8r.sql import (
    ColumnDef, FnDef, ForeignKey, MAX_FUNCTION_DEPTH, MAX_FUNCTION_EXPANSIONS,
    OpenTable, SchemaContext, SqlError, TableDef, expand_functions, function_query,
    is_identifier, lex_sql, open_tables, parse_sql, quote_identifier,
    relation_name, relation_parts, split_statements, validate_function,
)


def compact(text):
    return re.sub(r"\s+", " ", text).strip()


def test_lexer_preserves_text_offsets_and_opaque_regions():
    sql = "SELECT E'it\\'s;@x', 'it''s;@x', \"x;\"\"@x\", $tag$;@x$tag$, @x -- @x;\n/* outer /* @x; */ end */"
    tokens = lex_sql(sql)
    assert "".join(token.text for token in tokens) == sql
    assert all(sql[token.start:token.end] == token.text for token in tokens)
    assert [token.text for token in tokens if token.kind == "parameter"] == ["@x"]
    assert {token.kind for token in tokens} >= {
        "word", "string", "quoted_identifier", "dollar_string", "parameter", "comment", "whitespace", "symbol",
    }


@pytest.mark.parametrize("sql", ["SELECT 'x", 'SELECT "x', "/* outer /* inner */", "SELECT $tag$x"])
def test_unterminated_lexical_regions_are_errors(sql):
    with pytest.raises(SqlError, match="Unterminated"):
        lex_sql(sql)
    assert parse_sql(sql).errors


def test_ordinary_backslash_is_not_a_standard_string_escape():
    assert split_statements("SELECT '\\'; SELECT 2") == ("SELECT '\\'", "SELECT 2")


def test_dollar_tags_and_identifier_dollars_are_distinct():
    sql = "SELECT dollar$tag$, $_tag$one;two$_tag$; SELECT $$three;four$$"
    assert len(split_statements(sql)) == 2
    assert lex_sql("dollar$tag$")[0].kind == "word"


def test_split_native_ddl_dml_transactions_and_dollar_function():
    sql = """BEGIN;
CREATE TABLE t (id bigint);
INSERT INTO t VALUES (1), (2);
CREATE FUNCTION public.identity_sql(x int) RETURNS int LANGUAGE sql
AS $body$ SELECT x; -- @x and f() stay inside the body
$body$;
UPDATE t SET id = id + 1;
COMMIT; -- trailing comment
"""
    document = parse_sql(sql, [FnDef("f", body="SELECT 9")])
    assert document.sql == sql
    assert document.errors == ()
    assert len(document.statements) == 6
    assert "$body$ SELECT x;" in document.statements[3]


def test_comment_only_statements_are_not_executed():
    assert split_statements(" ; -- only a comment\n; /* also /* nested */ */ ; ") == ()
    assert split_statements("-- first\nSELECT 1; /* second */ SELECT 2;") == (
        "-- first\nSELECT 1", "/* second */ SELECT 2",
    )


@pytest.mark.parametrize("sql", [
    "SELECT (", "SELECT missing_column FROM unknown_table", "SELEC invalid grammar",
    "SELECT server_extension(1)", "CREATE EXTENSION some_extension", "ROLLBACK",
])
def test_server_is_sql_grammar_and_schema_authority(sql):
    document = parse_sql(sql)
    assert document.sql == sql
    assert document.errors == ()
    assert document.statements == (sql,)


def test_expression_parameters_preserve_precedence_and_overlap():
    fn = FnDef("calc", ("x", "xx"), "SELECT @x*@xx+@x AS value")
    sql = compact(expand_functions("SELECT * FROM calc(1 + 2, 3 + 4)", [fn]))
    assert "SELECT (1 + 2 )*(3 + 4 )+(1 + 2 ) AS value" in sql
    assert sql.endswith(") AS calc")


def test_parameter_substitution_never_enters_opaque_regions():
    fn = FnDef("opaque", ("x",), "SELECT @x, '@x', E'@x', \"@x\", $$@x$$, $tag$@x$tag$ /* @x */ -- @x\n")
    sql = expand_functions("SELECT * FROM opaque(7)", [fn])
    assert "SELECT (7\n), '@x', E'@x', \"@x\", $$@x$$, $tag$@x$tag$ /* @x */ -- @x" in sql


def test_array_nested_call_struct_and_quoted_commas_are_single_arguments():
    fn = FnDef("pair", ("a", "b"), "SELECT @a, @b")
    sql = expand_functions("SELECT * FROM pair(ARRAY[1, 2, 3], coalesce('a,b', 'c'))", [fn])
    assert "(ARRAY[1, 2, 3]\n)" in sql
    assert "(coalesce('a,b', 'c')\n)" in sql
    assert "({'a': 1, 'b': 2}\n)" in expand_functions(
        "SELECT * FROM pair({'a': 1, 'b': 2}, [3, 4])", [fn],
    )


def test_comment_terminated_argument_and_body_keep_closing_delimiters():
    fn = FnDef("echo", ("x",), "SELECT @x AS value -- body comment")
    sql = function_query(fn, "2 -- argument comment")
    assert "(2 -- argument comment\n) AS value -- body comment\n)" in sql


@pytest.mark.parametrize("suffix", ["", " AS chosen", " chosen", ' AS "Odd Alias"', " AS chosen(value)"])
def test_expansion_preserves_explicit_aliases_and_defaults_to_call_name(suffix):
    fn = FnDef("entry", body="SELECT 1 AS value")
    sql = compact(expand_functions("SELECT * FROM entry()" + suffix, [fn]))
    if suffix:
        assert sql.endswith(")" + suffix)
        assert "AS entry" not in sql
    else:
        assert sql.endswith(") AS entry")


def test_aliases_preserve_keyword_case_and_comments():
    sql = expand_functions("select * from ENTRY() /* alias */ aS e", [FnDef("entry", body="SELECT 1")])
    assert "/* alias */ aS e" in sql
    assert "AS ENTRY" not in sql


def test_from_join_comma_and_parenthesized_join_relations_expand():
    fn = FnDef("entry", body="SELECT 1 AS value")
    sql = expand_functions(
        "SELECT * FROM entry() a, entry() b JOIN entry() c ON true "
        "JOIN (entry() d CROSS JOIN entry() e) ON true", [fn],
    )
    assert sql.count("LATERAL (SELECT 1 AS value") == 5
    assert "AS entry" not in sql


def test_lateral_is_added_once_for_implicit_table_function_correlation():
    fn = FnDef("echo", ("x",), "SELECT @x AS value")
    sql = expand_functions("SELECT * FROM data d JOIN LATERAL echo(d.id) e ON true", [fn])
    assert sql.count("LATERAL") == 1
    assert "(d.id\n) AS value" in sql


def test_nested_cte_and_subquery_calls_expand():
    leaf = FnDef("leaf", ("x",), "SELECT @x AS value")
    outer = FnDef("outer_query", ("x",), "WITH q AS (SELECT * FROM leaf(@x)) SELECT * FROM q")
    sql = expand_functions("WITH s AS (SELECT * FROM outer_query(4)) SELECT * FROM s", [leaf, outer])
    assert "leaf(" not in sql
    assert "outer_query(" not in sql
    assert sql.count("LATERAL (") == 2


def test_finite_same_function_nested_in_argument_is_not_a_cycle():
    fn = FnDef("echo", ("x",), "SELECT @x AS value")
    sql = expand_functions("SELECT * FROM echo((SELECT value FROM echo(3)))", [fn])
    assert "echo(" not in sql
    assert sql.count("LATERAL (") == 2


def test_scalar_and_schema_qualified_calls_remain_native():
    fn = FnDef("entry", body="SELECT 1")
    sql = 'SELECT entry(), extract(day FROM entry()), trim(entry() FROM \'x\') FROM public.entry(), "public"."entry"()'
    assert expand_functions(sql, [fn]) == sql


def test_expression_parentheses_do_not_inherit_outer_from_scope():
    fn = FnDef("entry", body="SELECT 1")
    sql = "SELECT * FROM native_fn(1, entry()), native_fn(ARRAY[1, entry()])"
    assert expand_functions(sql, [fn]) == sql


def test_only_matching_saved_relation_calls_expand():
    fn = FnDef("entry", body="SELECT 1")
    sql = "SELECT 'FROM entry()', $$JOIN entry()$$ FROM server_fn() -- FROM entry()"
    assert expand_functions(sql, [fn]) == sql
    assert 'AS "entry"' in expand_functions('SELECT * FROM "entry"()', [fn])
    assert expand_functions('SELECT * FROM "ENTRY"()', [fn]) == 'SELECT * FROM "ENTRY"()'


@pytest.mark.parametrize("arguments", ["", "1, 2", "1,", ",1", "1,,2", "/* empty */"])
def test_wrong_arity_and_empty_arguments_are_expansion_errors(arguments):
    fn = FnDef("echo", ("x",), "SELECT @x")
    with pytest.raises(SqlError, match="argument"):
        function_query(fn, arguments)


@pytest.mark.parametrize("sql", ["SELECT * FROM echo(1", "SELECT * FROM echo(ARRAY[1,2))", "SELECT * FROM echo(1; SELECT 2)"])
def test_malformed_saved_call_boundaries_are_expansion_errors(sql):
    with pytest.raises(SqlError):
        expand_functions(sql, [FnDef("echo", ("x",), "SELECT @x")])


def test_preview_arguments_cannot_escape_the_call_boundary():
    with pytest.raises(SqlError):
        function_query(FnDef("echo", ("x",), "SELECT @x"), "1); DROP TABLE data; --")


def test_function_body_semicolon_in_literal_is_not_a_statement_separator():
    fn = FnDef("echo", ("x",), "SELECT @x, ';' AS semi;")
    validate_function(fn)
    sql = function_query(fn, "';'")
    assert "SELECT (';'\n), ';' AS semi" in sql
    assert len(split_statements(sql)) == 1


@pytest.mark.parametrize("body", [
    "", "-- comment", "SELECT 1; SELECT 2", "DELETE FROM data", "CREATE TABLE t(id int)",
    "BEGIN", "WITH q AS (SELECT 1) DELETE FROM data",
])
def test_function_bodies_must_be_one_query(body):
    with pytest.raises(SqlError):
        validate_function(FnDef("bad", body=body))


@pytest.mark.parametrize("body", [
    "SELECT no_such_column FROM no_such_table", "WITH q AS (SELECT 1) SELECT * FROM q",
    "VALUES (1), (2)", "TABLE data",
])
def test_function_validation_accepts_native_query_shapes_without_schema_check(body):
    validate_function(FnDef("query_body", body=body))


@pytest.mark.parametrize("fn", [
    FnDef("a.b", body="SELECT 1"), FnDef("bad", ("x", "x"), "SELECT @x"),
    FnDef("bad", ("x.y",), "SELECT 1"), FnDef("bad", ("x",), "SELECT @other"),
])
def test_function_signatures_and_undeclared_parameters_are_validated(fn):
    with pytest.raises(SqlError):
        validate_function(fn)


def test_direct_and_indirect_cycles_are_bounded():
    direct = FnDef("loop", body="SELECT * FROM loop()")
    with pytest.raises(SqlError, match="cycle"):
        validate_function(direct)
    first = FnDef("first_fn", body="SELECT * FROM second_fn()")
    second = FnDef("second_fn", body="SELECT * FROM first_fn()")
    with pytest.raises(SqlError, match="first_fn -> second_fn -> first_fn"):
        validate_function(first, [first, second])


def test_sibling_calls_do_not_count_as_cycles():
    fn = FnDef("entry", body="SELECT 1")
    assert expand_functions("SELECT * FROM entry() a, entry() b", [fn]).count("LATERAL (") == 2


def test_expansion_depth_is_bounded():
    functions = [FnDef(f"f{i}", body=f"SELECT * FROM f{i + 1}()") for i in range(MAX_FUNCTION_DEPTH)]
    functions.append(FnDef(f"f{MAX_FUNCTION_DEPTH}", body="SELECT 1"))
    with pytest.raises(SqlError, match="nested calls"):
        expand_functions("SELECT * FROM f0()", functions)


def test_total_expansion_count_is_bounded():
    fn = FnDef("entry", body="SELECT 1")
    sql = " UNION ALL ".join("SELECT * FROM entry()" for _ in range(MAX_FUNCTION_EXPANSIONS + 1))
    with pytest.raises(SqlError, match="expansions"):
        expand_functions(sql, [fn])


def test_expanded_size_is_bounded_before_repeated_parameter_allocation():
    fn = FnDef("big", ("x",), "SELECT " + ", ".join("@x" for _ in range(1000)))
    with pytest.raises(SqlError, match="too large"):
        function_query(fn, "'" + "x" * 2000 + "'")


def test_parse_errors_point_to_original_call_line():
    sql = "SELECT 1;\n\nSELECT * FROM echo()"
    document = parse_sql(sql, [FnDef("echo", ("x",), "SELECT @x")])
    assert document.sql == sql
    assert document.statements == ()
    assert len(document.errors) == 1
    assert document.errors[0].line == 3


def test_native_operators_are_not_expression_parameters():
    sql = "SELECT a <@array_col, b @> other, c @@ query, d @? '$.x'"
    assert not [token for token in lex_sql(sql) if token.kind == "parameter"]
    assert expand_functions(sql) == sql


def test_identifier_quoting_and_qualified_names_follow_native_case_rules():
    assert is_identifier("valid_name")
    assert not is_identifier("schema.table")
    assert quote_identifier('a"b') == '"a""b"'
    assert relation_name("Odd Schema", "items") == '"Odd Schema"."items"'
    assert relation_parts('Public."Mixed.Name"') == ("public", "Mixed.Name")
    assert relation_parts("a..b") is None
    assert relation_parts('"unterminated') is None
    assert relation_parts('""') is None
    with pytest.raises(SqlError):
        quote_identifier("a\x00b")


def test_schema_snapshots_are_immutable_and_resolve_default_schema():
    column = ColumnDef("id", "BIGINT", values=[1, 2])
    fk = ForeignKey(["id"], '"public"."parents"', ["id"])
    public = TableDef('"public"."items"', columns=[column], foreign_keys=[fk])
    other = TableDef('"other"."items"', columns=[ColumnDef("id", "BIGINT")])
    fn = FnDef("entry", ["x"], "SELECT @x")
    schema = SchemaContext([public, other], [fn], "public")
    assert schema.table_by_name("items") == public
    assert schema.table_by_name("other.items") == other
    assert SchemaContext([public, other]).table_by_name("items") is None
    assert schema.fn_by_name("ENTRY") == fn
    assert schema.column_by_name("id").tables == (public.name, other.name)
    assert column.values == (1, 2)
    assert fk.columns == ("id",)
    with pytest.raises(FrozenInstanceError):
        schema.default_schema = "other"


def test_completion_relation_hints_include_qualified_quoted_and_comma_relations():
    assert open_tables('SELECT * FROM public.items AS i, "Odd Schema"."Items" j JOIN entry(ARRAY[1,2]) e ON true') == (
        OpenTable("public.items", "i"), OpenTable('"Odd Schema"."Items"', "j"), OpenTable("entry", "e"),
    )
    assert open_tables("SELECT * FROM public.items") == (OpenTable("public.items", "items"),)
    assert open_tables("SELECT * FROM 'unterminated") == ()


def test_correlated_saved_functions_execute_as_native_lateral_relations(parquet_source):
    from d8r.engine import execute_sql

    fn = FnDef("twice", ("x",), "SELECT @x * 2 AS value")
    sql = expand_functions(
        "SELECT d.id, twice.value FROM data AS d, twice(d.id + 1) ORDER BY d.id", [fn],
    )
    assert execute_sql(parquet_source, sql)["rows"] == [[1, 4], [2, 6], [3, 8]]


def test_nested_saved_queries_execute_with_explicit_aliases(parquet_source):
    from d8r.engine import execute_sql

    fn = FnDef("echo", ("x",), "SELECT @x AS value")
    sql = expand_functions(
        "SELECT result.value FROM echo((SELECT value FROM echo(7))) AS result", [fn],
    )
    assert execute_sql(parquet_source, sql)["rows"] == [[7]]


@pytest.mark.parametrize("body", [
    "SELECT * FROM @x", "SELECT 1 AS @x", "SELECT t.@x FROM data t",
    "SELECT @x(1)", "TABLE @x", "WITH @x AS (SELECT 1) SELECT 1",
])
def test_parameters_cannot_be_sql_identifier_templates(body):
    with pytest.raises(SqlError, match="expression parameter"):
        validate_function(FnDef("query_body", ("x",), body))


def test_distinct_from_is_not_a_relation_introducer():
    fn = FnDef("entry", body="SELECT 1")
    sql = "SELECT 1 IS DISTINCT FROM entry() FROM data WHERE 2 IS NOT DISTINCT FROM entry()"
    assert expand_functions(sql, [fn]) == sql
    predicate = FnDef("predicate", ("x", "y"), "SELECT @x IS DISTINCT FROM @y AS result")
    validate_function(predicate)
    assert "IS DISTINCT FROM (NULL" in function_query(predicate, "1, NULL")


def test_sql_standard_atomic_routine_body_is_one_native_statement():
    routine = """CREATE FUNCTION public.classify(n integer) RETURNS integer
LANGUAGE SQL
BEGIN /* retain the routine body */ ATOMIC
    SELECT CASE WHEN n < 0 THEN -1 ELSE CASE WHEN n = 0 THEN 0 ELSE 1 END END;
    SELECT CASE n WHEN 0 THEN 10 ELSE 20 END;
END"""
    sql = "BEGIN;\n" + routine + ";\nSELECT public.classify(0);\nCOMMIT;"
    assert split_statements(sql) == ("BEGIN", routine, "SELECT public.classify(0)", "COMMIT")
    document = parse_sql(sql)
    assert document.sql == sql
    assert not document.errors
    assert len(document.statements) == 4


def test_parenthesized_rule_actions_keep_their_internal_statement_separators():
    rule = """CREATE RULE audit_insert AS ON INSERT TO data DO ALSO (
    INSERT INTO audit VALUES (NEW.id, 'insert;event');
    NOTIFY data_changed;
)"""
    assert split_statements(rule + "; SELECT 1;") == (rule, "SELECT 1")


def test_transaction_begin_and_case_expressions_do_not_start_atomic_blocks():
    sql = "BEGIN; SELECT CASE WHEN true THEN 'BEGIN ATOMIC; END' ELSE 'x' END; END;"
    assert split_statements(sql) == (
        "BEGIN", "SELECT CASE WHEN true THEN 'BEGIN ATOMIC; END' ELSE 'x' END", "END",
    )


def test_atomic_block_keywords_inside_opaque_regions_do_not_close_the_body():
    sql = "CREATE PROCEDURE p() LANGUAGE SQL BEGIN ATOMIC SELECT 'END;', $$END;$$; /* END; */ SELECT 2; END; SELECT 3"
    statements = split_statements(sql)
    assert len(statements) == 2
    assert statements[0].endswith("SELECT 2; END")
    assert statements[1] == "SELECT 3"


def test_case_insensitive_function_registry_collisions_are_never_silent():
    lower = FnDef("entry", body="SELECT 1")
    upper = FnDef("ENTRY", body="SELECT 2")
    with pytest.raises(SqlError, match="case-insensitive"):
        expand_functions("SELECT * FROM entry()", [lower, upper])
    with pytest.raises(SqlError, match="case-insensitive"):
        SchemaContext(fns=[lower, upper])
    with pytest.raises(SqlError, match="case-insensitive"):
        validate_function(upper, {"entry": lower})
    with pytest.raises(SqlError, match="case-insensitive"):
        function_query(upper, "", {"entry": lower})
    assert parse_sql("SELECT 1", [lower, upper]).errors


def test_exact_name_function_edits_replace_existing_definitions():
    old = FnDef("entry", body="SELECT 1")
    new = FnDef("entry", body="SELECT 2")
    validate_function(new, {"entry": old})
    assert "SELECT 2" in function_query(new, "", {"entry": old})
