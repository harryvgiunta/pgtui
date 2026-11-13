"""D1 SQL uses SQLite lexical rules and executes native, uncorrelated templates."""
import sqlite3

import pytest

import d8r.sql as sql_module
from d8r.sql import (
    FnDef, OpenTable, SchemaContext, SqlError, TableDef, expand_functions,
    function_query, lex_sql, open_tables, parse_sql, relation_parts,
    split_statements, validate_function,
)


@pytest.fixture
def sqlite_connection():
    connection = sqlite3.connect(":memory:")
    try:
        yield connection
    finally:
        connection.close()


def test_sqlite_tokens_preserve_offsets_and_every_quote_form():
    text = "SELECT 'it''s;@x', \"a\"\";@x\", `a``;@x`, [a;@x], @x -- @x;\n/* @x; */"
    tokens = lex_sql(text, dialect="sqlite")
    assert "".join(token.text for token in tokens) == text
    assert all(text[token.start:token.end] == token.text for token in tokens)
    assert [token.text for token in tokens if token.kind == "parameter"] == ["@x"]
    assert len([token for token in tokens if token.kind == "quoted_identifier"]) == 3
    assert len(split_statements(text, dialect="sqlite")) == 1


@pytest.mark.parametrize("text", ["SELECT 'x", 'SELECT "x', "SELECT `x", "SELECT [x", "/* x"])
def test_sqlite_unterminated_regions_report_original_line(text):
    document = parse_sql("SELECT 1;\n" + text, dialect="sqlite")
    assert document.errors
    assert document.errors[0].line == 2
    assert document.statements == ()


def test_sqlite_comments_are_not_nested(sqlite_connection):
    text = "/* outer /* inner */ SELECT 7; -- trailing\n/* comment only */"
    statements = split_statements(text, dialect="sqlite")
    assert statements == ("/* outer /* inner */ SELECT 7",)
    assert sqlite_connection.execute(statements[0]).fetchall() == [(7,)]


def test_sqlite_has_no_postgres_escape_or_dollar_strings(sqlite_connection):
    tokens = lex_sql("E'plain'; $tag$; SELECT 2", dialect="sqlite")
    assert [(token.kind, token.text) for token in tokens[:2]] == [("word", "E"), ("string", "'plain'")]
    assert not any(token.kind == "dollar_string" for token in tokens)
    text = "SELECT '\\'; SELECT 'it''s'"
    statements = split_statements(text, dialect="sqlite")
    assert [sqlite_connection.execute(statement).fetchone() for statement in statements] == [("\\",), ("it's",)]
    assert len(split_statements("SELECT $$a;b$$; SELECT 2", dialect="sqlite")) == 3


def test_sqlite_backticks_adjacent_to_operators_stay_opaque(sqlite_connection):
    text = "SELECT 2+`x;@x` FROM (SELECT 3 AS `x;@x`)"
    assert len(split_statements(text, dialect="sqlite")) == 1
    assert not any(token.kind == "parameter" for token in lex_sql(text, dialect="sqlite"))
    assert sqlite_connection.execute(text).fetchone() == (5,)


@pytest.mark.parametrize("temporary", ["", "TEMP ", "TEMPORARY "])
def test_sqlite_trigger_case_and_internal_semicolons_execute_as_one_statement(sqlite_connection, temporary):
    text = f"""CREATE TABLE records(id INTEGER, label TEXT);
CREATE TABLE audit(value TEXT, [end] INTEGER);
CREATE {temporary}TRIGGER IF NOT EXISTS [record;change] AFTER INSERT ON records
WHEN CASE WHEN NEW.id > 0 THEN 1 ELSE 0 END
BEGIN
  INSERT INTO audit VALUES (
    CASE WHEN NEW.id = 1 THEN CASE WHEN NEW.label IS NULL THEN 'null;END' ELSE NEW.label END
         ELSE 'other;BEGIN' END, 0);
  UPDATE audit SET end = end + 1;
  INSERT INTO audit VALUES ('literal; END', 2); /* END; BEGIN; */
END;
INSERT INTO records VALUES (1, NULL);
SELECT value, [end] FROM audit ORDER BY rowid;
"""
    document = parse_sql(text, dialect="sqlite")
    assert document.errors == ()
    assert document.sql == text
    assert len(document.statements) == 5
    assert document.statements[2].endswith("END")
    for statement in document.statements[:-1]:
        sqlite_connection.execute(statement)
    assert sqlite_connection.execute(document.statements[-1]).fetchall() == [("null;END", 1), ("literal; END", 2)]


def test_sqlite_trigger_scope_resets_and_transactions_stay_separate(sqlite_connection):
    text = """CREATE TABLE records(id INTEGER);
CREATE TABLE audit(id INTEGER);
CREATE TRIGGER first_trigger AFTER INSERT ON records BEGIN INSERT INTO audit VALUES (NEW.id); END;
CREATE TRIGGER second_trigger AFTER INSERT ON records BEGIN INSERT INTO audit VALUES (NEW.id + 1); END;
BEGIN; INSERT INTO records VALUES (4); COMMIT;
SELECT id FROM audit ORDER BY id;"""
    statements = split_statements(text, dialect="sqlite")
    assert len(statements) == 8
    for statement in statements[:-1]:
        sqlite_connection.execute(statement)
    assert sqlite_connection.execute(statements[-1]).fetchall() == [(4,), (5,)]


def test_sqlite_completion_and_schema_resolve_native_identifier_quotes():
    assert relation_parts('`Odd``Schema`.[a.b]', dialect="sqlite") == ("Odd`Schema", "a.b")
    assert relation_parts("[unterminated", dialect="sqlite") is None
    assert open_tables("SELECT * FROM [a;b] AS `a``lias`, `other` [second]", dialect="sqlite") == (
        OpenTable("[a;b]", "a`lias"), OpenTable("`other`", "second"),
    )
    table = TableDef('"a;b"')
    schema = SchemaContext((table,))
    assert schema.table_by_name("[a;b]", dialect="sqlite") == table
    assert schema.table_by_name("`a;b`", dialect="sqlite") == table


def test_sqlite_templates_protect_all_quotes_and_comments(sqlite_connection):
    fn = FnDef("opaque", ("x",), """SELECT @x AS value, '@x' AS literal, "@x", `@x`, [@x]
FROM (SELECT 9 AS "@x") /* @unknown */ -- @unknown
""")
    validate_function(fn, dialect="sqlite")
    query = function_query(fn, "7 -- argument comment", dialect="sqlite")
    assert "LATERAL" not in query
    assert "'@x' AS literal, \"@x\", `@x`, [@x]" in query
    assert "/* @unknown */ -- @unknown" in query
    assert sqlite_connection.execute(query).fetchall() == [(7, "@x", 9, 9, 9)]


@pytest.mark.parametrize("alias", ["", " AS chosen", " chosen", ' AS "Odd Alias"', " AS `Odd``Alias`", " AS [Odd Alias]"])
def test_sqlite_template_aliases_and_precedence_execute(sqlite_connection, alias):
    fn = FnDef("calc", ("x", "xx"), "SELECT @x * @xx + @x AS value")
    query = expand_functions("SELECT * FROM calc(1 + 2, 3 + 4)" + alias, [fn], dialect="sqlite")
    assert "LATERAL" not in query
    assert query.endswith(alias if alias else " AS calc")
    assert sqlite_connection.execute(query).fetchall() == [(24,)]


def test_sqlite_quoted_call_and_comment_alias_preserve_spelling(sqlite_connection):
    fn = FnDef("entry", body="SELECT 1 AS value")
    query = expand_functions("SELECT * FROM [entry]() /* alias */ aS `result`", [fn], dialect="sqlite")
    assert query.endswith("/* alias */ aS `result`")
    assert sqlite_connection.execute(query).fetchall() == [(1,)]


def test_sqlite_nested_calls_ctes_and_scalar_arguments_execute(sqlite_connection):
    leaf = FnDef("leaf", ("x",), "SELECT @x AS value")
    outer = FnDef("outer_query", ("x",), "WITH q AS (SELECT * FROM leaf(@x)) SELECT * FROM q")
    query = expand_functions(
        "SELECT result.value FROM outer_query((SELECT value FROM leaf(coalesce(NULL, 'a,b;')))) AS result",
        [leaf, outer], dialect="sqlite",
    )
    assert "LATERAL" not in query
    assert sqlite_connection.execute(query).fetchall() == [("a,b;",)]
    finite = function_query(leaf, "(SELECT value FROM leaf(3))", dialect="sqlite")
    assert sqlite_connection.execute(finite).fetchall() == [(3,)]


def test_sqlite_from_join_comma_and_parenthesized_calls_execute(sqlite_connection):
    fn = FnDef("entry", body="SELECT 1 AS value")
    query = expand_functions(
        "SELECT a.value+b.value+c.value+d.value+e.value FROM entry() a, entry() b "
        "JOIN entry() c ON true JOIN (entry() d CROSS JOIN entry() e) ON true",
        [fn], dialect="sqlite",
    )
    assert "LATERAL" not in query
    assert sqlite_connection.execute(query).fetchall() == [(5,)]


def test_sqlite_rejects_explicit_local_lateral_without_claiming_correlation_support(sqlite_connection):
    fn = FnDef("echo", ("x",), "SELECT @x AS value")
    with pytest.raises(SqlError, match="LATERAL.*not supported"):
        expand_functions("SELECT * FROM LATERAL /* comment */ echo(1)", [fn], dialect="sqlite")
    query = expand_functions("SELECT * FROM (SELECT 1 AS id) d, echo(d.id)", [fn], dialect="sqlite")
    assert "LATERAL" not in query
    assert "d.id" in query
    with pytest.raises(sqlite3.OperationalError, match="no such column"):
        sqlite_connection.execute(query)
    native = "SELECT * FROM LATERAL native_function(1)"
    assert expand_functions(native, [fn], dialect="sqlite") == native


def test_sqlite_local_with_ordinality_remains_unsupported():
    with pytest.raises(SqlError, match="WITH ORDINALITY"):
        expand_functions("SELECT * FROM entry() WITH ORDINALITY", [FnDef("entry", body="SELECT 1")], dialect="sqlite")


@pytest.mark.parametrize("arguments", ["", "1,2", "1,", ",1", "1,,2", "/* empty */", "1; SELECT 2", "1); DROP TABLE records; --"])
def test_sqlite_template_arity_and_argument_boundaries(arguments):
    with pytest.raises(SqlError):
        function_query(FnDef("echo", ("x",), "SELECT @x"), arguments, dialect="sqlite")


@pytest.mark.parametrize("body", ["SELECT 1; SELECT 2", "DELETE FROM records", "SELECT @other", "SELECT * FROM @x", "SELECT 1 AS @x"])
def test_sqlite_template_body_and_parameter_validation(body):
    with pytest.raises(SqlError):
        validate_function(FnDef("invalid", ("x",), body), dialect="sqlite")


def test_sqlite_template_cycles_and_expansion_bounds(monkeypatch):
    direct = FnDef("loop", body="SELECT * FROM loop()")
    with pytest.raises(SqlError, match="cycle"):
        validate_function(direct, dialect="sqlite")
    first = FnDef("first_fn", body="SELECT * FROM second_fn()")
    second = FnDef("second_fn", body="SELECT * FROM first_fn()")
    with pytest.raises(SqlError, match="cycle"):
        validate_function(first, [first, second], dialect="sqlite")
    monkeypatch.setattr(sql_module, "MAX_FUNCTION_DEPTH", 2)
    functions = [FnDef("f0", body="SELECT * FROM f1()"), FnDef("f1", body="SELECT * FROM f2()"), FnDef("f2", body="SELECT 1")]
    with pytest.raises(SqlError, match="nested calls"):
        expand_functions("SELECT * FROM f0()", functions, dialect="sqlite")
    monkeypatch.setattr(sql_module, "MAX_FUNCTION_EXPANSIONS", 2)
    with pytest.raises(SqlError, match="expansions"):
        expand_functions("SELECT * FROM entry() a, entry() b, entry() c", [FnDef("entry", body="SELECT 1")], dialect="sqlite")
    monkeypatch.setattr(sql_module, "MAX_EXPANDED_SQL", 100)
    with pytest.raises(SqlError, match="too large"):
        function_query(FnDef("big", ("x",), "SELECT @x, @x, @x"), "'" + "x" * 50 + "'", dialect="sqlite")


def test_sqlite_native_statements_are_not_translated():
    text = "SELECT json_extract('{\"x\":1}', '$.x'); PRAGMA table_info([records]); SELECT native_function(1)"
    document = parse_sql(text, dialect="sqlite")
    assert document.sql == text
    assert document.errors == ()
    assert len(document.statements) == 3


@pytest.mark.parametrize("identifier", ['"ECHO"', '[ECHO]', '`ECHO`'])
def test_sqlite_quoted_function_names_follow_native_case_rules(sqlite_connection, identifier):
    query = expand_functions(f"SELECT * FROM {identifier}(7)",
                             [FnDef("echo", ("x",), "SELECT @x AS value")], dialect="sqlite")
    assert sqlite_connection.execute(query).fetchall() == [(7,)]
