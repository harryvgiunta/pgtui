"""Native dialect inventories for completion, not SQL translation or validation."""
from __future__ import annotations

from functools import lru_cache

from d8r.sql import RoutineDef

_COMMON = """SELECT FROM WHERE JOIN LEFT RIGHT INNER FULL CROSS OUTER ON USING AS DISTINCT
GROUP BY HAVING ORDER ASC DESC NULLS FIRST LAST LIMIT OFFSET WITH RECURSIVE UNION ALL
INTERSECT EXCEPT INSERT INTO VALUES UPDATE SET DELETE RETURNING CREATE ALTER DROP TABLE
VIEW INDEX BEGIN COMMIT ROLLBACK SAVEPOINT RELEASE CASE WHEN THEN ELSE END AND OR NOT
NULL IS IN EXISTS BETWEEN LIKE ESCAPE OVER PARTITION FILTER WINDOW ROWS RANGE PRECEDING
FOLLOWING UNBOUNDED CURRENT ROW CAST TRUE FALSE DEFAULT PRIMARY KEY FOREIGN REFERENCES
CHECK UNIQUE CONSTRAINT EXPLAIN ANALYZE NATURAL TEMP TEMPORARY IF"""
_POSTGRES = """ILIKE SIMILAR LATERAL FETCH NEXT ONLY MATERIALIZED CONFLICT DO NOTHING
EXCLUDED ARRAY ANY SOME ALL COLLATE TABLESAMPLE WITHIN GROUPS EXCLUDE TIES
TRUNCATE RESTART IDENTITY CASCADE RESTRICT COPY VACUUM SCHEMA DATABASE SEQUENCE
GENERATED ALWAYS STORED VIRTUAL IDENTITY MERGE MATCHED CALL GRANT REVOKE
JSON_TABLE JSON_EXISTS JSON_QUERY JSON_VALUE"""
_SQLITE = """GLOB REGEXP MATCH PRAGMA REPLACE IGNORE ABORT FAIL RAISE CONFLICT DO NOTHING
EXCLUDED ATTACH DETACH DATABASE VACUUM AUTOINCREMENT STRICT WITHOUT ROWID
MATERIALIZED GROUPS EXCLUDE TIES INDEXED"""
_DUCKDB = """ILIKE LATERAL QUALIFY PIVOT UNPIVOT SUMMARIZE DESCRIBE SHOW INSTALL LOAD
COPY SAMPLE TABLESAMPLE REPLACE EXCLUDE RENAME COLUMNS POSITIONAL ASOF SEMI ANTI
BY NAME STRUCT MAP LIST ARRAY UNION ALL MACRO SECRET ATTACH DETACH DATABASE
CONFLICT DO NOTHING EXCLUDED MATERIALIZED GROUPS CALL"""


def dialect_name(dialect: str) -> str:
    name = dialect.casefold()
    if name in {"sqlite", "sqlite3", "d1", "cloudflare", "cloudflare-d1"}:
        return "sqlite"
    return "duckdb" if name == "duckdb" else "postgres"


@lru_cache(maxsize=8)
def keywords_for(dialect: str) -> tuple[str, ...]:
    extra = {"postgres": _POSTGRES, "sqlite": _SQLITE, "duckdb": _DUCKDB}[dialect_name(dialect)]
    return tuple(sorted(set((_COMMON + " " + extra).split())))


KEYWORDS = tuple(sorted(set((_COMMON + " " + _POSTGRES + " " + _SQLITE + " " + _DUCKDB).split())))

_TYPES = {
    "postgres": ("smallint", "integer", "bigint", "numeric", "decimal", "real", "double precision",
                 "text", "varchar", "char", "boolean", "date", "time", "timestamp", "timestamptz",
                 "interval", "uuid", "json", "jsonb", "bytea", "inet", "cidr", "tsvector", "tsquery"),
    "sqlite": ("INTEGER", "REAL", "TEXT", "BLOB", "NUMERIC"),
    "duckdb": ("BOOLEAN", "TINYINT", "SMALLINT", "INTEGER", "BIGINT", "HUGEINT", "UTINYINT",
               "USMALLINT", "UINTEGER", "UBIGINT", "UHUGEINT", "FLOAT", "DOUBLE", "DECIMAL",
               "VARCHAR", "BLOB", "DATE", "TIME", "TIMESTAMP", "TIMESTAMPTZ", "INTERVAL", "UUID",
               "JSON", "BIT", "STRUCT", "MAP", "UNION"),
}


def types_for(dialect: str) -> tuple[str, ...]:
    return _TYPES[dialect_name(dialect)]


def _routine(name: str, parameters: str, returns: str, doc: str, *, kind: str = "scalar",
             variadic: bool = False) -> RoutineDef:
    params = tuple(parameters.split("|")) if parameters else ()
    types = tuple(param.partition(" ")[2].rstrip("?") for param in params)
    return RoutineDef(name, params, returns, doc, kind, variadic, types)


# These signatures describe native syntax. Discovered catalog overloads supplement
# this offline inventory and take precedence for the connected source.
_COMMON_ROUTINES = (
    _routine("count", "expression any", "bigint", "Count non-NULL values; count(*) counts rows.", kind="aggregate"),
    _routine("sum", "expression numeric", "numeric", "Sum non-NULL input values.", kind="aggregate"),
    _routine("avg", "expression numeric", "numeric", "Arithmetic mean of non-NULL values.", kind="aggregate"),
    _routine("min", "expression any", "any", "Smallest non-NULL input value.", kind="aggregate"),
    _routine("max", "expression any", "any", "Largest non-NULL input value.", kind="aggregate"),
    _routine("coalesce", "value any|fallback any", "any", "First non-NULL argument.", variadic=True),
    _routine("nullif", "value any|other any", "any", "NULL if the arguments are equal."),
    _routine("lower", "text text", "text", "Convert text to lowercase."),
    _routine("upper", "text text", "text", "Convert text to uppercase."),
    _routine("length", "value text", "integer", "Length of a string."),
    _routine("trim", "text text", "text", "Trim leading and trailing spaces."),
    _routine("ltrim", "text text|characters text?", "text", "Trim leading characters."),
    _routine("rtrim", "text text|characters text?", "text", "Trim trailing characters."),
    _routine("replace", "text text|from text|to text", "text", "Replace occurrences of a substring."),
    _routine("substr", "text text|start integer|length integer?", "text", "Extract a substring."),
    _routine("abs", "value numeric", "numeric", "Absolute value."),
    _routine("round", "value numeric|digits integer?", "numeric", "Round to the given precision."),
    _routine("row_number", "", "bigint", "Number rows within a window partition.", kind="window"),
    _routine("rank", "", "bigint", "Rank within a window, leaving gaps for ties.", kind="window"),
    _routine("dense_rank", "", "bigint", "Rank within a window, without gaps for ties.", kind="window"),
    _routine("lag", "value any|offset integer?|default any?", "any", "Value from an earlier window row.", kind="window"),
    _routine("lead", "value any|offset integer?|default any?", "any", "Value from a later window row.", kind="window"),
    _routine("first_value", "value any", "any", "First value in the window frame.", kind="window"),
    _routine("last_value", "value any", "any", "Last value in the window frame.", kind="window"),
    _routine("nth_value", "value any|n integer", "any", "Nth value in the window frame.", kind="window"),
    _routine("ntile", "buckets integer", "integer", "Divide a partition into buckets.", kind="window"),
)
_POSTGRES_ROUTINES = (
    _routine("string_agg", "value text|delimiter text", "text", "Concatenate non-NULL inputs.", kind="aggregate"),
    _routine("array_agg", "value any", "array", "Collect inputs into an array.", kind="aggregate"),
    _routine("json_agg", "value any", "json", "Collect inputs into a JSON array.", kind="aggregate"),
    _routine("jsonb_agg", "value any", "jsonb", "Collect inputs into a JSONB array.", kind="aggregate"),
    _routine("bool_and", "value boolean", "boolean", "True when all non-NULL inputs are true.", kind="aggregate"),
    _routine("bool_or", "value boolean", "boolean", "True when any input is true.", kind="aggregate"),
    _routine("concat", "value any", "text", "Concatenate arguments, ignoring NULL.", variadic=True),
    _routine("concat_ws", "separator text|value any", "text", "Concatenate values with a separator.", variadic=True),
    _routine("substring", "text text|start integer|count integer?", "text", "Extract a substring."),
    _routine("split_part", "text text|delimiter text|field integer", "text", "Extract one delimited field."),
    _routine("regexp_replace", "text text|pattern text|replacement text|flags text?", "text", "Replace regular-expression matches."),
    _routine("regexp_match", "text text|pattern text|flags text?", "text[]", "First regular-expression match."),
    _routine("to_char", "value timestamp|format text", "text", "Format a timestamp or number."),
    _routine("to_date", "text text|format text", "date", "Parse a date using a template."),
    _routine("to_timestamp", "text text|format text", "timestamptz", "Parse a timestamp using a template."),
    _routine("date_trunc", "precision text|source timestamp", "timestamp", "Truncate a timestamp to a precision."),
    _routine("date_part", "field text|source timestamp", "double precision", "Extract a date/time field."),
    _routine("age", "later timestamp|earlier timestamp?", "interval", "Symbolic interval between timestamps."),
    _routine("now", "", "timestamptz", "Transaction start timestamp."),
    _routine("clock_timestamp", "", "timestamptz", "Current wall-clock timestamp."),
    _routine("greatest", "value any", "any", "Largest argument.", variadic=True),
    _routine("least", "value any", "any", "Smallest argument.", variadic=True),
    _routine("jsonb_build_object", "key text|value any", "jsonb", "Construct a JSONB object.", variadic=True),
    _routine("jsonb_extract_path_text", "document jsonb|path text", "text", "Extract JSONB text at a path.", variadic=True),
    _routine("jsonb_array_elements", "document jsonb", "setof jsonb", "Expand JSONB array elements.", kind="table"),
    _routine("generate_series", "start integer|stop integer|step integer?", "setof integer", "Generate an inclusive series.", kind="table"),
    _routine("unnest", "array anyarray", "setof anyelement", "Expand an array into rows.", kind="table"),
    _routine("format", "format text|value any", "text", "Format arguments into a string.", variadic=True),
    _routine("pg_typeof", "value any", "regtype", "Return the native type of an expression."),
)
_SQLITE_ROUTINES = (
    _routine("group_concat", "value text|separator text?", "text", "Concatenate non-NULL inputs.", kind="aggregate"),
    _routine("total", "value numeric", "real", "Floating-point sum, zero for no rows.", kind="aggregate"),
    _routine("ifnull", "value any|fallback any", "any", "Return fallback for a NULL value."),
    _routine("iif", "condition boolean|then any|else any", "any", "Choose a value by a condition."),
    _routine("instr", "text text|substring text", "integer", "One-based position of a substring."),
    _routine("hex", "value blob", "text", "Hexadecimal representation."),
    _routine("quote", "value any", "text", "Produce an SQL literal."),
    _routine("typeof", "value any", "text", "Return the SQLite storage class."),
    _routine("random", "", "integer", "Pseudo-random signed integer."),
    _routine("randomblob", "length integer", "blob", "Pseudo-random bytes."),
    _routine("printf", "format text|value any", "text", "Format values as text.", variadic=True),
    _routine("date", "time_value text|modifier text?", "text", "Native SQLite date conversion.", variadic=True),
    _routine("time", "time_value text|modifier text?", "text", "Native SQLite time conversion.", variadic=True),
    _routine("datetime", "time_value text|modifier text?", "text", "Native SQLite datetime conversion.", variadic=True),
    _routine("julianday", "time_value text|modifier text?", "real", "Julian day number.", variadic=True),
    _routine("unixepoch", "time_value text|modifier text?", "integer", "Unix timestamp.", variadic=True),
    _routine("strftime", "format text|time_value text|modifier text?", "text", "Format a SQLite date/time.", variadic=True),
    _routine("json", "value text", "text", "Validate and normalize JSON."),
    _routine("json_extract", "document text|path text", "any", "Extract values from JSON.", variadic=True),
    _routine("json_object", "key text|value any", "text", "Construct a JSON object.", variadic=True),
    _routine("json_array", "value any", "text", "Construct a JSON array.", variadic=True),
    _routine("json_group_array", "value any", "text", "Aggregate into a JSON array.", kind="aggregate"),
    _routine("json_each", "document text|path text?", "table", "Walk immediate JSON children.", kind="table"),
    _routine("json_tree", "document text|path text?", "table", "Recursively walk JSON values.", kind="table"),
)
_DUCKDB_ROUTINES = (
    _routine("string_agg", "value varchar|separator varchar?", "varchar", "Concatenate non-NULL inputs.", kind="aggregate"),
    _routine("list", "value any", "list", "Collect inputs into a list.", kind="aggregate"),
    _routine("arg_max", "argument any|value any", "any", "Argument at the largest value.", kind="aggregate"),
    _routine("arg_min", "argument any|value any", "any", "Argument at the smallest value.", kind="aggregate"),
    _routine("median", "value numeric", "numeric", "Median input value.", kind="aggregate"),
    _routine("quantile_cont", "value numeric|fraction double", "numeric", "Interpolated quantile.", kind="aggregate"),
    _routine("count_if", "condition boolean", "bigint", "Count true values.", kind="aggregate"),
    _routine("concat", "value any", "varchar", "Concatenate values.", variadic=True),
    _routine("concat_ws", "separator varchar|value any", "varchar", "Concatenate with a separator.", variadic=True),
    _routine("contains", "text varchar|search varchar", "boolean", "Test substring containment."),
    _routine("regexp_matches", "text varchar|pattern varchar|options varchar?", "boolean", "Test a regular expression."),
    _routine("regexp_replace", "text varchar|pattern varchar|replacement varchar|options varchar?", "varchar", "Replace regular-expression matches."),
    _routine("string_split", "text varchar|separator varchar", "varchar[]", "Split text into a list."),
    _routine("date_trunc", "part varchar|date timestamp", "timestamp", "Truncate a date/time."),
    _routine("date_diff", "part varchar|start timestamp|end timestamp", "bigint", "Count date-part boundaries."),
    _routine("strftime", "date timestamp|format varchar", "varchar", "Format a timestamp."),
    _routine("strptime", "text varchar|format varchar", "timestamp", "Parse a timestamp."),
    _routine("try_strptime", "text varchar|format varchar", "timestamp", "Parse a timestamp, returning NULL on failure."),
    _routine("epoch", "date timestamp", "double", "Seconds since the Unix epoch."),
    _routine("now", "", "timestamptz", "Current transaction timestamp."),
    _routine("greatest", "value any", "any", "Largest argument.", variadic=True),
    _routine("least", "value any", "any", "Smallest argument.", variadic=True),
    _routine("list_transform", "list any[]|lambda lambda", "list", "Apply a lambda to list elements."),
    _routine("list_filter", "list any[]|lambda lambda", "list", "Keep elements satisfying a lambda."),
    _routine("list_extract", "list any[]|index integer", "any", "Extract a one-based list element."),
    _routine("struct_extract", "struct struct|key varchar", "any", "Extract a struct field."),
    _routine("json_extract", "json json|path varchar", "json", "Extract JSON using a path."),
    _routine("json_extract_string", "json json|path varchar", "varchar", "Extract JSON as text."),
    _routine("to_json", "value any", "json", "Convert a native value to JSON."),
    _routine("read_parquet", "path varchar", "table", "Read native Parquet files.", kind="table"),
    _routine("read_csv", "path varchar", "table", "Read CSV files with native options.", kind="table"),
    _routine("read_json", "path varchar", "table", "Read JSON files with native options.", kind="table"),
    _routine("range", "start bigint|stop bigint|step bigint?", "table", "Generate a half-open series.", kind="table"),
    _routine("generate_series", "start bigint|stop bigint|step bigint?", "table", "Generate an inclusive series.", kind="table"),
    _routine("unnest", "list any[]", "any", "Expand a list or struct in SELECT."),
)


@lru_cache(maxsize=8)
def functions_for(dialect: str) -> tuple[RoutineDef, ...]:
    native = {"postgres": _POSTGRES_ROUTINES, "sqlite": _SQLITE_ROUTINES,
              "duckdb": _DUCKDB_ROUTINES}[dialect_name(dialect)]
    return _COMMON_ROUTINES + native


_EXPRESSIONS = (
    _routine("coalesce", "value any|fallback any", "any", "First non-NULL argument.", variadic=True),
    _routine("nullif", "value any|other any", "any", "NULL if the arguments are equal."),
    _routine("cast", "value any|type type", "target type", "Native CAST(value AS type) expression.", kind="syntax"),
)
_EXTRACT = _routine("extract", "field keyword|source timestamp", "numeric",
                    "Native EXTRACT(field FROM source) expression.", kind="syntax")
_PG_COMPARISONS = (
    _routine("greatest", "value any", "any", "Largest argument.", variadic=True),
    _routine("least", "value any", "any", "Smallest argument.", variadic=True),
)


def expressions_for(dialect: str) -> tuple[RoutineDef, ...]:
    """Grammar forms which need not appear in native function catalogs."""
    native = dialect_name(dialect)
    if native == "postgres":
        return _EXPRESSIONS + (_EXTRACT,) + _PG_COMPARISONS
    if native == "duckdb":
        return _EXPRESSIONS + (_EXTRACT,)
    return _EXPRESSIONS
