"""Native database sources and metadata-only discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import threading

from d8r.sql import CatalogObject, ColumnDef, ForeignKey, IndexDef, RoutineDef, quote_identifier, relation_name

POSTGRES_VERSION = "18.7"
_RUN_PROBE_SECONDS = 2.0
_TRANSACTION_TIMEOUTS = {"25P03", "25P04"}
_TERMINATED = _TRANSACTION_TIMEOUTS | {"57P01", "57P02", "57P04", "57P05"}


class EngineError(ValueError):
    """A safe, public execution or connection error."""

    def __init__(self, message: str, *, sqlstate: str = "") -> None:
        super().__init__(message)
        self.sqlstate = sqlstate


@dataclass
class DataSource:
    id: str
    display: str
    doc: str
    kind: str
    dialect: str
    con: object = field(default=None, repr=False)
    datasets: dict[str, dict] = field(default_factory=dict)
    postgres: dict = field(default_factory=dict, repr=False)
    d1: dict = field(default_factory=dict, repr=False)
    connection_key: str = ""
    default_schema: str | None = None
    server_version: str = ""
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    _transaction: bool = field(default=False, repr=False, compare=False)
    routines: tuple[RoutineDef, ...] = ()
    objects: tuple[CatalogObject, ...] = ()
    # Origins belong to this exact live connection and discovery snapshot.
    _pg_origin_catalog: tuple[object, dict[str, dict], dict[tuple[int, int], tuple[str, str]]] | None = field(
        default=None, init=False, repr=False, compare=False,
    )


def load() -> dict[str, DataSource]:
    return {"postgres": DataSource("postgres", "PostgreSQL", "Connect to PostgreSQL or Cloudflare D1, or open a local Parquet file.", "disconnected", "postgres")}


# These are application-authored messages, never server DETAIL/HINT or identifiers.
_SQL_ERRORS = {
    "42601": ("SQL syntax error", "Check clause order, commas between expressions, and matching parentheses near the error position."),
    "42P01": ("Relation does not exist", "Check the table or CTE name, schema qualification, and aliases visible in this query scope."),
    "42703": ("Column does not exist", "Check the column spelling and quoted case against the schema; use alias.column from a visible relation."),
    "42702": ("Ambiguous column reference", "Qualify the reference as alias.column using the intended table or subquery alias."),
    "42P09": ("Ambiguous relation reference", "Give each relation a distinct alias and qualify column references as alias.column."),
    "42803": ("Invalid grouping", "Include every nonaggregate SELECT expression in GROUP BY, including all columns expanded by * or alias.*; aggregate expressions stay out of GROUP BY. Check HAVING and ORDER BY references too."),
    "42883": ("Function or operator does not exist for these types", "Check the function or operator name, argument count and argument types; use explicit casts where appropriate."),
    "42725": ("Ambiguous function call", "Use explicit argument casts and schema qualification to select the intended function signature."),
    "42804": ("SQL type mismatch", "Check expression, operator and target column types; use compatible values or explicit casts."),
    "42846": ("Unsupported type conversion", "Check the source and destination types and choose a supported conversion."),
    "22P02": ("Invalid input for the requested type", "Check the input format and the target type of explicit or implicit casts."),
    "22007": ("Invalid date or time format", "Use a date or time value in a format accepted by the target type."),
    "22008": ("Date or time value out of range", "Check date and time fields and the supported range of the target type."),
    "22003": ("Numeric value out of range", "Check numeric values and arithmetic against the range of the target type."),
    "22012": ("Division by zero", "Check the divisor; use NULLIF with zero if a NULL result is intended."),
    "42501": ("Permission denied", "Check privileges for the requested operation."),
    "25006": ("Read-only execution rejected a write", "Choose Run native SQL only if the write is intended."),
    "57014": ("Query cancelled or deadline exceeded", ""),
    "25P02": ("Transaction is aborted; issue ROLLBACK", ""),
    "23505": ("Unique constraint violation", "Check for duplicate values in a unique or primary key."),
    "23503": ("Foreign key constraint violation", "Check that referenced rows exist and are not still referenced when deleting or updating them."),
    "28P01": ("Authentication failed", "Check the selected source credentials."),
}


def _sql_error_message(state: str, suffix: str = "") -> str:
    label, hint = _SQL_ERRORS[state]
    return label + suffix + (". Hint: " + hint if hint else "")


_DUCK_ERROR_PATTERNS = tuple((re.compile(pattern), state) for pattern, state in (
    (r'Binder Error: Ambiguous reference to column name ', "42702"),
    (r'Binder Error: (?:column .{1,512} must appear in the GROUP BY clause|GROUP BY clause cannot contain aggregates|Aggregates cannot be present in a GROUP BY clause|Cannot mix aggregates with non-aggregated columns|aggregate function calls cannot be nested)', "42803"),
    (r'Binder Error: (?:Referenced column .{1,512} not found in FROM clause|Values list .{1,512} does not have a column named)', "42703"),
    (r'(?:Catalog Error: Table with name .{1,512} does not exist|Binder Error: Referenced table .{1,512} not found)', "42P01"),
    (r'(?:Catalog Error: (?:Scalar|Aggregate|Table) Function with name .{1,512} does not exist|Binder Error: No function matches the given name and argument types)', "42883"),
    (r'Binder Error: (?:Cannot mix values of type|Cannot compare values of type|Cannot combine types|Cannot concatenate types)', "42804"),
))


def _safe_error(exc: Exception, operation: str = "SQL execution", *, position_offset: int = 0) -> EngineError:
    # Only real driver diagnostics qualify. Arbitrary exceptions are never read
    # or stringified; DuckDB has no SQLSTATE, so classify bounded known formats.
    import duckdb
    import psycopg

    if isinstance(exc, psycopg.Error):
        state = exc.sqlstate
        valid_state = isinstance(state, str) and re.fullmatch(r"[0-9A-Z]{5}", state) is not None
        suffix = f" [SQLSTATE {state}]" if valid_state else ""
        position = exc.diag.statement_position
        if (isinstance(position, str) and re.fullmatch(r"[0-9]{1,10}", position)
                and position_offset < int(position) <= 2147483647):
            suffix += f" at character {int(position) - position_offset}"
        if valid_state and state in _SQL_ERRORS:
            return EngineError(_sql_error_message(state, suffix), sqlstate=state)
        return EngineError(f"{operation} failed" + suffix, sqlstate=state if valid_state else "")
    if isinstance(exc, duckdb.Error):
        state = None
        if isinstance(exc, duckdb.ParserException):
            state = "42601"
        elif isinstance(exc, duckdb.ConversionException):
            state = "22P02"
        elif isinstance(exc, duckdb.OutOfRangeException):
            state = "22003"
        elif isinstance(exc, duckdb.InterruptException):
            state = "57014"
        elif isinstance(exc, (duckdb.BinderException, duckdb.CatalogException)):
            message = str(exc)[:2048].split("\n", 1)[0]
            state = next((code for pattern, code in _DUCK_ERROR_PATTERNS if pattern.match(message)), None)
        if state is not None:
            return EngineError(_sql_error_message(state), sqlstate=state)
    return EngineError(f"{operation} failed")


def add_postgres_source(source_id, *, host, port=5432, database, user, password,
                        schema="public", sslmode="prefer", display=None) -> DataSource:
    metadata = {}
    for key, value in (("host", host), ("database", database), ("user", user), ("schema", schema)):
        if not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value):
            raise EngineError(f"A valid PostgreSQL {key} is required")
        metadata[key] = value.strip()
    if "://" in metadata["host"] or "," in metadata["host"]:
        raise EngineError("PostgreSQL host must be a single hostname or address")
    try:
        if isinstance(port, bool) or not isinstance(port, (int, str)):
            raise ValueError
        port = int(port)
        if not 1 <= port <= 65535:
            raise ValueError
    except (TypeError, ValueError):
        raise EngineError("PostgreSQL port must be an integer from 1 to 65535") from None
    if sslmode not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
        raise EngineError("Unsupported PostgreSQL SSL mode")
    if not isinstance(password, str) or "\x00" in password:
        raise EngineError("A valid PostgreSQL password is required")
    if schema.lower().startswith("pg_") or schema.lower() == "information_schema":
        raise EngineError("Select a PostgreSQL user schema")
    metadata.update(port=str(port), sslmode=sslmode)
    con = None
    try:
        import psycopg
        con = psycopg.connect(host=metadata["host"], port=port, dbname=metadata["database"],
                              user=metadata["user"], password=password, sslmode=sslmode,
                              connect_timeout=10, autocommit=True)
        with con.cursor() as cursor:
            cursor.execute("SELECT has_schema_privilege(%s, 'USAGE')", (metadata["schema"],))
            if not cursor.fetchone()[0]:
                raise EngineError("Selected PostgreSQL schema is unavailable")
            cursor.execute(psycopg.sql.SQL("SET search_path TO {}").format(psycopg.sql.Identifier(metadata["schema"])))
            cursor.execute("SHOW server_version")
            version = str(cursor.fetchone()[0])
        def public(value):
            return value.replace(password, "[redacted]") if password else value
        source = DataSource(str(source_id), public(display or f"PostgreSQL · {metadata['database']}"),
                            public(f"PostgreSQL · {metadata['database']} · {metadata['schema']}"),
                            "postgres-live", "postgres", con=con, postgres=metadata,
                            default_schema=metadata["schema"], server_version=public(version))
        refresh_source_schema(source)
        return source
    except Exception as exc:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        if isinstance(exc, EngineError):
            raise
        raise _safe_error(exc, "PostgreSQL connection") from None


# Catalog queries intentionally never count, sample, or scan user relations.
_PG_RELATION_KINDS = {"r": "table", "p": "partitioned table", "v": "view",
                      "m": "materialized view", "f": "foreign table"}
_PG_USER_NAMESPACE = """
(pg_catalog.left(n.nspname, 3) <> 'pg_' AND n.nspname <> 'information_schema'
 OR n.oid = pg_catalog.pg_my_temp_schema())
"""
_PG_SCHEMA_USAGE = "pg_catalog.has_schema_privilege(n.oid, 'USAGE')"
# PostgreSQL may evaluate filters before catalog joins. Guard privilege
# functions that reject OIDs belonging to a different kind of pg_class row.
_PG_RELATION_ACCESS = f"""
{_PG_USER_NAMESPACE} AND {_PG_SCHEMA_USAGE}
AND CASE WHEN c.relkind IN ('r','p','v','m','f') THEN
        pg_catalog.has_table_privilege(c.oid, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
        OR pg_catalog.has_any_column_privilege(c.oid, 'SELECT,INSERT,UPDATE,REFERENCES')
    ELSE false END
"""
_PG_ROUTINE_ACCESS = f"""
(n.nspname = 'pg_catalog' OR {_PG_USER_NAMESPACE}) AND {_PG_SCHEMA_USAGE}
AND pg_catalog.has_function_privilege(p.oid, 'EXECUTE')
"""
_PG_SEQUENCE_ACCESS = f"""
{_PG_USER_NAMESPACE} AND {_PG_SCHEMA_USAGE}
AND CASE WHEN c.relkind = 'S' THEN pg_catalog.has_sequence_privilege(c.oid, 'USAGE,SELECT,UPDATE') ELSE false END
"""
_PG_COLLATION_ACCESS = f"""
(n.nspname = 'pg_catalog' OR {_PG_USER_NAMESPACE}) AND {_PG_SCHEMA_USAGE}
AND c.collencoding IN (-1, pg_catalog.pg_char_to_encoding(pg_catalog.current_setting('server_encoding')))
"""
_PG_TYPE_ACCESS = f"""
{_PG_USER_NAMESPACE} AND {_PG_SCHEMA_USAGE} AND pg_catalog.has_type_privilege(t.oid, 'USAGE')
AND (t.typtype = 'd' OR (t.typtype = 'c' AND c.relkind = 'c'))
"""

_PG_COLUMNS = f"""
SELECT c.oid, n.nspname, c.relname, pg_catalog.obj_description(c.oid, 'pg_class'),
       a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod),
       pg_catalog.col_description(c.oid, a.attnum), c.relpersistence = 't',
       (SELECT pg_catalog.array_agg(e.enumlabel::text ORDER BY e.enumsortorder)
        FROM pg_catalog.pg_enum e
        WHERE e.enumtypid = CASE WHEN column_type.typtype = 'd'
                               THEN column_type.typbasetype ELSE a.atttypid END), c.relkind, a.attnum
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
LEFT JOIN pg_catalog.pg_type column_type ON column_type.oid = a.atttypid
WHERE c.relkind IN ('r','p','v','m','f') AND {_PG_RELATION_ACCESS}
ORDER BY n.nspname, c.relname, a.attnum
"""
_PG_FKS = """
SELECT k.conrelid, k.oid, ns.nspname, target.relname, src.attname, dst.attname
FROM pg_catalog.pg_constraint k
JOIN pg_catalog.pg_class target ON target.oid = k.confrelid
JOIN pg_catalog.pg_namespace ns ON ns.oid = target.relnamespace
CROSS JOIN LATERAL unnest(k.conkey, k.confkey) WITH ORDINALITY AS pair(srcnum, dstnum, position)
JOIN pg_catalog.pg_attribute src ON src.attrelid = k.conrelid AND src.attnum = pair.srcnum
JOIN pg_catalog.pg_attribute dst ON dst.attrelid = k.confrelid AND dst.attnum = pair.dstnum
WHERE k.contype = 'f'
ORDER BY k.conrelid, k.oid, pair.position
"""
_PG_PARTITIONS = """
SELECT edge.inhrelid, edge.inhparent
FROM pg_catalog.pg_inherits edge
JOIN pg_catalog.pg_class child ON child.oid = edge.inhrelid AND child.relispartition
JOIN pg_catalog.pg_class parent ON parent.oid = edge.inhparent AND parent.relkind = 'p'
"""
_PG_INDEXES = f"""
SELECT i.indrelid, n.nspname, ix.relname, am.amname, i.indnkeyatts,
       i.indisunique, i.indisprimary, i.indisvalid AND i.indisready AND i.indislive,
       i.indpred IS NOT NULL, i.indimmediate,
       ARRAY(SELECT a.attname::text
             FROM pg_catalog.unnest(i.indkey) WITH ORDINALITY AS k(attnum, position)
             LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum = k.attnum
             ORDER BY k.position)
FROM pg_catalog.pg_index i
JOIN pg_catalog.pg_class ix ON ix.oid = i.indexrelid
JOIN pg_catalog.pg_am am ON am.oid = ix.relam
JOIN pg_catalog.pg_class c ON c.oid = i.indrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE {_PG_RELATION_ACCESS}
ORDER BY n.nspname, c.relname, ix.relname
"""
_PG_SEQUENCES = f"""
SELECT n.nspname, c.relname, pg_catalog.format_type(s.seqtypid, NULL),
       pg_catalog.obj_description(c.oid, 'pg_class')
FROM pg_catalog.pg_sequence s
JOIN pg_catalog.pg_class c ON c.oid = s.seqrelid
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
WHERE {_PG_SEQUENCE_ACCESS}
ORDER BY n.nspname, c.relname
"""
_PG_EXTENSIONS = f"""
SELECT e.extname, e.extversion, n.nspname, pg_catalog.obj_description(e.oid, 'pg_extension')
FROM pg_catalog.pg_extension e
JOIN pg_catalog.pg_namespace n ON n.oid = e.extnamespace
WHERE {_PG_SCHEMA_USAGE}
ORDER BY e.extname
"""
_PG_COLLATIONS = f"""
SELECT DISTINCT ON (n.nspname, c.collname) n.nspname, c.collname, c.collprovider,
       pg_catalog.obj_description(c.oid, 'pg_collation')
FROM pg_catalog.pg_collation c
JOIN pg_catalog.pg_namespace n ON n.oid = c.collnamespace
WHERE {_PG_COLLATION_ACCESS}
ORDER BY n.nspname, c.collname, c.collencoding DESC
"""
_PG_TYPES = f"""
SELECT n.nspname, t.typname, t.typtype, pg_catalog.format_type(t.typbasetype, t.typtypmod),
       t.typnotnull, pg_catalog.obj_description(t.oid, 'pg_type'),
       a.attname, pg_catalog.format_type(a.atttypid, a.atttypmod),
       pg_catalog.col_description(c.oid, a.attnum)
FROM pg_catalog.pg_type t
JOIN pg_catalog.pg_namespace n ON n.oid = t.typnamespace
LEFT JOIN pg_catalog.pg_class c ON c.oid = t.typrelid
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
WHERE {_PG_TYPE_ACCESS}
ORDER BY n.nspname, t.typname, a.attnum
"""
_PG_COLLATION_PROVIDERS = {"b": "builtin", "c": "libc", "i": "icu", "d": "default"}

_PG_ROUTINES = f"""
SELECT n.nspname, p.proname, p.prokind, p.proretset,
       pg_catalog.pg_get_function_result(p.oid),
       pg_catalog.obj_description(p.oid, 'pg_proc'), p.provariadic <> 0,
       p.pronargdefaults,
       p.proargnames, p.proargmodes,
       ARRAY(SELECT pg_catalog.format_type(arg.type_oid, NULL)
             FROM pg_catalog.unnest(COALESCE(p.proallargtypes, p.proargtypes::oid[]))
                  WITH ORDINALITY AS arg(type_oid, position)
             ORDER BY arg.position),
       ARRAY(SELECT a.attname FROM pg_catalog.pg_attribute a
             WHERE a.attrelid = t.typrelid AND a.attnum > 0 AND NOT a.attisdropped
             ORDER BY a.attnum),
       ARRAY(SELECT pg_catalog.format_type(a.atttypid, a.atttypmod)
             FROM pg_catalog.pg_attribute a
             WHERE a.attrelid = t.typrelid AND a.attnum > 0 AND NOT a.attisdropped
             ORDER BY a.attnum),
       p.oid
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
JOIN pg_catalog.pg_type t ON t.oid = p.prorettype
WHERE {_PG_ROUTINE_ACCESS}
ORDER BY n.nspname, p.proname, p.oid
"""


def _postgres_objects(cursor) -> tuple[CatalogObject, ...]:
    objects = []
    cursor.execute(_PG_SEQUENCES)
    objects.extend(CatalogObject(relation_name(schema, name), "sequence", dtype, doc or "")
                   for schema, name, dtype, doc in cursor)
    cursor.execute(_PG_EXTENSIONS)
    objects.extend(CatalogObject(relation_name(name), "extension", f"{version} · schema {quote_identifier(schema)}", doc or "")
                   for name, version, schema, doc in cursor)
    cursor.execute(_PG_COLLATIONS)
    objects.extend(CatalogObject(relation_name(schema, name), "collation", _PG_COLLATION_PROVIDERS.get(provider, provider), doc or "")
                   for schema, name, provider, doc in cursor)
    cursor.execute(_PG_TYPES)
    types = {}
    for schema, name, kind, base, not_null, doc, column, dtype, column_doc in cursor:
        entry = types.setdefault(relation_name(schema, name), (kind, base, not_null, doc or "", []))
        if column is not None:
            entry[-1].append(ColumnDef(column, dtype, column_doc or ""))
    for name, (kind, base, not_null, doc, fields) in types.items():
        detail = base + (" NOT NULL" if not_null else "") if kind == "d" else ""
        objects.append(CatalogObject(name, "domain" if kind == "d" else "composite", detail, doc, tuple(fields)))
    return tuple(objects)


def _postgres_routines(cursor) -> tuple[RoutineDef, ...]:
    cursor.execute(_PG_ROUTINES)
    routines = []
    for (namespace, name, native_kind, returns_set, result, doc, variadic,
         defaults, names, modes, types, column_names, column_types, oid) in cursor:
        parameters, parameter_types, outputs = [], [], []
        for index, dtype in enumerate(types):
            mode = modes[index] if modes else "i"
            parameter_name = names[index] if names else ""
            if mode in {"o", "b", "t"} and parameter_name:
                outputs.append(ColumnDef(parameter_name, dtype))
            if mode in {"o", "t"}:
                continue
            prefix = "VARIADIC " if mode == "v" else "INOUT " if mode == "b" else ""
            named = quote_identifier(parameter_name) + " " if parameter_name else ""
            parameters.append(prefix + named + dtype)
            parameter_types.append(dtype)
        columns = tuple(outputs) or tuple(ColumnDef(name, dtype)
                                          for name, dtype in zip(column_names, column_types))
        kind = {"a": "aggregate", "w": "window", "p": "procedure"}.get(native_kind, "scalar")
        if kind == "scalar" and (returns_set or column_names or len(outputs) > 1):
            kind = "table"
        if kind == "aggregate" and not parameters:
            parameters.append("*")
        routines.append(RoutineDef(
            relation_name(namespace, name), tuple(parameters), result or "", doc or "",
            kind, bool(variadic), tuple(parameter_types), columns if kind == "table" else (),
            min_args=max(0, len(parameters) - defaults - int(variadic)),
            oid=oid,
        ))
    return tuple(routines)


def _duckdb_routines(con) -> tuple[RoutineDef, ...]:
    rows = con.execute("""
        SELECT CASE WHEN database_name IN ('system', 'temp', current_database())
                    THEN NULL ELSE database_name END,
               schema_name, function_name, function_type, description,
               return_type, parameters, parameter_types, varargs
        FROM duckdb_functions()
        WHERE function_type IN ('scalar', 'aggregate', 'window', 'table', 'macro', 'table_macro')
        ORDER BY database_name, schema_name, function_name, function_oid
    """).fetchall()
    routines = []
    for database, namespace, name, native_kind, doc, result, names, types, varargs in rows:
        names = names or ()
        parameter_types = [types[index] or "" if types and index < len(types) else ""
                           for index in range(len(names))]
        parameters = [f"{name} {dtype}".strip() for name, dtype in zip(names, parameter_types)]
        if varargs:
            parameters.append(f"VARIADIC {varargs}")
            parameter_types.append(varargs)
        kind = {"macro": "scalar", "table_macro": "table"}.get(native_kind, native_kind)
        parts = (database, namespace, name) if database else (namespace, name)
        # duckdb_functions exposes arguments, not table output schemas. Never
        # bind or execute user routines to guess their result columns.
        routines.append(RoutineDef(relation_name(*parts), tuple(parameters), result or "",
                                   doc or "", kind, bool(varargs), tuple(parameter_types),
                                   min_args=len(names) if native_kind in {"scalar", "aggregate", "window"} else None))
    return tuple(routines)


def refresh_source_schema(source: DataSource) -> None:
    with source._lock:
        source._pg_origin_catalog = None
        if source.con is None:
            source.datasets, source.routines, source.objects = {}, (), ()
            return
        try:
            datasets = {}
            routines, objects = (), ()
            origin_index = None
            if source.kind == "d1-live":
                with source.con.operation() as operation:
                    datasets = source.con.schema(operation)
                source.datasets, source.routines, source.objects = datasets, routines, objects
                return
            if source.kind == "postgres-live":
                by_oid = {}
                names_by_oid = {}
                origin_index = {}
                # A savepoint inside an existing transaction isolates discovery
                # failures without committing or rolling back the user's work.
                with source.con.transaction(), source.con.cursor() as cursor:
                    cursor.execute(_PG_COLUMNS)
                    for oid, namespace, name, doc, column, dtype, column_doc, temporary, enum_values, kind, attnum in cursor:
                        key = relation_name(namespace, name)
                        item = datasets.setdefault(key, {"columns": [], "doc": doc or "", "rows": None,
                                                          "foreign_keys": [], "temp": bool(temporary),
                                                          "kind": _PG_RELATION_KINDS[kind], "indexes": []})
                        if column is not None:
                            item["columns"].append(ColumnDef(column, dtype, column_doc or "",
                                                             tuple(enum_values) if enum_values is not None else None))
                            origin_index[oid, attnum] = (key, column)
                        by_oid[oid] = item
                        names_by_oid[oid] = key
                    cursor.execute(_PG_FKS)
                    keys = {}
                    for oid, constraint, namespace, target, column, target_column in cursor:
                        if oid not in by_oid:
                            continue
                        entry = keys.setdefault((oid, constraint), (relation_name(namespace, target), [], []))
                        entry[1].append(column)
                        entry[2].append(target_column)
                    for (oid, _), (target, columns, target_columns) in keys.items():
                        by_oid[oid]["foreign_keys"].append(ForeignKey(tuple(columns), target, tuple(target_columns)))
                    cursor.execute(_PG_PARTITIONS)
                    for child, parent in cursor:
                        if child in by_oid and parent in by_oid:
                            by_oid[child]["partition_parent"] = names_by_oid[parent]
                    cursor.execute(_PG_INDEXES)
                    for oid, namespace, name, method, key_count, unique, primary, valid, partial, immediate, columns in cursor:
                        if oid in by_oid:
                            by_oid[oid]["indexes"].append(IndexDef(
                                relation_name(namespace, name), method, tuple(columns[:key_count]),
                                tuple(columns[key_count:]), unique, primary, valid, partial, immediate,
                            ))
                    routines = _postgres_routines(cursor)
                    objects = _postgres_objects(cursor)
            elif source.kind == "parquet":
                rows = source.con.execute("""
                    SELECT table_schema, table_name, column_name, data_type
                    FROM information_schema.columns
                    WHERE table_catalog = current_database() OR table_catalog = 'temp'
                    ORDER BY table_schema, table_name, ordinal_position
                """).fetchall()
                for namespace, name, column, dtype in rows:
                    key = "data" if namespace == "main" and name == "data" else relation_name(namespace, name)
                    item = datasets.setdefault(key, {"columns": [], "doc": "", "rows": None, "foreign_keys": []})
                    item["columns"].append(ColumnDef(column, dtype))
                routines = _duckdb_routines(source.con)
            else:
                raise EngineError("Source is disconnected")
            for item in datasets.values():
                item["columns"] = tuple(item["columns"])
                item["foreign_keys"] = tuple(item["foreign_keys"])
                if "indexes" in item:
                    item["indexes"] = tuple(item["indexes"])
            source.datasets, source.routines, source.objects = datasets, routines, objects
            if origin_index is not None:
                source._pg_origin_catalog = (source.con, datasets, origin_index)
        except EngineError:
            raise
        except Exception as exc:
            raise _safe_error(exc, "Schema discovery") from None


def _postgres_disconnected(con, exc=None) -> bool:
    """Distinguish a lost session from server errors on a working session."""
    import psycopg
    from psycopg.pq import TransactionStatus

    if con.closed or con.info.transaction_status == TransactionStatus.UNKNOWN:
        return True
    if isinstance(exc, psycopg.Error):
        state = exc.sqlstate or ""
        # LOCK/statement timeouts and user cancellation are statement errors,
        # not transport timeouts. Never match localized driver message text.
        if state:
            # SQL can explicitly RAISE an arbitrary SQLSTATE at ERROR severity
            # without terminating the connection (including in a user txn).
            return (exc.diag.severity_nonlocalized != "ERROR"
                    and (state.startswith("08") or state in _TERMINATED))
        return isinstance(exc, psycopg.OperationalError)
    return isinstance(exc, (ConnectionError, TimeoutError))


def _discard_connection(source: DataSource) -> None:
    """Retire an unusable handle under the source lock. Do not cancel or retry."""
    con, source.con = source.con, None
    source._transaction = False
    source._pg_origin_catalog = None
    if con is not None:
        try:
            con.close()
        except Exception:
            pass  # Already unusable; never expose driver close diagnostics.


def preflight_source(source: DataSource) -> bool:
    """Worker-only liveness check BEFORE dispatch. False permits reconnect."""
    if source.kind != "postgres-live":
        return source.con is not None and not getattr(source.con, "closed", False)

    from psycopg import errors, generators
    from psycopg.pq import ExecStatus, TransactionStatus

    if not source._lock.acquire(blocking=False):
        raise EngineError("PostgreSQL connection is busy; nothing was executed")
    try:
        con = source.con
        if con is None:
            return False
        if _postgres_disconnected(con):
            lost_transaction = source._transaction
            _discard_connection(source)
            if lost_transaction:
                raise EngineError("PostgreSQL connection lost during your transaction; this submission was not executed")
            return False
        status = con.info.transaction_status
        if status in (TransactionStatus.INTRANS, TransactionStatus.INERROR):
            # Even SELECT 1 (or a cancelled probe) can change an open
            # transaction. Let the user's own SQL run exactly once instead.
            source._transaction = True
            return True
        if status != TransactionStatus.IDLE:
            raise EngineError("PostgreSQL connection is busy; nothing was executed")
        source._transaction = False
        try:
            # Empty simple-query round trip, no relation/function lookup or
            # result rows. Use libpq's nonblocking path with a client-side
            # deadline: a server-side timeout cannot bound a broken socket.
            con.pgconn.send_query(b"")
            results = con.wait(generators.execute(con.pgconn), timeout=_RUN_PROBE_SECONDS)
            if results and results[0].status == ExecStatus.FATAL_ERROR:
                raise errors.error_from_result(results[0])
        except Exception as exc:
            if _postgres_disconnected(con, exc):
                _discard_connection(source)
                if isinstance(exc, errors.Error) and exc.sqlstate in _TRANSACTION_TIMEOUTS:
                    raise EngineError("PostgreSQL transaction timed out; this submission was not executed") from None
                return False
            if con.info.transaction_status == TransactionStatus.ACTIVE:
                _discard_connection(source)  # Probe interrupted before ReadyForQuery.
            raise _safe_error(exc, "PostgreSQL health check") from None
        if len(results) != 1 or results[0].status != ExecStatus.EMPTY_QUERY:
            _discard_connection(source)
            raise EngineError("Unexpected PostgreSQL health check response; nothing was executed")
        return True
    finally:
        source._lock.release()


def transaction_open(source: DataSource) -> bool:
    con = source.con
    if con is None:
        return False
    if source.kind == "postgres-live":
        from psycopg.pq import TransactionStatus
        return con.info.transaction_status in (TransactionStatus.INTRANS, TransactionStatus.INERROR, TransactionStatus.ACTIVE)
    if source.kind == "d1-live":
        return False
    return source._transaction


def cancel_source(source: DataSource) -> None:
    con = source.con
    if con is None:
        return
    try:
        if source.kind == "postgres-live":
            # Cancellation must not wait on the execution lock.
            con.cancel_safe(timeout=2.0)
        elif source.kind == "parquet":
            con.interrupt()
        elif source.kind == "d1-live":
            con.cancel()
    except Exception:
        # A concurrent close/completion is harmless; no driver text is exposed.
        pass


def close_source(source: DataSource) -> None:
    cancel_source(source)
    with source._lock:
        con, source.con = source.con, None
        source._transaction = False
        source._pg_origin_catalog = None
        if con is not None:
            try:
                con.close()
            except Exception as exc:
                raise _safe_error(exc, "Connection close") from None
