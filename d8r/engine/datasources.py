"""Native database sources and metadata-only discovery."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
import threading

from d8r.sql import ColumnDef, ForeignKey, RoutineDef, quote_identifier, relation_name

POSTGRES_VERSION = "18.7"


class EngineError(ValueError):
    """A safe, public execution or connection error."""


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
    "25006": ("Read-only execution rejected a write", "Use ordinary Run only if the write is intended."),
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


def _safe_error(exc: Exception, operation: str = "SQL execution") -> EngineError:
    # Only real driver diagnostics qualify. Arbitrary exceptions are never read
    # or stringified; DuckDB has no SQLSTATE, so classify bounded known formats.
    import duckdb
    import psycopg

    if isinstance(exc, psycopg.Error):
        state = exc.sqlstate
        valid_state = isinstance(state, str) and re.fullmatch(r"[0-9A-Z]{5}", state) is not None
        suffix = f" [SQLSTATE {state}]" if valid_state else ""
        position = exc.diag.statement_position
        if isinstance(position, str) and re.fullmatch(r"[0-9]{1,10}", position) and 0 < int(position) <= 2147483647:
            suffix += f" at character {int(position)}"
        if valid_state and state in _SQL_ERRORS:
            return EngineError(_sql_error_message(state, suffix))
        return EngineError(f"{operation} failed" + suffix)
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
            return EngineError(_sql_error_message(state))
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
_PG_COLUMNS = """
SELECT c.oid, n.nspname, c.relname, obj_description(c.oid, 'pg_class'),
       a.attname, format_type(a.atttypid, a.atttypmod),
       col_description(c.oid, a.attnum), c.relpersistence = 't',
       (SELECT pg_catalog.array_agg(e.enumlabel::text ORDER BY e.enumsortorder)
        FROM pg_catalog.pg_enum e
        WHERE e.enumtypid = CASE WHEN column_type.typtype = 'd'
                               THEN column_type.typbasetype ELSE a.atttypid END)
FROM pg_catalog.pg_class c
JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
LEFT JOIN pg_catalog.pg_type column_type ON column_type.oid = a.atttypid
WHERE c.relkind IN ('r','p','v','m','f')
  AND ((n.nspname NOT LIKE 'pg\\_%%' ESCAPE '\\' AND n.nspname <> 'information_schema')
       OR n.oid = pg_my_temp_schema())
  AND has_schema_privilege(n.oid, 'USAGE')
  AND (has_table_privilege(c.oid, 'SELECT,INSERT,UPDATE,DELETE,TRUNCATE,REFERENCES,TRIGGER')
       OR has_any_column_privilege(c.oid, 'SELECT,INSERT,UPDATE,REFERENCES'))
ORDER BY n.nspname, c.relname, a.attnum
""".replace("%%", "%")
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
_PG_ROUTINES = """
SELECT n.nspname, p.proname, p.prokind, p.proretset,
       pg_catalog.pg_get_function_result(p.oid),
       pg_catalog.obj_description(p.oid, 'pg_proc'), p.provariadic <> 0,
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
             ORDER BY a.attnum)
FROM pg_catalog.pg_proc p
JOIN pg_catalog.pg_namespace n ON n.oid = p.pronamespace
JOIN pg_catalog.pg_type t ON t.oid = p.prorettype
WHERE (n.nspname = 'pg_catalog'
       OR (n.nspname NOT LIKE 'pg\\_%%' ESCAPE '\\' AND n.nspname <> 'information_schema')
       OR n.oid = pg_catalog.pg_my_temp_schema())
  AND pg_catalog.has_schema_privilege(n.oid, 'USAGE')
  AND pg_catalog.has_function_privilege(p.oid, 'EXECUTE')
ORDER BY n.nspname, p.proname, p.oid
""".replace("%%", "%")


def _postgres_routines(cursor) -> tuple[RoutineDef, ...]:
    cursor.execute(_PG_ROUTINES)
    routines = []
    for (namespace, name, native_kind, returns_set, result, doc, variadic,
         names, modes, types, column_names, column_types) in cursor:
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
                                   doc or "", kind, bool(varargs), tuple(parameter_types)))
    return tuple(routines)


def refresh_source_schema(source: DataSource) -> None:
    with source._lock:
        if source.con is None:
            source.datasets, source.routines = {}, ()
            return
        try:
            datasets = {}
            routines = ()
            if source.kind == "d1-live":
                with source.con.operation() as operation:
                    datasets = source.con.schema(operation)
                source.datasets, source.routines = datasets, routines
                return
            if source.kind == "postgres-live":
                by_oid = {}
                # A savepoint inside an existing transaction isolates discovery
                # failures without committing or rolling back the user's work.
                with source.con.transaction(), source.con.cursor() as cursor:
                    cursor.execute(_PG_COLUMNS)
                    for oid, namespace, name, doc, column, dtype, column_doc, temporary, enum_values in cursor:
                        key = relation_name(namespace, name)
                        item = datasets.setdefault(key, {"columns": [], "doc": doc or "", "rows": None,
                                                          "foreign_keys": [], "temp": bool(temporary)})
                        if column is not None:
                            item["columns"].append(ColumnDef(column, dtype, column_doc or "",
                                                             tuple(enum_values) if enum_values is not None else None))
                        by_oid[oid] = item
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
                    routines = _postgres_routines(cursor)
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
            source.datasets, source.routines = datasets, routines
        except EngineError:
            raise
        except Exception as exc:
            raise _safe_error(exc, "Schema discovery") from None


def transaction_open(source: DataSource) -> bool:
    if source.con is None:
        return False
    if source.kind == "postgres-live":
        from psycopg.pq import TransactionStatus
        return source.con.info.transaction_status in (TransactionStatus.INTRANS, TransactionStatus.INERROR, TransactionStatus.ACTIVE)
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
        if con is not None:
            try:
                con.close()
            except Exception as exc:
                raise _safe_error(exc, "Connection close") from None
