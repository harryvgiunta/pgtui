"""Bounded native execution; ordinary Run never translates or adds SQL LIMIT."""
from __future__ import annotations

import math
import threading
from time import perf_counter
from uuid import uuid4

from d8r.sql import SqlError, lex_sql, quote_identifier, relation_name, relation_parts, split_statements

from .datasources import DataSource, EngineError, _safe_error, cancel_source, transaction_open

_DEFAULT_ROWS = 10000
_WRITE_WORDS = frozenset("INSERT UPDATE DELETE MERGE INTO CREATE ALTER DROP TRUNCATE COPY CALL DO EXECUTE PREPARE GRANT REVOKE VACUUM ANALYZE REFRESH LOCK SET RESET".split())
_QUERY_WORDS = frozenset(("SELECT", "WITH", "VALUES", "TABLE"))
_DUCK_EXTERNAL = frozenset(("COPY", "EXPORT", "IMPORT", "ATTACH", "DETACH", "LOAD", "INSTALL", "EXTENSION", "PREPARE", "EXECUTE", "CALL"))
_UNSAFE_FUNCTIONS = frozenset(("set_config", "nextval", "setval", "pg_notify", "pg_stat_file", "pg_file_rename", "pg_file_unlink", "pg_file_write", "pg_promote", "http", "system", "shell", "exec"))
_UNSAFE_PREFIXES = ("lo_", "dblink", "http_", "pg_read_", "pg_write_", "pg_ls_", "pg_advisory_", "pg_try_advisory_", "pg_log_", "pg_logical_", "pg_replication_", "pg_create_", "pg_drop_", "pg_backup_", "pg_wal_", "pg_stat_reset", "pg_reload_", "pg_rotate_", "pg_terminate_", "pg_cancel_", "pg_switch_")


def _row_limit(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EngineError("Row limit must be a nonnegative integer")
    return value or _DEFAULT_ROWS


def _words(sql: str, *, dialect="postgres") -> list[str]:
    return [token.text.upper() for token in lex_sql(sql, dialect=dialect) if token.kind == "word"]


def _check_readonly_functions(sql):
    tokens = [token for token in lex_sql(sql) if token.kind not in {"whitespace", "comment"}]
    for index, token in enumerate(tokens):
        if token.kind not in {"word", "quoted_identifier"}:
            continue
        name = token.text.replace('""', '"')[1:-1] if token.kind == "quoted_identifier" else token.text
        name = name.lower()
        following = tokens[index + 1].text if index + 1 < len(tokens) else ""
        if name == "u" and following == "&":
            raise EngineError("Escaped Unicode identifiers are not accepted by read-only tools")
        if following == "(" and (name in _UNSAFE_FUNCTIONS or name.startswith(_UNSAFE_PREFIXES)):
            raise EngineError("Read-only tools cannot call file, network, session-control or mutation functions")


def _pg_result(con, sql, cap, readonly):
    # Read-only tools already own a transaction, so a server-side cursor also
    # bounds libpq memory. Normal native writes/transactions use the ordinary
    # cursor: stopping a stream early could cancel or roll back their effects.
    cursor = con.cursor(name=f"pgtui_{uuid4().hex}") if readonly else con.cursor()
    with cursor:
        if readonly:
            cursor.execute(sql)
        else:
            cursor.execute(sql, prepare=False)
        if cursor.description:
            columns = [column.name for column in cursor.description]
            dtypes = []
            for column in cursor.description:
                info = con.adapters.types.get(column.type_code)
                dtypes.append(info.name if info is not None else str(column.type_code))
            buffered = cursor.fetchmany(cap + 1)
            total = len(buffered)
            rows = [list(row) for row in buffered[:cap]]
        else:
            columns, dtypes, rows = [], [], []
            total = max(0, cursor.rowcount)
        return columns, dtypes, rows, total, cursor.statusmessage or "OK"


def _duck_result(con, sql, cap, statement_type):
    cursor = con.execute(sql)
    description = cursor.description
    if description:
        buffered = cursor.fetchmany(cap + 1)
        return ([column[0] for column in description], [str(column[1]) for column in description],
                [list(row) for row in buffered[:cap]], len(buffered), statement_type)
    return [], [], [], 0, statement_type


def execute_sql(source: DataSource, sql: str, *, max_rows=10000, readonly=False, timeout=30.0) -> dict:
    """Execute one statement with a visible ``max_rows + 1`` lower-bound count.

    ``total`` is an exact count only when the result was exhausted; when rows are
    truncated it is the observed lower bound (displayed rows + one). Non-result
    PostgreSQL statements report the driver's affected-row count. A zero preview
    limit means the default 10,000-row cap, never an unbounded allocation.

    Read-only PostgreSQL transactions are database protection, not a sandbox for
    privileged server extensions or arbitrary side effects of installed functions.
    DuckDB permits only native SELECT statements for read-only execution and has
    permanently locked external access limited to the original local Parquet file.
    D1 buffers a bounded (50 MB) HTTP response, which can exceed the visible cap;
    ordinary SQL is unchanged. D1 read tools inspect remote EXPLAIN opcodes and
    then use an outer LIMIT, not a connection-local read-only transaction. This
    does not bound scan cost. Cancellation/deadlines cannot roll back remote work.
    """
    cap = _row_limit(max_rows)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise EngineError("Query timeout must be a positive finite number")
    if not isinstance(sql, str):
        raise EngineError("SQL must be text")
    try:
        statements = split_statements(sql, dialect=source.dialect)
        words = _words(sql, dialect=source.dialect)
    except SqlError:
        raise EngineError("SQL contains an unterminated literal or comment") from None
    if len(statements) != 1:
        raise EngineError("Execute exactly one SQL statement at a time")
    if source.kind == "d1-live":
        from .d1api import execute_d1
        return execute_d1(source, sql, statements[0], cap=cap, readonly=readonly, timeout=timeout)
    first = words[0] if words else ""
    with source._lock:
        if source.con is None or source.kind not in {"postgres-live", "parquet"}:
            raise EngineError("Source is disconnected")
        if readonly and transaction_open(source):
            raise EngineError("Read-only tools cannot run inside your open transaction")
        started = perf_counter()
        expired = threading.Event()
        def interrupt():
            expired.set()
            cancel_source(source)
        timer = threading.Timer(float(timeout), interrupt)
        timer.daemon = True
        own_transaction = False
        statement_type = first
        timer.start()
        try:
            if source.kind == "parquet":
                statements_native = source.con.extract_statements(sql)
                if len(statements_native) != 1:
                    raise EngineError("Execute exactly one SQL statement at a time")
                statement_type = statements_native[0].type.name
                if statement_type in _DUCK_EXTERNAL or first in _DUCK_EXTERNAL:
                    raise EngineError("External writes, attachments, extensions and prepared execution are disabled for Parquet sources")
                if readonly and statement_type != "SELECT":
                    raise EngineError("Read-only tools accept a single SELECT query only")
            elif readonly and (first not in _QUERY_WORDS or any(word in _WRITE_WORDS for word in words)):
                # PostgreSQL READ ONLY allows writes to temporary tables. This
                # additional lexical prohibition closes explicit temp DML/INTO;
                # the server transaction remains the authority for nested writes.
                raise EngineError("Read-only tools accept a single non-mutating query only")
            if readonly and source.kind == "postgres-live":
                _check_readonly_functions(sql)
            if readonly:
                if source.kind == "postgres-live":
                    with source.con.cursor() as cursor:
                        cursor.execute("BEGIN READ ONLY")
                        own_transaction = True
                        cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(max(1, math.ceil(timeout * 1000))),))
                else:
                    source.con.execute("BEGIN TRANSACTION READ ONLY")
                    own_transaction = True
                    source._transaction = True
            if source.kind == "postgres-live":
                columns, dtypes, rows, total, status = _pg_result(source.con, sql, cap, readonly)
            else:
                columns, dtypes, rows, total, status = _duck_result(source.con, sql, cap, statement_type)
                if statement_type == "TRANSACTION" and not readonly:
                    if first in {"BEGIN", "START"}:
                        source._transaction = True
                    elif first in {"COMMIT", "END", "ROLLBACK", "ABORT"}:
                        source._transaction = False
            if expired.is_set():
                raise EngineError("Query deadline exceeded")
            schema_changed = not readonly and (first in {"CREATE", "ALTER", "DROP", "COMMENT", "GRANT", "REVOKE", "SECURITY", "DO", "CALL", "COMMIT", "END", "ROLLBACK", "ABORT"})
            return {"columns": columns, "dtypes": dtypes, "rows": rows, "total": total,
                    "ms": (perf_counter() - started) * 1000, "sql": sql,
                    "status": status, "schema_changed": schema_changed}
        except EngineError:
            raise
        except Exception as exc:
            if expired.is_set():
                raise EngineError("Query deadline exceeded") from None
            raise _safe_error(exc) from None
        finally:
            # Join before another query acquires the lane: a stale timer must
            # never cancel the next user statement or the cleanup ROLLBACK.
            timer.cancel()
            timer.join()
            if own_transaction:
                try:
                    if source.kind == "postgres-live":
                        source.con.rollback()
                    else:
                        source.con.execute("ROLLBACK")
                        source._transaction = False
                except Exception:
                    # Do not return an unsafe connection to subsequent callers.
                    con, source.con = source.con, None
                    source._transaction = False
                    try:
                        con.close()
                    except Exception:
                        pass
                    raise EngineError("Read-only transaction cleanup failed; reconnect the source") from None


def _registered_table(source, table):
    if table not in source.datasets:
        raise EngineError("Select an exact registered table name")
    parts = relation_parts(table, dialect=source.dialect)
    if not parts:
        raise EngineError("Invalid registered table identifier")
    return relation_name(*parts)


def sample_rows(source: DataSource, table: str, columns, limit=20) -> dict:
    quoted_table = _registered_table(source, table)
    available = {column.name for column in source.datasets[table]["columns"]}
    selected = tuple(columns) if columns is not None else tuple(column.name for column in source.datasets[table]["columns"])
    if not selected or any(column not in available for column in selected):
        raise EngineError("Select exact registered column names")
    cap = _row_limit(limit)
    sql = f"SELECT {', '.join(quote_identifier(column) for column in selected)} FROM {quoted_table}"
    return execute_sql(source, sql, max_rows=cap, readonly=True)


def column_values(source: DataSource, dataset: str, column: str, limit=100) -> tuple[list[object], bool]:
    quoted_table = _registered_table(source, dataset)
    if column not in {item.name for item in source.datasets[dataset]["columns"]}:
        raise EngineError("Select an exact registered column name")
    cap = _row_limit(limit)
    result = execute_sql(source, f"SELECT DISTINCT {quote_identifier(column)} FROM {quoted_table} ORDER BY 1",
                         max_rows=cap, readonly=True)
    return [row[0] for row in result["rows"]], result["total"] > len(result["rows"])
