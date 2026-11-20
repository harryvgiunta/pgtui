"""Bounded native execution, guarded result pages and explicit native runs."""
from __future__ import annotations

import csv
import json
import math
import threading
from contextlib import nullcontext
from time import perf_counter
from typing import TextIO
from uuid import uuid4

from d8r.sql import SqlError, cte_declarations, lex_sql, quote_identifier, relation_name, relation_parts, split_statements

from .datasources import (
    DataSource, EngineError, _discard_connection, _postgres_disconnected, _safe_error,
    cancel_source, transaction_open,
)

_DEFAULT_ROWS = 10000
_PLAN_CHARS = 64000
# Standalone SET is already excluded by _QUERY_WORDS. Within a query SET
# also names the generated column of a recursive CTE's SEARCH/CYCLE clause;
# UPDATE/MERGE and other writes remain prohibited independently.
_WRITE_WORDS = frozenset("INSERT UPDATE DELETE MERGE INTO CREATE ALTER DROP TRUNCATE COPY CALL DO EXECUTE PREPARE GRANT REVOKE VACUUM ANALYZE REFRESH LOCK RESET".split())
_QUERY_WORDS = frozenset(("SELECT", "WITH", "VALUES", "TABLE"))
_DUCK_EXTERNAL = frozenset(("COPY", "EXPORT", "IMPORT", "ATTACH", "DETACH", "LOAD", "INSTALL", "EXTENSION", "PREPARE", "EXECUTE", "CALL"))
_UNSAFE_FUNCTIONS = frozenset(("set_config", "nextval", "setval", "pg_notify", "pg_stat_file", "pg_file_rename", "pg_file_unlink", "pg_file_write", "pg_promote", "http", "system", "shell", "exec"))
_UNSAFE_PREFIXES = ("lo_", "dblink", "http_", "pg_read_", "pg_write_", "pg_ls_", "pg_advisory_", "pg_try_advisory_", "pg_log_", "pg_logical_", "pg_replication_", "pg_create_", "pg_drop_", "pg_backup_", "pg_wal_", "pg_stat_reset", "pg_reload_", "pg_rotate_", "pg_terminate_", "pg_cancel_", "pg_switch_")


def _row_limit(value) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EngineError("Row limit must be a nonnegative integer")
    return value or _DEFAULT_ROWS


def _read_functions(tokens):
    """Check explicit calls and collect their names for the native volatility check."""
    names = set()
    declarations = {name for name, _ in cte_declarations(tokens)}
    for index, token in enumerate(tokens):
        if token.kind not in {"word", "quoted_identifier"}:
            continue
        name = token.text[1:-1].replace('""', '"') if token.kind == "quoted_identifier" else token.text.lower()
        following = tokens[index + 1].text if index + 1 < len(tokens) else ""
        if name.lower() == "u" and following == "&":
            raise EngineError("Escaped Unicode identifiers are not accepted by read-only tools")
        if following == "(" and index not in declarations:
            if name.lower() in _UNSAFE_FUNCTIONS or name.lower().startswith(_UNSAFE_PREFIXES):
                raise EngineError("Read-only tools cannot call file, network, session-control or mutation functions")
            names.add(name)
    return names


def _page_read_by_default(words):
    # Routing only, not a read-only proof. The chosen read still goes through
    # the engine guard; a rejected read is never retried as native SQL.
    if not words or words[0] not in _QUERY_WORDS:
        return False
    if "INTO" in words or (words[0] == "WITH" and {"INSERT", "DELETE", "UPDATE", "MERGE"}.intersection(words)):
        return False  # SELECT INTO or a modifying CTE/terminal DML.
    locks = (["UPDATE"], ["NO", "KEY", "UPDATE"], ["SHARE"], ["KEY", "SHARE"])
    return not any(words[i + 1:i + 1 + len(clause)] == clause
                   for i, word in enumerate(words) if word == "FOR" for clause in locks)


def _pg_previewable(con, words, tokens):
    # DECLARE cannot run DML/SELECT INTO. Don't partially apply row locks or
    # stop a directly called volatile routine (including user functions) early.
    if not words or words[0] not in _QUERY_WORDS or "FOR" in words or _WRITE_WORDS.intersection(words):
        return False
    try:
        names = _read_functions(tokens)
    except EngineError:
        return False
    if names:
        # Check all overloads conservatively. No extra round trip for plain scans.
        with con.cursor() as cursor:
            cursor.execute("SELECT EXISTS (SELECT FROM pg_catalog.pg_proc "
                           "WHERE proname = ANY(%s::name[]) AND provolatile = 'v')", (list(names),))
            if cursor.fetchone()[0]:
                return False
    return True


def _pg_fetch_page(cursor, sql, cap, offset):
    """Leave only the fetched PGresult on cursor; release the portal before returning."""
    import psycopg
    from psycopg.pq import TransactionStatus

    # Only generated identifiers and validated integers enter the control SQL.
    # Owning the prefix lets us map DECLARE diagnostics back to the user's text.
    name = f"pgtui_{uuid4().hex}"
    prefix = f"DECLARE {name} NO SCROLL CURSOR WITHOUT HOLD FOR "
    try:
        cursor.execute(prefix + sql, prepare=False)
    except psycopg.Error as exc:
        if _postgres_disconnected(cursor.connection, exc):
            raise  # Keep native transport diagnostics for the connection owner.
        raise _safe_error(exc, position_offset=len(prefix)) from None
    try:
        if offset:
            cursor.execute(f"MOVE FORWARD {offset} FROM {name}")
        cursor.execute(f"FETCH FORWARD {cap + 1} FROM {name}")
    finally:
        # In a failed transaction PostgreSQL releases the portal on rollback.
        # Don't mask the error or roll back a transaction owned by the caller.
        if cursor.connection.info.transaction_status == TransactionStatus.INTRANS:
            with cursor.connection.cursor() as control:
                control.execute(f"CLOSE {name}")


def _pg_result(con, sql, cap, *, server_cursor, offset):
    with con.cursor() as cursor:
        if server_cursor:
            _pg_fetch_page(cursor, sql, cap, offset)
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


def _pg_explain(con, sql):
    from psycopg.types.string import TextLoader

    # EXPLAIN is not valid in DECLARE CURSOR. libpq buffers this single JSON
    # cell; bound it before JSON decoding and never publish a partial plan tree.
    with con.cursor() as cursor:
        cursor.adapters.register_loader("json", TextLoader)
        cursor.execute("EXPLAIN (FORMAT JSON, ANALYZE FALSE) " + sql, prepare=False)
        text = cursor.fetchone()[0]
    truncated = len(text) > _PLAN_CHARS
    return {"plan": None if truncated else json.loads(text),
            "truncated": truncated, "plan_limit": _PLAN_CHARS}


def _duck_result(con, sql, cap, statement_type, offset):
    # DuckDB streams normal SELECTs. For a requested later page, skip rows in
    # the native relation rather than materializing the prefix in Python.
    cursor = con.sql(sql).limit(cap + 1, offset) if offset else con.execute(sql)
    description = cursor.description
    if description:
        buffered = cursor.fetchmany(cap + 1)
        return ([column[0] for column in description], [str(column[1]) for column in description],
                [list(row) for row in buffered[:cap]], len(buffered), statement_type)
    return [], [], [], 0, statement_type


def _cursor_to_csv(cursor, writer, batch_size, expired):
    writer.writerow(column[0] for column in cursor.description)
    count = 0
    while not expired.is_set():
        rows = cursor.fetchmany(batch_size)
        if not rows:
            return count
        writer.writerows(rows)
        count += len(rows)
    raise EngineError("Query deadline exceeded")


def execute_sql(source: DataSource, sql: str, *, max_rows=10000, readonly=False, timeout=30.0,
                offset=0, page_reads=False) -> dict:
    """Execute one statement with a visible ``max_rows + 1`` lower-bound count.

    ``total`` is an exact count only when the result was exhausted; when rows are
    truncated it is the observed lower bound (displayed rows + one). Non-result
    PostgreSQL statements report the driver's affected-row count. A zero preview
    limit means the default 10,000-row cap, never an unbounded allocation.
    An offset requests a fresh read-only page; totals are relative to that page.
    Eligible manual PostgreSQL reads use a short-lived server cursor, without
    retaining an open transaction between pages or rewriting the query.
    ``page_reads`` selects guarded paging for reads while preserving native
    writes and caller-owned transactions. ``readonly`` always requires a read.

    Read-only PostgreSQL transactions are database protection, not a sandbox for
    privileged server extensions or arbitrary side effects of installed functions.
    DuckDB permits only native SELECT statements for read-only execution and has
    permanently locked external access limited to the original local Parquet file.
    D1 buffers a bounded (50 MB) HTTP response, which can exceed the visible cap;
    native SQL is unchanged. Paged D1 reads inspect remote EXPLAIN opcodes and
    then use an outer LIMIT, not a connection-local read-only transaction. This
    does not bound scan cost. Cancellation/deadlines cannot roll back remote work.
    """
    return _execute_statement(source, sql, max_rows=max_rows, readonly=readonly, timeout=timeout,
                              offset=offset, page_reads=page_reads)


def export_query_csv(source: DataSource, sql: str, output: TextIO, *, d1_page_size=1000, timeout=30.0) -> int:
    """Write a fresh, guarded read from row one; return the actual CSV data row count."""
    writer = csv.writer(output)
    if source.kind == "d1-live":
        # HTTP has no cursor. Keep the original response bound and reapply the
        # read-only EXPLAIN guard on every page, including the first.
        offset = 0
        columns = None
        while True:
            page = execute_sql(source, sql, readonly=True, max_rows=d1_page_size, offset=offset, timeout=timeout)
            if columns is None:
                columns = page["columns"]
                writer.writerow(columns)
            elif columns != page["columns"]:
                raise EngineError("Result columns changed during export; run the query again")
            rows = page["rows"]
            writer.writerows(rows)
            offset += len(rows)
            if page["total"] <= len(rows):
                return offset
    result = _execute_statement(source, sql, readonly=True, max_rows=1024, timeout=timeout, csv_writer=writer)
    return result["total"]


def prepare_sql(source: DataSource, sql: str, *, timeout=30.0) -> None:
    """Plan one read-only query on its native engine without running the query."""
    _execute_statement(source, sql, max_rows=1, readonly=True, timeout=timeout, prepare_only=True)


def explain_sql(source: DataSource, sql: str, *, timeout=30.0) -> dict:
    """Return a bounded PostgreSQL JSON estimate, never EXPLAIN ANALYZE.

    Planning can evaluate immutable functions; the ordinary read-only guards
    still apply. This is not an execution check or a privileged-function sandbox.
    """
    if source.kind != "postgres-live":
        raise EngineError("Query plans are available only for connected PostgreSQL sources")
    return _execute_statement(source, sql, max_rows=1, readonly=True, timeout=timeout, explain=True)


def _execute_statement(source, sql, *, max_rows, readonly, timeout, explain=False, prepare_only=False,
                       offset=0, page_reads=False, csv_writer=None):
    cap = _row_limit(max_rows)
    if type(offset) is not int or not 0 <= offset <= 2**63 - 1 or (offset and not readonly):
        raise EngineError("Page offset must be a nonnegative integer and is only supported for read-only queries")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise EngineError("Query timeout must be a positive finite number")
    if not isinstance(sql, str):
        raise EngineError("SQL must be text")
    if explain:
        if len(sql) > 64000:
            raise EngineError("Supply a query of at most 64,000 characters")
        timeout = min(timeout, 30.0)
    try:
        statements = split_statements(sql, dialect=source.dialect)
        tokens = [t for t in lex_sql(sql, dialect=source.dialect) if t.kind not in {"whitespace", "comment"}]
        words = [t.text.upper() for t in tokens if t.kind == "word"]
    except SqlError:
        raise EngineError("SQL contains an unterminated literal or comment. Hint: Check matching quotes and close any block comment.") from None
    if len(statements) != 1:
        raise EngineError("Execute exactly one SQL statement at a time")
    if source.kind == "d1-live":
        from .d1api import execute_d1
        readonly = readonly or (page_reads and _page_read_by_default(words))
        return execute_d1(source, sql, statements[0], cap=cap, readonly=readonly, timeout=timeout,
                          prepare_only=prepare_only, offset=offset)
    first = words[0] if words else ""
    with source._lock:
        if source.con is None or source.kind not in {"postgres-live", "parquet"}:
            raise EngineError("Source is disconnected")
        if source.kind == "postgres-live" and _postgres_disconnected(source.con):
            lost = " Your transaction is no longer available." if source._transaction else ""
            _discard_connection(source)
            raise EngineError("PostgreSQL connection lost; this statement was not sent." + lost)
        user_transaction = transaction_open(source)
        if readonly and user_transaction:
            raise EngineError("Read-only tools cannot run inside your open transaction")
        if page_reads and not user_transaction and source.kind == "postgres-live":
            readonly = readonly or _page_read_by_default(words)
        started = perf_counter()
        expired = threading.Event()
        def interrupt():
            expired.set()
            cancel_source(source)
        timer = threading.Timer(float(timeout), interrupt)
        timer.daemon = True
        own_transaction = False
        disconnected = False
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
                if page_reads and not user_transaction and statement_type == "SELECT":
                    readonly = True
                if readonly and statement_type != "SELECT":
                    raise EngineError("Read-only tools accept a single SELECT query only")
            elif readonly and (first not in _QUERY_WORDS or any(word in _WRITE_WORDS for word in words)):
                # PostgreSQL READ ONLY allows writes to temporary tables. This
                # additional lexical prohibition closes explicit temp DML/INTO;
                # the server transaction remains the authority for nested writes.
                raise EngineError("Read-only tools accept a single non-mutating query only")
            if readonly and source.kind == "postgres-live":
                _read_functions(tokens)
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
            if prepare_only or explain:
                if explain:
                    result = _pg_explain(source.con, sql)
                else:
                    # Fixed EXPLAIN, never caller-supplied options or ANALYZE.
                    if source.kind == "postgres-live":
                        with source.con.cursor() as cursor:
                            cursor.execute("EXPLAIN (ANALYZE FALSE) " + sql, prepare=False)
                    else:
                        source.con.execute("EXPLAIN " + sql)
                    result = {}
                if expired.is_set():
                    raise EngineError("Query deadline exceeded")
                return {**result, "ms": (perf_counter() - started) * 1000}
            if csv_writer is not None:
                if source.kind == "postgres-live":
                    # A named cursor fetches each batch from one server-side result;
                    # fetchmany on a regular psycopg cursor would buffer it in libpq.
                    with source.con.cursor(name=f"pgtui_{uuid4().hex}") as cursor:
                        cursor.execute(sql)
                        count = _cursor_to_csv(cursor, csv_writer, cap, expired)
                else:
                    count = _cursor_to_csv(source.con.sql(sql), csv_writer, cap, expired)
                return {"total": count}
            if source.kind == "postgres-live":
                server_cursor = readonly or _pg_previewable(source.con, words, tokens)
                # Never use WITH HOLD (commit would spool the remaining rows),
                # nor a savepoint that would mask an error in the user's transaction.
                transaction = source.con.transaction() if server_cursor and not (readonly or user_transaction) else nullcontext()
                with transaction:
                    columns, dtypes, rows, total, status = _pg_result(
                        source.con, sql, cap, server_cursor=server_cursor, offset=offset)
                pageable = server_cursor and not user_transaction
            else:
                columns, dtypes, rows, total, status = _duck_result(source.con, sql, cap, statement_type, offset)
                pageable = statement_type == "SELECT" and not user_transaction
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
                    "status": status, "schema_changed": schema_changed, "pageable": pageable}
        except EngineError:
            raise
        except Exception as exc:
            if source.kind == "postgres-live" and _postgres_disconnected(source.con, exc):
                disconnected = True
                lost = " Your transaction is no longer available." if user_transaction else ""
                raise EngineError("PostgreSQL connection lost during execution. The outcome may be unknown; "
                                  "nothing was replayed. Check the database before retrying." + lost) from None
            if expired.is_set():
                raise EngineError("Query deadline exceeded") from None
            raise _safe_error(exc) from None
        finally:
            # Join before another query acquires the lane: a stale timer must
            # never cancel the next user statement or the cleanup ROLLBACK.
            timer.cancel()
            timer.join()
            if disconnected:
                _discard_connection(source)
            elif own_transaction:
                try:
                    if source.kind == "postgres-live":
                        source.con.rollback()
                    else:
                        source.con.execute("ROLLBACK")
                        source._transaction = False
                except Exception:
                    # Do not return an unsafe connection to subsequent callers.
                    _discard_connection(source)
                    raise EngineError("Read-only transaction cleanup failed; reconnect the source") from None
            if source.kind == "postgres-live" and source.con is not None and not source.con.closed:
                # Keep the last known user state in case libpq reports UNKNOWN
                # by the next Run (e.g. idle_in_transaction_session_timeout).
                source._transaction = transaction_open(source)


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
    result = execute_sql(source,
                         f"SELECT DISTINCT {quote_identifier(column)} FROM {quoted_table} ORDER BY 1 LIMIT {cap + 1}",
                         max_rows=cap, readonly=True)
    return [row[0] for row in result["rows"]], result["total"] > len(result["rows"])
