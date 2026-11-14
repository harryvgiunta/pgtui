"""Native Cloudflare D1 REST client; never a local SQLite connection.

Cloudflare documents /raw as positional columns/rows (including duplicate names).
Normal Run sends authored SQL unchanged: its preview cap does not cap the remote
response. HTTP bodies are bounded separately at 50 MB. Read tools prepare EXPLAIN
before a bounded outer SELECT; this does not bound scan cost or provide a session
transaction. Cancellation/deadlines discard late results, not remote effects.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import math
import re
import threading
from time import monotonic, perf_counter
from uuid import UUID

import httpx

from d8r.sql import ColumnDef, ForeignKey, lex_sql, relation_name
from .datasources import DataSource, EngineError, _sql_error_message

_API = "https://api.cloudflare.com/client/v4"
MAX_RESPONSE_BYTES = 50_000_000
MAX_OPERATION_SECONDS = 30.0
_MAX_LIST_PAGES = 100
_MAX_TABLES = 2000
_UNCERTAIN = "Remote execution may have completed; cancellation does not roll back changes. Do not retry writes automatically."
_UNSAFE_FUNCTIONS = frozenset({
    "load_extension", "readfile", "writefile", "edit", "eval", "fts3_tokenizer",
    "sqlite_rename_table", "sqlite_rename_column", "sqlite_drop_column",
    "sqlite_dbpage", "sqlite_dbdata", "sqlite_dbptr", "zipfile", "fsdir",
    "shell", "system", "exec", "http", "https", "curl", "set_config",
})
_UNSAFE_PREFIXES = ("http_", "https_", "fileio_", "sqlite_rename_", "sqlite_unsafe_", "pragma_")
_CONTROL_OPCODES = frozenset({
    "OpenWrite", "SetCookie", "CreateBtree", "Destroy", "Clear", "ParseSchema",
    "LoadAnalysis", "DropTable", "DropIndex", "DropTrigger", "VCreate", "VDestroy",
    "VUpdate", "VRename", "Vacuum", "IncrVacuum", "Checkpoint", "JournalMode",
    "AutoCommit", "Savepoint", "SqlExec", "Expire", "Pagecount", "MaxPgcnt",
})

# Cloudflare/SQLite expose text rather than PostgreSQL diagnostics. Match only
# bounded, anchored native error formats, then discard every remote character.
_D1_ERROR_PATTERNS = tuple((re.compile(pattern), state) for pattern, state in (
    (r'(?:near ".{0,512}": syntax error|incomplete input|unrecognized token:)', "42601"),
    (r'ambiguous column name:', "42702"),
    (r'no such table:', "42P01"),
    (r'(?:no such column:|table .{1,512} has no column named)', "42703"),
    (r'(?:no such function:|wrong number of arguments to function)', "42883"),
    (r'(?:misuse of aggregate(?: function)?|aggregate functions are not allowed in the GROUP BY clause|HAVING clause on a non-aggregate query)', "42803"),
    (r'(?:datatype mismatch|cannot store .{1,512} value in .{1,512} column)', "42804"),
    (r'UNIQUE constraint failed:', "23505"),
    (r'FOREIGN KEY constraint failed', "23503"),
))


def _d1_error(payload, fallback):
    errors = payload.get("errors", [])
    messages = [payload.get("error")]
    if isinstance(errors, list):
        messages.extend(error.get("message") for error in errors[:8] if isinstance(error, dict))
    for message in messages:
        if not isinstance(message, str):
            continue
        message = message[:2048].removeprefix("D1_ERROR: ")
        for pattern, state in _D1_ERROR_PATTERNS:
            if pattern.match(message):
                return EngineError("Cloudflare D1: " + _sql_error_message(state) + " " + _UNCERTAIN)
    return EngineError(fallback + " " + _UNCERTAIN)


def create_client() -> httpx.Client:
    """Fixed-origin transport; no environment proxy, redirects, or retries."""
    return httpx.Client(timeout=30.0, follow_redirects=False, trust_env=False,
                        transport=httpx.HTTPTransport(retries=0))


def _malformed():
    return EngineError("Cloudflare D1 returned an invalid response")


def _uuid(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = str(UUID(value))
    except ValueError:
        return None
    return parsed if value.lower() == parsed else None


@dataclass
class _Operation:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)

    def remaining(self):
        if self.cancelled.is_set():
            raise EngineError("Cloudflare D1 request cancelled. " + _UNCERTAIN)
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise EngineError("Cloudflare D1 deadline exceeded. " + _UNCERTAIN)
        return remaining


@dataclass
class CloudflareD1:
    account_id: str
    database: str
    api_token: str = field(repr=False)
    _client: httpx.Client = field(repr=False)
    _active: _Operation | None = field(default=None, repr=False)
    _state: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _closed: bool = field(default=False, repr=False)

    @contextmanager
    def operation(self, timeout=MAX_OPERATION_SECONDS):
        operation = _Operation(monotonic() + min(float(timeout), MAX_OPERATION_SECONDS))
        with self._state:
            if self._closed:
                raise EngineError("Cloudflare D1 source is closed")
            if self._active is not None:
                raise EngineError("Cloudflare D1 source is busy")
            self._active = operation
        try:
            yield operation
            operation.remaining()
        finally:
            with self._state:
                self._active = None

    def cancel(self):
        # Never acquire the source execution lock or close a socket on the UI lane.
        with self._state:
            if self._active is not None:
                self._active.cancelled.set()

    def close(self):
        self.cancel()
        with self._state:
            self._closed = True
        try:
            self._client.close()
        except Exception:
            raise EngineError("Cloudflare D1 connection close failed") from None
        finally:
            self.api_token = ""

    def _request(self, operation, method, suffix="", *, params=None, body=None):
        url = f"{_API}/accounts/{self.account_id}/d1/database{suffix}"
        try:
            remaining = operation.remaining()
            with self._client.stream(method, url, params=params, json=body,
                                     headers={"Authorization": f"Bearer {self.api_token}",
                                              "Accept-Encoding": "identity"},
                                     follow_redirects=False, timeout=remaining) as response:
                operation.remaining()
                http_failed = response.status_code < 200 or response.status_code >= 300
                sql_request = method == "POST" and suffix.endswith("/raw")
                if http_failed and not (sql_request and response.status_code == 400):
                    raise EngineError(f"Cloudflare D1 request failed (HTTP {response.status_code}). " + _UNCERTAIN)
                if response.headers.get("content-encoding", "identity").lower() != "identity":
                    raise EngineError("Cloudflare D1 returned an unsupported compressed response")
                length = response.headers.get("content-length")
                if length is not None and (not length.isascii() or not length.isdigit()):
                    raise _malformed()
                if length is not None and int(length) > MAX_RESPONSE_BYTES:
                    raise EngineError("Cloudflare D1 response exceeds the 50 MB safety limit. " + _UNCERTAIN)
                data = bytearray()
                for chunk in response.iter_bytes():
                    operation.remaining()
                    if len(data) + len(chunk) > MAX_RESPONSE_BYTES:
                        raise EngineError("Cloudflare D1 response exceeds the 50 MB safety limit. " + _UNCERTAIN)
                    data.extend(chunk)
                operation.remaining()
                payload = json.loads(data)
            if not isinstance(payload, dict):
                raise _malformed()
            if "errors" in payload and not isinstance(payload["errors"], list):
                raise _malformed()
            if http_failed:
                raise _d1_error(payload, f"Cloudflare D1 request failed (HTTP {response.status_code}).")
            if payload.get("success") is not True or payload.get("errors"):
                fallback = "Cloudflare D1 rejected the request."
                if sql_request:
                    raise _d1_error(payload, fallback)
                raise EngineError(fallback + " " + _UNCERTAIN)
            if not isinstance(payload.get("result"), list):
                raise _malformed()
            operation.remaining()
            return payload
        except EngineError:
            raise
        except httpx.TimeoutException:
            raise EngineError("Cloudflare D1 deadline exceeded. " + _UNCERTAIN) from None
        except (ValueError, TypeError, UnicodeError, RecursionError):
            raise _malformed() from None
        except Exception:
            raise EngineError("Cloudflare D1 request failed. " + _UNCERTAIN) from None

    def resolve(self, name, operation):
        direct = _uuid(name)
        if direct is not None:
            self.database = direct
            return
        matches = set()
        for page in range(1, _MAX_LIST_PAGES + 1):
            payload = self._request(operation, "GET", params={"name": name, "page": page, "per_page": 1000})
            rows = payload["result"]
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("name"), str):
                    raise _malformed()
                if row["name"] == name:
                    target = _uuid(row.get("uuid"))
                    if target is None:
                        raise _malformed()
                    matches.add(target)
            info = payload.get("result_info", {})
            if not isinstance(info, dict):
                raise _malformed()
            per_page = info.get("per_page", 1000)
            reported_page = info.get("page", page)
            if type(per_page) is not int or not 1 <= per_page <= 10000 or reported_page != page:
                raise _malformed()
            total_pages = info.get("total_pages")
            if total_pages is not None and (type(total_pages) is not int or total_pages < 0):
                raise _malformed()
            if not rows or len(rows) < per_page or (total_pages is not None and page >= total_pages):
                break
        else:
            raise EngineError("Cloudflare D1 database lookup exceeded its page limit; use the database UUID")
        if len(matches) != 1:
            raise EngineError("Cloudflare D1 database name is missing or ambiguous; use the exact database UUID")
        self.database = matches.pop()

    def raw(self, sql, operation):
        if len(sql.encode("utf-8")) > 100_000:
            raise EngineError("Cloudflare D1 SQL exceeds the 100 KB statement limit")
        payload = self._request(operation, "POST", f"/{self.database}/raw", body={"sql": sql})
        results = payload["result"]
        if len(results) != 1 or not isinstance(results[0], dict):
            raise _malformed()
        result = results[0]
        if "errors" in result and not isinstance(result["errors"], list):
            raise _malformed()
        if result.get("success") is not True or result.get("error") or result.get("errors"):
            raise _d1_error(result, "Cloudflare D1 statement failed.")
        raw = result.get("results")
        if not isinstance(raw, dict):
            raise _malformed()
        columns, rows = raw.get("columns"), raw.get("rows")
        if not isinstance(columns, list) or any(not isinstance(name, str) for name in columns):
            raise _malformed()
        if not isinstance(rows, list):
            raise _malformed()
        for row in rows:
            if not isinstance(row, list) or len(row) != len(columns):
                raise _malformed()
            for value in row:
                scalar = value is None or isinstance(value, (str, int, bool)) or (isinstance(value, float) and math.isfinite(value))
                blob = isinstance(value, list) and all(type(byte) is int and 0 <= byte <= 255 for byte in value)
                if not scalar and not blob:
                    raise _malformed()
        meta = result.get("meta", {})
        if not isinstance(meta, dict):
            raise _malformed()
        changes = meta.get("changes", 0)
        if type(changes) is not int or changes < 0:
            raise _malformed()
        operation.remaining()
        return columns, rows, changes

    def schema(self, operation):
        _, tables, _ = self.raw("SELECT name, type FROM sqlite_schema WHERE type IN ('table', 'view') ORDER BY name", operation)
        if len(tables) > _MAX_TABLES:
            raise EngineError("Cloudflare D1 schema exceeds the metadata discovery limit")
        datasets, primary_keys, pending = {}, {}, {}
        for row in tables:
            if len(row) != 2 or not isinstance(row[0], str):
                raise _malformed()
            name, kind = row
            if name.lower().startswith(("sqlite_", "_cf_")):
                continue
            key = relation_name(name)
            literal = "'" + name.replace("'", "''") + "'"
            names, columns, _ = self.raw(f"PRAGMA table_xinfo({literal})", operation)
            required = {"cid", "name", "type", "pk", "hidden"}
            if not required.issubset(names):
                raise _malformed()
            entries = [dict(zip(names, column)) for column in columns]
            entries.sort(key=lambda column: column["cid"])
            if any(not isinstance(column["name"], str) or not isinstance(column["type"], str) for column in entries):
                raise _malformed()
            datasets[key] = {"columns": tuple(ColumnDef(column["name"], column["type"]) for column in entries if column["hidden"] != 1),
                             "doc": "", "rows": None, "foreign_keys": (), "temp": False}
            primary_keys[name] = tuple(column["name"] for column in sorted(entries, key=lambda column: column["pk"]) if column["pk"])
            if kind == "table":
                names, keys, _ = self.raw(f"PRAGMA foreign_key_list({literal})", operation)
                if not {"id", "seq", "table", "from", "to"}.issubset(names):
                    raise _malformed()
                pending[key] = [dict(zip(names, entry)) for entry in keys]
        for key, entries in pending.items():
            grouped = {}
            for entry in sorted(entries, key=lambda entry: (entry["id"], entry["seq"])):
                grouped.setdefault(entry["id"], []).append(entry)
            fks = []
            for group in grouped.values():
                target = group[0]["table"]
                target_columns = tuple(entry["to"] for entry in group)
                if all(column is None for column in target_columns):
                    target_columns = primary_keys.get(target, ())
                if len(target_columns) != len(group) or any(not isinstance(column, str) for column in target_columns):
                    continue
                fks.append(ForeignKey(tuple(entry["from"] for entry in group), relation_name(target), target_columns))
            datasets[key]["foreign_keys"] = tuple(fks)
        operation.remaining()
        return datasets


def add_d1_source(source_id, *, account_id, database, api_token, display=None):
    if not isinstance(account_id, str) or re.fullmatch(r"[0-9a-fA-F]{32}", account_id.strip()) is None:
        raise EngineError("A Cloudflare account ID must contain 32 hexadecimal characters")
    if (not isinstance(api_token, str) or not api_token or api_token != api_token.strip()
            or not api_token.isascii() or any(ord(char) < 33 or ord(char) == 127 for char in api_token)
            or set(api_token) <= {"*"}):
        raise EngineError("A nonempty, unmasked Cloudflare API token is required")
    if not isinstance(database, str) or not database.strip() or any(ord(char) < 32 for char in database):
        raise EngineError("A Cloudflare D1 database name or UUID is required")
    if display is not None and not isinstance(display, str):
        raise EngineError("A valid Cloudflare D1 display name is required")
    con = None
    try:
        con = CloudflareD1(account_id.strip().lower(), "", api_token, create_client())
        with con.operation() as operation:
            con.resolve(database.strip(), operation)
            _, rows, _ = con.raw("SELECT sqlite_version()", operation)
            if len(rows) != 1 or len(rows[0]) != 1 or not isinstance(rows[0][0], str) or re.fullmatch(r"\d+\.\d+\.\d+(?:\.\d+)?", rows[0][0]) is None:
                raise _malformed()
            source = DataSource(str(source_id).replace(api_token, "[redacted]"),
                                (display or f"Cloudflare D1 · {database.strip()}").replace(api_token, "[redacted]"),
                                "Cloudflare D1 · native SQLite over HTTPS", "d1-live", "sqlite", con=con,
                                d1={"account_id": con.account_id, "database": con.database},
                                connection_key=f"d1:{con.account_id}:{con.database}", server_version=rows[0][0])
            source.datasets = con.schema(operation)
        return source
    except Exception as exc:
        if con is not None:
            try:
                con.close()
            except Exception:
                pass
        if isinstance(exc, EngineError):
            raise
        raise EngineError("Cloudflare D1 connection failed") from None


def _function_name(token):
    text = token.text
    if token.kind == "quoted_identifier":
        if text.startswith("["):
            return text[1:-1].lower()
        return text[1:-1].replace(text[0] * 2, text[0]).lower()
    return text.lower()


def _unsafe_function(name):
    return name in _UNSAFE_FUNCTIONS or name.startswith(_UNSAFE_PREFIXES)


def _guard_readonly(con, sql, tokens, operation):
    for index, token in enumerate(tokens[:-1]):
        if token.kind in {"word", "quoted_identifier"} and tokens[index + 1].text == "(" and _unsafe_function(_function_name(token)):
            raise EngineError("Read-only D1 tools cannot call file, network, extension or mutation functions")
    names, rows, _ = con.raw("EXPLAIN " + sql, operation)
    required = {"addr", "opcode", "p1", "p2", "p3", "p4", "p5"}
    if not required.issubset(names) or len(set(names)) != len(names) or not rows:
        raise EngineError("Cloudflare D1 did not return a usable EXPLAIN program")
    program = [dict(zip(names, row)) for row in rows]
    if (any(type(instruction["addr"]) is not int or instruction["addr"] != index
            for index, instruction in enumerate(program))
            or program[0]["opcode"] != "Init"
            or type(program[0]["p2"]) is not int
            or not 0 <= program[0]["p2"] < len(program)
            or not any(instruction["opcode"] == "Halt" for instruction in program)
            or program[-1]["opcode"] not in {"Goto", "Halt"}):
        raise EngineError("Cloudflare D1 returned an incomplete EXPLAIN program")
    ephemeral = set()
    for instruction in program:
        opcode, p1, p2, p4 = (instruction[name] for name in ("opcode", "p1", "p2", "p4"))
        if (not isinstance(opcode, str)
                or any(type(instruction[name]) is not int for name in ("p1", "p2", "p3", "p5"))
                or (p4 is not None and not isinstance(p4, str))):
            raise _malformed()
        if opcode in _CONTROL_OPCODES or (opcode == "Transaction" and p2 != 0):
            raise EngineError("Read-only D1 EXPLAIN rejected a write or session-control operation")
        if opcode in {"OpenEphemeral", "OpenAutoindex", "SorterOpen"}:
            ephemeral.add(p1)
        elif opcode == "OpenDup" and p2 in ephemeral:
            ephemeral.add(p1)
        elif opcode in {"OpenRead", "OpenPseudo", "Close"}:
            ephemeral.discard(p1)
        if opcode in {"Insert", "Delete", "IdxInsert", "IdxDelete", "SorterInsert"} and p1 not in ephemeral:
            raise EngineError("Read-only D1 EXPLAIN rejected a persistent write")
        if opcode in {"Function", "PureFunc", "AggStep", "AggStep1", "AggFinal", "AggValue", "AggInverse"}:
            if not isinstance(p4, str) or _unsafe_function(p4.split("(", 1)[0].lower()):
                raise EngineError("Read-only D1 EXPLAIN rejected an unsafe function")


def execute_d1(source, sql, statement, *, cap, readonly, timeout):
    tokens = [token for token in lex_sql(statement, dialect="sqlite") if token.kind not in {"whitespace", "comment"}]
    words = [token.text.upper() for token in tokens if token.kind == "word"]
    first = words[0] if words else ""
    if first in {"BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE", "ABORT", "START"}:
        raise EngineError("Cloudflare D1 does not support session transactions across HTTP requests")
    if first == "CREATE" and len(words) > 1 and words[1] in {"TEMP", "TEMPORARY"}:
        raise EngineError("Cloudflare D1 does not support session TEMP/TEMPORARY objects")
    if first in {"ATTACH", "DETACH"} or (first == "CREATE" and any(_function_name(tokens[index]) == "temp" and tokens[index + 1].text == "." for index in range(len(tokens) - 1))):
        raise EngineError("Cloudflare D1 does not support attached or temporary session databases")
    if readonly and first not in {"SELECT", "WITH", "VALUES"}:
        raise EngineError("Read-only D1 tools accept one SELECT, WITH or VALUES query")
    with source._lock:
        con = source.con
        if con is None:
            raise EngineError("Source is disconnected")
        started = perf_counter()
        with con.operation(timeout) as operation:
            native = sql
            if readonly:
                _guard_readonly(con, statement, tokens, operation)
                # The newline protects the closing wrapper from a trailing line comment.
                native = f"SELECT * FROM (\n{statement}\n) AS _pgtui_read LIMIT {cap + 1}"
            columns, rows, changes = con.raw(native, operation)
            total = min(len(rows), cap + 1) if columns else changes
            result = {"columns": columns, "dtypes": ["" for _ in columns], "rows": rows[:cap],
                      "total": total, "ms": (perf_counter() - started) * 1000, "sql": sql,
                      "status": first or "OK", "schema_changed": not readonly and first in {"CREATE", "ALTER", "DROP", "REINDEX"}}
        return result
