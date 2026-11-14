"""D1 REST behavior with real SQLite execution supplied only by the test harness."""
from contextlib import contextmanager
import json
import threading

import httpx
import pytest

from d8r.engine import (EngineError, add_d1_source, cancel_source, close_source,
                        column_values, execute_sql, refresh_source_schema,
                        sample_rows, transaction_open)
from d8r.engine import d1api
from d8r.sql import relation_name
from tests.d1_support import ACCOUNT_ID, API_TOKEN, DATABASE_ID, DATABASE_NAME


def _source(**kwargs):
    return add_d1_source("d1-test", account_id=ACCOUNT_ID, database=DATABASE_ID,
                         api_token=API_TOKEN, **kwargs)


def _queries(harness):
    return [json.loads(request.content)["sql"] for request in harness.requests if request.method == "POST"]


@contextmanager
def _transport(source, handler):
    original = source.con._client
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        source.con._client = client
        try:
            yield
        finally:
            source.con._client = original


def test_uuid_source_metadata_and_secret_boundaries(d1_source, d1_harness):
    assert d1_source.kind == "d1-live"
    assert d1_source.dialect == "sqlite"
    assert d1_source.default_schema is None
    assert d1_source.d1 == {"account_id": ACCOUNT_ID, "database": DATABASE_ID}
    assert d1_source.connection_key == f"d1:{ACCOUNT_ID}:{DATABASE_ID}"
    assert d1_source.server_version == d1_harness.connection.execute("SELECT sqlite_version()").fetchone()[0]
    assert isinstance(d1_source.con, d1api.CloudflareD1)
    assert API_TOKEN not in repr(d1_source)
    assert API_TOKEN not in repr(d1_source.con)
    assert all(request.method == "POST" for request in d1_harness.requests)
    assert all(request.url.path.endswith("/raw") for request in d1_harness.requests)
    assert all(request.url.host == "api.cloudflare.com" for request in d1_harness.requests)
    assert not any("FROM records" in sql or "COUNT(" in sql.upper() for sql in _queries(d1_harness))
    assert not transaction_open(d1_source)


def test_name_resolution_canonical_identity(d1_harness):
    source = add_d1_source("named", account_id=ACCOUNT_ID.upper(), database=DATABASE_NAME,
                           api_token=API_TOKEN, display=f"name {API_TOKEN}")
    try:
        assert source.connection_key == f"d1:{ACCOUNT_ID}:{DATABASE_ID}"
        assert API_TOKEN not in source.display
        assert d1_harness.requests[0].method == "GET"
        assert d1_harness.requests[0].url.params["name"] == DATABASE_NAME
    finally:
        close_source(source)


@pytest.mark.parametrize("field,value", [("account_id", "bad"), ("account_id", "g" * 32),
    ("api_token", ""), ("api_token", "********"), ("api_token", "••••••"),
    ("api_token", "header\ninjection"), ("database", "")])
def test_invalid_credentials_never_contact_network(d1_harness, field, value):
    args = dict(account_id=ACCOUNT_ID, database=DATABASE_ID, api_token=API_TOKEN)
    args[field] = value
    with pytest.raises(EngineError):
        add_d1_source("bad", **args)
    assert d1_harness.requests == []


def test_paginated_exact_name_lookup(monkeypatch, d1_harness):
    requests = []
    def handler(request):
        if request.method != "GET":
            return d1_harness.handle(request)
        requests.append(request)
        page = int(request.url.params["page"])
        rows = ([{"name": DATABASE_NAME + "-other", "uuid": DATABASE_ID}] if page == 1
                else [{"name": DATABASE_NAME, "uuid": DATABASE_ID}])
        return httpx.Response(200, json={"success": True, "errors": [], "result": rows,
            "result_info": {"page": page, "per_page": 1, "total_pages": 2}})
    monkeypatch.setattr(d1api, "create_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    source = add_d1_source("named", account_id=ACCOUNT_ID, database=DATABASE_NAME, api_token=API_TOKEN)
    try:
        assert source.d1["database"] == DATABASE_ID
        assert [request.url.params["page"] for request in requests] == ["1", "2"]
    finally:
        close_source(source)


@pytest.mark.parametrize("rows", [[], [
    {"name": DATABASE_NAME, "uuid": DATABASE_ID},
    {"name": DATABASE_NAME, "uuid": "22222222-2222-4333-8444-555555555555"}]])
def test_missing_or_ambiguous_names_fail_closed(monkeypatch, d1_harness, rows):
    def handler(request):
        return httpx.Response(200, json={"success": True, "errors": [], "result": rows})
    monkeypatch.setattr(d1api, "create_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(EngineError):
        add_d1_source("named", account_id=ACCOUNT_ID, database=DATABASE_NAME, api_token=API_TOKEN)


def test_lookup_page_bound(monkeypatch, d1_harness):
    requests = []
    def handler(request):
        requests.append(request)
        return httpx.Response(200, json={"success": True, "errors": [],
            "result": [{"name": "other", "uuid": DATABASE_ID}],
            "result_info": {"page": int(request.url.params["page"]), "per_page": 1}})
    monkeypatch.setattr(d1api, "_MAX_LIST_PAGES", 2)
    monkeypatch.setattr(d1api, "create_client", lambda: httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(EngineError, match="page limit"):
        add_d1_source("named", account_id=ACCOUNT_ID, database=DATABASE_NAME, api_token=API_TOKEN)
    assert len(requests) == 2


def test_schema_declared_types_generated_columns_and_foreign_key_pairs(d1_source, d1_harness):
    d1_harness.connection.executescript('''
        CREATE TABLE "odd parent"("first id" TEXT, second INTEGER, PRIMARY KEY(second,"first id"));
        CREATE TABLE child(x INTEGER, y TEXT,
            total TEXT GENERATED ALWAYS AS (y || x) VIRTUAL,
            FOREIGN KEY(x,y) REFERENCES "odd parent"(second,"first id"));
        CREATE TABLE implicit(x INTEGER, y TEXT, FOREIGN KEY(x,y) REFERENCES "odd parent");
        CREATE VIEW "quoted \' view" AS SELECT x, y FROM child;
        CREATE TABLE _cf_private(value TEXT);
    ''')
    d1_harness.requests.clear()
    refresh_source_schema(d1_source)
    child = d1_source.datasets[relation_name("child")]
    assert [(column.name, column.type) for column in child["columns"]] == [("x", "INTEGER"), ("y", "TEXT"), ("total", "TEXT")]
    key = child["foreign_keys"][0]
    assert key.columns == ("x", "y")
    assert key.target_table == relation_name("odd parent")
    assert key.target_columns == ("second", "first id")
    assert d1_source.datasets[relation_name("implicit")]["foreign_keys"][0].target_columns == key.target_columns
    assert relation_name("quoted ' view") in d1_source.datasets
    assert relation_name("_cf_private") not in d1_source.datasets
    assert all(item["rows"] is None for item in d1_source.datasets.values())
    assert all("COUNT(" not in sql.upper() and "SELECT *" not in sql.upper() for sql in _queries(d1_harness))


def test_duplicate_empty_columns_and_empty_rows(d1_source):
    result = execute_sql(d1_source, 'SELECT id AS same, label AS same, NULL AS "" FROM records ORDER BY id')
    assert result["columns"] == ["same", "same", ""]
    assert result["rows"] == [[1, "alpha", None], [2, "beta", None], [3, None, None]]
    empty = execute_sql(d1_source, 'SELECT id AS "", label AS "" FROM records WHERE 0')
    assert empty["columns"] == ["", ""]
    assert empty["rows"] == []
    assert empty["total"] == 0


def test_normal_preview_does_not_rewrite_sql(d1_source, d1_harness):
    sql = "SELECT id FROM records ORDER BY id; -- preserved"
    result = execute_sql(d1_source, sql, max_rows=1)
    assert result["rows"] == [[1]]
    assert result["total"] == 2
    assert result["sql"] == sql
    assert _queries(d1_harness)[-1] == sql


def test_native_writes_and_schema_change(d1_source):
    created = execute_sql(d1_source, 'CREATE TABLE "new table"(x TEXT)')
    assert created["schema_changed"]
    inserted = execute_sql(d1_source, "INSERT INTO records VALUES (4, 'delta')")
    assert inserted["total"] == 1
    assert inserted["columns"] == []
    assert not inserted["schema_changed"]
    assert execute_sql(d1_source, "SELECT label FROM records WHERE id=4")["rows"] == [["delta"]]
    refresh_source_schema(d1_source)
    assert relation_name("new table") in d1_source.datasets


@pytest.mark.parametrize("sql", ["BEGIN", "COMMIT", "ROLLBACK", "SAVEPOINT x", "RELEASE x",
    "CREATE TEMP TABLE t(x)", "CREATE TEMPORARY VIEW v AS SELECT 1",
    'CREATE TABLE "temp".t(x)', "ATTACH ':memory:' AS other", "DETACH other",
    "SELECT 1; SELECT 2"])
def test_unsupported_session_semantics_before_http(d1_source, d1_harness, sql):
    d1_harness.requests.clear()
    with pytest.raises(EngineError):
        execute_sql(d1_source, sql)
    assert d1_harness.requests == []


@pytest.mark.parametrize("sql,rows", [
    ("SELECT DISTINCT label FROM records ORDER BY label", [[None], ["alpha"]]),
    ("SELECT id FROM records UNION SELECT id FROM records ORDER BY id", [[1], [2]]),
    ("WITH RECURSIVE n(x) AS (VALUES(1) UNION ALL SELECT x+1 FROM n WHERE x<5) SELECT x FROM n", [[1], [2]]),
    ("WITH a AS MATERIALIZED (SELECT id FROM records) SELECT * FROM a ORDER BY id", [[1], [2]]),
    ("VALUES (1), (2), (3)", [[1], [2]]),
    ("SELECT id FROM records ORDER BY id; -- trailing comment", [[1], [2]]),
])
def test_readonly_real_explain_and_bounded_outer_query(d1_source, d1_harness, sql, rows):
    d1_harness.requests.clear()
    result = execute_sql(d1_source, sql, readonly=True, max_rows=2)
    assert result["rows"] == rows
    assert result["total"] == 3
    assert result["sql"] == sql
    queries = _queries(d1_harness)
    assert len(queries) == 2
    assert queries[0].startswith("EXPLAIN ")
    assert queries[1].startswith("SELECT * FROM (")
    assert queries[1].endswith("LIMIT 3")


@pytest.mark.parametrize("sql", ["DELETE FROM records", "PRAGMA user_version=7", "VACUUM",
    "SELECT load_extension('secret')", 'SELECT "writefile"(\'secret\', label) FROM records',
    "SELECT * FROM pragma_writable_schema(1)", "SELECT 1; DROP TABLE records"])
def test_readonly_unsafe_statements_never_execute(d1_source, d1_harness, sql):
    d1_harness.requests.clear()
    with pytest.raises(EngineError):
        execute_sql(d1_source, sql, readonly=True)
    assert d1_harness.requests == []
    assert d1_harness.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 3


def test_with_write_is_rejected_by_real_sqlite_opcodes(d1_source, d1_harness):
    d1_harness.requests.clear()
    sql = "WITH selected AS (SELECT 1) DELETE FROM records WHERE id IN (SELECT * FROM selected) RETURNING id"
    with pytest.raises(EngineError, match="EXPLAIN"):
        execute_sql(d1_source, sql, readonly=True)
    assert _queries(d1_harness) == ["EXPLAIN " + sql]
    assert d1_harness.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 3


def test_registered_samples_share_readonly_guard(d1_source, d1_harness):
    d1_harness.requests.clear()
    table = relation_name("records")
    assert sample_rows(d1_source, table, ["id"], limit=1)["rows"] == [[1]]
    values, more = column_values(d1_source, table, "label", limit=2)
    assert values == [None, "alpha"] and more
    assert sum(sql.startswith("EXPLAIN ") for sql in _queries(d1_harness)) == 2
    count = len(d1_harness.requests)
    with pytest.raises(EngineError):
        sample_rows(d1_source, table, ["id; DELETE FROM records"])
    assert len(d1_harness.requests) == count


@pytest.mark.parametrize("payload", [
    {}, [], {"success": False, "errors": [{"message": API_TOKEN}], "result": []},
    {"success": True, "errors": [{"message": API_TOKEN}], "result": []},
    {"success": True, "result": [{"success": False, "error": API_TOKEN}]},
    {"success": True, "result": []},
    {"success": True, "result": [{"success": True, "results": {"columns": ["a"], "rows": [[]]}}]},
    {"success": True, "result": [{"success": True, "results": {"columns": [1], "rows": []}}]},
    {"success": True, "result": [{"success": True, "results": {"columns": ["a"], "rows": [{"a": API_TOKEN}]}}]},
    {"success": True, "result": [{"success": True, "results": {"columns": ["a"], "rows": [[{"secret": API_TOKEN}]]}}]},
])
def test_invalid_envelopes_are_safe_and_never_retried(d1_source, payload):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(200, json=payload)
    with _transport(d1_source, handler):
        with pytest.raises(EngineError) as error:
            execute_sql(d1_source, "SELECT 1")
    assert API_TOKEN not in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [302, 401, 403, 429, 500, 503])
def test_http_errors_never_follow_or_retry(d1_source, status):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, headers={"Location": "https://example.invalid/"}, text=API_TOKEN)
    with _transport(d1_source, handler):
        with pytest.raises(EngineError) as error:
            execute_sql(d1_source, "INSERT INTO records VALUES(4, 'delta')")
    assert API_TOKEN not in str(error.value)
    assert len(calls) == 1


def test_native_error_detail_not_exposed(d1_source):
    with pytest.raises(EngineError) as error:
        execute_sql(d1_source, f'SELECT * FROM "{API_TOKEN}"')
    assert API_TOKEN not in str(error.value)


def test_response_byte_bound(d1_source, monkeypatch):
    monkeypatch.setattr(d1api, "MAX_RESPONSE_BYTES", 64)
    class Stream(httpx.SyncByteStream):
        def __iter__(self):
            yield b" " * 40
            yield b" " * 40
    with _transport(d1_source, lambda request: httpx.Response(200, stream=Stream())):
        with pytest.raises(EngineError, match="safety limit"):
            execute_sql(d1_source, "SELECT 1")


def test_request_timeout_safe_and_not_retried(d1_source):
    calls = []
    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout(API_TOKEN, request=request)
    with _transport(d1_source, handler):
        with pytest.raises(EngineError, match="may have completed") as error:
            execute_sql(d1_source, "DELETE FROM records", timeout=0.1)
    assert len(calls) == 1
    assert API_TOKEN not in str(error.value)


def test_deadline_across_explain_and_execution(d1_source, d1_harness, monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(d1api, "monotonic", lambda: clock[0])
    calls = []
    def handler(request):
        calls.append(request)
        result = d1_harness.handle(request)
        clock[0] += 2
        return result
    with _transport(d1_source, handler):
        with pytest.raises(EngineError, match="deadline"):
            execute_sql(d1_source, "SELECT 1", readonly=True, timeout=1)
    assert len(calls) == 1


def test_cancel_does_not_wait_for_source_lock_or_claim_rollback(d1_source, d1_harness):
    entered, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    errors = []
    def handler(request):
        entered.set()
        assert release.wait(5)
        return d1_harness.handle(request)
    def run():
        try:
            execute_sql(d1_source, "INSERT INTO records VALUES(4,'late')")
        except EngineError as exc:
            errors.append(str(exc))
    def cancel():
        cancel_source(d1_source)
        cancelled.set()
    with _transport(d1_source, handler):
        worker = threading.Thread(target=run)
        worker.start()
        try:
            assert entered.wait(5)
            canceller = threading.Thread(target=cancel)
            canceller.start()
            try:
                assert cancelled.wait(1), "Cancellation blocked on the execution lock"
            finally:
                release.set()
                canceller.join(5)
        finally:
            release.set()
            worker.join(5)
    assert errors and "may have completed" in errors[0]
    assert d1_harness.connection.execute("SELECT label FROM records WHERE id=4").fetchone() == ("late",)
    assert execute_sql(d1_source, "SELECT 1")["rows"] == [[1]]


def test_truncated_explain_never_runs_target(d1_source, d1_harness):
    calls = []
    def handler(request):
        calls.append(request)
        response = d1_harness.handle(request)
        payload = response.json()
        payload["result"][0]["results"]["rows"] = payload["result"][0]["results"]["rows"][:2]
        return httpx.Response(200, json=payload)
    with _transport(d1_source, handler):
        with pytest.raises(EngineError, match="EXPLAIN"):
            execute_sql(d1_source, "SELECT id FROM records", readonly=True)
    assert len(calls) == 1


def test_close_clears_connection_and_token(d1_source):
    con = d1_source.con
    close_source(d1_source)
    assert d1_source.con is None
    assert con.api_token == ""
    assert con._client.is_closed
    assert not transaction_open(d1_source)
    with pytest.raises(EngineError):
        execute_sql(d1_source, "SELECT 1")


def test_native_blob_bracket_and_backtick_syntax(d1_source):
    result = execute_sql(d1_source, "SELECT x'0001ff' AS [blob value], 2 AS `odd;name`", max_rows=0)
    assert result["columns"] == ["blob value", "odd;name"]
    assert result["rows"] == [[[0, 1, 255], 2]]
    assert execute_sql(d1_source, "SELECT id FROM records", max_rows=0)["total"] == 3


def test_manual_trigger_is_one_native_statement(d1_source):
    execute_sql(d1_source, "CREATE TABLE audit(value TEXT)")
    result = execute_sql(d1_source, '''CREATE TRIGGER record_audit AFTER INSERT ON records
        BEGIN INSERT INTO audit VALUES(CASE WHEN NEW.label IS NULL THEN 'none' ELSE NEW.label END);
        INSERT INTO audit VALUES('done'); END;''')
    assert result["schema_changed"]
    execute_sql(d1_source, "INSERT INTO records VALUES(4, 'native')")
    assert execute_sql(d1_source, "SELECT value FROM audit ORDER BY rowid")["rows"] == [["native"], ["done"]]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_deadlines_make_no_request(d1_source, d1_harness, timeout):
    d1_harness.requests.clear()
    with pytest.raises(EngineError):
        execute_sql(d1_source, "SELECT 1", timeout=timeout)
    assert d1_harness.requests == []


def test_transport_deadline_is_capped(d1_source, d1_harness):
    execute_sql(d1_source, "SELECT 1", timeout=1000000)
    timeout = d1_harness.requests[-1].extensions["timeout"]
    assert all(0 < value <= 30 for value in timeout.values())


def test_malformed_json_is_sanitized(d1_source):
    with _transport(d1_source, lambda request: httpx.Response(200, text=API_TOKEN)):
        with pytest.raises(EngineError) as error:
            execute_sql(d1_source, "SELECT 1")
    assert API_TOKEN not in str(error.value)


def test_metadata_failure_keeps_previous_schema(d1_source):
    previous = d1_source.datasets
    with _transport(d1_source, lambda request: httpx.Response(503, text=API_TOKEN)):
        with pytest.raises(EngineError):
            refresh_source_schema(d1_source)
    assert d1_source.datasets is previous


@pytest.mark.parametrize("sql,hint", [
    ("SELECT FROM records", "commas"),
    ("SELECT id FROM records AS a JOIN records AS b ON a.id = b.id", "alias.column"),
    (f'SELECT * FROM "{API_TOKEN}"', "schema qualification"),
    (f'SELECT "{API_TOKEN}".missing FROM records', "column spelling"),
    ("SELECT private_missing_function(id) FROM records", "argument count"),
    ("SELECT sum(count(*)) FROM records", "GROUP BY"),
])
def test_native_sqlite_errors_offer_safe_hints(d1_source, d1_harness, sql, hint):
    d1_harness.requests.clear()
    with pytest.raises(EngineError) as error:
        execute_sql(d1_source, sql)
    message = str(error.value)
    assert hint in message
    assert API_TOKEN not in message
    assert "private_missing_function" not in message
    assert "Do not retry writes automatically" in message
    assert _queries(d1_harness) == [sql]
    assert execute_sql(d1_source, "SELECT 42")["rows"] == [[42]]


@pytest.mark.parametrize("status,nested", [(200, False), (400, False), (200, True)])
def test_remote_sql_error_envelopes_use_curated_hints_only(d1_source, status, nested):
    remote = f'D1_ERROR: near "{API_TOKEN}": syntax error at offset 7: SQLITE_ERROR'
    result = {"success": False, "error": remote}
    payload = ({"success": True, "errors": [], "result": [result]} if nested else
               {"success": False, "errors": [{"code": 7500, "message": remote}], "result": []})
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, json=payload)

    with _transport(d1_source, handler):
        with pytest.raises(EngineError) as error:
            execute_sql(d1_source, "SELECT 1")
    message = str(error.value)
    assert "SQL syntax error" in message
    assert "commas" in message
    assert API_TOKEN not in message
    assert "offset 7" not in message
    assert len(calls) == 1


@pytest.mark.parametrize("remote", [
    API_TOKEN + ' near "private": syntax error',
    "x" * 2048 + ' near "private": syntax error',
    {"message": API_TOKEN}, None,
])
def test_remote_error_classification_is_bounded_and_allowlisted(d1_source, remote):
    payload = {"success": False, "errors": [{"message": remote}], "result": []}
    with _transport(d1_source, lambda request: httpx.Response(400, json=payload)):
        with pytest.raises(EngineError) as error:
            execute_sql(d1_source, "SELECT 1")
    message = str(error.value)
    assert "HTTP 400" in message
    assert "Hint:" not in message
    assert API_TOKEN not in message
    assert "private" not in message
