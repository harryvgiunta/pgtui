"""Run preflight, uncertain outcomes, and cancellation across statement boundaries."""
from __future__ import annotations

import select
import socket
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import psycopg
import pytest
from psycopg.pq import TransactionStatus

from d8r.engine import DataSource, EngineError, close_source, execute_sql
from d8r.engine import datasources
from d8r.tui.session import Session
from tests.test_native_engine import postgres_source


def pg_session(con):
    source = DataSource("pg", "PostgreSQL", "", "postgres-live", "postgres", con=con, connection_key="test-pg")
    return Session({source.id: source})


def test_unresponsive_probe_has_client_deadline_and_retires_only_the_captured_handle(monkeypatch):
    # No server reply, no database or credentials needed. Exercise the actual
    # driver wait/deadline on an unreadable socket, not a timer that only sends
    # cancellation and leaves a blocked SQL worker behind.
    from psycopg.waiting import Wait

    sent = []
    stop = threading.Event()
    reader, writer = socket.socketpair()
    with reader, writer:
        wire = SimpleNamespace(socket=reader.fileno(), send_query=sent.append)
        con = SimpleNamespace(pgconn=wire, closed=False, info=SimpleNamespace(transaction_status=TransactionStatus.IDLE))
        con.wait = lambda gen, **kw: psycopg.Connection.wait(con, gen, **kw)
        con.close = reader.close

        def no_reply(pgconn):
            assert pgconn is wire
            while not stop.is_set():
                yield Wait.R

        monkeypatch.setattr(psycopg.generators, "execute", no_reply)
        monkeypatch.setattr(datasources, "_RUN_PROBE_SECONDS", 0.05)
        session = pg_session(con)
        captured = session.source
        assert session.source_connected()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(session.preflight_run, captured)
            try:
                assert pending.result(timeout=2) is False
            finally:
                stop.set()
                writer.close()
        assert sent == [b""]
        assert not session.source_connected() and reader.fileno() == -1
        assert session.source is captured and session.source_key() == "test-pg"


def test_preflight_detects_postgres_idle_timeout_before_dispatch(postgres_source):
    session = Session({postgres_source.id: postgres_source})
    assert session.preflight_run(postgres_source)
    execute_sql(postgres_source, "SET idle_session_timeout = '100ms'")
    # Wait for the server's close without consuming any libpq input. Merely
    # checking Connection.closed here cannot detect this stale session.
    assert select.select([postgres_source.con.fileno()], [], [], 5)[0]
    assert session.source_connected()
    assert not session.preflight_run(postgres_source)
    assert not session.source_connected() and not session.history


def test_probe_does_not_disturb_uncommitted_or_failed_transaction(postgres_source):
    session = Session({postgres_source.id: postgres_source})
    assert session.run("BEGIN").ok
    assert session.run("CREATE TEMP TABLE run_probe_tx(n int)").ok
    assert session.preflight_run(postgres_source)
    assert session.run("INSERT INTO run_probe_tx VALUES (1)").ok
    assert session.run("SET LOCAL statement_timeout = '30ms'").ok
    error = session.run("SELECT pg_sleep(3)")
    assert "57014" in error.error and session.source_connected()
    assert postgres_source.con.info.transaction_status == TransactionStatus.INERROR
    assert session.preflight_run(postgres_source)
    assert postgres_source.con.info.transaction_status == TransactionStatus.INERROR
    assert session.run("ROLLBACK").ok
    assert session.preflight_run(postgres_source) and session.run("SELECT 42").rows == [[42]]


def test_known_lost_transaction_cannot_resume_in_autocommit(postgres_source):
    session = Session({postgres_source.id: postgres_source})
    assert session.run("BEGIN").ok
    postgres_source.con.close()
    with pytest.raises(EngineError, match="transaction"):
        session.preflight_run(postgres_source)
    assert not session.source_connected()
    # A later, separate user Run may reconnect.
    assert not session.preflight_run(postgres_source)


@pytest.mark.parametrize("failure", [psycopg.errors.IdleInTransactionSessionTimeout,
                                    psycopg.OperationalError, TimeoutError])
def test_ambiguous_commit_stops_document_without_retry(monkeypatch, failure):
    from d8r.engine import execute

    sent = []
    con = SimpleNamespace(closed=False, info=SimpleNamespace(transaction_status=TransactionStatus.INTRANS))
    con.close = lambda: setattr(con, "closed", True)
    session = pg_session(con)
    # A probe must not send even an empty query inside a transaction. The
    # connection double deliberately has no cursor, pgconn, rollback or wait.
    assert session.preflight_run(session.source)

    def lost_reply(handle, sql, *args, **kwargs):
        assert handle is con
        sent.append(sql)
        raise failure("postgresql://private-secret@private-host")

    monkeypatch.setattr(execute, "_pg_result", lost_reply)
    outcomes = session.run_all("COMMIT; INSERT INTO log VALUES (1)")
    assert sent == ["COMMIT"] and len(outcomes) == 1
    assert "unknown" in outcomes[0].error and "transaction" in outcomes[0].error
    assert "private" not in outcomes[0].error
    assert not session.history and con.closed and not session.source_connected()


def test_statement_interrupt_does_not_trigger_reconnect(monkeypatch):
    from d8r.engine import execute

    con = SimpleNamespace(closed=False, info=SimpleNamespace(transaction_status=TransactionStatus.INTRANS))
    session = pg_session(con)

    def interrupted(*args, **kwargs):
        raise psycopg.errors.QueryCanceled("private SQL and driver diagnostics")

    monkeypatch.setattr(execute, "_pg_result", interrupted)
    outcome, = session.run_all("UPDATE t SET n = 1; DELETE FROM t")
    assert "57014" in outcome.error and "private" not in outcome.error
    assert session.source_connected() and session.source.con is con


def test_cancel_stops_queued_statements_while_preserving_completed_work(parquet_source, monkeypatch):
    session = Session({parquet_source.id: parquet_source})
    session.run("CREATE TEMP TABLE effects(n int)", record=False)
    done, release, cancelled = threading.Event(), threading.Event(), threading.Event()
    original = session._record_history

    def after_statement(entry, outcome):
        original(entry, outcome)
        done.set()
        assert release.wait(5)

    sql = "INSERT INTO effects VALUES (1); INSERT INTO effects VALUES (2)"
    with monkeypatch.context() as guarded, ThreadPoolExecutor(max_workers=1) as pool:
        guarded.setattr(session, "_record_history", after_statement)
        work = pool.submit(session.run_all, sql, cancel_event=cancelled)
        try:
            assert done.wait(5)
            cancelled.set()
            session.cancel()
            assert not work.done()  # cancellation is not proof of worker completion
        finally:
            release.set()
        first, stopped = work.result(timeout=5)
    assert first.ok and "cancelled" in stopped.error
    assert session.history[0].doc == first.doc and len(session.history) == 1
    assert session.run_all(sql, cancel_event=cancelled)[0].error
    assert session.run("SELECT n FROM effects").rows == [[1]]


def test_preflight_never_probes_d1_or_a_replaced_source(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    count = len(d1_harness.requests)
    assert session.preflight_run(d1_source)
    different = replace(d1_source, con=None)
    with pytest.raises(EngineError, match="target changed"):
        session.preflight_run(different)
    assert len(d1_harness.requests) == count


def test_busy_or_cancelled_probe_stops_submission_without_reconnecting():
    sent = []

    def interrupted(sql):
        sent.append(sql)
        raise psycopg.errors.QueryCanceled("private probe detail")

    con = SimpleNamespace(closed=False, info=SimpleNamespace(transaction_status=TransactionStatus.IDLE),
                          pgconn=SimpleNamespace(send_query=interrupted))
    session = pg_session(con)
    with ThreadPoolExecutor(max_workers=1) as pool:
        with session.source._lock:
            future = pool.submit(session.preflight_run, session.source)
            with pytest.raises(EngineError, match="busy"):
                future.result(timeout=2)
    assert not sent
    with pytest.raises(EngineError, match="57014") as error:
        session.preflight_run(session.source)
    assert "private" not in str(error.value)
    assert sent == [b""] and session.source_connected() and not session.history


def test_d1_lost_write_reply_is_not_replayed_or_followed_by_queued_sql(d1_source, d1_harness):
    import httpx
    from tests.test_d1_engine import _transport

    session = Session({d1_source.id: d1_source})
    before = len(d1_harness.requests)

    def lose_reply(request):
        d1_harness.handle(request)  # Native SQLite write really completed.
        raise httpx.ReadTimeout("private-token", request=request)

    with _transport(d1_source, lose_reply):
        assert session.preflight_run(d1_source)
        outcome, = session.run_all("INSERT INTO records(label) VALUES ('once'); DELETE FROM records")
    assert "may have completed" in outcome.error and "private-token" not in outcome.error
    assert len(d1_harness.requests) == before + 1 and not session.history
    assert session.run("SELECT count(*) FROM records").rows == [[4]]


def test_manual_schema_reload_rejects_stale_handles_and_refreshes_only_live_metadata(parquet_source):
    source = parquet_source
    session = Session({source.id: source})
    session.update_settings(value_cache_enabled=True)
    session.distinct_values(source, "data", "id")
    assert session.cached_values(source, "data", "id") is not None
    execute_sql(source, "CREATE TEMP TABLE externally_created(n int)")
    assert session.schema.table_by_name("externally_created") is None
    with pytest.raises(EngineError, match="original live"):
        session.reload_schema(replace(source))
    session.reload_schema(source)
    assert session.schema.table_by_name("externally_created") is not None
    assert session.cached_values(source, "data", "id") is None
    metadata = source.datasets
    close_source(source)
    with pytest.raises(EngineError, match="original live"):
        session.reload_schema(source)
    assert source.datasets is metadata
