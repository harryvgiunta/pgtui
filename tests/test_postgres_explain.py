"""PostgreSQL estimate-only AI plans and shared read-only safety boundaries."""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import replace
import json
import threading
from types import SimpleNamespace

import pytest

from d8r.ai import context as context_module
from d8r.ai.context import AIContext
from d8r.engine import EngineError, cancel_source, execute_sql, explain_sql, transaction_open
from d8r.tui.ai import AIChat, AIChats
from d8r.tui.session import Session
from tests.test_native_ai import run_async, tool
from tests.test_native_engine import postgres_source


@pytest.fixture
def plan_session(postgres_source, tmp_path):
    return Session({postgres_source.id: postgres_source}, data_dir=tmp_path / "plan-state")


def plan_context(session):
    return AIContext(session, session.active_id, "SELECT 1")


def test_plan_has_native_estimates_without_executing_target(postgres_source):
    execute_sql(postgres_source, "CREATE TEMP TABLE plan_items(id INTEGER PRIMARY KEY)")
    execute_sql(postgres_source, "INSERT INTO plan_items VALUES (1), (2)")
    execute_sql(postgres_source, "CREATE TEMP TABLE plan_other(id INTEGER)")
    execute_sql(postgres_source, "INSERT INTO plan_other VALUES (1), (3)")
    # The target would divide by zero if executed; column references prevent
    # PostgreSQL from doing that calculation during constant folding.
    result = explain_sql(postgres_source,
                         "SELECT a.id / (a.id - a.id) FROM plan_items a "
                         "JOIN plan_other b USING (id) WHERE b.id > 0; -- trailing comment")
    assert result["truncated"] is False
    plan = result["plan"][0]
    nodes = [plan["Plan"]]
    relations = []
    joins = []
    while nodes:
        node = nodes.pop()
        assert node["Plan Rows"] >= 0
        assert node["Total Cost"] >= node["Startup Cost"]
        assert "Actual Rows" not in node
        assert "Actual Total Time" not in node
        if "Relation Name" in node:
            relations.append(node["Relation Name"])
        if "Join Type" in node:
            joins.append(node["Join Type"])
        nodes.extend(node.get("Plans", []))
    assert set(relations) == {"plan_items", "plan_other"}
    assert joins == ["Inner"]
    assert "Execution Time" not in plan
    assert not transaction_open(postgres_source)
    # Manual native EXPLAIN remains allowed; test_query's path still rejects it.
    assert execute_sql(postgres_source, "EXPLAIN SELECT 1")["rows"]
    with pytest.raises(EngineError, match="non-mutating"):
        execute_sql(postgres_source, "EXPLAIN SELECT 1", readonly=True)


@run_async
async def test_plan_opt_in_expands_functions_without_execution_receipts(plan_session):
    session = plan_session
    source = session.source
    execute_sql(source, "CREATE TEMP TABLE plan_items(id INTEGER)")
    execute_sql(source, "INSERT INTO plan_items VALUES (1), (2)")
    session.save_fn("plan_filter", "n", "SELECT * FROM plan_items WHERE id >= @n", "")
    session.update_settings(ai_test_queries=False)
    disabled = plan_context(session)
    assert "explain_query" not in {item["function"]["name"] for item in disabled.tools}
    refusal = await tool(disabled, "explain_query", text="SELECT 1")
    assert refusal["explained"] is False
    assert "disabled" in refusal["error"]
    session.update_settings(ai_test_queries=True)
    context = plan_context(session)
    before = deepcopy(session.workspace)
    assert not session.ai_allow_sample_data
    prepared = await context.check_query("SELECT * FROM plan_filter(2)", execute=False)
    assert prepared == {"prepared": True, "executed": False}
    before_explain = dict(context.query_checks)
    result = await tool(context, "explain_query", text="SELECT * FROM plan_filter(2)")
    assert result["explained"] is True
    assert result["analyze"] is False
    assert result["plan"][0]["Plan"]["Relation Name"] == "plan_items"
    assert "2" in result["plan"][0]["Plan"]["Filter"]
    assert "rows" not in result and "executed" not in result
    assert context.query_checks == before_explain
    assert session.workspace == before
    rejected = await tool(context, "explain_query", text="SELECT 1", analyze=True)
    assert rejected["explained"] is False and "plan" not in rejected
    session.update_settings(ai_test_queries=False)
    assert (await tool(context, "explain_query", text="SELECT 1"))["explained"] is False


@pytest.mark.parametrize("backend", ["parquet_source", "d1_source"])
@run_async
async def test_plan_is_unavailable_for_other_backends(request, backend, tmp_path):
    source = request.getfixturevalue(backend)
    session = Session({source.id: source}, data_dir=tmp_path / "other-plan-state")
    session.update_settings(ai_test_queries=True)
    context = plan_context(session)
    assert "explain_query" not in {item["function"]["name"] for item in context.tools}
    result = await tool(context, "explain_query", text="SELECT 1")
    assert result["explained"] is False
    assert "PostgreSQL" in result["error"]
    with pytest.raises(EngineError, match="PostgreSQL"):
        explain_sql(source, "SELECT 1")


@pytest.mark.parametrize("query", [
    "EXPLAIN SELECT 1",
    "EXPLAIN (ANALYZE TRUE, FORMAT JSON) SELECT pg_sleep(5)",
    "SELECT 1; DELETE FROM plan_guard",
    "WITH changed AS (DELETE FROM plan_guard RETURNING *) SELECT * FROM changed",
    "SELECT * INTO TEMP plan_stolen FROM plan_guard",
    "SELECT pg_catalog.lo_export(1, '/tmp/forbidden')",
    'SELECT pg_catalog."pg_read_file"(\'/tmp/private\')',
    "SELECT pg_advisory_lock(1)",
    "SELECT set_config('statement_timeout', '0', false)",
    'SELECT U&"pg_read_file"(\'/tmp/private\')',
])
def test_plan_rejects_options_mutations_and_unsafe_functions(postgres_source, query):
    execute_sql(postgres_source, "CREATE TEMP TABLE plan_guard(id INTEGER)")
    execute_sql(postgres_source, "INSERT INTO plan_guard VALUES (1)")
    with pytest.raises(EngineError):
        explain_sql(postgres_source, query)
    assert not transaction_open(postgres_source)
    assert execute_sql(postgres_source, "SELECT * FROM plan_guard")["rows"] == [[1]]


def test_plan_refuses_user_transaction_without_rolling_it_back(postgres_source):
    execute_sql(postgres_source, "BEGIN")
    try:
        execute_sql(postgres_source, "CREATE TEMP TABLE plan_uncommitted(id INTEGER)")
        execute_sql(postgres_source, "INSERT INTO plan_uncommitted VALUES (7)")
        with pytest.raises(EngineError, match="open transaction"):
            explain_sql(postgres_source, "SELECT * FROM plan_uncommitted")
        assert transaction_open(postgres_source)
        assert execute_sql(postgres_source, "SELECT * FROM plan_uncommitted")["rows"] == [[7]]
    finally:
        execute_sql(postgres_source, "ROLLBACK")


def test_planning_uses_readonly_transaction(postgres_source):
    execute_sql(postgres_source, """
        CREATE FUNCTION pg_temp.plan_readonly() RETURNS BOOLEAN LANGUAGE plpgsql IMMUTABLE AS $$
        BEGIN
            IF current_setting('transaction_read_only') <> 'on' THEN
                RAISE EXCEPTION 'planning transaction is not read only';
            END IF;
            RETURN true;
        END $$
    """)
    result = explain_sql(postgres_source, "SELECT 1 WHERE pg_temp.plan_readonly()")
    assert result["plan"][0]["Plan"]["Node Type"] == "Result"
    assert "One-Time Filter" not in result["plan"][0]["Plan"]
    assert not transaction_open(postgres_source)


@pytest.mark.parametrize("cancel", [False, True])
def test_plan_deadline_and_cancellation_leave_connection_usable(postgres_source, cancel):
    execute_sql(postgres_source, """
        CREATE FUNCTION pg_temp.slow_plan() RETURNS BOOLEAN LANGUAGE plpgsql IMMUTABLE AS $$
        BEGIN
            PERFORM pg_sleep(5);
            RETURN true;
        END $$
    """)
    timer = threading.Timer(0.15, cancel_source, args=(postgres_source,)) if cancel else None
    if timer:
        timer.start()
    try:
        with pytest.raises(EngineError, match="cancelled" if cancel else "deadline|cancelled"):
            explain_sql(postgres_source, "SELECT 1 WHERE pg_temp.slow_plan()",
                        timeout=2 if cancel else 0.05)
    finally:
        if timer:
            timer.cancel()
            timer.join()
    assert not transaction_open(postgres_source)
    assert execute_sql(postgres_source, "SELECT 42", readonly=True)["rows"] == [[42]]


@run_async
async def test_plan_limit_withholds_whole_tree_and_driver_errors(plan_session):
    plan_session.update_settings(ai_test_queries=True)
    context = plan_context(plan_session)
    large = " UNION ALL ".join(f"SELECT {n} AS n" for n in range(400))
    result = await tool(context, "explain_query", text=large)
    assert result["explained"] is False
    assert result["truncated"] is True
    assert "plan" not in result
    assert len(json.dumps(result)) < 1000
    error = await context.call_tool("explain_query", {"text": "SELECT 'private-password'::integer"})
    assert json.loads(error)["explained"] is False
    assert "private-password" not in error
    assert not transaction_open(plan_session.source)
    recovery = await tool(context, "explain_query", text="VALUES (42)")
    assert recovery["explained"] is True
    assert recovery["plan"][0]["Plan"]["Node Type"] == "Result"


@pytest.mark.parametrize("change", ["permission", "source", "cancel"])
@run_async
async def test_inflight_plans_respect_revocation_and_connection_ownership(plan_session, monkeypatch, change):
    session = plan_session
    session.update_settings(ai_test_queries=True)
    context = plan_context(session)
    manager = AIChats(SimpleNamespace(), session)
    chat = AIChat("plan-chat", "target", provider_snapshot=(session.ai_provider_revision, session.ai_config))
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    original = context_module.explain_sql

    def blocked_plan(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5)
        return original(*args, **kwargs)

    monkeypatch.setattr(context_module, "explain_sql", blocked_plan)
    waiter = asyncio.create_task(manager._call_tool(chat, context, "explain_query", {"text": "SELECT 1"}, 0))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        if change == "permission":
            session.update_settings(ai_test_queries=False)
        elif change == "source":
            session.sources[session.active_id] = replace(context.source)
        else:
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert session.busy == "AI context lookup running"
            blocked = json.loads(await manager._call_tool(chat, context, "schema", {}, 0))
            assert "try again" in blocked["error"]
    finally:
        release.set()
        await asyncio.gather(*manager.context_tasks)
    if change != "cancel":
        result = json.loads(await waiter)
        assert result["explained"] is False and "plan" not in result
        assert ("disabled" if change == "permission" else "source changed") in result["error"].lower()
    assert not manager.context_tasks
    assert session.busy == ""
