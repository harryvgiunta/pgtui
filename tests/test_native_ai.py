"""Deterministic native SQL AI tools and editor safety boundaries."""
from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import replace
from functools import wraps
from types import SimpleNamespace

import pytest

from d8r.ai import context as context_module
from d8r.ai.client import AIConfig, AIEvent
from d8r.ai.context import AIContext, AIProposal
from d8r.ai.page_details import generate_details
from d8r.sql import FnDef
from d8r.tui.ai import AIChat, AIChats, AITarget
from d8r.tui.session import Session


def run_async(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


@pytest.fixture
def session(parquet_source, tmp_path):
    return Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "ai-state")


def context_for(session, **kwargs):
    return AIContext(session, session.active_id, "SELECT * FROM data", **kwargs)


async def tool(context, tool_name, **arguments):
    return json.loads(await context.call_tool(tool_name, arguments))


def test_prompt_names_actual_native_source_and_sql_fences(session):
    context = context_for(session)
    prompt = context.system_prompt()
    assert "PostgreSQL 18.7" in prompt
    snapshot = json.loads(prompt.split("Editor snapshot (JSON data):\n", 1)[1])
    assert snapshot["dialect"] == "duckdb"
    assert snapshot["source_kind"] == "parquet"
    assert snapshot["server_version"] == context.source.server_version
    assert context.read_proposal("An explanation.\n```text\nSELECT 1\n```") is None
    proposal = context.read_proposal("```sql\nSELECT * FROM data\n```")
    assert proposal.body == "SELECT * FROM data"
    assert context.validate_replacement(proposal) is None
    with pytest.raises(ValueError, match="incomplete"):
        context.read_proposal("```sql\nSELECT 1")


def test_multiple_sql_pages_and_function_metadata(session):
    context = context_for(session)
    proposal = context.read_proposal(
        '```sql\nSELECT id FROM data\n```\n'
        '```sql\nSELECT label FROM data\n```\n'
        '```json\n{"kind":"queries","titles":["IDs","Labels"]}\n```'
    )
    assert [query.title for query in proposal.queries] == ["IDs", "Labels"]
    assert context.validate_replacement(proposal) is None
    with pytest.raises(ValueError, match="matching titles"):
        context.read_proposal("```sql\nSELECT 1\n```\n```sql\nSELECT 2\n```")
    function = context.read_proposal(
        '```sql\nSELECT * FROM data WHERE id = @n\n```\n'
        '```json\n{"kind":"function","original_name":null,"name":"by_id",'
        '"description":"One ID","parameters":["n"],"arguments":"1"}\n```'
    )
    assert function.function.params == ("n",)
    assert context.validate_replacement(function) is None


@run_async
async def test_offline_validation_is_honest_and_never_executes(session, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Offline validation must not execute SQL")

    monkeypatch.setattr(context_module, "execute_sql", forbidden)
    context = context_for(session)
    result = await tool(context, "validate_sql", text="SELECT missing FROM nonexistent")
    assert result["valid"] is True
    assert result["scope"] == "lexical/template-only"
    result = await tool(context, "validate_sql", text="SELECT 'unterminated")
    assert result["valid"] is False
    result = await tool(context, "validate_sql", text="SELECT @n AS n", parameters=["n"])
    assert result["valid"] is True
    assert (await tool(context, "validate_sql", text="SELECT @missing", parameters=[]))["valid"] is False


@run_async
async def test_sample_and_execution_permissions_default_off(session):
    assert session.ai_allow_sample_data is False
    assert session.ai_test_queries is False
    context = context_for(session)
    tools = {item["function"]["name"] for item in context.tools}
    assert "sample_rows" not in tools
    assert "test_query" not in tools
    assert "disabled" in (await tool(context, "sample_rows", table="data"))["error"]
    result = await tool(context, "test_query", text="SELECT * FROM data")
    assert result["executed"] is False
    assert "disabled" in result["error"]


@run_async
async def test_metadata_is_cached_native_columns(session, monkeypatch):
    session.update_settings(ai_allow_schema_refreshes=True)
    context = context_for(session)
    monkeypatch.setattr(context_module, "execute_sql", lambda *a, **kw: pytest.fail("metadata executed SQL"))
    result = await tool(context, "schema", table="data")
    assert result["dialect"] == "duckdb"
    assert result["server_version"] == context.source.server_version
    assert [col["name"] for col in result["tables"][0]["columns"]] == ["id", "label"]
    assert "capabilities" not in result
    assert (await tool(context, "schema", query="not_a_column"))["tables"] == []


@run_async
async def test_readonly_checks_withhold_values_and_never_change_history(session):
    session.update_settings(ai_test_queries=True, ai=replace(session.ai_config, sample_rows=2))
    context = context_for(session)
    before = list(session.history_for(context.source_key))
    result = await tool(context, "test_query", text="SELECT * FROM data ORDER BY id")
    assert result["executed"] is True
    assert result["row_count"] == 2
    assert result["has_rows"] is True
    assert "rows" not in result
    assert "Row values withheld" in result["note"]
    assert session.history_for(context.source_key) == before
    assert "rows" not in context.query_checks["SELECT * FROM data ORDER BY id"]
    assert (await tool(context, "test_query", text="SELECT 1; SELECT 2"))["executed"] is False
    assert (await tool(context, "test_query", text="CREATE TABLE forbidden(id INTEGER)"))["executed"] is False
    assert "forbidden" not in context.source.datasets
    empty = await tool(context, "test_query", text="SELECT * FROM data WHERE id < 0")
    assert empty["executed"] is True
    assert empty["has_rows"] is False


@run_async
async def test_samples_are_projected_bounded_and_revocable(session):
    session.update_settings(ai_allow_sample_data=True, ai=replace(session.ai_config, sample_rows=2))
    context = context_for(session)
    result = await tool(context, "sample_rows", table="data", columns=["label"], limit=1)
    assert result["columns"] == ["label"]
    assert len(result["rows"]) == 1
    assert all(len(row) == 1 for row in result["rows"])
    assert "error" in await tool(context, "sample_rows", table="data", columns=["missing"])
    assert "error" in await tool(context, "sample_rows", table="data", limit=3)
    assert "error" in await tool(context, "sample_rows", table="data", columns=["id", "id"])
    session.update_settings(ai_allow_sample_data=False)
    assert "disabled" in (await tool(context, "sample_rows", table="data"))["error"]


@run_async
async def test_authorized_function_save_is_guarded_and_offline(session, monkeypatch):
    monkeypatch.setattr(context_module, "execute_sql", lambda *a, **kw: pytest.fail("save executed SQL"))
    context = context_for(session, save_guard=lambda: None)
    args = dict(name="by_id", original_name=None, description="Filter by ID",
                parameters=["n"], body="SELECT * FROM data WHERE id = @n")
    result = await tool(context, "save_function", **args)
    assert result == {"saved": True, "name": "by_id"}
    assert session.functions_for()["by_id"] == context.saved_functions["by_id"]
    assert context.document == "SELECT * FROM data"
    edited = {**args, "original_name": "by_id", "body": "SELECT id FROM data WHERE id = @n"}
    assert (await tool(context, "save_function", **edited))["saved"] is True
    session.save_fn("by_id", "n", "SELECT label FROM data WHERE id = @n", "Manual change")
    assert (await tool(context, "save_function", **edited))["saved"] is False
    blocked = context_for(session, save_guard=lambda: "The editor target changed.")
    assert (await tool(blocked, "save_function", **{**args, "name": "other"}))["saved"] is False
    assert "other" not in session.functions_for()
    unavailable = context_for(session)
    assert (await tool(unavailable, "save_function", **args))["saved"] is False


def test_frozen_definition_and_source_guards(session):
    saved = session.save_fn("by_id", "n", "SELECT * FROM data WHERE id = @n", "Saved")
    context = context_for(session)
    with pytest.raises(TypeError):
        context._functions["other"] = saved
    candidate = AIProposal(saved.body, saved, original_name="by_id")
    session.save_fn("by_id", "n", "SELECT id FROM data WHERE id = @n", "Changed")
    assert "changed" in context.replacement_guard(candidate)
    source = context.source
    session.sources[source.id] = replace(source)
    assert "source changed" in context.replacement_guard(AIProposal("SELECT 1")).lower()


@run_async
async def test_questions_validate_before_callback_and_resume(session):
    received = []

    async def answer(question):
        received.append(question)
        return "Use all rows"

    context = context_for(session, ask_user=answer)
    result = await tool(context, "ask_user", question="Which rows?",
                        options=[{"label": "All"}, {"label": "Recent"}], recommended=0)
    assert result == {"answer": "Use all rows"}
    assert received[0].recommended == 0
    assert "error" in await tool(context, "ask_user", question="Which rows?",
                                 options=[{"label": "All"}, {"label": "Other"}])
    assert len(received) == 1


@run_async
async def test_cancelled_turn_holds_shared_lane_until_read_finishes(session, monkeypatch):
    session.update_settings(ai_test_queries=True)
    context = context_for(session)
    manager = AIChats(SimpleNamespace(), session)
    chat = AIChat("chat", "target")
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()
    observed = {}

    def execute(source, sql, **options):
        observed.update(options)
        loop.call_soon_threadsafe(entered.set)
        release.wait(5)
        return {"rows": [[1]], "columns": ["id"]}

    monkeypatch.setattr(context_module, "execute_sql", execute)
    waiter = asyncio.create_task(manager._call_tool(chat, context, "test_query", {"text": "SELECT id FROM data"}, 0))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert session.busy == "AI context lookup running"
        assert len(manager.context_tasks) == 1
        blocked = json.loads(await manager._call_tool(chat, context, "schema", {}, 0))
        assert "try again" in blocked["error"]
        assert observed["readonly"] is True
        assert observed["max_rows"] == context.sample_rows
        assert observed["timeout"] <= 30
    finally:
        release.set()
        await asyncio.gather(*manager.context_tasks)
    assert session.busy == ""
    assert not manager.context_tasks


@run_async
async def test_sample_permission_revoked_during_read_discards_data(session, monkeypatch):
    session.update_settings(ai_allow_sample_data=True)
    context = context_for(session)
    entered = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def sample(*args):
        loop.call_soon_threadsafe(entered.set)
        release.wait(5)
        return {"rows": [["private"]], "columns": ["label"]}

    monkeypatch.setattr(context_module, "sample_rows", sample)
    task = asyncio.create_task(context.call_tool("sample_rows", {"table": "data"}))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        session.update_settings(ai_allow_sample_data=False)
    finally:
        release.set()
    result = await task
    assert "private" not in result
    assert "disabled" in json.loads(result)["error"]


@run_async
async def test_page_details_preserves_sql_and_disables_tools(monkeypatch):
    from d8r.ai import page_details

    async def run_turn(config, messages, tools, callback):
        assert tools == []
        assert json.loads(messages[1]["content"])["document"] == "SELECT id FROM data"
        messages.append({"role": "assistant", "content": json.dumps({"title": "IDs", "description": "Select IDs."})})
        yield AIEvent("text", "")

    monkeypatch.setattr(page_details.client, "run_turn", run_turn)
    assert await generate_details(AIConfig(), "SELECT id FROM data") == ("IDs", "Select IDs.")


def test_apply_guards_target_lane_and_automatic_function_saves(session, monkeypatch):
    context = context_for(session)
    target = AITarget((session.source_key(), "page"), session.active_id, context.document)
    manager = AIChats(SimpleNamespace(screen_stack=[]), session)
    chat = AIChat("chat", target.key, target=target, context=context,
                  proposal=AIProposal("SELECT id FROM data"))
    manager.chats[chat.id] = chat
    applied = []
    panel = SimpleNamespace(snapshot=lambda: target, apply_document=applied.append)
    monkeypatch.setattr(context_module, "execute_sql", lambda *a, **kw: pytest.fail("Apply executed SQL"))
    stale = replace(target, document="SELECT label FROM data")
    panel.snapshot = lambda: stale
    assert manager.apply(chat, panel) is False
    panel.snapshot = lambda: target
    session.busy = "Another query running"
    assert manager.apply(chat, panel) is False
    session.busy = ""
    assert not applied
    assert manager.apply(chat, panel, automatic=True) is True
    assert applied[0].body == "SELECT id FROM data"
    assert chat.proposal is None
    function = FnDef("saved", body="SELECT 1 AS value")
    chat.proposal = AIProposal(function.body, function)
    assert manager.apply(chat, panel, automatic=True) is False
    assert "saved" not in session.functions_for()


def test_postgres_prompt_includes_actual_server_version(session):
    source = session.sources[session.active_id]
    session.sources[source.id] = replace(source, kind="postgres-live", dialect="postgres", server_version="18.7")
    prompt = context_for(session).system_prompt()
    snapshot = json.loads(prompt.split("Editor snapshot (JSON data):\n", 1)[1])
    assert snapshot["server_version"] == "18.7"
    assert snapshot["dialect"] == "postgres"


@run_async
async def test_driver_error_text_is_never_sent_to_provider(session, monkeypatch):
    session.update_settings(ai_test_queries=True)

    def fail(*args, **kwargs):
        raise RuntimeError("password=private-password api-key=private-key")

    monkeypatch.setattr(context_module, "execute_sql", fail)
    result = await context_for(session).call_tool("test_query", {"text": "SELECT id FROM data"})
    assert json.loads(result)["executed"] is False
    assert "private-password" not in result
    assert "private-key" not in result
