"""D1 AI behavior through the native HTTP adapter and real SQLite execution."""
import asyncio
import json
from copy import deepcopy

from d8r.ai.context import AIContext
from d8r.tui.session import Session
from tests.d1_support import API_TOKEN


def test_d1_context_uses_actual_sqlite_version_without_credentials(d1_source):
    session = Session({d1_source.id: d1_source})
    context = AIContext(session, d1_source.id, "SELECT id FROM records")
    snapshot = json.loads(context.system_prompt().split("Editor snapshot (JSON data):\n", 1)[1])
    assert snapshot["dialect"] == "sqlite"
    assert snapshot["source_kind"] == "d1-live"
    assert snapshot["server_version"] == d1_source.server_version
    assert API_TOKEN not in context.system_prompt()
    proposal = context.read_proposal("```sql\nSELECT [id], `label` FROM records\n```")
    assert context.validate_replacement(proposal) is None
    assert session.history == []


def test_d1_ai_checks_expand_templates_and_keep_samples_opt_in(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    session.save_fn("by_id", "minimum", "SELECT [id], `label` FROM records WHERE id >= @minimum ORDER BY id", "")
    disabled = AIContext(session, d1_source.id, "SELECT * FROM records")
    before = len(d1_harness.requests)
    async def scenario():
        result = json.loads(await disabled.call_tool("test_query", {"text": "SELECT * FROM by_id(2)"}))
        assert not result["executed"]
        assert len(d1_harness.requests) == before
        session.update_settings(ai_test_queries=True)
        context = AIContext(session, d1_source.id, "SELECT * FROM records")
        workspace = deepcopy(session.workspace)
        history = list(session.history)
        evidence = json.loads(await context.call_tool("test_query", {"text": "SELECT * FROM by_id(2)"}))
        assert evidence["executed"] and evidence["row_count"] == 2
        assert "rows" not in evidence
        session.update_settings(ai_allow_sample_data=True)
        evidence = json.loads(await context.call_tool("test_query", {"text": "SELECT * FROM by_id(2)"}))
        assert evidence["rows"] == [[2, "beta"], [3, None]]
        assert session.history == history
        # Preference changes do not rewrite editor documents or result history.
        assert session.workspace["document"] == workspace["document"]
        assert session.workspace["history"] == workspace["history"]
        blocked = json.loads(await context.call_tool("test_query", {"text": "DELETE FROM records"}))
        assert not blocked["executed"]
        assert d1_harness.connection.execute("SELECT count(*) FROM records").fetchone() == (3,)
    asyncio.run(scenario())


def test_d1_ai_sample_projection_and_source_replacement_are_guarded(d1_source):
    session = Session({d1_source.id: d1_source})
    session.update_settings(ai_allow_sample_data=True)
    context = AIContext(session, d1_source.id, "")
    table = session.schema.table_by_name("records", dialect="sqlite").name
    async def scenario():
        sampled = json.loads(await context.call_tool("sample_rows", {"table": table, "columns": ["label"], "limit": 1}))
        assert sampled["columns"] == ["label"] and sampled["rows"] == [["alpha"]]
        session.update_settings(ai_allow_sample_data=False)
        assert "error" in json.loads(await context.call_tool("sample_rows", {"table": table}))
        session.sources.pop(d1_source.id)
        assert context.validate_proposal("SELECT * FROM records") is not None
    asyncio.run(scenario())


def test_d1_authorized_function_save_stays_offline_and_native(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    context = AIContext(session, d1_source.id, "SELECT * FROM records", save_guard=lambda: None)
    before = len(d1_harness.requests)
    args = {"name": "by_label", "original_name": None, "description": "", "parameters": ["needle"],
            "body": "SELECT `id`, [label] FROM records WHERE label = @needle"}
    async def scenario():
        saved = json.loads(await context.call_tool("save_function", args))
        assert saved["saved"]
        assert len(d1_harness.requests) == before
        assert not session.history
        session.save_fn("by_label", "needle", "SELECT id FROM records WHERE label = @needle", "manual edit")
        denied = json.loads(await context.call_tool("save_function", {**args, "original_name": "by_label"}))
        assert not denied["saved"]
    asyncio.run(scenario())
    outcome = session.fn_preview("by_label", "'beta'")
    assert outcome.ok and outcome.rows == [[2]]
    assert not session.history
