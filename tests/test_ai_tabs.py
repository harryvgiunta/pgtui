"""Isolated real Pilot conversation navigation; provider turns are controlled seams."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps
from pathlib import Path

import pytest
from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Button, Static, TextArea, Tooltip
from textual.worker import WorkerCancelled

from d8r.ai import client
from d8r.ai.client import AIConfig, AIEvent
from d8r.ai.context import AIContext
from d8r.tui.ai import (
    AIChats, AIPanel, AITarget, ConversationHistory, ConversationTab,
    ConversationTabs, DeleteOptionList, stage_saying,
)
from d8r.tui.session import Session


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


class ConversationHost(App):
    """Mount the shipped widgets/styles without the full workspace or live providers."""

    CSS_PATH = Path(__file__).resolve().parents[1] / "d8r" / "tui" / "app.tcss"
    CSS = "#editor { height: 4; } AIPanel { height: 1fr; }"

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.ai_chats = AIChats(self, session)
        self.applied = []

    def compose(self) -> ComposeResult:
        yield ConversationTabs(id="conversation-tabs")
        yield TextArea("SELECT 1", id="editor")
        yield AIPanel(self.session, self.target, self.apply_proposal, self.focus_editor, id="panel")

    @property
    def panel(self):
        return self.query_one(AIPanel)

    def target(self):
        return AITarget((self.session.source_key(), "page"), self.session.active_id,
                        self.query_one("#editor", TextArea).text)

    def focus_editor(self):
        self.query_one("#editor", TextArea).focus()

    def apply_proposal(self, proposal):
        self.applied.append(proposal)
        self.query_one("#editor", TextArea).load_text(proposal.body)

    def open_ai_chat(self, chat_id):
        self.panel.select_chat(chat_id)

    @on(ConversationTabs.Selected)
    def selected(self, event):
        self.open_ai_chat(event.chat_id)

    @on(ConversationTabs.NewChatRequested)
    def new_chat(self):
        self.panel.action_clear()

    def on_unmount(self):
        self.ai_chats.shutdown()


@pytest.fixture
def session(parquet_source, tmp_path):
    session = Session({parquet_source.id: parquet_source}, data_dir=tmp_path / "chat-state")
    session.update_settings(ai=AIConfig("https://provider.invalid/v1", "test", "test-only-key"),
                            remember_ai_key=False, ai_allow_schema_refreshes=True)
    return session


async def send(app, pilot, prompt="Show one value"):
    await pilot.click("#conversation-new")
    composer = app.panel.query_one("#ai-input", TextArea)
    composer.load_text(prompt)
    composer.focus()
    await pilot.press("enter")
    return app.panel.chat


@pytest.mark.parametrize("mode", ["default", "execute", "off"])
@async_test
async def test_final_pages_checked_before_apply_without_provider_tools(session, monkeypatch, mode):
    session.update_settings(ai_auto_accept=True)
    if mode == "off":
        session.update_settings(ai_test_queries=False)
    elif mode == "execute":
        session.update_settings(ai_allow_query_execution=True)
    session.install_quality_functions()
    # The default plan check must not evaluate this failing row-level cast.
    first = ("SELECT CAST(label AS INTEGER) FROM data" if mode == "default" else
             "SELECT value FROM dq_row_count(data)")
    second = "SELECT unknown_column FROM data"
    drafts, feedback = [], []

    async def provider(config, messages, tools, callback):
        if drafts:
            feedback.append(messages[-1]["content"])
        detail = (second if not drafts else "SELECT id FROM data" if mode == "default" else
                  f"SELECT unknown_column_{len(drafts)} FROM data")
        drafts.append(detail)
        answer = (f"```sql\n{first}\n```\n```sql\n{detail}\n```\n"
                  '```json\n{"kind":"queries","titles":["Quality","Detail"]}\n```')
        messages.append({"role": "assistant", "content": answer})
        yield AIEvent("text", answer)

    monkeypatch.setattr(client, "run_turn", provider)
    app = ConversationHost(session)
    async with app.run_test(size=(100, 42)) as pilot:
        chat = await send(app, pilot, "QC the table; one summary and one detail page")
        await app.workers.wait_for_complete()
        evidence = chat.context.query_checks
        assert not session.history
        if mode == "off":
            assert evidence == {} and len(app.applied) == 1
            assert drafts == [second]
            return
        assert evidence[second]["error"] and not evidence[second]["executed"]
        assert all(evidence[sql]["error"] in message and "Detail" in message
                   for sql, message in zip(drafts, feedback))
        if mode == "default":
            assert evidence[first] == {"prepared": True, "executed": False}
            # First repair passes; accept the complete batch without another request.
            assert drafts == [second, "SELECT id FROM data"]
            assert chat.status == "idle" and evidence[drafts[-1]]["prepared"]
            assert [q.body for q in app.applied[0].queries] == [first, drafts[-1]]
        else:
            assert evidence[first]["executed"] and evidence[first]["row_count"] == 1
            # Two different failed repairs; never automatically accept any of them.
            assert len(drafts) == 3 and len(feedback) == 2 and not app.applied
            assert chat.status == "error" and "Detail" in chat.query_warning
            assert not app.ai_chats.apply(chat, app.panel, automatic=True)
            assert app.query_one("#editor", TextArea).text == "SELECT 1"
            apply = app.panel.query_one("#ai-apply", Button)
            assert not apply.disabled
            apply.focus()
            await pilot.press("enter")
            assert [q.body for q in app.applied[0].queries] == [first, drafts[-1]]
            assert not session.history


@async_test
async def test_unusable_repair_preserves_last_query_for_manual_edit(session, monkeypatch):
    session.update_settings(ai_auto_accept=True)
    draft = "SELECT 'unterminated"
    answers = iter([f"```sql\n{draft}\n```", "No replacement."])

    async def provider(config, messages, tools, callback):
        answer = next(answers, None)
        if answer is None:
            # Even a provider ignoring the empty tool list cannot save a function.
            await callback("save_function", {"name": "unrequested", "original_name": None,
                           "parameters": [], "description": "", "body": "SELECT 1"})
            pytest.fail("Repair was allowed to invoke a tool")
        messages.append({"role": "assistant", "content": answer})
        yield AIEvent("text", answer)

    monkeypatch.setattr(client, "run_turn", provider)
    app = ConversationHost(session)
    async with app.run_test(size=(100, 42)) as pilot:
        chat = await send(app, pilot)
        await app.workers.wait_for_complete()
        assert chat.status == "error" and chat.query_warning and not app.applied
        assert "unrequested" not in session.functions_for() and not chat.context.query_checks
        app.panel.query_one("#ai-apply", Button).focus()
        await pilot.press("enter")
        assert app.query_one("#editor", TextArea).text == draft
        assert not session.history and not chat.save_error


@async_test
async def test_hover_conversation_tab_paints_literal_details(session):
    app = ConversationHost(session)
    app.TOOLTIP_DELAY = 0.01
    async with app.run_test(size=(90, 36), tooltips=True) as pilot:
        app.panel.action_clear()
        chat = app.panel.chat
        chat.title = "[dim]Q4 orders[/]"
        app.ai_chats.changed(chat)
        await pilot.pause()
        tab = app.query_one(ConversationTab)
        for widget, offset in ((tab, (1, tab.region.height - 1)),
                               (tab.query_one(".conversation-select"), (1, 0))):
            await pilot.hover(widget, offset=offset)
            await pilot.pause(0.1)
            tip = app.screen.query_one(Tooltip)
            assert tip.display
            x, y = tip.content_region.offset
            assert app.screen.get_widget_at(x, y)[0] is tip
            assert f"{chat.title} · {chat.label}" in tip.render_line(0).text
            style = app.screen.get_style_at(x, y)
            assert style.color != style.bgcolor


@async_test
async def test_switch_close_reopen_preserves_running_request_and_drafts(session, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()

    async def provider(config, messages, tools, callback):
        started.set()
        await release.wait()
        answer = "```sql\nSELECT 2\n```"
        messages.append({"role": "assistant", "content": answer})
        yield AIEvent("text", answer)

    monkeypatch.setattr(client, "run_turn", provider)
    app = ConversationHost(session)
    async with app.run_test(size=(100, 42)) as pilot:
        first = await send(app, pilot)
        await asyncio.wait_for(started.wait(), 2)
        generation, worker = first.generation, first.worker
        composer = app.panel.query_one("#ai-input", TextArea)
        composer.load_text("Keep this follow-up draft")
        await pilot.pause(app.query_one("#conversation-new", Button).active_effect_duration)
        await pilot.click("#conversation-new")
        await pilot.pause()
        assert app.panel.chat.id != first.id
        assert first.input == "Keep this follow-up draft"
        await pilot.click(f"#conversation-{first.id} .conversation-select")
        assert app.panel.chat_id == first.id and composer.text == first.input
        await pilot.click(f"#conversation-{first.id} .conversation-close")
        await pilot.pause()
        assert first.closed and first.id in app.ai_chats.chats
        assert first.worker is worker and first.generation == generation
        assert not app.query(f"#conversation-{first.id}")
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert first.status == "idle" and first.unread and first.closed
        assert not app.applied and session.load_chats()[first.id]["closed"]
        restored = AIChats(app, session).chats[first.id]
        assert restored.closed and restored.transcript == first.transcript
        await pilot.click("#conversation-history")
        options = app.screen.query_one(DeleteOptionList)
        options.highlighted = options.get_option_index(first.id)
        options.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.panel.chat_id == first.id and not first.closed
        assert composer.text == "Keep this follow-up draft"
        assert "SELECT 2" in str(app.panel.query_one("#ai-transcript", Static).content)
        assert not app.panel.query_one("#ai-apply", Button).disabled
        editor = app.query_one("#editor", TextArea)
        editor.load_text("SELECT 99")
        app.panel.target_changed()
        await pilot.pause()
        assert app.panel.query_one("#ai-apply", Button).disabled
        assert not app.ai_chats.apply(first, app.panel)
        editor.load_text("SELECT 1")
        app.panel.target_changed()
        composer.focus()
        await pilot.press("escape")
        await pilot.pause()
        assert not app.panel.display and not first.closed
        assert app.query(f"#conversation-{first.id}")
        await pilot.click(f"#conversation-{first.id} .conversation-select")
        await pilot.click("#ai-apply")
        assert app.applied[0].body == "SELECT 2"


@async_test
async def test_narrow_keyboard_history_delete_and_source_isolation(session):
    app = ConversationHost(session)
    async with app.run_test(size=(42, 36)) as pilot:
        for index in range(4):
            app.query_one(ConversationTabs).focus()
            await pilot.press("ctrl+n")
            app.panel.chat.title = f"Conversation {index} with a long readable name"
            app.ai_chats.changed(app.panel.chat)
        current = app.panel.chat
        foreign = app.ai_chats.create(AITarget(("another-source", "page"), "elsewhere", "SELECT 1"))
        await pilot.pause()
        tabs = app.query_one(ConversationTabs)
        assert len(tabs.query(ConversationTab)) == 4
        assert not tabs.query(f"#conversation-{foreign.id}")
        tabs.focus()
        await pilot.press("end")
        oldest = tabs._focused_tab().chat_id
        await pilot.press("enter")
        assert app.panel.chat_id == oldest
        tabs.focus()
        await pilot.press("home", "delete")
        await pilot.pause()
        assert current.closed
        await pilot.click("#conversation-history")
        assert isinstance(app.screen, ConversationHistory)
        options = app.screen.query_one(DeleteOptionList)
        assert options.option_count == 4
        options.highlighted = options.get_option_index(current.id)
        options.focus()
        await pilot.press("delete")
        await pilot.pause()
        assert current.id not in app.ai_chats.chats
        assert current.id not in session.load_chats()
        assert foreign.id in app.ai_chats.chats
        await pilot.press("escape")
        assert not isinstance(app.screen, ConversationHistory)




@async_test
async def test_actual_tool_stages_and_retry_status_without_timers(session, monkeypatch):
    retrying, release = asyncio.Event(), asyncio.Event()
    observed = []
    original_tool = AIContext.call_tool
    app = ConversationHost(session)

    async def inspect_tool(context, name, arguments):
        chat = app.panel.chat
        observed.append((name, chat.stage, chat.status_text))
        return await original_tool(context, name, arguments)

    async def provider(config, messages, tools, callback):
        yield AIEvent("status", "Retrying AI request in 1s (2/3).")
        retrying.set()
        await release.wait()
        for name in ("schema", "query_history", "functions", "validate_sql"):
            await callback(name, {"text": "SELECT 2"} if name == "validate_sql" else {})
            assert app.panel.chat.status_text == stage_saying("generating", 1)
        answer = "```sql\nSELECT 2\n```"
        yield AIEvent("text", answer)
        messages.append({"role": "assistant", "content": answer})

    monkeypatch.setattr(AIContext, "call_tool", inspect_tool)
    monkeypatch.setattr(client, "run_turn", provider)
    async with app.run_test(size=(100, 42)) as pilot:
        chat = await send(app, pilot)
        await asyncio.wait_for(retrying.wait(), 2)
        assert chat.status_text == "Retrying AI request in 1s (2/3)."
        await pilot.pause()
        assert chat.status_text == "Retrying AI request in 1s (2/3)."
        release.set()
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert [(name, stage) for name, stage, _ in observed] == [
            (stage, stage) for stage in ("schema", "query_history", "functions", "validate_sql")]
        assert all(text == stage_saying(name, 1) for name, _, text in observed)
        assert chat.status == "idle" and chat.proposal.body == "SELECT 2"
        assert not app.applied and not session.history


@async_test
async def test_cancel_and_provider_error_remain_honest(session, monkeypatch):
    started = asyncio.Event()

    async def pending(config, messages, tools, callback):
        started.set()
        await asyncio.Event().wait()
        yield AIEvent("text", "unreachable")

    monkeypatch.setattr(client, "run_turn", pending)
    app = ConversationHost(session)
    async with app.run_test(size=(100, 42)) as pilot:
        chat = await send(app, pilot)
        await asyncio.wait_for(started.wait(), 2)
        assert chat.status_text == stage_saying("general", 1)
        generation = chat.generation
        await pilot.click("#ai-cancel")
        await pilot.pause()
        assert chat.status == "cancelled" and "Cancelled" in chat.status_text
        app.ai_chats.set_stage(chat, "schema", generation)
        assert "Cancelled" in chat.status_text
        assert chat.proposal is None and not app.applied

        async def failing(config, messages, tools, callback):
            raise client.AIError("The AI connection failed. Retry manually.")
            yield AIEvent("text", "unreachable")

        monkeypatch.setattr(client, "run_turn", failing)
        await send(app, pilot, "Try a new conversation")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert app.panel.chat.status == "error"
        assert "The AI connection failed. Retry manually." in app.panel.chat.status_text
        assert not app.applied


@async_test
async def test_provider_roundtrip_revokes_chat_tools_and_redacts_headers(session, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    session.update_settings(ai=replace(session.ai_config, headers="X-Device: synthetic-header-secret"))

    async def provider(config, messages, tools, callback):
        yield AIEvent("diagnostic", "Gateway echoed synthetic-header-secret")
        started.set()
        await release.wait()
        await callback("schema", {})
        answer = "```sql\nSELECT 99\n```"
        messages.append({"role": "assistant", "content": answer})
        yield AIEvent("text", answer)

    monkeypatch.setattr(client, "run_turn", provider)
    app = ConversationHost(session)
    async with app.run_test(size=(100, 42)) as pilot:
        chat = await send(app, pilot)
        await asyncio.wait_for(started.wait(), 2)
        assert "synthetic-header-secret" not in chat.diagnostics.text
        original = session.active_ai_provider
        session.save_ai_provider("Another provider", session.ai_config, remember=False)
        session.select_ai_provider(original)
        release.set()
        try:
            await app.workers.wait_for_complete()
        except WorkerCancelled:
            pass
        await pilot.pause()
        assert chat.status == "cancelled"
        assert chat.proposal is None and not app.applied
        assert "SELECT 99" not in chat.transcript
        assert app.query_one("#editor", TextArea).text == "SELECT 1"
