"""Approval isolation and concurrent-create protection for new dbt projects."""
from __future__ import annotations

import asyncio
import json
import os

import httpx
import pytest
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static, TextArea

from d8r.ai import client
from d8r.dbt import DbtError, DbtProject
from d8r.dbt_setup import SetupInput, plan_scaffold, write_scaffold
from d8r.tui.add_source import AddSourceModal
from d8r.tui.app import D8RApp
from d8r.tui.dbt_setup import DbtSetupScreen
from d8r.tui.session import Session
from tests.d1_support import ACCOUNT_ID, API_TOKEN, DATABASE_NAME


def test_scaffold_late_collision_preserves_external_file_and_rolls_back_own_files(tmp_path, monkeypatch):
    root = tmp_path / "project"
    root.mkdir()
    plan = plan_scaffold(str(root), "orders", "postgres", SetupInput("public", "sales"),
                         model="stg_records", materialization="view", check="summary")
    external = root / "models" / "sources.yml"
    real_link = os.link

    def link(source, destination):
        if destination == external:
            external.write_bytes(b"concurrent creation")
        real_link(source, destination)
    monkeypatch.setattr(os, "link", link)
    with pytest.raises(DbtError):
        write_scaffold(plan)
    assert external.read_bytes() == b"concurrent creation"
    assert [path for path in root.rglob("*") if not path.is_dir()] == [external]


def test_setup_with_d1_keeps_secure_credentials_out_of_advice_and_requires_approval(
    tmp_path, d1_harness, credential_backend, monkeypatch,
):
    root = tmp_path / "private-folder"
    root.mkdir()
    state = tmp_path / "app-state"
    ai_key = "dbt-interview-provider-key"
    requests = []

    def provider(request):
        requests.append(json.loads(request.content))
        assert request.headers["Authorization"] == f"Bearer {ai_key}"
        choices = [{"index": 0, "delta": {"content": '{"materialization":"table","check":"nonempty"}'},
                    "finish_reason": "stop"}]
        return httpx.Response(200, headers={"Content-Type": "text/event-stream"},
                              text=f'data: {json.dumps({"choices": choices})}\n\ndata: [DONE]\n\n')

    monkeypatch.setattr(client, "create_client", lambda config: httpx.AsyncClient(transport=httpx.MockTransport(provider)))
    session = Session(data_dir=state)
    session.save_ai_provider("Interview", client.AIConfig(base_url="https://example.invalid/v1", model="test", api_key=ai_key))
    session.update_settings(ai_auto_accept=True)

    async def interview() -> None:
        app = D8RApp(session)
        async with app.run_test(size=(120, 42)) as pilot:
            app.action_settings()
            await pilot.pause()
            navigation = app.screen.query_one("#settings-sidebar", OptionList)
            navigation.highlighted = navigation.get_option_index("workspace")
            await pilot.pause()
            await pilot.press("enter", "down", "enter")
            await pilot.pause()
            setup = app.screen
            assert isinstance(setup, DbtSetupScreen)
            setup.query_one("#dbt-setup-target", Select).value = "d1"
            setup.query_one("#dbt-setup-directory", Input).value = str(root)
            setup.query_one("#dbt-setup-name", Input).value = "private_project"
            await pilot.click("#dbt-setup-next")
            await pilot.click("#dbt-setup-connect")
            form = app.screen
            assert isinstance(form, AddSourceModal)
            assert form.query_one("#source-backend", Select).value == "d1"
            form.query_one("#d1-account", Input).value = ACCOUNT_ID
            form.query_one("#d1-database", Input).value = DATABASE_NAME
            token = form.query_one("#d1-token", Input)
            assert token.password
            token.value = API_TOKEN
            await pilot.pause()
            form.action_add()
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert app.screen is setup and session.active_id == "postgres" and not session.dbt_enabled
            connection_requests = len(d1_harness.requests)
            setup.query_one("#dbt-setup-expects-rows", Checkbox).value = True
            await pilot.click("#dbt-setup-next")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert not setup.query_one("#dbt-setup-use-advice", Button).disabled
            await pilot.click("#dbt-setup-use-advice")
            setup.query_one("#dbt-setup-key", Select).value = "id"
            await pilot.click("#dbt-setup-next")
            preview = setup.query_one("#dbt-setup-preview", TextArea)
            assert setup.stage == 4 and preview.read_only and not any(root.iterdir())
            assert not session.dbt_enabled
            assert all(secret not in preview.text for secret in (API_TOKEN, ai_key, ACCOUNT_ID))
            await pilot.click("#dbt-setup-create")
            await pilot.pause()
            assert app.screen is not setup, str(setup.query_one("#dbt-setup-message", Static).content)
            assert session.dbt_enabled and session.dbt_project_dir == str(root)
            assert session.source.kind == "d1-live" and session.dialect == "sqlite"
            assert len(d1_harness.requests) == connection_requests  # Setup doesn't run dbt or starter SQL.

    asyncio.run(interview())
    assert len(requests) == 1 and "tools" not in requests[0]
    assert json.loads(requests[0]["messages"][1]["content"]) == {"target": "d1", "workload": "explore", "expects_rows": True}
    assert all(secret not in json.dumps(requests) for secret in (API_TOKEN, ai_key, ACCOUNT_ID, str(root), "private_project"))
    assert {API_TOKEN, ai_key} <= set(credential_backend.values.values())
    assert not session.load_chats() and not session.history
    assert not list(root.glob("profiles*"))
    for path in [*root.rglob("*"), *state.glob("*.json")]:
        if path.is_file():
            assert all(secret not in path.read_bytes() for secret in (API_TOKEN.encode(), ai_key.encode()))
    assert (root / "tests" / "nonempty_input.sql").exists()
    assert (root / "tests" / "valid_input_key.sql").exists()
    project = DbtProject(root, dialect="sqlite")
    stage = project.render_query((root / "models" / "stg_records.sql").read_text("utf-8"), database="")
    assert len(d1_harness.connection.execute(stage).fetchall()) == 3
    restored = Session(data_dir=state)
    assert restored.dbt_enabled and restored.dbt_project_dir == str(root) and restored.dialect == "sqlite"
