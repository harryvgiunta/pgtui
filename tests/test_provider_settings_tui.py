"""Provider form Save persists public configuration independently of key storage."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps
import json

import httpx
import pytest
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from d8r import credentials
from d8r.ai import client
from d8r.ai.client import AIConfig
from d8r.credentials import CredentialError
from d8r.tui.app import D8RApp
from d8r.tui.session import Session
from d8r.tui.settings import AIProviderScreen


def async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))
    return run


async def save_form(app, pilot):
    app.screen.query_one("#ai-settings-save", Button).focus()
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    assert not isinstance(app.screen, AIProviderScreen)


@pytest.mark.parametrize("remember", [True, False])
@async_test
async def test_provider_form_saves_every_option_and_remember_choice(remember, credential_backend):
    config = AIConfig("https://provider.invalid/v1", "chosen-model", "synthetic-provider-key",
                      max_attempts=4, timeout=75, max_tool_rounds=12, max_tool_calls=7, sample_rows=9,
                      headers="X-Device: synthetic-device-token")
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        for field, value in {"ai-base-url": config.base_url, "ai-model": config.model, "ai-api-key": config.api_key,
                             "ai-attempts": "4", "ai-timeout": "75", "ai-tool-rounds": "12",
                             "ai-tool-calls": "7", "ai-sample-rows": "9"}.items():
            app.screen.query_one("#" + field, Input).value = value
        app.screen.query_one("#ai-headers", TextArea).load_text(config.headers)
        app.screen.query_one("#ai-remember-credentials", Checkbox).value = remember
        await save_form(app, pilot)
        assert session.ai_config == config
        app.push_screen(AIProviderScreen(app, focus_field="ai-tool-rounds"))
        await pilot.pause()
        assert app.screen.query_one("#ai-remember-credentials", Checkbox).value is remember
        app.screen.query_one("#ai-tool-rounds", Input).value = "13"
        await save_form(app, pilot)
        session.update_settings(ai_completion=True, ai_test_queries=True, default_rows=23)
        session.save_fn("one", "", "SELECT 1", "")
    restored = Session()
    assert restored.ai_config == replace(config, max_tool_rounds=13, api_key=config.api_key if remember else "",
                                         headers=config.headers if remember else "")
    assert restored.ai_remember_key is remember
    assert restored.ai_completion and restored.ai_test_queries
    assert config.api_key not in session.settings_path.read_text(encoding="utf-8")
    assert "synthetic-device-token" not in session.settings_path.read_text(encoding="utf-8")
    assert bool(credential_backend.values) is remember


@async_test
async def test_limits_save_before_provider_is_configured():
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app, focus_field="ai-tool-rounds"))
        await pilot.pause()
        app.screen.query_one("#ai-tool-rounds", Input).value = "23"
        await save_form(app, pilot)
    restored = Session()
    assert restored.ai_config.max_tool_rounds == 23
    assert restored.ai_config.base_url == restored.ai_config.model == restored.ai_config.api_key == ""


@async_test
async def test_locked_keyring_keeps_reference_but_saves_provider_options(monkeypatch):
    configured = Session()
    config = AIConfig("https://provider.invalid/v1", "chosen-model", "synthetic-vault-key")
    configured.update_settings(ai=config)
    reference = json.loads(configured.settings_path.read_text())["ai_providers"][0]["config"]["api_key_ref"]

    def locked():
        raise CredentialError("Secure system keyring unavailable or locked.")

    with monkeypatch.context() as context:
        context.setattr(credentials, "_secure_backend", locked)
        session = Session()
        assert session.ai_credentials_missing
        app = D8RApp(session)
        async with app.run_test(size=(120, 45)) as pilot:
            app.push_screen(AIProviderScreen(app, focus_field="ai-tool-rounds"))
            await pilot.pause()
            app.screen.query_one("#ai-tool-rounds", Input).value = "19"
            await save_form(app, pilot)
            assert session.ai_credentials_missing
            with pytest.raises(CredentialError):
                session.ai_config.validate()
        saved = json.loads(session.settings_path.read_text())
        assert saved["ai_providers"][0]["config"]["api_key_ref"] == reference
        assert saved["ai_providers"][0]["config"]["max_tool_rounds"] == 19
    restored = Session()
    assert restored.ai_config == replace(config, max_tool_rounds=19)
    assert not restored.ai_credentials_missing


@async_test
async def test_session_key_saves_provider_without_accessing_unavailable_vault(monkeypatch):
    def locked():
        raise CredentialError("Secure system keyring unavailable or locked.")

    monkeypatch.setattr(credentials, "_secure_backend", locked)
    app = D8RApp(Session())
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        app.screen.query_one("#ai-base-url", Input).value = "https://provider.invalid/v1"
        app.screen.query_one("#ai-model", Input).value = "session-model"
        app.screen.query_one("#ai-api-key", Input).value = "synthetic-session-key"
        app.screen.query_one("#ai-remember-credentials", Checkbox).value = False
        await save_form(app, pilot)
    restored = Session()
    assert restored.ai_config.base_url == "https://provider.invalid/v1"
    assert restored.ai_config.model == "session-model"
    assert restored.ai_config.api_key == ""
    assert not restored.ai_remember_key


def test_new_provider_never_inherits_old_key_when_remember_is_off(credential_backend):
    session = Session()
    session.update_settings(ai=AIConfig("https://old.invalid/v1", "old-model", "synthetic-old-key"))
    new = AIConfig("https://new.invalid/v1", "new-model", "synthetic-new-key")
    session.update_settings(ai=new, remember_ai_key=False)
    session.update_settings(default_rows=29)
    assert Session().ai_config == replace(new, api_key="")
    assert not credential_backend.values
    assert "synthetic-" not in session.settings_path.read_text(encoding="utf-8")


@async_test
async def test_model_discovery_selection_and_profiles_keep_secrets_isolated(parquet_source, monkeypatch):
    requests = []

    def respond(request):
        requests.append(request)
        assert request.method == "GET"
        assert str(request.url) == "https://gateway.invalid/api/v1/models"
        assert not request.content
        assert request.headers["Authorization"] == "Bearer synthetic-api-key"
        assert request.headers["x-ntnx-harness-device-id"] == "synthetic-device-key"
        return httpx.Response(200, json={"data": [{"id": "internal-model"}, {"id": "second-model"}]})

    monkeypatch.setattr(client, "create_client", lambda config: httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    session = Session({parquet_source.id: parquet_source})
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        screen = app.screen
        for field, value in {"ai-provider-name": "Internal", "ai-base-url": "https://gateway.invalid/api/v1/chat/completions/",
                             "ai-api-key": "synthetic-api-key"}.items():
            screen.query_one("#" + field, Input).value = value
        screen.query_one("#ai-headers", TextArea).load_text("x-ntnx-harness-device-id: synthetic-device-key")
        screen.query_one("#ai-remember-credentials", Checkbox).value = False
        await pilot.pause()
        assert requests == []
        screen.query_one("#ai-models-populate", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        await pilot.pause()
        assert len(requests) == 1
        assert screen.query_one("#ai-model", Input).value == ""
        screen.query_one("#ai-model-list", Select).focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert screen.query_one("#ai-model", Input).value == "internal-model"
        await save_form(app, pilot)
        first_id = session.active_ai_provider
        first_config = session.ai_config
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        app.screen.query_one("#ai-provider-new", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.screen.query_one("#ai-api-key", Input).value == ""
        assert app.screen.query_one("#ai-headers", TextArea).text == ""
        for field, value in {"ai-provider-name": "Local", "ai-base-url": "http://localhost:8000/v1",
                             "ai-model": "manual-model"}.items():
            app.screen.query_one("#" + field, Input).value = value
        await save_form(app, pilot)
        second_id = session.active_ai_provider
        assert second_id != first_id
        app.push_screen(AIProviderScreen(app, focus_field="ai-provider"))
        await pilot.pause()
        app.screen.query_one("#ai-provider", Select).value = first_id
        await pilot.pause()
        assert session.active_ai_provider == second_id
        assert app.screen.query_one("#ai-api-key", Input).value == first_config.api_key
        await save_form(app, pilot)
        assert session.active_ai_provider == first_id
        assert session.ai_config == first_config
        assert len(requests) == 1
    restored = Session()
    assert [profile["name"] for profile in restored.ai_providers] == ["Internal", "Local"]
    assert restored.active_ai_provider == first_id
    assert restored.ai_config.model == "internal-model"
    assert restored.ai_config.api_key == restored.ai_config.headers == ""
    assert "synthetic-" not in session.settings_path.read_text()


@pytest.mark.parametrize("change", ["headers", "url", "provider", "cancel", "close"])
@async_test
async def test_discovery_discards_late_results_after_retarget_or_cancel(monkeypatch, change):
    started, release, completed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def discover(config):
        started.set()
        deadline = asyncio.get_running_loop().time() + 10
        while not release.is_set():
            try:
                await asyncio.wait_for(release.wait(), max(0, deadline - asyncio.get_running_loop().time()))
            except asyncio.CancelledError:
                pass
        completed.set()
        return ("obsolete-model",)

    monkeypatch.setattr(client, "discover_models", discover)
    session = Session()
    first = session.save_ai_provider("First", AIConfig("https://first.invalid/v1", "keep-model"), remember=False)
    second = session.save_ai_provider("Second", AIConfig("https://second.invalid/v1", "keep-model"), remember=False)
    session.select_ai_provider(first)
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        screen = app.screen
        screen.query_one("#ai-models-populate", Button).focus()
        await pilot.press("enter")
        await asyncio.wait_for(started.wait(), 2)
        if change == "headers":
            screen.query_one("#ai-headers", TextArea).load_text("X-Device: changed")
        elif change == "url":
            screen.query_one("#ai-base-url", Input).value = "https://new.invalid/v1"
        elif change == "provider":
            screen.query_one("#ai-provider", Select).value = second
        elif change == "cancel":
            screen.query_one("#ai-models-cancel", Button).focus()
            await pilot.press("enter")
        else:
            await pilot.press("escape")
        await pilot.pause()
        release.set()
        await asyncio.wait_for(completed.wait(), 2)
        await pilot.pause()
        if change == "close":
            app.push_screen(AIProviderScreen(app))
            await pilot.pause()
            screen = app.screen
        screen.query_one("#ai-model-list", Select).focus()
        await pilot.press("enter", "down", "enter")
        await pilot.pause()
        assert screen.query_one("#ai-model", Input).value == "keep-model"
        assert session.active_ai_provider == first
        assert session.ai_config.headers == ""


@async_test
async def test_discovery_requires_missing_header_reentry(monkeypatch):
    session = Session()
    session.save_ai_provider("Internal", AIConfig("https://gateway.invalid/v1", "chosen", "synthetic-api-key",
                                                headers="X-Device: synthetic-device-key"))

    def locked():
        raise CredentialError("Secure system keyring unavailable or locked.")

    monkeypatch.setattr(credentials, "_secure_backend", locked)
    calls = []

    async def discover(config):
        calls.append(config)
        return ("available",)

    monkeypatch.setattr(client, "discover_models", discover)
    app = D8RApp(Session())
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        app.screen.query_one("#ai-api-key", Input).value = "synthetic-reentered-key"
        await pilot.pause()
        app.screen.query_one("#ai-models-populate", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert calls == []
        assert "unavailable" in str(app.screen.query_one("#ai-settings-error", Static).content)
        app.screen.query_one("#ai-headers", TextArea).load_text("X-Device: synthetic-reentered-device")
        await pilot.pause()
        app.screen.query_one("#ai-models-populate", Button).focus()
        await pilot.press("enter")
        await app.workers.wait_for_complete()
        assert len(calls) == 1
        assert app.session.ai_credentials_missing


@async_test
async def test_provider_delete_requires_confirmation_and_cancel_never_commits_drafts():
    session = Session()
    first = session.save_ai_provider("First", AIConfig("https://first.invalid/v1", "first"), remember=False)
    second = session.save_ai_provider("Second", AIConfig("https://second.invalid/v1", "second"), remember=False)
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        app.screen.query_one("#ai-provider-name", Input).value = "Unsaved rename"
        await pilot.press("escape")
        await pilot.pause()
        assert [profile["name"] for profile in session.ai_providers] == ["First", "Second"]
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        app.screen.query_one("#ai-provider-delete", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert session.active_ai_provider == second
        await pilot.pause(app.screen.query_one("#ai-provider-delete", Button).active_effect_duration)
        await pilot.press("enter")
        await pilot.pause()
        assert session.active_ai_provider == first
        assert [profile["id"] for profile in Session().ai_providers] == [first]
