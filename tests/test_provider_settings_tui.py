"""Provider form Save persists public configuration independently of key storage."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from functools import wraps
import json

import pytest
from textual.widgets import Button, Checkbox, Input

from d8r import credentials
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
                      max_attempts=4, timeout=75, max_tool_rounds=12, max_tool_calls=7, sample_rows=9)
    session = Session()
    app = D8RApp(session)
    async with app.run_test(size=(120, 45)) as pilot:
        app.push_screen(AIProviderScreen(app))
        await pilot.pause()
        for field, value in {"ai-base-url": config.base_url, "ai-model": config.model, "ai-api-key": config.api_key,
                             "ai-attempts": "4", "ai-timeout": "75", "ai-tool-rounds": "12",
                             "ai-tool-calls": "7", "ai-sample-rows": "9"}.items():
            app.screen.query_one("#" + field, Input).value = value
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
    assert restored.ai_config == replace(config, max_tool_rounds=13, api_key=config.api_key if remember else "")
    assert restored.ai_remember_key is remember
    assert restored.ai_completion and restored.ai_test_queries
    assert config.api_key not in session.settings_path.read_text(encoding="utf-8")
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
    reference = json.loads(configured.settings_path.read_text())["ai"]["api_key_ref"]

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
        assert saved["ai"]["api_key_ref"] == reference
        assert saved["ai"]["max_tool_rounds"] == 19
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
