"""Native D1 session lifecycle, source isolation, and offline SQL behavior."""
import asyncio
import json

import pytest

from d8r import credentials
from d8r.completion.engine import complete
from d8r.credentials import CredentialError
from d8r.engine import EngineError, close_source
from d8r.storage import MemoryStore
from d8r.tui import session as session_module
from d8r.tui.connection import build_connection, build_saved_source, saved_source_label
from d8r.tui.session import Session
from tests.d1_support import ACCOUNT_ID, API_TOKEN, DATABASE_ID, DATABASE_NAME


KEY = f"d1:{ACCOUNT_ID}:{DATABASE_ID}"
OTHER_DATABASE = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def profile(**changes):
    return {"account_id": ACCOUNT_ID, "database": DATABASE_ID,
            "api_token": API_TOKEN, "display": "D1 saved", **changes}


def postgres_profile():
    return {"host": "localhost", "port": "5432", "database": "app", "user": "reader",
            "password": "pg-token", "schema": "public", "sslmode": "prefer", "display": "PG"}


def test_d1_builder_does_not_register_or_persist(d1_harness, credential_backend):
    session = Session()
    original = dict(session.sources)
    source = session.build_d1_source(account_id=ACCOUNT_ID.upper(), database=DATABASE_NAME,
                                     api_token=API_TOKEN, display="Cloud")
    try:
        assert source.con is not None
        assert source.kind == "d1-live" and source.dialect == "sqlite"
        assert source.connection_key == KEY
        assert source.d1 == {"account_id": ACCOUNT_ID, "database": DATABASE_ID}
        assert session.sources == original
        assert session.d1_profiles == []
        assert not session.storage_path.exists()
        assert not credential_backend.values
    finally:
        close_source(source)


def test_session_only_d1_never_leaks_through_other_saves(d1_source):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile(), persist=False)
    session.save_fn("labels", "", "SELECT [label] FROM [records]", "Labels")
    session.remember_postgres(postgres_profile())
    saved = MemoryStore(session.storage_path.parent)
    assert saved.document["d1_profiles"] == []
    assert saved.document["postgres_profiles"] == [postgres_profile()]
    assert session.saved_source_profile(d1_source.id)["api_token"] == API_TOKEN
    assert API_TOKEN not in session.storage_path.read_text(encoding="utf-8")


def test_session_only_rotated_token_preserves_saved_token(d1_source):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile())
    session.remember_d1(profile(api_token="temporary-token", display="Session display"), persist=False)
    session.remember_postgres(postgres_profile(), persist=False)
    session.save_fn("ids", "", "SELECT id FROM records", "")
    saved = MemoryStore(session.storage_path.parent)
    assert saved.document["d1_profiles"] == [profile()]
    assert saved.document["postgres_profiles"] == []
    assert session.saved_source_profile(KEY)["api_token"] == "temporary-token"
    session.remember_d1(profile(api_token="remembered-token"))
    assert MemoryStore(session.storage_path.parent).document["d1_profiles"] == [profile(api_token="remembered-token")]


def test_d1_identity_is_canonical_not_display_or_token(d1_source):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile(account_id=ACCOUNT_ID.upper()))
    session.remember_d1(profile(api_token="new-token", display="Renamed"))
    assert session.source_key() == KEY
    assert len(session.d1_profiles) == 1
    assert session.saved_source_profile(KEY) == {"kind": "d1-live", **profile(api_token="new-token", display="Renamed")}
    assert sum(value == d1_source.id for _, value in session.source_options()) == 1


def test_d1_display_and_reconnect_labels_redact_token(d1_source):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile(display=f"Cloud {API_TOKEN}"))
    assert API_TOKEN not in session.storage_path.read_text(encoding="utf-8")
    assert API_TOKEN not in session.d1_profiles[0]["display"]
    assert API_TOKEN not in saved_source_label(profile={"kind": "d1-live", **profile(display=API_TOKEN)})


def test_d1_saved_restore_is_offline_and_reconnect_replaces_placeholder(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile())
    session.remember_d1(profile(database=OTHER_DATABASE, display="Other"))
    session.update_settings(source=d1_source.id)
    session.save_workspace(document="SELECT [id] FROM [records]", source=KEY, dialect="sqlite")
    request_count = len(d1_harness.requests)
    restored = Session()
    assert len(d1_harness.requests) == request_count
    assert restored.source_key() == KEY and restored.dialect == "sqlite"
    assert not restored.source_connected()
    assert restored.workspace["document"] == "SELECT [id] FROM [records]"
    assert not restored.run("SELECT 1").ok
    assert len(d1_harness.requests) == request_count
    saved = restored.saved_source_profile(restored.active_id)
    connected = asyncio.run(build_connection(lambda: build_saved_source(restored, profile=saved)))
    try:
        restored.register(connected)
        assert restored.source is connected and restored.source_connected()
        assert KEY not in restored.sources
        assert restored.source_key() == KEY
        assert restored.run("SELECT [label] FROM [records] WHERE id = 1").rows == [["alpha"]]
        assert not any(source.connection_key.endswith(OTHER_DATABASE) for source in restored.sources.values())
    finally:
        close_source(connected)


def test_locked_d1_target_restores_but_cannot_reconnect(d1_source, d1_harness, monkeypatch):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile())
    session.save_workspace(source=KEY, dialect="sqlite")

    def locked():
        raise CredentialError("Vault locked")

    monkeypatch.setattr(credentials, "_secure_backend", locked)
    count = len(d1_harness.requests)
    restored = Session()
    saved = restored.saved_source_profile(KEY)
    assert saved["credentials_missing"] and saved["api_token"] == ""
    assert restored.dialect == "sqlite" and not restored.source_connected()
    with pytest.raises(CredentialError):
        build_saved_source(restored, profile=saved)
    assert len(d1_harness.requests) == count
    restored.save_fn("offline_ids", "", "SELECT [id] FROM [records]", "")
    assert json.loads(restored.storage_path.read_text(encoding="utf-8"))["d1_profiles"][0]["credential_ref"]


def test_d1_functions_history_pages_and_chats_stay_source_scoped(d1_source, parquet_source):
    session = Session({d1_source.id: d1_source, parquet_source.id: parquet_source})
    session.remember_d1(profile())
    session.save_fn("labels", "minimum", "SELECT [label] FROM [records] WHERE id > @minimum", "D1 labels")
    query = "SELECT * FROM labels(1)"
    assert session.run(query).rows == [["beta"], [None]]
    session.save_workspace(source=KEY, dialect="sqlite", document=query)
    page_id = session.workspace["document_id"]
    session.save_chat("d1-chat", {"messages": [{"role": "user", "content": "Explain records"},
        {"role": "assistant", "content": "Three records."}], "transcript": "Three records.",
        "input": "", "turns": 1, "visible": True, "target_key": KEY, "title": "D1 chat",
        "status": "idle", "status_text": "", "unread": False, "selected": True})
    session.set_active(parquet_source.id)
    assert "labels" not in session.fns and session.history == []
    session.save_fn("local_ids", "", "SELECT id FROM data", "")
    assert session.run("SELECT id FROM data WHERE id = 1").rows == [[1]]
    assert session.history[0].target != KEY
    assert [entry.doc for entry in session.history_for(KEY)] == [query]
    restored = Session()
    assert restored.source_key() == KEY
    assert set(restored.fns) == {"labels"}
    assert [entry.doc for entry in restored.history] == [query]
    assert next(page for page in restored.workspace["pages"] if page["id"] == page_id)["source"] == KEY
    assert restored.load_chat("d1-chat")["target_key"] == KEY
    preview = session.fn_preview("labels", "1", source_id=d1_source.id)
    assert preview.ok and preview.rows == [["beta"], [None]]
    assert session.source is parquet_source
    restored.restore_source(session.source_key(parquet_source.id))
    assert set(restored.fns) == {"local_ids"}
    assert all(entry.target != KEY for entry in restored.history)


def test_d1_functions_and_completion_use_sqlite_without_network(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    count = len(d1_harness.requests)
    session.save_fn("selected", "minimum", "SELECT `label` FROM [records] WHERE id > @minimum", "")
    compiled = session.compile("SELECT * FROM selected(1)")
    assert compiled.ok and "LATERAL" not in compiled.sql.upper()
    assert not session.compile("SELECT * FROM LATERAL selected(1)").ok
    document = "SELECT r.[la] FROM [records] r"
    offers = complete(document, len("SELECT r.[la"), session.schema, dialect="sqlite")
    assert any(item.insert == "[label]" for item in offers.items)
    assert len(d1_harness.requests) == count
    preview = session.fn_preview("selected", "1")
    assert preview.ok and preview.rows == [["beta"], [None]]
    assert session.history == []


def test_d1_trigger_document_splits_and_executes_native_statements(d1_source):
    session = Session({d1_source.id: d1_source})
    outcomes = session.run_all("CREATE TABLE audit (label TEXT); "
        "CREATE TRIGGER record_insert AFTER INSERT ON records BEGIN "
        "INSERT INTO audit VALUES (CASE WHEN NEW.label IS NULL THEN 'none' ELSE NEW.label END); "
        "INSERT INTO audit VALUES ('second'); END; "
        "INSERT INTO records VALUES (4, 'delta'); SELECT label FROM audit ORDER BY rowid")
    assert len(outcomes) == 4
    assert all(outcome.ok for outcome in outcomes)
    assert outcomes[-1].rows == [["delta"], ["second"]]
    assert len(session.history) == 4
    assert session.history[-2].doc.upper().startswith("CREATE TRIGGER")


def test_d1_successful_write_survives_schema_refresh_failure(d1_source, monkeypatch):
    session = Session({d1_source.id: d1_source})

    def fail_refresh(source):
        raise EngineError("Metadata unavailable")

    monkeypatch.setattr(session_module, "refresh_source_schema", fail_refresh)
    outcome = session.run("CREATE TABLE persisted (id INTEGER)")
    assert outcome.ok and not outcome.error and outcome.schema_changed
    assert "schema not refreshed" in outcome.status
    assert session.history[0].doc == "CREATE TABLE persisted (id INTEGER)"
    assert session.run("SELECT count(*) FROM persisted").rows == [[0]]


def test_d1_disconnected_workspace_keeps_sqlite_without_saved_credentials(d1_source, d1_harness):
    session = Session({d1_source.id: d1_source})
    session.remember_d1(profile(), persist=False)
    session.save_workspace(source=KEY, dialect="sqlite", document="SELECT `id` FROM [records]")
    request_count = len(d1_harness.requests)
    restored = Session()
    assert restored.dialect == "sqlite"
    saved = restored.saved_source_profile(KEY)
    assert saved == {"kind": "d1-live", "account_id": ACCOUNT_ID, "database": DATABASE_ID,
                     "api_token": "", "display": d1_source.display, "credentials_missing": True}
    assert len(d1_harness.requests) == request_count
    with pytest.raises(CredentialError):
        build_saved_source(restored, profile=saved)
    assert len(d1_harness.requests) == request_count
    assert restored.d1_profiles == []
    assert not restored.source_connected()
    assert restored.compile("SELECT `id` FROM [records]").ok
    with pytest.raises(ValueError, match="intrinsic"):
        restored.update_settings(dialect="postgres")
