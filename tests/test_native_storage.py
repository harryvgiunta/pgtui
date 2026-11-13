"""Deterministic native SQL persistence and OS-vault boundary tests."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

import pytest

from d8r import credentials, storage
from d8r.credentials import CredentialError, CredentialVault
from d8r.storage import MemoryStore, SettingsStore, WorkspaceStore


def profile(password="postgres-secret", **changes):
    return {
        "host": "localhost", "port": "5432", "database": "app", "user": "reader",
        "password": password, "schema": "public", "sslmode": "prefer", "display": "Local",
        **changes,
    }


def functions():
    return {"postgres:local": [{
        "name": "by_id", "params": ["id"], "body": "SELECT * FROM public.items WHERE id = @id",
        "description": "Native SQL, preserved without execution",
    }]}


def locked_backend(monkeypatch):
    def unavailable():
        raise CredentialError("Secure system keyring unavailable or locked.")
    monkeypatch.setattr(credentials, "_secure_backend", unavailable)


def test_data_directory_is_independent(tmp_path, monkeypatch):
    monkeypatch.delenv("PGTUI_DATA_DIR", raising=False)
    monkeypatch.setenv("D8R_DATA_DIR", str(tmp_path / "original"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert storage.data_directory() == tmp_path / ".pgtui"
    assert MemoryStore().path.parent == tmp_path / ".pgtui"
    assert not (tmp_path / ".pgtui").exists()
    monkeypatch.setenv("PGTUI_DATA_DIR", str(tmp_path / "isolated"))
    assert storage.data_directory() == tmp_path / "isolated"


def test_vault_namespace_never_reads_original_entries(tmp_path, credential_backend):
    reference = "a" * 32
    canonical = os.path.normcase(str(tmp_path.resolve()))
    digest = hashlib.sha256(os.fsencode(canonical)).hexdigest()
    original = (f"d8r:{digest}:{reference}", "credential")
    credential_backend.values[original] = "original-secret"
    vault = CredentialVault(tmp_path)
    assert vault.get(reference) is None
    saved = vault.put("fork-secret")
    assert (f"pgtui:{digest}:{saved}", "credential") in credential_backend.values
    vault.delete(saved)
    assert credential_backend.values == {original: "original-secret"}


def test_postgres_functions_and_password_roundtrip(tmp_path, credential_backend):
    store = MemoryStore(tmp_path)
    store.save(functions(), [profile()])
    disk = json.loads(store.path.read_text(encoding="utf-8"))
    assert disk["version"] == 2
    assert "password" not in disk["postgres_profiles"][0]
    assert "postgres-secret" not in store.path.read_text(encoding="utf-8")
    reference = disk["postgres_profiles"][0]["credential_ref"]
    assert CredentialVault(tmp_path).get(reference) == "postgres-secret"
    loaded = MemoryStore(tmp_path)
    assert not loaded.error
    assert loaded.document["functions"] == functions()
    assert loaded.document["postgres_profiles"] == [profile()]
    loaded.save({})
    assert loaded.document["postgres_profiles"] == [profile()]
    assert len(credential_backend.values) == 1
    loaded.save({}, [])
    assert not credential_backend.values


@pytest.mark.parametrize("changes", [
    {"port": "0"}, {"port": "65536"}, {"port": "１２"}, {"sslmode": "unsafe"},
    {"database": ""}, {"backend": "sqlite"},
])
def test_postgres_profile_validation(changes):
    with pytest.raises(ValueError):
        MemoryStore.validate_profile(profile(**changes))


def test_validation_precedes_any_vault_write(tmp_path, credential_backend):
    store = MemoryStore(tmp_path)
    with pytest.raises(ValueError, match="duplicate PostgreSQL"):
        store.save(functions(), [profile(), profile()])
    assert not store.path.exists()
    assert not credential_backend.values


def test_passwordless_profiles_work_without_keyring(tmp_path, monkeypatch):
    locked_backend(monkeypatch)
    store = MemoryStore(tmp_path)
    store.save(functions(), [profile("")])
    assert json.loads(store.path.read_text())["postgres_profiles"][0]["credential_ref"] is None
    loaded = MemoryStore(tmp_path)
    assert not loaded.credential_warning
    assert loaded.document["postgres_profiles"] == [profile("")]


def test_locked_keyring_preserves_profiles_and_references(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    store.save({}, [profile()])
    reference = json.loads(store.path.read_text())["postgres_profiles"][0]["credential_ref"]
    locked_backend(monkeypatch)
    loaded = MemoryStore(tmp_path)
    missing = loaded.document["postgres_profiles"][0]
    assert missing["credentials_missing"] is True
    assert missing["password"] == ""
    assert loaded.credential_warning
    loaded.save(functions())
    assert json.loads(store.path.read_text())["postgres_profiles"][0]["credential_ref"] == reference
    with pytest.raises(CredentialError, match="Re-enter"):
        loaded.save({}, [{**missing, "host": "elsewhere"}])
    with pytest.raises(CredentialError):
        loaded.save({}, [profile("new-secret")])
    assert json.loads(store.path.read_text())["postgres_profiles"][0]["credential_ref"] == reference


def test_stale_writer_rolls_back_new_credential(tmp_path, credential_backend):
    first, stale = MemoryStore(tmp_path), MemoryStore(tmp_path)
    first.save(functions(), [profile()])
    original = first.path.read_bytes()
    entries = dict(credential_backend.values)
    with pytest.raises(ValueError, match="changed outside"):
        stale.save({}, [profile("stale-secret")])
    assert first.path.read_bytes() == original
    assert credential_backend.values == entries
    assert stale.document["postgres_profiles"] == []


def test_failed_replace_preserves_old_file_and_secret(tmp_path, monkeypatch, credential_backend):
    store = MemoryStore(tmp_path)
    store.save(functions(), [profile()])
    original = store.path.read_bytes()
    entries = dict(credential_backend.values)

    def fail_replace(*args):
        raise OSError("driver-secret must not be exposed")

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(ValueError, match="Nothing was changed") as caught:
        store.save({}, [profile("replacement-secret")])
    assert "driver-secret" not in str(caught.value)
    assert store.path.read_bytes() == original
    assert store.document["functions"] == functions()
    assert credential_backend.values == entries
    assert not list(tmp_path.glob(".memory-*.tmp"))
    assert not store.path.with_suffix(".lock").exists()


@pytest.mark.parametrize("contents", [
    b'{"version":2,"version":2}',
    b'{"version":2,"functions":{},"postgres_profiles":[],"backend_profiles":[]}',
    b'{"version":1,"functions":{},"postgres_profiles":[]}',
    b'{"version":2,"functions":{},"postgres_profiles":NaN}',
    b'not-json-secret',
])
def test_corrupt_and_legacy_memory_are_not_migrated(tmp_path, contents, credential_backend):
    path = tmp_path / "memory.json"
    path.write_bytes(contents)
    store = MemoryStore(tmp_path)
    assert store.error
    assert "not-json-secret" not in store.error
    with pytest.raises(ValueError):
        store.save({}, [])
    assert path.read_bytes() == contents
    assert not credential_backend.values


def test_fresh_locks_block_and_stale_locks_recover(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.time, "time", lambda: 10_000.0)
    store = MemoryStore(tmp_path)
    lock = tmp_path / "memory.lock"
    lock.write_text("", encoding="utf-8")
    os.utime(lock, (10_000, 10_000))
    with pytest.raises(ValueError, match="busy"):
        store.save({}, [])
    assert lock.exists()
    os.utime(lock, (9_000, 9_000))
    store.save({}, [])
    assert store.path.exists()
    assert not lock.exists()


def test_only_stale_temporary_files_are_swept(tmp_path, monkeypatch):
    monkeypatch.setattr(storage.time, "time", lambda: 10_000.0)
    old, fresh = tmp_path / ".memory-old.tmp", tmp_path / ".memory-fresh.tmp"
    old.write_text("old")
    fresh.write_text("fresh")
    os.utime(old, (9_000, 9_000))
    os.utime(fresh, (10_000, 10_000))
    MemoryStore(tmp_path)
    assert not old.exists()
    assert fresh.exists()


def ai_settings(store, api_key="ai-secret"):
    document = deepcopy(store.document)
    document["ai"].update(base_url="https://api.example.test/v1", model="test-model", api_key=api_key)
    return document


def test_settings_have_no_cross_dialect_preferences(tmp_path):
    store = SettingsStore(tmp_path)
    assert store.document["source"] == "postgres"
    assert "dialect" not in store.document
    assert "source_dialects" not in store.document
    for key, value in (("dialect", "duckdb"), ("source_dialects", {})):
        with pytest.raises(ValueError, match="invalid settings fields"):
            store.save({**store.document, key: value})
    store.save(store.document)
    assert SettingsStore(tmp_path).document == store.document


def test_ai_secret_roundtrip_locked_preferences_and_forgetting(tmp_path, monkeypatch, credential_backend):
    store = SettingsStore(tmp_path)
    store.save(ai_settings(store))
    assert "ai-secret" not in store.path.read_text()
    assert SettingsStore(tmp_path).document["ai"]["api_key"] == "ai-secret"
    reference = json.loads(store.path.read_text())["ai"]["api_key_ref"]
    with monkeypatch.context() as locked:
        locked_backend(locked)
        loaded = SettingsStore(tmp_path)
        assert loaded.ai_credentials_missing
        assert loaded.document["ai"]["api_key"] == ""
        loaded.save({**loaded.document, "default_rows": 75})
        assert json.loads(store.path.read_text())["ai"]["api_key_ref"] == reference
    loaded = SettingsStore(tmp_path)
    loaded.save(ai_settings(loaded, ""), replace_ai=True)
    assert not loaded.ai_credentials_missing
    assert not credential_backend.values
    assert json.loads(store.path.read_text())["ai"]["api_key_ref"] is None


def test_plaintext_settings_rejected_without_migration(tmp_path, credential_backend):
    store = SettingsStore(tmp_path)
    original = json.dumps(ai_settings(store)).encode()
    store.path.write_bytes(original)
    loaded = SettingsStore(tmp_path)
    assert loaded.error
    assert loaded.document["ai"]["api_key"] == ""
    assert loaded.path.read_bytes() == original
    assert not credential_backend.values


def workspace(store):
    document = deepcopy(store.document)
    document.update(document="SELECT * FROM data", document_id="page1", source="parquet:local", dialect="duckdb")
    document["sources"] = {"parquet:local": {
        "id": "local", "display": "Local Parquet", "kind": "parquet", "path": "/local/data.parquet",
    }}
    document["pages"] = [{
        "id": "page1", "title": "Data", "description": "Native SQL", "document": "SELECT * FROM data",
        "cursor": [0, 4], "source": "parquet:local", "dialect": "duckdb",
    }]
    document["history"] = [{
        "at": "2026-11-12T12:00:00", "source": "local", "dialect": "duckdb", "rows": 3,
        "ms": 1.25, "doc": "SELECT * FROM data", "target": "parquet:local",
    }]
    document["chats"] = {"chat1": {
        "messages": [{"role": "user", "content": "Show data"}, {"role": "assistant", "content": "SELECT * FROM data"}],
        "transcript": "SQL discussion", "input": "", "turns": 1, "visible": True,
        "target_key": "parquet:local", "title": "Data", "status": "idle", "status_text": "",
        "unread": False, "selected": True,
    }}
    return document


def test_native_pages_history_and_chats_roundtrip(tmp_path):
    store = WorkspaceStore(tmp_path)
    document = workspace(store)
    store.save(document)
    assert WorkspaceStore(tmp_path).document == document


@pytest.mark.parametrize("location", ["workspace", "page", "history"])
def test_non_native_dialect_metadata_rejected(tmp_path, location):
    store = WorkspaceStore(tmp_path)
    document = workspace(store)
    target = document if location == "workspace" else document["pages" if location == "page" else "history"][0]
    target["dialect"] = "mysql"
    with pytest.raises(ValueError, match="dialect"):
        store.save(document)
    assert not store.path.exists()


def test_incomplete_chat_exchange_never_persists(tmp_path):
    store = WorkspaceStore(tmp_path)
    document = workspace(store)
    document["chats"]["chat1"]["messages"].pop()
    with pytest.raises(ValueError, match="incomplete saved conversation"):
        store.save(document)
    assert not store.path.exists()


def test_locked_vault_rejects_new_remembered_secret_without_file(tmp_path, monkeypatch):
    locked_backend(monkeypatch)
    store = MemoryStore(tmp_path)
    with pytest.raises(CredentialError):
        store.save({}, [profile()])
    assert not store.path.exists()
    assert store.document["postgres_profiles"] == []


def test_rotating_password_retires_only_previous_reference(tmp_path, credential_backend):
    store = MemoryStore(tmp_path)
    store.save({}, [profile()])
    old = json.loads(store.path.read_text())["postgres_profiles"][0]["credential_ref"]
    unrelated = CredentialVault(tmp_path / "another-directory").put("unrelated-secret")
    store.save(functions(), [profile("replacement-secret")])
    new = json.loads(store.path.read_text())["postgres_profiles"][0]["credential_ref"]
    vault = CredentialVault(tmp_path)
    assert old != new
    assert vault.get(old) is None
    assert vault.get(new) == "replacement-secret"
    assert CredentialVault(tmp_path / "another-directory").get(unrelated) == "unrelated-secret"
    assert len(credential_backend.values) == 2


def test_vault_cleanup_failure_does_not_undo_committed_metadata(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    store.save({}, [profile()])

    def denied(self, reference):
        raise CredentialError("Denied deletion")

    monkeypatch.setattr(CredentialVault, "delete", denied)
    store.save(functions(), [])
    assert store.document["postgres_profiles"] == []
    assert json.loads(store.path.read_text())["postgres_profiles"] == []
    assert "unused credential could not be removed" in store.credential_warning


def test_complete_tool_exchange_roundtrips_without_execution(tmp_path):
    store = WorkspaceStore(tmp_path)
    document = workspace(store)
    document["chats"]["chat1"]["messages"] = [
        {"role": "user", "content": "Check SQL"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call1", "type": "function",
            "function": {"name": "validate_sql", "arguments": '{"sql":"SELECT 1"}'},
        }]},
        {"role": "tool", "tool_call_id": "call1", "content": "Lexically valid"},
        {"role": "assistant", "content": "Ready"},
    ]
    store.save(document)
    assert WorkspaceStore(tmp_path).document == document
