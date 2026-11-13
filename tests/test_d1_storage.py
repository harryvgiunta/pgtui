"""D1 metadata compatibility and atomic vault persistence boundaries."""
from copy import deepcopy
import json

import pytest

from d8r import credentials, storage
from d8r.credentials import CredentialError, CredentialVault
from d8r.storage import MemoryStore, WorkspaceStore
from tests.d1_support import ACCOUNT_ID, API_TOKEN, DATABASE_ID


KEY = f"d1:{ACCOUNT_ID}:{DATABASE_ID}"


def profile(**changes):
    return {"account_id": ACCOUNT_ID, "database": DATABASE_ID,
            "api_token": API_TOKEN, "display": "Cloud database", **changes}


def postgres_profile():
    return {"host": "localhost", "port": "5432", "database": "app", "user": "reader",
            "password": "postgres-secret", "schema": "public", "sslmode": "prefer", "display": "PG"}


def test_version_two_without_d1_profiles_remains_readable(tmp_path):
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({"version": 2, "functions": {}, "postgres_profiles": []}), encoding="utf-8")
    store = MemoryStore(tmp_path)
    assert not store.error
    assert store.document["d1_profiles"] == []
    store.save({})
    assert json.loads(path.read_text(encoding="utf-8"))["d1_profiles"] == []


def test_d1_profile_canonicalization_and_vault_roundtrip(tmp_path, credential_backend):
    store = MemoryStore(tmp_path)
    normalized = store.validate_d1_profile(profile(account_id=ACCOUNT_ID.upper(), database=DATABASE_ID.upper()))
    assert normalized == profile()
    store.save({}, d1_profiles=[normalized])
    raw = store.path.read_text(encoding="utf-8")
    saved = json.loads(raw)["d1_profiles"][0]
    assert API_TOKEN not in raw and "api_token" not in saved
    assert set(saved) == {"account_id", "database", "display", "credential_ref"}
    assert CredentialVault(tmp_path).get(saved["credential_ref"]) == API_TOKEN
    assert MemoryStore(tmp_path).document["d1_profiles"] == [profile()]
    store.save({KEY: []})
    assert len(credential_backend.values) == 1


@pytest.mark.parametrize("changes", [
    {"account_id": "bad"}, {"database": "name-not-resolved"}, {"api_token": ""},
    {"api_token": "   "}, {"extra": "not allowed"},
])
def test_invalid_d1_profiles_are_rejected(changes):
    with pytest.raises(ValueError):
        MemoryStore.validate_d1_profile(profile(**changes))


def test_d1_duplicate_validation_precedes_vault_changes(tmp_path, credential_backend):
    store = MemoryStore(tmp_path)
    with pytest.raises(ValueError, match="duplicate D1"):
        store.save({}, [postgres_profile()], [profile(), profile(account_id=ACCOUNT_ID.upper())])
    assert not credential_backend.values
    assert not store.path.exists()


def test_vault_reference_union_preserves_other_backend(tmp_path, credential_backend):
    store = MemoryStore(tmp_path)
    store.save({}, [postgres_profile()], [profile()])
    original = json.loads(store.path.read_text(encoding="utf-8"))
    pg_ref = original["postgres_profiles"][0]["credential_ref"]
    d1_ref = original["d1_profiles"][0]["credential_ref"]
    store.save({}, d1_profiles=[profile(api_token="rotated-d1-secret")])
    vault = CredentialVault(tmp_path)
    assert vault.get(pg_ref) == "postgres-secret"
    assert vault.get(d1_ref) is None
    assert len(credential_backend.values) == 2
    store.save({}, postgres_profiles=[])
    assert len(credential_backend.values) == 1
    assert MemoryStore(tmp_path).document["d1_profiles"][0]["api_token"] == "rotated-d1-secret"
    store.save({}, d1_profiles=[])
    assert not credential_backend.values


def test_locked_vault_preserves_both_reference_groups(tmp_path, monkeypatch):
    store = MemoryStore(tmp_path)
    store.save({}, [postgres_profile()], [profile()])
    original = json.loads(store.path.read_text(encoding="utf-8"))

    def locked():
        raise CredentialError("Vault locked")

    monkeypatch.setattr(credentials, "_secure_backend", locked)
    loaded = MemoryStore(tmp_path)
    missing = loaded.document["d1_profiles"][0]
    assert missing["credentials_missing"] and missing["api_token"] == ""
    assert loaded.credential_warning
    loaded.save({KEY: []})
    saved = json.loads(store.path.read_text(encoding="utf-8"))
    for group in ("postgres_profiles", "d1_profiles"):
        assert saved[group] == original[group]
    with pytest.raises(CredentialError, match="Re-enter"):
        loaded.save({}, d1_profiles=[{**missing, "display": "Changed"}])


def test_missing_d1_vault_reference_is_not_keyless_authentication(tmp_path):
    stored = {key: value for key, value in profile().items() if key != "api_token"}
    (tmp_path / "memory.json").write_text(json.dumps({"version": 2, "functions": {},
        "postgres_profiles": [], "d1_profiles": [{**stored, "credential_ref": None}]}), encoding="utf-8")
    store = MemoryStore(tmp_path)
    assert not store.error
    assert store.document["d1_profiles"][0]["credentials_missing"]
    assert store.credential_warning
    store.save({})
    assert json.loads(store.path.read_text(encoding="utf-8"))["d1_profiles"][0]["credential_ref"] is None


def test_stale_d1_save_discards_only_new_credentials(tmp_path, credential_backend):
    store, stale = MemoryStore(tmp_path), MemoryStore(tmp_path)
    store.save({}, [postgres_profile()], [profile()])
    original = store.path.read_bytes()
    secrets = dict(credential_backend.values)
    with pytest.raises(ValueError, match="changed outside"):
        stale.save({}, d1_profiles=[profile(api_token="stale-token")])
    assert store.path.read_bytes() == original
    assert credential_backend.values == secrets
    assert stale.document["d1_profiles"] == []


def test_failed_d1_commit_preserves_metadata_and_vault(tmp_path, monkeypatch, credential_backend):
    store = MemoryStore(tmp_path)
    store.save({}, [postgres_profile()], [profile()])
    original, secrets = store.path.read_bytes(), dict(credential_backend.values)

    def fail_replace(*args):
        raise OSError("private driver details")

    monkeypatch.setattr(storage.os, "replace", fail_replace)
    with pytest.raises(ValueError):
        store.save({}, d1_profiles=[profile(api_token="new-token")])
    assert store.path.read_bytes() == original
    assert credential_backend.values == secrets
    assert store.document["d1_profiles"] == [profile()]


def test_sqlite_workspace_metadata_requires_d1_identity(tmp_path):
    store = WorkspaceStore(tmp_path)
    document = deepcopy(store.document)
    document.update(source=KEY, dialect="sqlite", sources={KEY: {
        "id": "d1", "display": "D1", "kind": "d1-live", "path": "",
    }})
    store.save(document)
    assert WorkspaceStore(tmp_path).document["dialect"] == "sqlite"
    for target, kind, path in [("local.sqlite", "d1-live", "local.sqlite"),
                                (KEY, "parquet", "local.sqlite"), ("ordinary", "disconnected", "")]:
        invalid = {**document, "source": target, "sources": {target: {
            "id": "other", "display": "Other", "kind": kind, "path": path,
        }}}
        with pytest.raises(ValueError):
            store.save(invalid)
