"""Isolated local state, native vault boundary, and generated Parquet test data."""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("PGTUI_DATA_DIR", str(tmp_path / "pgtui-state"))


@pytest.fixture(autouse=True)
def credential_backend(monkeypatch):
    from keyring.backend import KeyringBackend
    from keyring.errors import PasswordDeleteError
    from d8r import credentials

    class MemoryKeyring(KeyringBackend):
        priority = 1

        def __init__(self):
            self.values = {}

        def get_password(self, service, username):
            return self.values.get((service, username))

        def set_password(self, service, username, password):
            self.values[service, username] = password

        def delete_password(self, service, username):
            try:
                del self.values[service, username]
            except KeyError:
                raise PasswordDeleteError("missing test credential") from None

    backend = MemoryKeyring()
    monkeypatch.setattr(credentials, "_secure_backend", lambda: backend)
    return backend


@pytest.fixture
def parquet_source(tmp_path):
    import duckdb
    from d8r.engine import add_parquet_source, close_source

    path = tmp_path / "records.parquet"
    connection = duckdb.connect()
    try:
        connection.execute(
            "COPY (SELECT * FROM (VALUES (1::BIGINT, 'alpha'), (2, 'beta'), (3, NULL)) AS t(id, label)) "
            "TO ? (FORMAT PARQUET)", [str(path)],
        )
    finally:
        connection.close()
    source = add_parquet_source(str(path))
    try:
        yield source
    finally:
        close_source(source)


@pytest.fixture
def sources(parquet_source):
    return {parquet_source.id: parquet_source}


@pytest.fixture
def d1_harness(monkeypatch):
    from d8r.engine import d1api
    from tests.d1_support import D1Harness

    harness = D1Harness()
    monkeypatch.setattr(d1api, "create_client", harness.client)
    try:
        yield harness
    finally:
        harness.close()


@pytest.fixture
def d1_source(d1_harness):
    from d8r.engine import add_d1_source, close_source
    from tests.d1_support import ACCOUNT_ID, DATABASE_ID, API_TOKEN

    source = add_d1_source("test-d1", account_id=ACCOUNT_ID, database=DATABASE_ID, api_token=API_TOKEN)
    try:
        yield source
    finally:
        close_source(source)
