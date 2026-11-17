"""The TUI's headless core: registry, schema seam, palette data, runs, history.

Everything the widgets need to know is computed here in plain Python, so the
Textual layer stays a thin shell and the tests can drive a `Session` without a
terminal attached. Provider configuration is loaded from local settings; AI
transport lives in `d8r.ai`, while database networking stays in the engine.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from uuid import UUID, uuid4

from d8r.ai.client import AIConfig
from d8r.completion.context import analyze
from d8r.credentials import CredentialError

from d8r.engine import (
    DataSource, EngineError, add_d1_source, add_parquet_source, add_postgres_source,
    cancel_source, close_source, column_values, execute_sql, load,
    refresh_source_schema, transaction_open,
)
from d8r.sql import (
    FnDef, SQLDocument, SchemaContext, SqlError, TableDef, function_query,
    lex_sql, open_tables, parse_parameters, parse_sql, quote_identifier,
    relation_parts, split_statements, validate_function,
)
from d8r.storage import MemoryStore, SettingsStore, WorkspaceStore
from d8r.schema_cache import SchemaCacheStore, SchemaSnapshot, capture_schema

PREVIEW_ROW_CAP = 10000
"""Maximum buffered preview rows; never an implicit SQL LIMIT or full count."""


# ---------------------------------------------------------------------------
# Source -> schema seam
# ---------------------------------------------------------------------------


def tables_of(source: DataSource) -> list[TableDef]:
    """Build the completion snapshot from cached metadata, without database IO."""
    return [TableDef(name=name, doc=entry.get("doc", ""),
                     columns=tuple(entry["columns"]),
                     foreign_keys=tuple(entry.get("foreign_keys", ())))
            for name, entry in source.datasets.items()]


def _source_id(hint: str, taken) -> str:
    """A stable, unique registry id derived from a human hint."""
    base = re.sub(r"[^a-z0-9]+", "-", hint.strip().lower()).strip("-") or "source"
    candidate = base
    suffix = 2
    while candidate in taken:
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate




# ---------------------------------------------------------------------------
# Results of one run
# ---------------------------------------------------------------------------


@dataclass
class HistoryEntry:
    """One executed document, as the History tab lists it."""

    at: str
    source: str
    dialect: str
    rows: int
    ms: float
    doc: str
    target: str = ""

    @property
    def title(self) -> str:
        """The document's first non-empty line, for the History table."""
        for line in self.doc.split("\n"):
            if line.strip():
                return line.strip()
        return ""


@dataclass
class RunOutcome:
    """One native statement's result; total is a buffered row-count lower bound."""

    status: str = ""
    error: str = ""
    ok: bool = False
    source: str = ""
    dialect: str = ""
    columns: list[str] = field(default_factory=list)
    dtypes: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    total: int = 0
    ms: float = 0.0
    sql: str = ""
    schema_changed: bool = False
    doc: str = ""

    @property
    def capped(self) -> bool:
        """True when the preview buffer, not the document, decided the row count."""
        return self.total > len(self.rows)


@dataclass
class TxState:
    """A current engine-derived transaction snapshot, never persisted."""

    open: bool = False


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------


class _MissingAIConfig(AIConfig):
    """Keep provider metadata editable while rejecting every network request."""

    def validate_connection(self) -> None:
        raise CredentialError("Saved AI credentials are unavailable. Unlock the OS vault or re-enter the key and headers in Settings → AI provider.")

    def validate(self) -> None:
        self.validate_connection()


class Session:
    """Native SQL workspace with source-scoped history and saved functions."""

    def __init__(
        self, sources: dict[str, DataSource] | None = None, *, data_dir: Path | None = None,
    ) -> None:
        self.sources: dict[str, DataSource] = load() if not sources else dict(sources)
        self.active_id: str = next(iter(self.sources))
        self.dialect: str = self.source.dialect
        # UI-owned reservation: a background run/context lookup owns the shared
        # connection until its real work finishes. Headless
        # callers keep the synchronous run API and need no reservation.
        self.busy: str = ""
        self._value_cache: dict[tuple[str, str, str], tuple[DataSource, tuple[list[object], bool]]] = {}
        self._value_cache_lock = RLock()
        self.value_cache_epoch = 0
        # Definitions load without parsing bodies or connecting to their source.
        self._memory = MemoryStore(data_dir)
        saved_functions = self._memory.document["functions"]
        self._source_functions: dict[str, dict[str, FnDef]] = {
            key: {item["name"]: FnDef(
                name=item["name"], params=tuple(item["params"]), body=item["body"], doc=item["description"],
                param_types=tuple(item.get("param_types", ())),
            ) for item in definitions}
            for key, definitions in saved_functions.items()
        }
        self.postgres_profiles: list[dict[str, str]] = self._memory.document["postgres_profiles"]
        self.d1_profiles: list[dict[str, str]] = self._memory.document["d1_profiles"]
        self._session_only: set[str] = set()
        self._settings = SettingsStore(self._memory.path.parent)
        settings = self._settings.document
        if settings["source"] in self.sources:
            self.active_id = settings["source"]
            self.dialect = self.source.dialect
        self.intellisense: bool = settings["intellisense"]
        self.default_rows: int = settings["default_rows"]
        self.value_cache_enabled: bool = settings["value_cache_enabled"]
        self.value_cache_limit: int = settings["value_cache_limit"]
        self.pane_visibility: dict[str, bool] = dict(settings["panes"])
        self.pane_sizes: dict[str, int] = dict(settings["pane_sizes"])
        self.ai_provider_revision = 0
        self.ai_credentials_missing = self._settings.ai_credentials_missing
        self.ai_config = self.ai_provider_config(self.active_ai_provider) if self.active_ai_provider else AIConfig()
        self._workspace = WorkspaceStore(self._memory.path.parent)
        self._workspace_lock = RLock()
        self.workspace = deepcopy(self._workspace.document)
        if self.workspace["document"] is None:
            self.workspace["document"] = ""
        self.workspace_error = self._workspace.error
        self.workspace["document_id"] = self.workspace["document_id"] or uuid4().hex
        self._history = [HistoryEntry(**entry) for entry in self.workspace["history"]]
        self._source_history: dict[str, list[HistoryEntry]] = {}
        for entry in self._history:
            if not entry.target:
                # Saved identities outrank reusable registry slugs. Ambiguous
                # legacy rows stay durable but belong to no active workspace.
                owners = [key for key, saved in self.workspace["sources"].items()
                          if saved["id"] == entry.source]
                entry.target = (owners[0] if len(owners) == 1 else
                                self.source_key(entry.source) if not owners else "")
            if entry.target:
                self._source_history.setdefault(entry.target, []).append(entry)
        self.workspace["history"] = [asdict(entry) for entry in self._history]
        self._completion_usage: dict[str, tuple[int, tuple[TableDef, ...], dict[tuple[str, str, str], int]]] = {}
        self._schema_snapshots: dict[str, tuple[DataSource, dict, SchemaContext]] = {}
        self._schema_cache = SchemaCacheStore(self._memory.path.parent)
        self._schema_cache_lock = RLock()
        self._cached_schemas = dict(self._schema_cache.snapshots)
        self._hydrated_schemas: dict[str, tuple[DataSource, SchemaSnapshot, dict]] = {}
        self.metadata_cache_warning = self._schema_cache.error
        for source in self.sources.values():
            if source.con is not None:
                self.cache_source_schema(source)
        if not self.workspace["pages"]:
            seen: set[tuple[str, str, str]] = set()
            for entry in reversed(self._history):
                if not entry.target:
                    continue
                identity = (entry.doc, entry.target, entry.dialect)
                if identity in seen or identity == (
                    self.workspace["document"], self.workspace["source"], self.workspace["dialect"],
                ):
                    continue
                seen.add(identity)
                self.workspace["pages"].append({
                    "id": uuid4().hex, "title": "", "document": entry.doc,
                    "cursor": [0, 0], "source": identity[1], "dialect": entry.dialect,
                })
        if self.workspace["source"]:
            self.restore_source(self.workspace["source"])
        else:
            selected = settings["source"]
            if (selected in self.sources or any(self.source_key(name) == selected for name in self.sources)
                    or self.saved_source_profile(selected) is not None or selected in self.workspace["sources"]):
                self.restore_source(selected)
            else:
                self.refresh_schema()
            self.workspace["source"] = self.source_key()
            self.workspace["dialect"] = self.dialect
        source_workspaces = self.workspace["source_workspaces"]
        draft = self.workspace["function_draft"]
        draft_source = (draft["source"] or self.workspace["source"]) if draft is not None else None
        source_workspaces.setdefault(self.workspace["source"], {
            "document_id": self.workspace["document_id"], "function_draft": None,
            "active_view": self.workspace["active_view"] if draft_source in (None, self.workspace["source"]) else "workspace",
        })
        if draft is not None:
            draft = {**draft, "source": draft_source}
            state = source_workspaces.setdefault(draft_source, {
                "document_id": "", "function_draft": None, "active_view": self.workspace["active_view"],
            })
            if state["function_draft"] is None:
                state["function_draft"] = deepcopy(draft)
        selected_workspace = source_workspaces[self.workspace["source"]]
        self.workspace["function_draft"] = deepcopy(selected_workspace["function_draft"])
        self.workspace["active_view"] = selected_workspace["active_view"]

    @property
    def dbt_project_dir(self) -> str:
        """The selected source workspace's project home; reading never opens it."""
        state = self.workspace["source_workspaces"].get(self.source_key(), {})
        return state.get("dbt_project_dir", "")

    @property
    def history(self) -> list[HistoryEntry]:
        """Newest-first history for the active connection, including target scopes."""
        return self.history_for()

    def history_for(self, source_id: str | None = None) -> list[HistoryEntry]:
        """Read one stable target's history without switching the active source."""
        key = self.source_key(source_id)
        history = self._source_history.get(key)
        if history is None:
            history = self._source_history[key] = []
        return history

    @property
    def fns(self) -> dict[str, FnDef]:
        """Definitions for the active source, including temporary target scopes."""
        return self.functions_for()

    def functions_for(self, source_id: str | None = None) -> dict[str, FnDef]:
        """Read one stable target's library without switching the active source."""
        key = self.source_key(source_id)
        functions = self._source_functions.get(key)
        if functions is None:
            functions = self._source_functions[key] = {}
        return functions

    @property
    def storage_path(self) -> Path:
        """The local JSON file containing functions and public connection metadata."""
        return self._memory.path

    @property
    def settings_path(self) -> Path:
        """The editable JSON file containing preferences and AI configuration."""
        return self._settings.path

    @property
    def memory_error(self) -> str:
        """An actionable startup read error, or an empty string."""
        return "\n".join(error for error in (self._memory.error, self._settings.error, self.workspace_error) if error)

    @property
    def credential_warning(self) -> str:
        """Sanitized migration, unavailable-vault, and cleanup status."""
        return "\n".join(dict.fromkeys(warning for warning in (
            self._memory.credential_warning, self._settings.credential_warning,
        ) if warning))

    @property
    def ai_auto_accept(self) -> bool:
        """Whether complete, validated AI proposals may update unchanged drafts."""
        return self._settings.document["ai_auto_accept"]

    @property
    def ai_allow_sample_data(self) -> bool:
        """Whether AI tools may read sample records from the current source."""
        return self._settings.document["ai_allow_sample_data"]

    @property
    def ai_allow_schema_refreshes(self) -> bool:
        """Whether AI tools may look up source schema metadata."""
        return self._settings.document["ai_allow_schema_refreshes"]

    @property
    def ai_test_queries(self) -> bool:
        """Whether AI tools may execute bounded, read-only query checks."""
        return self._settings.document["ai_test_queries"]

    @property
    def ai_completion(self) -> bool:
        """Whether typing pauses may request provider SQL completion proposals."""
        return self._settings.document["ai_completion"]

    @property
    def entry_screen(self) -> bool:
        """Whether startup plays the optional terminal entry animation."""
        return self._settings.document["entry_screen"]

    @property
    def theme(self) -> str:
        """The saved application palette, independent of the current workspace."""
        return self._settings.document["theme"]

    @property
    def ai_remember_key(self) -> bool:
        """Whether the selected provider's key and headers may enter the OS vault."""
        return self._ai_provider(self.active_ai_provider)["remember"] if self.active_ai_provider else True

    @property
    def ai_providers(self) -> list[dict]:
        """Saved provider profiles, including session-only secrets, as isolated copies."""
        return deepcopy(self._settings.document["ai_providers"])

    @property
    def active_ai_provider(self) -> str:
        return self._settings.document["ai_active_provider"]

    def _ai_provider(self, provider_id: str) -> dict:
        for profile in self._settings.document["ai_providers"]:
            if profile["id"] == provider_id:
                return profile
        raise ValueError("The selected AI provider is not registered.")

    def ai_provider_missing_secrets(self, provider_id: str) -> frozenset[str]:
        self._ai_provider(provider_id)
        return frozenset(self._settings.missing_ai_credentials.get(provider_id, ()))

    def ai_provider_config(self, provider_id: str) -> AIConfig:
        profile = self._ai_provider(provider_id)
        config_type = _MissingAIConfig if self.ai_provider_missing_secrets(provider_id) else AIConfig
        return config_type(**profile["config"])

    def _save_provider_settings(self, document: dict) -> None:
        before = (self.active_ai_provider, asdict(self.ai_config), self.ai_credentials_missing)
        self._settings.save(document)
        self.ai_credentials_missing = self._settings.ai_credentials_missing
        self.ai_config = self.ai_provider_config(self.active_ai_provider) if self.active_ai_provider else AIConfig()
        after = (self.active_ai_provider, asdict(self.ai_config), self.ai_credentials_missing)
        if before != after:
            self.ai_provider_revision += 1

    def save_ai_provider(
        self, name: str, config: AIConfig, *, provider_id: str | None = None, remember: bool = True,
    ) -> str:
        """Atomically create/update and activate a named provider without network IO."""
        document = deepcopy(self._settings.document)
        if provider_id is not None:
            self._ai_provider(provider_id)
        else:
            provider_id = str(uuid4())
        profile = {"id": provider_id, "name": name, "config": asdict(config), "remember": remember}
        for index, previous in enumerate(document["ai_providers"]):
            if previous["id"] == provider_id:
                document["ai_providers"][index] = profile
                break
        else:
            document["ai_providers"].append(profile)
        document["ai_active_provider"] = provider_id
        self._save_provider_settings(document)
        return provider_id

    def select_ai_provider(self, provider_id: str) -> None:
        self._ai_provider(provider_id)
        if provider_id != self.active_ai_provider:
            self._save_provider_settings({**self._settings.document, "ai_active_provider": provider_id})

    def delete_ai_provider(self, provider_id: str) -> None:
        self._ai_provider(provider_id)
        document = deepcopy(self._settings.document)
        document["ai_providers"] = [profile for profile in document["ai_providers"] if profile["id"] != provider_id]
        if document["ai_active_provider"] == provider_id:
            document["ai_active_provider"] = document["ai_providers"][0]["id"] if document["ai_providers"] else ""
        self._save_provider_settings(document)

    def update_settings(
        self, *, intellisense: bool | None = None,
        panes: dict[str, bool] | None = None, source: str | None = None,
        dialect: str | None = None, ai: AIConfig | None = None,
        default_rows: int | None = None, ai_auto_accept: bool | None = None,
        ai_allow_sample_data: bool | None = None,
        ai_allow_schema_refreshes: bool | None = None,
        ai_test_queries: bool | None = None,
        ai_completion: bool | None = None,
        pane_sizes: dict[str, int] | None = None,
        value_cache_enabled: bool | None = None,
        value_cache_limit: int | None = None,
        entry_screen: bool | None = None,
        theme: str | None = None,
        remember_ai_key: bool | None = None,
    ) -> None:
        """Save explicit preference changes before applying them to the session.

        Temporary source contexts and unsaved provider forms never call this.
        A failed write leaves both the live state and the saved file untouched.
        """
        document = deepcopy(self._settings.document)
        if intellisense is not None:
            document["intellisense"] = intellisense
        if entry_screen is not None:
            document["entry_screen"] = entry_screen
        if theme is not None:
            document["theme"] = theme
        if panes is not None:
            document["panes"] = {**document["panes"], **panes}
        if pane_sizes is not None:
            document["pane_sizes"] = {**document["pane_sizes"], **pane_sizes}
        if source is not None:
            if source not in self.sources:
                raise ValueError("The selected data source is not registered.")
            document["source"] = self.source_key(source)
        intrinsic = self.sources[source].dialect if source is not None else self.source.dialect
        if dialect is not None and dialect != intrinsic:
            raise ValueError("SQL dialect is intrinsic to the selected source.")
        if ai is not None or remember_ai_key is not None:
            if self.active_ai_provider:
                profile = next(profile for profile in document["ai_providers"] if profile["id"] == self.active_ai_provider)
            else:
                profile = {"id": str(uuid4()), "name": "Default", "config": asdict(AIConfig()), "remember": True}
                document["ai_providers"].append(profile)
                document["ai_active_provider"] = profile["id"]
            if ai is not None:
                profile["config"] = asdict(ai)
            if remember_ai_key is not None:
                profile["remember"] = remember_ai_key
        if default_rows is not None:
            document["default_rows"] = default_rows
        if ai_auto_accept is not None:
            document["ai_auto_accept"] = ai_auto_accept
        if ai_allow_sample_data is not None:
            document["ai_allow_sample_data"] = ai_allow_sample_data
        if ai_allow_schema_refreshes is not None:
            document["ai_allow_schema_refreshes"] = ai_allow_schema_refreshes
        if ai_test_queries is not None:
            document["ai_test_queries"] = ai_test_queries
        if ai_completion is not None:
            document["ai_completion"] = ai_completion
        if value_cache_enabled is not None:
            document["value_cache_enabled"] = value_cache_enabled
        if value_cache_limit is not None:
            document["value_cache_limit"] = value_cache_limit
        self._save_provider_settings(document)
        with self._value_cache_lock:
            enabled = self._settings.document["value_cache_enabled"]
            limit = self._settings.document["value_cache_limit"]
            if (enabled, limit) != (self.value_cache_enabled, self.value_cache_limit):
                self.value_cache_enabled, self.value_cache_limit = enabled, limit
                self.clear_value_cache()
        if intellisense is not None:
            self.intellisense = intellisense
        if panes is not None:
            self.pane_visibility = dict(self._settings.document["panes"])
        if pane_sizes is not None:
            self.pane_sizes = dict(self._settings.document["pane_sizes"])
        if source is not None:
            self.set_active(source)
        if default_rows is not None:
            self.default_rows = default_rows

    def save_workspace(self, **changes) -> None:
        """Merge owned fields under one lock; failures preserve the live draft."""
        with self._workspace_lock:
            document = {**self.workspace, **deepcopy(changes)}
            if document["document"] is not None:
                page = next((page for page in document["pages"]
                             if page["id"] == document["document_id"]), None)
                snapshot = {"id": document["document_id"], "title": page["title"] if page else "",
                            "description": page.get("description", "") if page else "",
                            **{key: deepcopy(document[key]) for key in
                               ("document", "cursor", "source", "dialect")}}
                document["pages"] = [snapshot if item["id"] == snapshot["id"] else item
                                     for item in document["pages"]]
                if page is None:
                    document["pages"].append(snapshot)
            source_workspaces = deepcopy(document["source_workspaces"])
            state = source_workspaces.setdefault(document["source"], {
                "document_id": "", "function_draft": None, "active_view": "workspace",
            })
            state["document_id"] = document["document_id"]
            state["active_view"] = document["active_view"]
            if "function_draft" in changes:
                draft = document["function_draft"]
                key = draft["source"] if draft is not None else document["source"]
                state = source_workspaces.setdefault(key, {
                    "document_id": "", "function_draft": None, "active_view": "workspace",
                })
                state["function_draft"] = deepcopy(draft)
                state["active_view"] = document["active_view"]
            document["source_workspaces"] = source_workspaces
            sources = dict(document["sources"])
            for source in self.sources.values():
                if source.con is not None:
                    sources[self.source_key(source.id)] = {
                        "id": source.id, "display": source.display, "kind": source.kind,
                        "path": source.connection_key if source.kind == "parquet" else "",
                    }
            document["sources"] = sources
            self.workspace = document
            if document == self._workspace.document and not self.workspace_error:
                return
            try:
                self._workspace.save(document)
            except ValueError as exc:
                self.workspace_error = str(exc)
                raise
            self.workspace_error = ""

    def load_chats(self) -> dict[str, dict]:
        """Return independent saved conversations, never executable request state."""
        with self._workspace_lock:
            return deepcopy(self.workspace["chats"])

    def load_chat(self, key: str) -> dict | None:
        with self._workspace_lock:
            return deepcopy(self.workspace["chats"].get(key))

    def save_chat(self, key: str, chat: dict | None) -> None:
        with self._workspace_lock:
            chats = dict(self.workspace["chats"])
            if chat is None:
                chats.pop(key, None)
            else:
                chats[key] = deepcopy(chat)
            self.save_workspace(chats=chats)

    def delete_chats(self, keys: list[str]) -> None:
        """Keep removals in memory on failure, so later autosaves cannot revive them."""
        with self._workspace_lock:
            removed = set(keys)
            self.save_workspace(chats={key: chat for key, chat in self.workspace["chats"].items()
                                       if key not in removed})

    def _record_history(self, entry: HistoryEntry, outcome: RunOutcome) -> None:
        with self._workspace_lock:
            self._history.insert(0, entry)
            self.history_for(entry.target).insert(0, entry)
            try:
                self.save_workspace(history=[asdict(item) for item in self._history])
            except ValueError as exc:
                # The query already ran. A disk error must not suggest a retry
                # of potentially stateful work or throw away its result.
                outcome.status += f" · history not saved: {exc}"


    @classmethod
    def _source_identity(cls, source: DataSource) -> str:
        if source.connection_key:
            return source.connection_key
        if source.kind == "postgres-live":
            return cls._postgres_profile_id(source.postgres)
        if source.kind == "d1-live":
            return cls._d1_profile_id(source.d1)
        return source.id

    def source_key(self, source_id: str | None = None) -> str:
        source_id = self.active_id if source_id is None else source_id
        source = self.sources.get(source_id)
        return self._source_identity(source) if source is not None else source_id

    def source_connected(self, source_id: str | None = None) -> bool:
        source = self.sources.get(self.active_id if source_id is None else source_id)
        return source is not None and source.con is not None and not getattr(source.con, "closed", False)


    def restore_source(self, key: str, *, activate: bool = True) -> str:
        """Restore saved target identity without opening a file or connection."""
        key = self.source_key(key)
        source_id = next((name for name in self.sources if self.source_key(name) == key), None)
        if source_id is None:
            saved = self.workspace["sources"].get(key, {})
            profile = self.saved_source_profile(key)
            label = saved.get("display") or (profile.get("display") if profile else None) or key
            is_parquet = saved.get("kind") == "parquet" or (profile and profile.get("kind") == "parquet")
            is_d1 = saved.get("kind") == "d1-live" or (profile and profile.get("kind") == "d1-live") or bool(
                re.fullmatch(r"d1:[0-9a-f]{32}:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", key)
            )
            source_id = key
            self.sources[key] = DataSource(
                id=key, display=f"{label} · disconnected", doc="Reconnect explicitly to use this target.",
                kind="disconnected", dialect="sqlite" if is_d1 else "duckdb" if is_parquet else "postgres",
                connection_key=key,
            )
        self._hydrate_cached_schema(self.sources[source_id])
        if activate:
            self.set_active(source_id)
        return source_id




    def _save_memory(self, fns: dict[str, FnDef],
                     postgres_profiles: list[dict[str, str]] | None = None,
                     d1_profiles: list[dict[str, str]] | None = None,
                     *, persistent: str | None = None) -> None:
        functions = {**self._source_functions, self.source_key(): fns}
        durable = {}
        for group, profiles, identify in (
            ("postgres_profiles", self.postgres_profiles if postgres_profiles is None else postgres_profiles, self._postgres_profile_id),
            ("d1_profiles", self.d1_profiles if d1_profiles is None else d1_profiles, self._d1_profile_id),
        ):
            saved = {identify(profile): profile for profile in self._memory.document[group]}
            durable[group] = []
            for profile in profiles:
                identity = identify(profile)
                if identity in self._session_only and identity != persistent:
                    if identity in saved:
                        durable[group].append(saved[identity])
                else:
                    durable[group].append(profile)
        self._memory.save({key: [
            {"name": fn.name, "params": list(fn.params), "body": fn.body, "description": fn.doc,
             **({"param_types": list(fn.param_types)} if fn.param_types else {})}
            for fn in definitions.values()
        ] for key, definitions in functions.items()}, **durable)



    def remember_postgres(self, profile: dict[str, str], *, persist: bool = True) -> None:
        """Keep credentials session-only unless explicitly remembered."""
        profile = {key: value for key, value in profile.items() if key != "kind"}
        profile["port"] = str(int(profile["port"]))
        profile = self._memory.validate_profile(profile)
        identity = self._postgres_profile_id(profile)
        profiles = list(self.postgres_profiles)
        for index, saved in enumerate(profiles):
            if self._postgres_profile_id(saved) == identity:
                profiles[index] = profile
                break
        else:
            profiles.append(profile)
        if persist:
            self._save_memory(self.fns, profiles, persistent=identity)
            self._session_only.discard(identity)
        else:
            self._session_only.add(identity)
        self.postgres_profiles = profiles


    @staticmethod
    def _postgres_profile_id(profile: dict[str, str]) -> str:
        # JSON preserves boundaries in arbitrary database/user/schema identifiers.
        identity = [str(profile[key]) for key in ("host", "port", "database", "user", "schema", "sslmode")]
        return "saved-postgres:" + json.dumps(identity, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _d1_profile_id(profile: dict[str, str]) -> str:
        return "d1:" + profile["account_id"].lower() + ":" + str(UUID(profile["database"]))

    def remember_d1(self, profile: dict[str, str], *, persist: bool = True) -> None:
        """Remember a resolved D1 target; later saves never promote session secrets."""
        profile = {key: value for key, value in profile.items() if key != "kind"}
        profile = self._memory.validate_d1_profile(profile)
        profile["display"] = profile["display"].replace(profile["api_token"], "[redacted]")
        identity = self._d1_profile_id(profile)
        profiles = list(self.d1_profiles)
        for index, saved in enumerate(profiles):
            if self._d1_profile_id(saved) == identity:
                profiles[index] = profile
                break
        else:
            profiles.append(profile)
        if persist:
            self._save_memory(self.fns, d1_profiles=profiles, persistent=identity)
            self._session_only.discard(identity)
        else:
            self._session_only.add(identity)
        self.d1_profiles = profiles

    # -- the active source --------------------------------------------------

    @property
    def source(self) -> DataSource:
        """The active datasource."""
        return self.sources[self.active_id]


    def saved_source_profile(self, source_id: str) -> dict | None:
        """Resolve an explicit reconnect request without contacting its target."""
        source_id = self.source_key(source_id)
        saved = self.workspace["sources"].get(source_id)
        if saved and saved["kind"] == "parquet" and saved["path"]:
            return {"kind": "parquet", "path": saved["path"], "display": saved["display"]}
        d1 = next((profile for profile in self.d1_profiles
                   if self._d1_profile_id(profile) == source_id), None)
        if d1 is not None:
            return {**d1, "kind": "d1-live"}
        if re.fullmatch(r"d1:[0-9a-f]{32}:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", source_id):
            _, account_id, database = source_id.split(":")
            return {"kind": "d1-live", "account_id": account_id, "database": database,
                    "api_token": "", "display": saved.get("display", database) if saved else database,
                    "credentials_missing": True}
        postgres = next((profile for profile in self.postgres_profiles
                         if self._postgres_profile_id(profile) == source_id), None)
        return {**postgres, "kind": "postgres-live"} if postgres is not None else None

    def source_options(self) -> list[tuple[str, str]]:
        """Registered sources plus saved targets that are not registered yet."""
        registered = {self.source_key(name) for name in self.sources}
        return [(source.display, source.id) for source in self.sources.values()] + [
            (f"{profile['display']} · disconnected", self._postgres_profile_id(profile))
            for profile in self.postgres_profiles
            if self._postgres_profile_id(profile) not in registered
        ] + [
            (f"{profile['display']} · disconnected", self._d1_profile_id(profile))
            for profile in self.d1_profiles
            if self._d1_profile_id(profile) not in registered
        ] + [
            (f"{source['display']} · disconnected", key)
            for key, source in self.workspace["sources"].items()
            if source["kind"] in {"parquet", "d1-live"} and key not in registered
            and not any(self._d1_profile_id(profile) == key for profile in self.d1_profiles)
        ]

    def set_active(self, source_id: str) -> None:
        """Point the session at another registered source and capture its schema."""
        if source_id not in self.sources:
            raise KeyError(source_id)
        self.active_id = source_id
        self.dialect = self.source.dialect
        self.refresh_schema()

    @contextmanager
    def target_source(self, source_id: str | None = None) -> Iterator[None]:
        """Use a source for synchronous editor work, then restore the workspace.

        Never hold this scope across an await: the session's active source changes.
        """
        if source_id is None or source_id == self.active_id:
            self._restore_schema_snapshot()
            yield
            return
        active_id, dialect = self.active_id, self.dialect
        try:
            self.active_id = source_id
            self.dialect = self.source.dialect
            self._restore_schema_snapshot()
            yield
        finally:
            self.active_id, self.dialect = active_id, dialect
            self._restore_schema_snapshot()

    def register(self, source: DataSource, activate: bool = True) -> None:
        """Register an explicitly built source, replacing its disconnected target."""
        key = self._source_identity(source)
        connected = source.con is not None and not getattr(source.con, "closed", False)
        placeholders = [name for name in self.sources if not self.source_connected(name) and self.source_key(name) == key]
        was_active = self.active_id in placeholders
        with self._value_cache_lock:
            for name in placeholders:
                if connected:
                    self.clear_value_cache(name)
                del self.sources[name]
            previous = self.sources.get(source.id)
            if previous is not None and previous is not source:
                if connected or self._source_identity(previous) != key:
                    self.clear_value_cache(source.id)
                close_source(previous)
            self.sources[source.id] = source
        self.cache_source_schema(source)
        if activate:
            self.set_active(source.id)
        elif was_active:
            self.set_active(source.id)
        elif source.id == self.active_id:
            self.refresh_schema()

    def cache_source_schema(self, source: DataSource) -> None:
        """Save completed native discovery, never from completion's snapshot path."""
        if self.sources.get(source.id) is not source:
            return
        try:
            snapshot = capture_schema(source)
            if snapshot is None:
                return
            key = self._source_identity(source)
            with self._schema_cache_lock:
                if self.sources.get(source.id) is not source:
                    return
                self._cached_schemas[key] = snapshot
                self._schema_cache.save_snapshot(key, snapshot)
                self.metadata_cache_warning = ""
        except ValueError as exc:
            self.metadata_cache_warning = "Metadata cache not saved: " + str(exc)
        except Exception:
            self.metadata_cache_warning = "Metadata cache could not be saved; live metadata remains available."

    def _hydrate_cached_schema(self, source: DataSource) -> None:
        """Attach last-known catalog metadata to an offline source, without I/O."""
        if source.con is not None and not getattr(source.con, "closed", False):
            return
        snapshot = self._cached_schemas.get(self._source_identity(source))
        if snapshot is None or snapshot.dialect != source.dialect:
            return
        hydrated = self._hydrated_schemas.get(source.id)
        if (hydrated is not None and hydrated[0] is source and hydrated[1] is snapshot
                and hydrated[2] is source.datasets and source.routines is snapshot.routines):
            return
        source.datasets = snapshot.datasets()
        source.routines = snapshot.routines
        source.default_schema = snapshot.default_schema
        source.server_version = snapshot.server_version
        self._hydrated_schemas[source.id] = (source, snapshot, source.datasets)

    def schema_cache_info(self, source_id: str | None = None) -> str:
        source = self.sources.get(self.active_id if source_id is None else source_id)
        if source is None or self.source_connected(source.id):
            return ""
        snapshot = self._cached_schemas.get(self._source_identity(source))
        if snapshot is None or snapshot.dialect != source.dialect:
            return ""
        return f"Cached metadata from {snapshot.captured_at} · may be stale"

    def refresh_schema(self) -> None:
        """Replace this session's snapshot; in-flight parses retain the old one."""
        source = self.source
        self._hydrate_cached_schema(source)
        self.schema = SchemaContext(
            tables=tuple(tables_of(source)), fns=tuple(self.fns.values()),
            default_schema=source.default_schema or source.postgres.get("schema"),
            routines=source.routines,
        )
        self._schema_snapshots[source.id] = (source, source.datasets, self.schema)

    def _restore_schema_snapshot(self) -> None:
        """Switch completion targets without rebuilding unchanged catalog metadata."""
        source = self.source
        self._hydrate_cached_schema(source)
        cached = self._schema_snapshots.get(source.id)
        if (cached is not None and cached[0] is source and cached[1] is source.datasets
                and cached[2].routines is source.routines
                and cached[2].fns == tuple(self.fns.values())
                and cached[2].default_schema == (source.default_schema or source.postgres.get("schema"))):
            self.schema = cached[2]
        else:
            self.refresh_schema()


    # -- palette data -------------------------------------------------------


    def column_reference(self, doc: str, dataset: str, column: str, *, offset: int) -> str:
        """Rebind an explorer field through the caret's actual query scope."""
        context = analyze(doc, offset, self.schema, dialect=self.source.dialect)
        for relation in context.relations:
            for field in relation.fields:
                if field.table == dataset and field.column == column:
                    prefix = relation.reference + "." if relation.reference and context.clause != "identifier" else ""
                    return prefix + quote_identifier(field.name)
        if context.clause == "identifier":
            return quote_identifier(column)
        parts = relation_parts(dataset, dialect=self.source.dialect) or (dataset,)
        return ".".join(quote_identifier(part) for part in (*parts, column))

    def completion_usage(self) -> dict[tuple[str, str, str], int]:
        """Rank cached relation/column/function offers by successful SQL history."""
        target = self.source_key()
        cached = self._completion_usage.get(target)
        if cached is not None and cached[0] == len(self.history) and cached[1] == self.schema.tables:
            return cached[2]
        usage: dict[tuple[str, str, str], int] = {}
        for rank, entry in enumerate(self.history):
            try:
                tokens = [token for token in lex_sql(entry.doc, dialect=entry.dialect)
                          if token.kind not in {"whitespace", "comment"}]
            except SqlError:
                continue
            words = {token.text.lower() for token in tokens if token.kind == "word"}
            quoted = {(relation_parts(token.text, dialect=entry.dialect) or (token.text,))[0]
                      for token in tokens if token.kind == "quoted_identifier"}
            for opened in open_tables(entry.doc, dialect=entry.dialect):
                table = self.schema.table_by_name(opened.dataset, dialect=self.source.dialect)
                usage.setdefault(("dataset", "", table.name if table else opened.dataset), rank)
                if table is not None:
                    for column in table.columns:
                        if column.name in quoted or column.name.lower() in words:
                            usage.setdefault(("column", table.name, column.name), rank)
            for token, following in zip(tokens, tokens[1:]):
                if token.kind == "word" and following.text == "(":
                    usage.setdefault(("function", "", token.text), rank)
        self._completion_usage[target] = (len(self.history), self.schema.tables, usage)
        return usage



    # -- saved table-valued functions --------------------------------------

    def save_fn(self, name: str, params_text: str, body: str, doc: str) -> FnDef:
        """Validate expansion offline, persist, then replace the live definition."""
        fn = self.validate_fn(name, params_text, body, doc)
        functions = dict(self.fns)
        functions[fn.name] = fn
        self._save_memory(functions)
        self.fns[fn.name] = fn  # replacing keeps definition order; a new name appends
        self.refresh_schema()
        return fn

    def validate_fn(self, name: str, params_text: str, body: str, doc: str) -> FnDef:
        """Check function lexical/expansion rules, not server syntax or schemas."""
        params, param_types = parse_parameters(params_text)
        fn = FnDef(name=name.strip(), params=params, body=body, doc=doc.strip(), param_types=param_types)
        validate_function(fn, self.fns, dialect=self.source.dialect)
        return fn

    def install_quality_functions(self) -> int:
        """Install this dialect's SQL library atomically, without executing SQL."""
        from d8r.quality import quality_functions

        functions = dict(self.fns)
        existing = {name.casefold(): fn for name, fn in functions.items()}
        additions = []
        for fn in quality_functions(self.source.dialect):
            previous = existing.get(fn.name.casefold())
            if previous is not None:
                if previous != fn:
                    raise SqlError(f'Function "{fn.name}" already exists; rename it before installing the quality library.')
                continue
            additions.append(fn)
            functions[fn.name] = fn
        for fn in additions:
            validate_function(fn, functions, dialect=self.source.dialect)
        if additions:
            self._save_memory(functions)
            self.fns.update({fn.name: fn for fn in additions})
            self.refresh_schema()
        return len(additions)

    def delete_fn(self, name: str) -> None:
        """Persist deletion before removing the function from the live registry."""
        if name not in self.fns:
            return
        functions = dict(self.fns)
        del functions[name]
        self._save_memory(functions)
        del self.fns[name]
        self.refresh_schema()


    def fn_preview(self, name: str, args_text: str = "", *, source_id: str | None = None) -> RunOutcome:
        """Preview one source's function without switching the visible workspace."""
        source = self.sources[self.active_id if source_id is None else source_id]
        functions = dict(self.functions_for(source.id))
        try:
            fn = functions.get(name)
            if fn is None:
                raise SqlError(f"Unknown saved function: {name}")
            sql = function_query(fn, args_text, functions, dialect=source.dialect)
        except SqlError as exc:
            return RunOutcome(status=str(exc), error=str(exc))
        return self._run_statement(sql, f"SELECT * FROM {fn.name}({args_text})", record=False, source=source)

    @property
    def cached_value_columns(self) -> int:
        """The number of successful column reads retained only in memory."""
        with self._value_cache_lock:
            return len(self._value_cache)

    def clear_value_cache(self, source_id: str | None = None) -> None:
        """Forget values and prevent reads already in flight from republishing them."""
        with self._value_cache_lock:
            if source_id is None:
                self._value_cache.clear()
            else:
                target = self.source_key(source_id)
                self._value_cache = {
                    key: entry for key, entry in self._value_cache.items()
                    if key[0] != target
                }
            self.value_cache_epoch += 1

    def cached_values(
        self, source: DataSource, dataset: str, column: str,
    ) -> tuple[list[object], bool] | None:
        """Read the shared typed value cache without touching a connection."""
        with self._value_cache_lock:
            if not self.value_cache_enabled:
                return None
            entry = self._value_cache.get((self._source_identity(source), dataset, column))
            return None if entry is None or entry[0].dialect != source.dialect else entry[1]

    def distinct_values(
        self, source: DataSource, dataset: str, column: str, *, epoch: int | None = None,
    ) -> tuple[list[object], bool]:
        """Inspect one bounded, typed distinct projection, reusing successful reads."""
        with self._value_cache_lock:
            if epoch is not None and epoch != self.value_cache_epoch:
                raise EngineError("Value cache changed before this read started.")
            cached = self.cached_values(source, dataset, column)
            if cached is not None:
                return cached
            limit = self.value_cache_limit
            epoch = self.value_cache_epoch
            enabled = self.value_cache_enabled
        values = column_values(source, dataset, column, limit)
        with self._value_cache_lock:
            if (enabled and self.value_cache_enabled and epoch == self.value_cache_epoch
                    and self.sources.get(source.id) is source):
                self._value_cache[(self._source_identity(source), dataset, column)] = (source, values)
        return values


    # -- building sources (the add-source modal's engine edge) --------------

    def next_source_id(self, hint: str) -> str:
        """A registry id for a new source, kept unique against the live registry."""
        return _source_id(hint, self.sources)


    def build_parquet_source(self, path: str, display: str = "") -> DataSource:
        """Build a lazy local source without registering or persisting anything."""
        return add_parquet_source(path, display)


    def build_postgres_source(
        self, *, host: str, database: str, user: str, password: str,
        port: int = 5432, schema: str = "public", sslmode: str = "prefer", display: str | None = None,
    ) -> DataSource:
        """Build a real PostgreSQL source; Test does not register or persist it."""
        hint = display or database or "postgres-live"
        if password:
            hint = hint.replace(password, "redacted")
        return add_postgres_source(
            self.next_source_id(hint),
            host=host, port=port, database=database, user=user, password=password,
            schema=schema, sslmode=sslmode, display=display,
        )

    def build_d1_source(
        self, *, account_id: str, database: str, api_token: str, display: str = "",
    ) -> DataSource:
        """Build the native HTTP source without registration or credential persistence."""
        hint = display or database or "d1-live"
        if api_token:
            hint = hint.replace(api_token, "redacted")
        return add_d1_source(
            self.next_source_id(hint), account_id=account_id, database=database,
            api_token=api_token, display=display or None,
        )

    def tx_state(self) -> TxState:
        """Read actual connection state, including transactions opened outside Run."""
        return TxState(open=transaction_open(self.source))

    def cancel(self, source_id: str | None = None) -> None:
        """Request cancellation without releasing the shared connection owner."""
        cancel_source(self.sources[self.active_id if source_id is None else source_id])

    def execution_error(
        self, exc: Exception, *, operation: str = "Query",
    ) -> str:
        """Only safe engine/SQL errors may cross the driver/UI boundary."""
        if isinstance(exc, (EngineError, SqlError, CredentialError)):
            return str(exc)
        return f"{operation} failed. Check the SQL, connection and permissions; driver details are withheld to protect credentials."

    def parse(self, doc: str) -> SQLDocument:
        """Expand saved functions and split native SQL; no server validation or IO."""
        return parse_sql(doc, self.fns, dialect=self.source.dialect)

    def compile(self, doc: str) -> RunOutcome:
        """Preview expanded native SQL offline, even for disconnected targets."""
        document = self.parse(doc)
        outcome = RunOutcome(source=self.active_id, dialect=self.source.dialect, doc=doc)
        if document.errors:
            first = document.errors[0]
            outcome.error = f"line {first.line}: {first.message}"
            outcome.status = f"not executed · {outcome.error}"
            return outcome
        outcome.ok = True
        outcome.sql = document.sql
        outcome.status = (f"expanded SQL · {self.source.dialect} · {len(document.statements)} statements"
                          " · server syntax/schema not validated")
        return outcome

    def run(self, doc: str, record: bool = True) -> RunOutcome:
        """Execute the document and return its first outcome; use run_all for all."""
        return self.run_all(doc, record=record)[0]

    def run_all(self, doc: str, record: bool = True) -> list[RunOutcome]:
        """Execute native statements in order, stopping at the first server error.

        Lexical/expansion errors prevent all execution. Native SQL transaction
        boundaries are exactly those authored; successful earlier statements are
        not silently rolled back when a later statement fails.
        """
        document = self.parse(doc)
        if document.errors:
            first = document.errors[0]
            message = f"line {first.line}: {first.message}"
            return [RunOutcome(status=f"not executed · {message}", error=message, doc=doc)]
        if not document.statements:
            return [RunOutcome(ok=True, status="nothing to run", source=self.active_id,
                               dialect=self.source.dialect, doc=doc)]
        if not self.source_connected():
            message = "The saved target is disconnected. Reconnect it or explicitly select another source before running."
            return [RunOutcome(status=message, error=message, doc=doc)]
        authored = split_statements(doc, dialect=self.source.dialect)
        outcomes = []
        for sql, text in zip(document.statements, authored):
            outcome = self._run_statement(sql, text, record=record)
            outcomes.append(outcome)
            if outcome.error:
                break
        return outcomes

    def _run_statement(self, sql: str, doc: str, *, record: bool,
                       source: DataSource | None = None) -> RunOutcome:
        source = self.source if source is None else source
        try:
            result = execute_sql(source, sql, max_rows=min(self.default_rows or PREVIEW_ROW_CAP, PREVIEW_ROW_CAP))
        except Exception as exc:
            message = self.execution_error(exc)
            if not self.source_connected(source.id) and source is self.source:
                self.refresh_schema()
            return RunOutcome(status=message, error=message, source=source.id,
                              dialect=source.dialect, sql=sql, doc=doc)
        self.clear_value_cache(self._source_identity(source))
        schema_warning = ""
        if result["schema_changed"]:
            try:
                refresh_source_schema(source)
                self.cache_source_schema(source)
                schema_warning = self.metadata_cache_warning
                if source is self.source:
                    self.refresh_schema()
            except Exception as exc:
                # The SQL already succeeded; a metadata failure must not invite
                # retrying a write or conceal its successful result/history.
                schema_warning = "schema not refreshed: " + self.execution_error(
                    exc, operation="Schema refresh",
                )
        outcome = RunOutcome(
            ok=True, source=source.id, dialect=source.dialect,
            columns=result["columns"], dtypes=result["dtypes"], rows=result["rows"],
            total=result["total"], ms=result["ms"], sql=result["sql"],
            schema_changed=result["schema_changed"], doc=doc,
        )
        outcome.status = " · ".join(part for part in (
            result["status"], self._status_for(outcome), schema_warning,
        ) if part)
        if record:
            self._record_history(HistoryEntry(
                at=time.strftime("%Y-%m-%d %H:%M:%S%z"), source=source.id,
                dialect=source.dialect, rows=outcome.total, ms=outcome.ms,
                doc=doc, target=self.source_key(source.id),
            ), outcome)
        return outcome

    @staticmethod
    def _status_for(outcome: RunOutcome) -> str:
        status = f"{len(outcome.rows)} rows · {outcome.ms:.1f} ms · {outcome.source} · {outcome.dialect}"
        if outcome.capped:
            status += " · more rows available (preview capped; full count not scanned)"
        return status
