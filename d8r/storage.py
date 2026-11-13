"""Atomic metadata/settings storage; credentials live only in the OS vault.

New directories and files are owner-only on POSIX; Windows uses filesystem ACLs.
PGTUI uses its own data directory and never migrates another application's files.
"""

from __future__ import annotations

import json
import os
import re
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import time
from uuid import UUID

from d8r.ai.client import AIConfig
from d8r.credentials import CredentialError, CredentialVault
from d8r.themes import DEFAULT_THEME, THEMES

# How long a lock or temp file may exist before it is assumed to belong to a
# crashed save. A save is two writes plus one fsync (sub-second), so anything
# older than a minute is debris, not an in-flight save.
_STALE_SECONDS = 60.0


def data_directory() -> Path:
    """Resolve the user data directory without creating it."""
    override = os.environ.get("PGTUI_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".pgtui"


def _object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("invalid JSON constant")


def _record(value: object, keys: set[str], label: str) -> dict:
    if not isinstance(value, dict) or value.keys() != keys:
        raise ValueError(f"{label} must contain exactly {', '.join(sorted(keys))}")
    return value


def _text(value: object, label: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ValueError(f"{label} must be {'a nonempty string' if nonempty else 'a string'}")


def _validate(value: object) -> dict:
    """Validate structure, never parse/execute bodies against the active schema."""
    if isinstance(value, dict):
        value = {"d1_profiles": [], **value}
    document = _record(value, {"version", "functions", "postgres_profiles", "d1_profiles"}, "memory")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("unsupported memory version (expected 1)")
    functions = document["functions"]
    if not isinstance(functions, dict):
        raise ValueError("functions must map source keys to definition lists")
    for key in functions:
        _text(key, "function source key", nonempty=True)
    groups = functions.values()
    for group in groups:
        if not isinstance(group, list):
            raise ValueError("source functions must be a list")
        names = set()
        for function in group:
            fn = _record(function, {"name", "params", "body", "description"}, "function")
            _text(fn["name"], "function name", nonempty=True)
            _text(fn["body"], "function body")
            _text(fn["description"], "function description")
            if fn["name"] in names:
                raise ValueError("duplicate function name")
            names.add(fn["name"])
            if not isinstance(fn["params"], list):
                raise ValueError("function params must be an ordered list")
            for param in fn["params"]:
                _text(param, "function parameter", nonempty=True)
            if len(set(fn["params"])) != len(fn["params"]):
                raise ValueError("duplicate function parameter")
    postgres_profiles = document["postgres_profiles"]
    if not isinstance(postgres_profiles, list):
        raise ValueError("postgres_profiles must be a list")
    identities = set()
    for profile in postgres_profiles:
        _record(profile, {"host", "port", "database", "user", "password", "schema", "sslmode", "display"}, "PostgreSQL profile")
        for key, text in profile.items():
            _text(text, f"PostgreSQL profile {key}", nonempty=key != "password")
        if not profile["port"].isascii() or not profile["port"].isdigit() or not 1 <= int(profile["port"]) <= 65535:
            raise ValueError("PostgreSQL port must be between 1 and 65535")
        if profile["sslmode"] not in {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}:
            raise ValueError("invalid PostgreSQL SSL mode")
        identity = tuple(profile[key] for key in ("host", "port", "database", "user", "schema", "sslmode"))
        if identity in identities:
            raise ValueError("duplicate PostgreSQL profile")
        identities.add(identity)
    if not isinstance(document["d1_profiles"], list):
        raise ValueError("d1_profiles must be a list")
    identities = set()
    for profile in document["d1_profiles"]:
        _record(profile, {"account_id", "database", "api_token", "display"}, "D1 profile")
        for key, text in profile.items():
            _text(text, f"D1 profile {key}", nonempty=key != "api_token")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", profile["account_id"]):
            raise ValueError("D1 account ID must contain 32 hexadecimal characters")
        try:
            database = str(UUID(profile["database"]))
        except ValueError:
            raise ValueError("D1 saved database must be a resolved UUID") from None
        profile = dict(profile, account_id=profile["account_id"].lower(), database=database)
        identity = (profile["account_id"], profile["database"])
        if identity in identities:
            raise ValueError("duplicate D1 profile")
        identities.add(identity)
    document = deepcopy(document)
    for profile in document["d1_profiles"]:
        profile["account_id"] = profile["account_id"].lower()
        profile["database"] = str(UUID(profile["database"]))
        if profile["api_token"]:
            profile["display"] = profile["display"].replace(profile["api_token"], "[redacted]")
    return deepcopy(document)


def _settings(value: object) -> dict:
    """Validate editable settings and fill omitted keys without retaining aliases."""
    defaults = {
        "version": 1, "intellisense": True,
        "entry_screen": False,
        "theme": DEFAULT_THEME,
        "panes": {"results": True, "sql": True, "history": True, "schema": True,
                  "agents": True, "ai": False},
        "pane_sizes": {"explorer": 36, "ai": 44, "editor": 22},
        "source": "postgres",
        "default_rows": 50,
        "value_cache_enabled": True,
        "value_cache_limit": 1000,
        "ai_auto_accept": False,
        "ai_allow_sample_data": False,
        "ai_allow_schema_refreshes": True,
        "ai_test_queries": False,
        "ai_completion": False,
        "ai": asdict(AIConfig()),
    }
    if not isinstance(value, dict) or not value.keys() <= defaults.keys():
        raise ValueError("invalid settings fields")
    document = defaults | value
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("unsupported settings version (expected 1)")
    if type(document["intellisense"]) is not bool:
        raise ValueError("intellisense must be a boolean")
    if type(document["entry_screen"]) is not bool:
        raise ValueError("entry_screen must be a boolean")
    if not isinstance(document["theme"], str) or document["theme"] not in THEMES:
        raise ValueError("Choose a supported theme from Settings → Theme.")
    if type(document["ai_auto_accept"]) is not bool:
        raise ValueError("ai_auto_accept must be a boolean")
    for field in ("ai_allow_sample_data", "ai_allow_schema_refreshes", "ai_test_queries", "ai_completion"):
        if type(document[field]) is not bool:
            raise ValueError(f"{field} must be a boolean")
    if type(document["value_cache_enabled"]) is not bool:
        raise ValueError("Cache column values must be a boolean.")
    if type(document["value_cache_limit"]) is not int or not 1 <= document["value_cache_limit"] <= 10_000:
        raise ValueError("Values per column must be an integer between 1 and 10000.")
    for field in ("panes", "pane_sizes", "ai"):
        provided = document[field]
        if not isinstance(provided, dict) or not provided.keys() <= defaults[field].keys():
            raise ValueError(f"invalid {field} settings fields")
        document[field] = defaults[field] | provided
    if any(type(visible) is not bool for visible in document["panes"].values()):
        raise ValueError("pane visibility must be a boolean")
    if any(type(size) is not int or not 1 <= size <= 1000 for size in document["pane_sizes"].values()):
        raise ValueError("Pane sizes must be integers between 1 and 1000 terminal cells.")
    _text(document["source"], "source", nonempty=True)
    if type(document["default_rows"]) is not int or not 0 <= document["default_rows"] <= 1_000_000:
        raise ValueError("Default rows returned must be an integer between 0 and 1000000 (0 uses the preview cap).")
    ai = document["ai"]
    for field in ("base_url", "model", "api_key"):
        _text(ai[field], f"AI {field}")
    if any(ai[field] for field in ("base_url", "model", "api_key")):
        AIConfig(**ai).validate()
    else:
        AIConfig(**ai).validate_limits()
    return document


def _reference(value: object) -> None:
    if value is not None:
        _text(value, "credential reference", nonempty=True)


def _memory_document(value: object) -> dict:
    if isinstance(value, dict):
        value = {"d1_profiles": [], **value}
    document = _record(value, {"version", "functions", "postgres_profiles", "d1_profiles"}, "memory")
    if type(document["version"]) is not int or document["version"] != 2:
        raise ValueError("unsupported saved memory version (expected 2)")
    runtime = {"version": 1, "functions": document["functions"]}
    for group, secret in (("postgres_profiles", "password"), ("d1_profiles", "api_token")):
        if not isinstance(document[group], list):
            raise ValueError("invalid saved connection profiles")
        profiles = []
        for profile in document[group]:
            if not isinstance(profile, dict) or "credential_ref" not in profile or secret in profile:
                raise ValueError("invalid credential metadata")
            _reference(profile["credential_ref"])
            profiles.append({**{key: item for key, item in profile.items() if key != "credential_ref"}, secret: ""})
        runtime[group] = profiles
    normalized = _validate(runtime)
    document = deepcopy(document)
    for stored, profile in zip(document["d1_profiles"], normalized["d1_profiles"]):
        stored["account_id"], stored["database"] = profile["account_id"], profile["database"]
    return deepcopy(document)


def _settings_document(value: object) -> dict:
    if not isinstance(value, dict) or type(value.get("version")) is not int or value["version"] != 2:
        raise ValueError("unsupported saved settings version (expected 2)")
    ai = value.get("ai")
    if not isinstance(ai, dict) or "api_key_ref" not in ai or "api_key" in ai:
        raise ValueError("invalid AI credential metadata")
    _reference(ai["api_key_ref"])
    document = _settings({**value, "version": 1, "ai": {
        **{key: item for key, item in ai.items() if key != "api_key_ref"}, "api_key": "",
    }})
    document["version"] = 2
    document["ai"].pop("api_key")
    document["ai"]["api_key_ref"] = ai["api_key_ref"]
    return document


class _JSONStore:
    """One loaded snapshot; errors preserve the file and block unsafe writes."""

    def __init__(self, data_dir: Path | None, name: str, default: dict, validate) -> None:
        self.path = (data_directory() if data_dir is None else Path(data_dir)) / f"{name}.json"
        self._name = name
        self._validate = validate
        self.error = ""
        self.document = default
        self._snapshot: bytes | None = None
        try:
            self._snapshot = self._read()
            if self._snapshot is not None:
                self.document = self._validate(json.loads(
                    self._snapshot.decode("utf-8"), object_pairs_hook=_object,
                    parse_constant=_constant,
                ))
        except (OSError, ValueError, RecursionError):
            self.error = (
                f"Cannot load local {self._name} at {self.path}: invalid or unreadable file. "
                "The original file is unchanged; repair it or move it aside, then restart PGTUI. "
                "Local saves are disabled until restart."
            )
        self._sweep()

    def _read(self) -> bytes | None:
        try:
            return self.path.read_bytes()
        except FileNotFoundError:
            return None

    def _sweep(self) -> None:
        """Remove temp files older than the grace period — orphaned debris of
        crashed saves. Fresh ones belong to a save in flight, active or not."""
        deadline = time.time() - _STALE_SECONDS
        try:
            artifacts = list(self.path.parent.glob(f".{self._name}-*.tmp"))
        except OSError:
            return
        for artifact in artifacts:
            try:
                if artifact.stat().st_mtime < deadline:
                    artifact.unlink(missing_ok=True)
            except OSError:
                pass

    def _acquire_lock(self, lock: Path) -> None:
        """Take the save lock, clearing it only when it is older than the
        grace period — a crashed save's lock, not one being held now."""
        for attempt in (0, 1):
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            except FileExistsError:
                if attempt:
                    raise ValueError(
                        f"Local {self._name} is busy at {self.path}. Retry after the other "
                        f"save finishes; a lock left by a crash clears itself after "
                        f"{int(_STALE_SECONDS)} seconds."
                    ) from None
                try:
                    age = time.time() - lock.stat().st_mtime
                except FileNotFoundError:
                    continue  # vanished between checks; retry the open
                if age <= _STALE_SECONDS:
                    raise ValueError(
                        f"Local {self._name} is busy at {self.path}. Retry after the other "
                        f"save finishes; a lock left by a crash clears itself after "
                        f"{int(_STALE_SECONDS)} seconds."
                    ) from None
                try:
                    lock.unlink(missing_ok=True)
                except OSError:
                    pass  # raced another taker; retry the open
                continue
            os.close(descriptor)
            return

    def _save(self, value: dict) -> None:
        """Commit a complete snapshot, only if the loaded file is still current."""
        if self.error:
            raise ValueError(self.error)
        document = self._validate(value)
        try:
            data = (json.dumps(document, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        except (ValueError, UnicodeError, RecursionError):
            raise ValueError(f"Invalid local {self._name}; nothing was saved.") from None
        lock = self.path.with_suffix(".lock")
        temporary: str | None = None
        locked = False
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._acquire_lock(lock)
            locked = True
            if self._read() != self._snapshot:
                raise ValueError(
                    f"Local {self._name} changed outside this session at {self.path}; nothing was saved. "
                    f"Copy your unsaved draft and restart PGTUI to load the latest {self._name}."
                )
            with tempfile.NamedTemporaryFile(
                mode="wb", prefix=f".{self._name}-", suffix=".tmp", dir=self.path.parent, delete=False,
            ) as stream:
                temporary = stream.name
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Detect external edits made while the temporary file was being written.
            if self._read() != self._snapshot:
                raise ValueError(
                    f"Local {self._name} changed outside this session at {self.path}; nothing was saved. "
                    f"Copy your unsaved draft and restart PGTUI to load the latest {self._name}."
                )
            os.replace(temporary, self.path)
            temporary = None
        except OSError:
            raise ValueError(
                f"Cannot save local {self._name} at {self.path}. "
                "Nothing was changed; check directory permissions and free disk space, then retry."
            ) from None
        finally:
            # Cleanup must not turn a successful replace into a reported failed save.
            for artifact in (temporary, lock if locked else None):
                if artifact is not None:
                    try:
                        Path(artifact).unlink(missing_ok=True)
                    except OSError:
                        pass
        self._snapshot = data
        self.document = document


class _CredentialStore(_JSONStore):
    """Keep the validated disk snapshot separate from hydrated runtime values."""

    _missing_warning = "Saved credentials are unavailable in the OS credential vault. Unlock the vault or re-enter credentials; affected connections are blocked. Metadata and references are preserved."

    def _credential_state(self) -> None:
        self._vault = CredentialVault(self.path.parent)
        self._disk = deepcopy(self.document)
        self._secret_values: dict[str, str] = {}
        self.credential_warning = ""

    def _warn(self, message: str) -> None:
        if message not in self.credential_warning:
            self.credential_warning = "\n".join(filter(None, (self.credential_warning, message)))

    def _missing(self) -> None:
        self._warn(self._missing_warning)

    def _clear_missing(self) -> None:
        self.credential_warning = "\n".join(
            line for line in self.credential_warning.splitlines() if line != self._missing_warning
        )

    def _get(self, reference: str | None) -> str | None:
        if reference is None:
            return None
        try:
            value = self._vault.get(reference)
        except CredentialError:
            value = None
        if value is None:
            self._missing()
        else:
            self._secret_values[reference] = value
        return value

    def _discard(self, references: set[str]) -> None:
        for reference in references:
            try:
                self._vault.delete(reference)
            except CredentialError:
                self._warn("An unused credential could not be removed from the OS vault. The committed file and its credentials remain valid.")
            self._secret_values.pop(reference, None)

    def _reference_for(self, payload: str | None, previous: str | None, created: set[str]) -> str | None:
        # An explicit empty secret is keyless authentication, not a lost entry.
        # Persist null so local/passwordless targets need no credential service.
        if payload is None or payload == "":
            return None
        if previous is not None and previous in self._secret_values and self._secret_values[previous] == payload:
            return previous
        reference = self._vault.put(payload)
        created.add(reference)
        self._secret_values[reference] = payload
        return reference

    @staticmethod
    def _references(document: dict) -> set[str]:
        if "ai" in document:
            reference = document["ai"]["api_key_ref"]
            return {reference} if reference is not None else set()
        return {profile["credential_ref"]
                for group in ("postgres_profiles", "d1_profiles") for profile in document[group]
                if profile["credential_ref"] is not None}

    def _commit(self, disk: dict, runtime: dict) -> None:
        previous = self._references(self._disk)
        self._save(disk)
        self._disk = self.document
        self.document = deepcopy(runtime)
        self._discard(previous - self._references(self._disk))


class MemoryStore(_CredentialStore):
    """Source-scoped SQL functions and native connection metadata with vault references."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__(data_dir, "memory", {
            "version": 2, "functions": {}, "postgres_profiles": [], "d1_profiles": [],
        }, _memory_document)
        self._credential_state()
        runtime = {**deepcopy(self.document), "version": 1, "postgres_profiles": [], "d1_profiles": []}
        if not self.error:
            for group, secret in (("postgres_profiles", "password"), ("d1_profiles", "api_token")):
                for stored in self._disk[group]:
                    reference = stored["credential_ref"]
                    value = self._get(reference)
                    profile = {key: item for key, item in stored.items() if key != "credential_ref"}
                    profile[secret] = value if value is not None else ""
                    if value is None and (reference is not None or group == "d1_profiles"):
                        profile["credentials_missing"] = True
                        self._missing()
                    runtime[group].append(profile)
        self.document = runtime

    @staticmethod
    def _identity(profile: dict) -> tuple[str, ...]:
        keys = (("account_id", "database") if "account_id" in profile else
                ("host", "port", "database", "user", "schema", "sslmode"))
        return tuple(profile[key] for key in keys)

    @staticmethod
    def validate_profile(profile: dict) -> dict:
        return _validate({"version": 1, "functions": {}, "postgres_profiles": [profile]})["postgres_profiles"][0]

    @staticmethod
    def validate_d1_profile(profile: dict) -> dict:
        validated = _validate({"version": 1, "functions": {}, "postgres_profiles": [],
                               "d1_profiles": [profile]})["d1_profiles"][0]
        _text(validated["api_token"], "D1 API token", nonempty=True)
        return validated

    def save(self, functions: dict[str, list[dict]],
             postgres_profiles: list[dict[str, str]] | None = None,
             d1_profiles: list[dict[str, str]] | None = None) -> None:
        if self.error:
            raise ValueError(self.error)
        supplied = {"postgres_profiles": self.document["postgres_profiles"] if postgres_profiles is None else postgres_profiles,
                    "d1_profiles": self.document["d1_profiles"] if d1_profiles is None else d1_profiles}
        complete = {"version": 1, "functions": functions}
        # Validate both groups before creating any vault entries.
        for group, profiles in supplied.items():
            if not isinstance(profiles, list):
                raise ValueError("invalid saved connection profiles")
            complete[group] = []
            for profile in profiles:
                if isinstance(profile, dict) and profile.get("credentials_missing"):
                    if profile not in self.document[group]:
                        raise CredentialError("Re-enter missing credentials before changing this connection.")
                    complete[group].append({key: value for key, value in profile.items() if key != "credentials_missing"})
                else:
                    complete[group].append(profile)
        normalized = _validate(complete)
        runtime = {"version": 1, "functions": normalized["functions"], "postgres_profiles": [], "d1_profiles": []}
        disk = {**runtime, "version": 2, "postgres_profiles": [], "d1_profiles": []}
        pending = []
        for group, secret in (("postgres_profiles", "password"), ("d1_profiles", "api_token")):
            old = {self._identity(profile): profile for profile in self._disk[group]}
            for original, profile in zip(supplied[group], normalized[group]):
                previous = old.get(self._identity(profile), {})
                if original.get("credentials_missing"):
                    metadata = deepcopy(previous)
                    profile["credentials_missing"] = True
                else:
                    if group == "d1_profiles":
                        _text(profile[secret], "D1 API token", nonempty=True)
                    metadata = {key: value for key, value in profile.items() if key != secret}
                    metadata["credential_ref"] = None
                    pending.append((metadata, profile[secret], previous.get("credential_ref")))
                disk[group].append(metadata)
                runtime[group].append(profile)
        _memory_document(disk)
        created: set[str] = set()
        try:
            for metadata, payload, previous in pending:
                metadata["credential_ref"] = self._reference_for(payload, previous, created)
            self._commit(disk, runtime)
        except Exception:
            self._discard(created)
            raise
        if not any(profile.get("credentials_missing")
                   for group in ("postgres_profiles", "d1_profiles") for profile in runtime[group]):
            self._clear_missing()


class SettingsStore(_CredentialStore):
    """Editable preferences; provider API keys never enter the JSON snapshot."""

    def __init__(self, data_dir: Path | None = None) -> None:
        default = _settings({})
        default["version"] = 2
        default["ai"].pop("api_key")
        default["ai"]["api_key_ref"] = None
        super().__init__(data_dir, "settings", default, _settings_document)
        self._credential_state()
        self.ai_credentials_missing = False
        if self.error:
            self.document = _settings({})
            return
        reference = self.document["ai"]["api_key_ref"]
        value = self._get(reference)
        self.ai_credentials_missing = reference is not None and value is None
        ai = {key: item for key, item in self.document["ai"].items() if key != "api_key_ref"}
        self.document = {**self.document, "version": 1, "ai": {**ai, "api_key": value if value is not None else ""}}

    def save(self, document: dict, *, replace_ai: bool = False) -> None:
        if self.error:
            raise ValueError(self.error)
        runtime = _settings(document)
        ai = runtime["ai"]
        unchanged = not replace_ai and ai == self.document["ai"]
        public = {key: value for key, value in ai.items() if key != "api_key"}
        disk = {**runtime, "version": 2, "ai": {**public, "api_key_ref": None}}
        created: set[str] = set()
        try:
            if unchanged:
                disk["ai"]["api_key_ref"] = self._disk["ai"]["api_key_ref"]
            elif ai["base_url"] or ai["model"] or ai["api_key"]:
                previous = self._disk["ai"]["api_key_ref"] if public == {
                    key: value for key, value in self._disk["ai"].items() if key != "api_key_ref"
                } else None
                disk["ai"]["api_key_ref"] = self._reference_for(ai["api_key"], previous, created)
            self._commit(disk, runtime)
        except Exception:
            self._discard(created)
            raise
        if not unchanged:
            self.ai_credentials_missing = False
            self._clear_missing()


def _cursor(value: object) -> None:
    if not isinstance(value, list) or len(value) != 2 or any(type(item) is not int or item < 0 for item in value):
        raise ValueError("invalid workspace cursor")


def _chat_messages(messages: object) -> None:
    """Only complete exchanges may be replayed; loading never executes tools."""
    if not isinstance(messages, list):
        raise ValueError("chat messages must be a list")
    next_role = "user"
    pending: set[str] = set()
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("invalid chat message")
        role = message.get("role")
        if role != next_role:
            raise ValueError("incomplete chat exchange")
        if role == "user":
            _record(message, {"role", "content"}, "user message")
            _text(message["content"], "user message")
            next_role = "assistant"
        elif role == "assistant":
            calls = message.get("tool_calls")
            if calls is None:
                _record(message, {"role", "content"}, "assistant message")
                _text(message["content"], "assistant message", nonempty=True)
                next_role = "user"
            else:
                _record(message, {"role", "content", "tool_calls"}, "assistant tool message")
                if message["content"] is not None:
                    _text(message["content"], "assistant message")
                if not isinstance(calls, list) or not calls:
                    raise ValueError("invalid saved tool calls")
                for call in calls:
                    _record(call, {"id", "type", "function"}, "tool call")
                    _text(call["id"], "tool call id", nonempty=True)
                    if call["type"] != "function" or call["id"] in pending:
                        raise ValueError("invalid saved tool call")
                    function = _record(call["function"], {"name", "arguments"}, "tool function")
                    _text(function["name"], "tool function name", nonempty=True)
                    _text(function["arguments"], "tool arguments")
                    pending.add(call["id"])
                next_role = "tool"
        else:
            _record(message, {"role", "tool_call_id", "content"}, "tool result")
            _text(message["tool_call_id"], "tool result id", nonempty=True)
            _text(message["content"], "tool result")
            if message["tool_call_id"] not in pending:
                raise ValueError("unmatched saved tool result")
            pending.remove(message["tool_call_id"])
            if not pending:
                next_role = "assistant"
    if next_role != "user":
        raise ValueError("incomplete saved conversation")


def _function_draft(draft: object) -> None:
    if draft is None:
        return
    _record(draft, {"identity", "source", "selected", "name", "description", "parameters",
                    "body", "arguments", "cursor"}, "function draft")
    for name, text in draft.items():
        if name != "cursor":
            _text(text, f"function draft {name}")
    _cursor(draft["cursor"])


def _d1_source_key(key: str) -> bool:
    return re.fullmatch(r"d1:[0-9a-f]{32}:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", key) is not None


def _workspace_document(value: object) -> dict:
    if isinstance(value, dict):
        value = {"pages": [], "explorer_tab": "schema", "source_workspaces": {}, **value}
    document = _record(value, {
        "version", "document", "document_id", "cursor", "source", "dialect", "sources",
        "active_view", "function_draft", "chats", "history", "pages", "explorer_tab", "source_workspaces",
    }, "workspace")
    if type(document["version"]) is not int or document["version"] != 1:
        raise ValueError("unsupported workspace version (expected 1)")
    if document["document"] is not None:
        _text(document["document"], "workspace document")
    for name in ("document_id", "source", "dialect"):
        _text(document[name], f"workspace {name}")
    if document["dialect"] and document["dialect"] not in {"postgres", "duckdb", "sqlite"}:
        raise ValueError("unsupported workspace dialect")
    if document["explorer_tab"] not in ("schema", "pages", "functions", "agents"):
        raise ValueError("invalid explorer tab")
    if not isinstance(document["pages"], list):
        raise ValueError("invalid query pages")
    page_ids: set[str] = set()
    for page in document["pages"]:
        if isinstance(page, dict):
            page.setdefault("description", "")
        _record(page, {"id", "title", "description", "document", "cursor", "source", "dialect"}, "query page")
        for name in ("id", "title", "description", "document", "source", "dialect"):
            _text(page[name], f"query page {name}", nonempty=name == "id")
        if page["id"] in page_ids:
            raise ValueError("duplicate query page identity")
        page_ids.add(page["id"])
        if page["dialect"] and page["dialect"] not in {"postgres", "duckdb", "sqlite"}:
            raise ValueError("unsupported query page dialect")
        _cursor(page["cursor"])
    _cursor(document["cursor"])
    if document["active_view"] not in ("workspace", "function"):
        raise ValueError("invalid workspace view")
    if not isinstance(document["sources"], dict):
        raise ValueError("invalid workspace sources")
    for key, source in document["sources"].items():
        _text(key, "source key", nonempty=True)
        _record(source, {"id", "display", "kind", "path"}, "workspace source")
        for name, text in source.items():
            _text(text, f"workspace source {name}")
        if source["kind"] not in {"postgres-live", "parquet", "d1-live", "disconnected"}:
            raise ValueError("unsupported workspace source kind")
        if source["kind"] == "d1-live" and (not _d1_source_key(key) or source["path"]):
            raise ValueError("invalid D1 workspace source")
    _function_draft(document["function_draft"])
    if not isinstance(document["source_workspaces"], dict):
        raise ValueError("invalid source workspaces")
    for key, workspace in document["source_workspaces"].items():
        _text(key, "workspace source key", nonempty=True)
        _record(workspace, {"document_id", "function_draft", "active_view"}, "source workspace")
        _text(workspace["document_id"], "source workspace document_id")
        if workspace["active_view"] not in ("workspace", "function"):
            raise ValueError("invalid source workspace view")
        _function_draft(workspace["function_draft"])
        if workspace["function_draft"] is not None and workspace["function_draft"]["source"] != key:
            raise ValueError("function draft belongs to a different source workspace")
    if not isinstance(document["chats"], dict):
        raise ValueError("invalid saved chats")
    for key, chat in document["chats"].items():
        _text(key, "chat key", nonempty=True)
        legacy = {"messages", "transcript", "input", "turns", "visible"}
        if isinstance(chat, dict) and chat.keys() == legacy:
            _chat_messages(chat["messages"])
            # The old target key remains a stable chat id; subsequent chats use
            # UUIDs. Migration adds only inert metadata, never a runnable job.
            chat = {**chat, "target_key": key, "title": next(
                (message["content"][:80] for message in chat["messages"]
                 if isinstance(message, dict) and message.get("role") == "user"), "New chat"),
                "status": "idle", "status_text": "Conversation restored; send for a fresh proposal.",
                "unread": False, "selected": True}
            document["chats"][key] = chat
        _record(chat, legacy | {"target_key", "title", "status", "status_text", "unread", "selected"}, "chat")
        _chat_messages(chat["messages"])
        for field in ("transcript", "input", "target_key", "title", "status_text"):
            _text(chat[field], f"chat {field}")
        if (type(chat["turns"]) is not int or chat["turns"] < 0
                or any(type(chat[field]) is not bool for field in ("visible", "unread", "selected"))
                or chat["status"] not in ("idle", "working", "error", "cancelled")):
            raise ValueError("invalid saved chat state")
    if not isinstance(document["history"], list):
        raise ValueError("invalid query history")
    for entry in document["history"]:
        if isinstance(entry, dict):
            entry.setdefault("target", "")
        _record(entry, {"at", "source", "dialect", "rows", "ms", "doc", "target"}, "query history entry")
        for field in ("at", "source", "dialect", "doc", "target"):
            _text(entry[field], f"history {field}")
        if entry["dialect"] not in {"postgres", "duckdb", "sqlite"}:
            raise ValueError("unsupported history dialect")
        if type(entry["rows"]) is not int or entry["rows"] < 0:
            raise ValueError("invalid history row count")
        if type(entry["ms"]) not in (int, float) or not 0 <= entry["ms"] < float("inf"):
            raise ValueError("invalid history duration")
    for target, dialect in [(document["source"], document["dialect"]),
                            *((page["source"], page["dialect"]) for page in document["pages"]),
                            *((entry["target"], entry["dialect"]) for entry in document["history"])]:
        source = document["sources"].get(target, {})
        if dialect == "sqlite" and (not _d1_source_key(target)
                                    or source.get("kind", "d1-live") not in {"d1-live", "disconnected"}):
            raise ValueError("SQLite workspace metadata requires a D1 target")
    return document


class WorkspaceStore(_JSONStore):
    """Autosaved drafts and histories; never executable state or credentials."""

    def __init__(self, data_dir: Path | None = None) -> None:
        super().__init__(data_dir, "workspace", {
            "version": 1, "document": None, "document_id": "", "cursor": [0, 0],
            "source": "", "dialect": "", "sources": {}, "active_view": "workspace",
            "function_draft": None, "chats": {}, "history": [],
            "pages": [], "explorer_tab": "schema", "source_workspaces": {},
        }, _workspace_document)

    def save(self, document: dict) -> None:
        self._save(document)
