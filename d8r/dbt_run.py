"""Plan explicit dbt commands and consume only their own structured results."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from d8r.dbt import DbtError, DbtProject, _MAX_ARTIFACT, _artifact_directory, _linked, _read_artifact, _safe_root, _yaml
from d8r.dbt_files import DbtFile, DbtFiles
from d8r.dbt_profiles import DbtTarget, DbtTargetSettings
from d8r.dbt_vars import validate_run_vars

DbtModelCommand = Literal["run", "run-with-deps", "build", "test", "compile", "debug", "show"]
_CLI = (sys.executable, "-I", "-c", "from dbt.cli.main import cli; cli()")
_SELECTOR_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,255}")
_NODE_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,511}")
_INVOCATION_ID = re.compile(r"[a-f0-9]{8}(?:-[a-f0-9]{4}){3}-[a-f0-9]{12}", re.IGNORECASE)
_ARTIFACTS = ("manifest.json", "run_results.json")
_EVENT_LIMIT = 1024 * 1024
_PREVIEW_MAX = 100
_CELL_MAX = 4096


@dataclass(frozen=True)
class DbtRuntime:
    label: str
    argv: tuple[str, ...]
    environment: Path | None = None


def discover_dbt_project(path: str | Path) -> Path:
    """Find the nearest dbt project without resolving away a refused link."""
    directory = _safe_root(path)
    for candidate in (directory, *directory.parents):
        config = candidate / "dbt_project.yml"
        if _linked(config):
            raise DbtError("dbt_project.yml cannot be a symlink or junction.")
        if config.is_file():
            return candidate
    raise DbtError("Open a directory at or below a local dbt_project.yml.")


def _native_dbt(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        if os.name != "nt":
            return os.access(path, os.X_OK)
        if path.suffix.lower() != ".exe":
            return False  # In particular do not let CreateProcess run a batch shim.
        with path.open("rb") as handle:
            header = handle.read(64)
            if len(header) != 64 or header[:2] != b"MZ":
                return False
            handle.seek(int.from_bytes(header[60:64], "little"))
            pe = handle.read(24)
        return (len(pe) == 24 and pe[:4] == b"PE\0\0" and
                int.from_bytes(pe[22:24], "little") & 0x2002 == 0x0002)
    except OSError:
        return False


def _dbt_in(directory: Path) -> Path | None:
    candidates = ((directory / "Scripts" / "dbt.exe", directory / "dbt.exe") if os.name == "nt"
                  else (directory / "bin" / "dbt", directory / "dbt"))
    return next((candidate for candidate in candidates if _native_dbt(candidate)), None)


def _absolute(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    # A drive-relative Windows path is neither a project-relative nor absolute path.
    if path.anchor and not path.is_absolute():
        raise ValueError
    return Path(os.path.abspath(root / path))


def _environment_of(executable: Path) -> Path | None:
    parent = executable.parent
    for candidate in (parent, parent.parent):
        if (candidate / "pyvenv.cfg").is_file() or (candidate / "conda-meta").is_dir():
            return candidate
    return None


def find_dbt_runtime(root: Path, override: str = "") -> DbtRuntime:
    """Resolve a dbt command with filesystem lookups only (no version or activation probes)."""
    if override:
        try:
            path = _absolute(root, override)
            environment = path if path.is_dir() else None
            executable = _dbt_in(path) if environment else path if _native_dbt(path) else None
            if executable:
                return DbtRuntime(str(executable), (str(executable),), environment or _environment_of(executable))
        except (OSError, RuntimeError, ValueError):
            pass
        raise DbtError("Invalid dbt runtime override. Choose an environment containing dbt or a dbt executable"
                       + (" (.exe on Windows)." if os.name == "nt" else "."))

    directories = [root / ".venv", root / "venv"]
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        if value := os.environ.get(key):
            try:
                directories.append(_absolute(root, value))
            except (OSError, RuntimeError, ValueError):
                continue
    for directory in directories:
        if executable := _dbt_in(directory.absolute()):
            return DbtRuntime(str(executable), (str(executable),), directory.absolute())
    filename = "dbt.exe" if os.name == "nt" else "dbt"
    for entry in os.get_exec_path():
        if not entry:
            continue  # Do not implicitly search the current directory.
        try:
            executable = _absolute(root, entry.strip('"') if os.name == "nt" else entry) / filename
            if _native_dbt(executable):
                return DbtRuntime(str(executable), (str(executable),), _environment_of(executable))
        except (OSError, RuntimeError, ValueError):
            continue
    return DbtRuntime("Bundled dbt Core", _CLI)


def dbt_named_selectors(home: str) -> tuple[str, ...]:
    project = DbtFiles(discover_dbt_project(home))
    try:
        document = project.read("selectors.yml", missing=True)
        if not document.revision:
            return ()
        data = _yaml(document.text, project.root / "selectors.yml")
        entries = data.get("selectors") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise ValueError
        names: dict[str, None] = {}
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not _SELECTOR_NAME.fullmatch(name) or name in names:
                raise ValueError
            names[name] = None
        return tuple(names)
    except (ValueError, yaml.YAMLError, RecursionError):
        raise DbtError("Cannot read named selectors. Check selectors.yml for a selectors list with unique literal names.") from None


@dataclass(frozen=True)
class DbtNodeResult:
    unique_id: str
    status: str
    reason: str
    path: str = ""
    row: int | None = None


@dataclass(frozen=True)
class DbtRunResult:
    exit_code: int
    nodes: tuple[DbtNodeResult, ...] = ()
    compiled_sql: str = ""
    preview: tuple[dict[str, Any], ...] = ()
    message: str = ""
    artifact_revision: tuple = ()
    preview_available: bool = False


@dataclass(frozen=True)
class DbtRun:
    root: Path
    verb: str
    selector: str
    target: DbtTarget | None
    artifact_dir: Path
    log_dir: Path
    runtime: DbtRuntime
    model_path: str = ""
    source_revision: str = ""
    named_selector: str = ""
    state_path: Path | None = None
    defer: bool = False
    preview_sql: str = ""
    limit: int = 50
    profiles_dir: Path | None = None  # Fallback for debug without a resolved target.
    vars_override: str = field(default="", repr=False)

    @property
    def args(self) -> tuple[str, ...]:
        return self._args(self.artifact_dir, self.log_dir)

    def _args(self, artifact_dir: Path, log_dir: Path) -> tuple[str, ...]:
        # Older Core releases only accept these global options before the command.
        args = ["--log-path", str(log_dir), "--log-format", "json", "--log-level", "info", "--no-use-colors",
                "--no-send-anonymous-usage-stats"]
        if self.verb != "debug":
            args.append("--no-write-json" if self.verb == "show" else "--write-json")
        if self.verb in {"show", "compile"}:
            args.append("--no-partial-parse")
        args.extend((self.verb, "--project-dir", str(self.root)))
        if self.target:
            args.extend(("--profiles-dir", str(self.target.profiles_dir), "--profile", self.target.profile,
                         "--target", self.target.name))
        elif self.profiles_dir:
            args.extend(("--profiles-dir", str(self.profiles_dir)))
        if self.vars_override:
            args.extend(("--vars", self.vars_override))
        if self.verb == "debug":
            return tuple(args)
        args.extend(("--target-path", str(artifact_dir), "--defer" if self.defer else "--no-defer"))
        if self.state_path:
            args.extend(("--state", str(self.state_path)))
        if self.verb == "show":
            args.extend(("--inline", _literal_sql(self.preview_sql), "--output", "json", "--limit", str(self.limit)))
        else:
            args.extend(("--selector" if self.named_selector else "--select", self.selector))
        return tuple(args)

    def _env(self) -> dict[str, str]:
        env = dict(os.environ)
        for name in ("DBT_STATE", "DBT_ARTIFACT_STATE_PATH", "DBT_DEFER_STATE", "DBT_DEFER_TO_STATE",
                     "DBT_FAVOR_STATE", "DBT_PARTIAL_PARSE_FILE_PATH", "DBT_EXCLUDE", "DBT_SELECT",
                     "DBT_MODELS", "DBT_QUIET", "DBT_INLINE", "DBT_INLINE_DIRECT", "DBT_ENABLE_LEGACY_LOGGER"):
            env.pop(name, None)
        env["DBT_SEND_ANONYMOUS_USAGE_STATS"] = "false"
        # Older Core ignores this env var; unlike a CLI option it cannot break startup.
        env["DBT_UPLOAD_TO_ARTIFACTS_INGEST_API"] = "false"
        if directory := self.runtime.environment:
            bins = ((directory / "Scripts", directory / "Library" / "bin", directory / "Library" / "mingw-w64" / "bin",
                     directory / "Library" / "usr" / "bin", directory) if os.name == "nt" else (directory / "bin",))
            path = env.get("PATH", os.defpath)
            env["PATH"] = os.pathsep.join((*map(str, bins), path))
        return env

    async def execute(self, progress: Callable[[str], None] | None = None) -> DbtRunResult:
        events = _Events(self.verb, self.limit, progress)
        if self.verb == "show":
            if self.model_path:
                await asyncio.to_thread(_verify_show_source, self)
            # Keep even the partial-parse cache, compiled SQL and Core's own log away
            # from the model artifacts (older versions also differ on write-json).
            try:
                with tempfile.TemporaryDirectory(prefix="pgtui-dbt-") as directory:
                    temp = Path(directory)
                    code = await _execute(self, self._args(temp / "target", temp / "logs"), events)
            except OSError:
                raise DbtError("Cannot use a temporary directory for dbt preview. Check its permissions.") from None
            if not code and self.model_path:
                await asyncio.to_thread(_verify_show_source, self)
            return _show_result(code, events)
        if self.verb == "debug":
            await asyncio.to_thread(_check_output_files, self.log_dir / "dbt.log")
            code = await _execute(self, self.args, events)
            message = (events.failure() if code else "dbt debug passed; the target connection works."
                       if events.debug_passed and not events.conflict else "dbt returned no verified connection check.")
            return DbtRunResult(code, message=message)

        before = await asyncio.to_thread(_before_run, self)
        code = await _execute(self, self.args, events)
        return await asyncio.to_thread(_run_result, self, before, code, events)


def _literal_sql(sql: str) -> str:
    # Jinja compiles the expression once. Delimiters in its returned string are
    # SQL data, not a second Jinja template; unlike a raw block this also handles
    # SQL containing a literal '{% endraw %}'.
    return "{{ " + json.dumps(sql, ensure_ascii=False) + " }}"


def _check_state(root: Path, output: Path, value: str, defer: bool) -> Path | None:
    if not value:
        if defer:
            raise DbtError("Defer requires a local state directory with manifest.json.")
        return None
    try:
        state = _absolute(root, value)
        if (str(state.anchor).startswith("\\\\") or not state.is_dir() or
                any(_linked(item) for item in (state, *state.parents))):
            raise DbtError("Select a real local dbt state directory, not a link or network path.")
        if state == output or output.exists() and state.samefile(output):
            raise DbtError("State input and dbt artifact output must use different directories.")
        if defer:
            manifest = state / "manifest.json"
            if _linked(manifest):
                raise DbtError("The defer manifest must be a regular local manifest.json, not a link.")
            try:
                data = _read_artifact(manifest)
            except (OSError, DbtError):
                data = None
            schema = _object(_object(data).get("metadata")).get("dbt_schema_version")
            if (data is None or not isinstance(schema, str) or not _MANIFEST_SCHEMA.fullmatch(schema)
                    or not isinstance(data.get("nodes"), dict)):
                raise DbtError("Defer requires a readable dbt manifest.json in the state directory.")
        return state
    except (OSError, RuntimeError):
        raise DbtError("Cannot read the dbt state directory. Check its path and permissions.") from None
    except ValueError as exc:
        if isinstance(exc, DbtError):
            raise
        raise DbtError("Choose an absolute or project-relative dbt state directory.") from None


def prepare_dbt_run(home: str, path: str, revision: str, settings: DbtTargetSettings,
                    command: DbtModelCommand, *, runtime_path: str = "", named_selector: str = "",
                    state_path: str = "", defer: bool = False, preview_sql: str = "", limit: int = 50,
                    vars_override: str = "") -> DbtRun:
    vars_override = validate_run_vars(vars_override)
    project = DbtFiles(discover_dbt_project(home))
    if project.config_error:
        raise DbtError("Invalid dbt_project.yml. Fix its YAML before running dbt.")
    runtime = find_dbt_runtime(project.root, runtime_path)
    try:
        target = settings.resolve(project.root, project.config)
    except DbtError:
        raise DbtError("Cannot select a dbt target. Check the profiles.yml path, YAML, profile and target settings.") from None
    if target is None and command != "debug":
        raise DbtError("No dbt profile found. Check profile: in dbt_project.yml and Path to dbt Profiles in Workspace Settings.")
    try:
        artifact_dir = _artifact_directory(project.config, target)
        # Don't publish override-specific compiled SQL or relations as workspace defaults.
        if vars_override and command not in {"debug", "show"}:
            artifact_dir /= ".pgtui-vars"
        output = project._path(artifact_dir.as_posix())
        logs = project._path(project.config.get("log-path", "logs"))
    except DbtError:
        raise DbtError("Set target-path and log-path to safe project-relative directories in dbt_project.yml.") from None
    selector = ""
    model_path = ""
    if command in {"show", "debug"} and (named_selector or state_path or defer):
        raise DbtError("Selectors and state cannot be used with dbt debug or inline preview.")
    if command == "show":
        if not preview_sql.strip() or "\0" in preview_sql:
            raise DbtError("Preview requires nonempty compiled SQL.")
        if len(preview_sql) > 24_000 or len(_literal_sql(preview_sql).encode("utf-16-le", "surrogatepass")) > 48_000:
            raise DbtError("This SQL is too long for an inline dbt preview. Preview a smaller query.")
        if type(limit) is not int or not 1 <= limit <= _PREVIEW_MAX:
            raise DbtError(f"The dbt preview limit must be between 1 and {_PREVIEW_MAX} rows.")
    if named_selector:
        if command not in {"run", "build", "test"} or path or revision:
            raise DbtError("Use a named selector without a model for dbt run, build or test.")
        if named_selector not in dbt_named_selectors(str(project.root)):
            raise DbtError("Choose a named selector from selectors.yml.")
        selector = named_selector
    elif command != "debug" and (command != "show" or path or revision):
        model = project.model(path)
        if model is None:
            raise DbtError("Open a dbt model under the project's configured model-paths.")
        model_path = path
        _check_source(project, model_path, revision)
        if command != "show":
            selector = f"{model.name}.v{model.version}" if model.version else model.name
            if not _SELECTOR_NAME.fullmatch(selector):
                raise DbtError("This model name cannot be used as a literal dbt model selector.")
            if command == "compile" and Path(model_path).suffix.lower() != ".sql":
                raise DbtError("Compiled SQL is available only for SQL models.")
            if command == "run-with-deps":
                selector = "+" + selector
    state = _check_state(project.root, output, state_path, defer)
    debug_profiles = None
    if target is None:
        # Mirror settings discovery: avoid running debug against a stray global profile.
        debug_profiles = project.root if (project.root / "profiles.yml").is_file() else Path.home() / ".dbt"
    return DbtRun(project.root, "run" if command == "run-with-deps" else command, selector, target, output,
                  logs, runtime, model_path, revision, named_selector, state, defer, preview_sql, limit, debug_profiles,
                  vars_override=vars_override)


def _verify_show_source(run: DbtRun) -> None:
    _check_source(DbtFiles(run.root), run.model_path, run.source_revision)


def _check_source(project: DbtFiles, path: str, revision: str) -> DbtFile:
    try:
        document = project.read(path)
        if document.revision == revision:
            return document
    except DbtError:
        pass
    raise DbtError("The model changed on disk. Reload it before running dbt.")


def _object(value: object) -> dict:
    return value if isinstance(value, dict) else {}


def _classification(value: object) -> str:
    """Recognize common error classes without ever returning matching input."""
    if not isinstance(value, str):
        return ""
    text = value[:8192].casefold()
    if "env var required" in text or "required but not provided" in text and "env var" in text:
        return "Set the required environment variables in the shell that starts pgtui."
    if "could not find adapter type" in text or "error importing adapter" in text or "invalid adapter type" in text:
        return "Install the profile's dbt adapter in the selected dbt runtime."
    if "password authentication failed" in text or "authentication failed" in text or "invalid credentials" in text:
        return "Authentication failed. Check the target credentials and environment variables."
    if any(token in text for token in ("connection refused", "connection timed out", "could not translate host name",
                                       "could not resolve host", "network is unreachable")):
        return "Could not reach the database. Check the target host, port and network access."
    if "certificate verify failed" in text or "ssl error" in text:
        return "Check the database target's TLS settings and certificates."
    if "permission denied" in text or "insufficient privilege" in text:
        return "Check the target role's privileges on the database objects used by this node."
    if "syntax error" in text:
        return "Check SQL syntax in the node and its macros."
    if "dbt deps" in text and ("packages" in text or "dependencies" in text):
        return "Install the project's dependencies with dbt deps in the selected runtime."
    return ""


_PROGRESS = {
    ("MainReportVersion", "A001"): "dbt Core started.",
    ("AdapterRegistered", "E034"): "Database adapter loaded.",
    ("FoundStats", "W006"): "Project graph loaded.",
    ("StatsLine", "Z023"): "dbt finished processing the selection.",
}
_START_EVENT = ("MainReportVersion", "A001")
_SHOW_EVENT = ("ShowNode", "Q041")
_DEBUG_EVENT = ("DebugCmdOut", "Z047")
_DEBUG_RESULT = ("DebugCmdResult", "Z048")
_CORE_ERROR = ("MainEncounteredError", "Z002")
_PROFILE_ERROR = ("LogDbtProfileError", "A011")
_PROJECT_ERROR = ("LogDbtProjectError", "A009")
_SETUP_PROFILE = "Check the selected profile and target in profiles.yml, and the shell environment."
_SETUP_PROJECT = "Check dbt_project.yml and the project's required dbt Core version."
_SETUP_CONNECTION = "Could not connect to the database. Check the target credentials and network access."


class _Events:
    def __init__(self, verb: str, limit: int, progress: Callable[[str], None] | None) -> None:
        self.verb = verb
        self.limit = limit
        self.progress = progress
        self.invocation = ""
        self.conflict = False
        self.preview: tuple[dict[str, Any], ...] | None = None
        self.seen_preview = False
        self.debug_passed = False
        self.issues: dict[str, None] = {}
        self.last_progress = ""

    def _progress(self, text: str) -> None:
        if self.progress is not None and text != self.last_progress:
            self.last_progress = text
            self.progress(text)

    def feed(self, line: bytes) -> None:
        if len(line) > _EVENT_LIMIT:
            return
        try:
            event = _object(_json(line))
        except (ValueError, RecursionError):
            return
        info, data = _object(event.get("info")), _object(event.get("data"))
        name, code, invocation = info.get("name"), info.get("code"), info.get("invocation_id")
        if (not isinstance(name, str) or not isinstance(code, str) or not isinstance(invocation, str)
                or not _INVOCATION_ID.fullmatch(invocation)):
            return
        key = name, code
        if key == _START_EVENT:
            if self.invocation and self.invocation != invocation:
                self.conflict = True
            self.invocation = invocation
        if self.conflict or not self.invocation or invocation != self.invocation:
            return
        if key in _PROGRESS:
            self._progress(_PROGRESS[key])
        elif key == ("LogStartLine", "Q011"):
            index, total = data.get("index"), data.get("total")
            if type(index) is int and type(total) is int and 0 < index <= total <= 1_000_000:
                self._progress(f"Processing node {index:,} of {total:,}…")
        elif key == _CORE_ERROR:
            if reason := _classification(data.get("exc")):
                self.issues[reason] = None
        elif key in {_PROFILE_ERROR, _PROJECT_ERROR}:
            reason = _classification(data.get("exc")) or (_SETUP_PROFILE if key == _PROFILE_ERROR else _SETUP_PROJECT)
            self.issues[reason] = None
        elif self.verb == "debug" and key in {_DEBUG_EVENT, _DEBUG_RESULT}:
            self._debug(data.get("msg"), key == _DEBUG_RESULT)
        elif self.verb == "show" and key == _SHOW_EVENT and data.get("is_inline") is True:
            # Second matching preview event makes the result ambiguous, even if
            # either payload is malformed. Do not choose the latest payload.
            self.preview = (None if self.seen_preview or data.get("node_name") != "inline_query"
                            or data.get("output_format") != "json" else _preview(data.get("preview"), self.limit))
            self.seen_preview = True

    def _debug(self, value: object, summary: bool) -> None:
        if not isinstance(value, str):
            return
        text = value.lstrip()
        if summary:
            if text.rstrip() == "All checks passed!":
                self.debug_passed = True
            elif text.startswith("dbt was unable to connect to the specified database."):
                self.issues.pop(_SETUP_CONNECTION, None)
                self.issues[_classification(text) or _SETUP_CONNECTION] = None
            elif text.startswith(("Profile loading failed", "dbt looked for a profiles.yml file", "The profiles.yml")):
                self.issues[_classification(text) or _SETUP_PROFILE] = None
            elif text.startswith("Project loading failed"):
                self.issues[_classification(text) or _SETUP_PROJECT] = None
            elif text.startswith(("- git [ERROR", "Error from git --help:")):
                self.issues["Install Git or put it on PATH in the shell that starts pgtui."] = None
        elif text.startswith("profiles.yml file [ERROR"):
            self.issues[_SETUP_PROFILE] = None
        elif text.startswith("dbt_project.yml file [ERROR"):
            self.issues[_SETUP_PROJECT] = None
        elif text.startswith("Connection test: [ERROR"):
            self.issues[_SETUP_CONNECTION] = None
        elif text == "Connection:":
            self._progress("Checking the target connection…")

    def failure(self) -> str:
        if self.issues and not self.conflict:
            return " ".join(self.issues)
        if not self.invocation or self.conflict:
            return ("dbt could not start with the required options. Check that the selected runtime has dbt Core "
                    "and the target adapter installed.")
        if self.verb == "debug":
            return "dbt debug failed. Check the project, target profile, adapter and database connection."
        return f"dbt {self.verb} failed. Check the project and target with dbt debug."


def _preview(value: object, limit: int) -> tuple[dict[str, Any], ...] | None:
    if not isinstance(value, str):
        return None
    try:
        rows = _json(value)
    except (ValueError, RecursionError):
        return None
    if not isinstance(rows, list) or len(rows) > limit:
        return None
    columns = None
    clean = []
    for row in rows:
        if not isinstance(row, dict) or len(row) > 256:
            return None
        if columns is None:
            columns = list(row)
            if any(len(column) > 256 or not column.isprintable() for column in columns):
                return None
        if list(row) != columns:
            return None
        cells = {}
        for key, cell in row.items():
            if cell is not None and type(cell) not in {str, bool, int, float}:
                return None
            if type(cell) is float and not math.isfinite(cell):
                return None
            if isinstance(cell, str):
                # Make data visible without allowing terminal control sequences.
                cell = "".join(ch if ch.isprintable() else f"\\u{ord(ch):04x}" for ch in cell[:_CELL_MAX])
                if len(cell) > _CELL_MAX or len(row[key]) > _CELL_MAX:
                    cell = cell[:_CELL_MAX - 1] + "…"
            cells[key] = cell
        clean.append(cells)
    return tuple(clean)


def _invalid_json_constant(value: str) -> None:
    raise ValueError


def _unique_json_object(items: list[tuple[str, Any]]) -> dict:
    obj = dict(items)
    if len(obj) != len(items):
        raise ValueError
    return obj


def _json(content: str | bytes | bytearray) -> Any:
    return json.loads(content, object_pairs_hook=_unique_json_object, parse_constant=_invalid_json_constant)


def _show_result(code: int, events: _Events) -> DbtRunResult:
    if code:
        return DbtRunResult(code, message=events.failure())
    if events.conflict or events.preview is None:
        return DbtRunResult(code, message="dbt returned no valid preview. Check that the runtime supports dbt show with JSON output.")
    return DbtRunResult(code, preview=events.preview, message=f"Preview returned {len(events.preview)} rows.",
                        preview_available=True)


async def _lines(stream: asyncio.StreamReader) -> AsyncIterator[bytes]:
    pending = bytearray()
    oversized = False
    while chunk := await stream.read(64 * 1024):
        start = 0
        while start < len(chunk):
            newline = chunk.find(b"\n", start)
            end = len(chunk) if newline < 0 else newline
            if not oversized:
                if len(pending) + end - start > _EVENT_LIMIT:
                    pending.clear()
                    oversized = True
                else:
                    pending.extend(chunk[start:end])
            if newline < 0:
                break
            if not oversized and pending:
                yield bytes(pending)
                pending.clear()
            oversized = False
            start = newline + 1


async def _kill_and_reap(spawn: asyncio.Task[asyncio.subprocess.Process]) -> None:
    try:
        process = await spawn
    except (OSError, ValueError, NotImplementedError):
        return  # Failed before a child existed.
    if process.returncode is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    try:
        if process.stdout:
            while await process.stdout.read(64 * 1024):
                pass  # Flush the pipe so wait() can complete after kill().
    finally:
        await process.wait()


async def _execute(run: DbtRun, args: tuple[str, ...], events: _Events) -> int:
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        *run.runtime.argv, *args, cwd=run.root, env=run._env(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ))
    try:
        process = await asyncio.shield(spawn)
        if process.stdout:
            # dbt owns its local logs; interpret allowlisted JSON events, never relay stdout.
            async for line in _lines(process.stdout):
                events.feed(line)
        return await process.wait()
    except BaseException as exc:
        cleanup = asyncio.create_task(_kill_and_reap(spawn))
        # A second cancel must not interrupt the kill/reap begun by the first.
        cancelled = isinstance(exc, asyncio.CancelledError)
        while True:
            try:
                await asyncio.shield(cleanup)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError from None
        if isinstance(exc, (OSError, ValueError, NotImplementedError)):
            raise DbtError("Could not launch or read dbt Core. Check the selected runtime and its permissions.") from None
        raise


type _StatKey = tuple[int, int, int, int, int]
type _Stamp = tuple[int, int, int, int, int, str]
type _Artifact = tuple[_Stamp, dict | None] | None
_MANIFEST_SCHEMA = re.compile(r"https://schemas\.getdbt\.com/dbt/manifest/v[1-9][0-9]*\.json")
_RESULT_SCHEMA = re.compile(r"https://schemas\.getdbt\.com/dbt/run-results/v[1-9][0-9]*\.json")
_STATUSES = {"success", "pass", "warn", "fail", "error", "runtime error", "skipped", "partial success", "no-op", "reused"}


def _stat_key(info: os.stat_result) -> _StatKey:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read_current(path: Path, *, parse: bool = False) -> _Artifact:
    """Fingerprint the bytes read, rejecting non-regular or concurrently replaced files."""
    try:
        if any(_linked(part) for part in (path, *path.parents)):
            return None
        original = path.stat()
        if not stat.S_ISREG(original.st_mode) or original.st_size > _MAX_ARTIFACT:
            return None
        key = _stat_key(original)
        digest = hashlib.sha256()
        content = bytearray() if parse else None
        with path.open("rb") as handle:
            # Windows can report a different ctime from a file handle. Compare
            # file identity here and the full path stat before/after the read.
            if _stat_key(os.fstat(handle.fileno()))[:4] != key[:4]:
                return None
            count = 0
            while chunk := handle.read(128 * 1024):
                count += len(chunk)
                if count > original.st_size:
                    return None
                digest.update(chunk)
                if content is not None:
                    content.extend(chunk)
        if count != original.st_size or not _unchanged(path, key):
            return None
        data = None if content is None else _json(content)
        if parse and not isinstance(data, dict):
            return None
        return (*key, digest.hexdigest()), data
    except (OSError, ValueError, RecursionError):
        return None


def _unchanged(path: Path, key: tuple) -> bool:
    try:
        return not any(_linked(part) for part in (path, *path.parents)) and _stat_key(path.stat()) == key
    except (OSError, ValueError):
        return False


def _artifact_pair(directory: Path, *, parse: bool = False) -> tuple[_Artifact, ...]:
    paths = tuple(directory / name for name in _ARTIFACTS)
    files = [_read_current(path, parse=parse) for path in paths]
    return tuple(item if item is not None and _unchanged(path, item[0][:-1]) else None
                 for path, item in zip(paths, files, strict=True))


def _compile_inputs(run: DbtRun) -> tuple[_StatKey, ...]:
    # Profile paths may be intentionally external or linked; only stat them, never read credentials here.
    if not run.model_path or run.target is None:
        return ()
    try:
        source = run.root / run.model_path
        config = run.root / "dbt_project.yml"
        if any(_linked(part) for part in (source, *source.parents, config)):
            return ()
        inputs = [item.stat() for item in (source, config, run.target.profiles_dir / "profiles.yml")]
        return tuple(_stat_key(info) for info in inputs) if all(stat.S_ISREG(info.st_mode) for info in inputs) else ()
    except (OSError, ValueError):
        return ()


def dbt_artifact_revision(run: DbtRun) -> tuple:
    """Snapshot current artifact bytes and input file identities, or () if unavailable."""
    manifest, results = _artifact_pair(run.artifact_dir)
    inputs = _compile_inputs(run)
    revision = (manifest[0], results[0], inputs) if manifest is not None and results is not None and inputs else ()
    return revision if dbt_artifact_current(run, revision) else ()


def dbt_artifact_current(run: DbtRun, revision: tuple) -> bool:
    """Cheap periodic check for changed or unavailable artifacts/model/config/profile."""
    if len(revision) != 3 or not revision[2]:
        return False
    manifest, results, inputs = revision
    return (_compile_inputs(run) == inputs and
            _unchanged(run.artifact_dir / _ARTIFACTS[0], manifest[:-1]) and
            _unchanged(run.artifact_dir / _ARTIFACTS[1], results[:-1]))


@dataclass(frozen=True)
class _Before:
    artifacts: tuple[_Artifact, ...]
    inputs: tuple[_StatKey, ...] = ()


def _check_output_files(*paths: Path) -> None:
    try:
        for path in paths:
            if any(_linked(part) for part in (path, *path.parents)) or path.exists() and not path.is_file():
                raise DbtError("dbt output must use regular local artifact and log files, not links.")
    except OSError:
        raise DbtError("Cannot access the dbt output directory. Check its permissions.") from None


def _before_run(run: DbtRun) -> _Before:
    project = DbtFiles(run.root)
    inputs = _compile_inputs(run) if run.verb == "compile" else ()
    if run.verb == "compile" and not inputs:
        raise DbtError("Cannot verify the model, project and profile files. Check their paths and permissions.")
    if run.model_path:
        _check_source(project, run.model_path, run.source_revision)
    _check_state(run.root, run.artifact_dir, str(run.state_path) if run.state_path else "", run.defer)
    _check_output_files(*(run.artifact_dir / name for name in _ARTIFACTS), run.log_dir / "dbt.log")
    return _Before(_artifact_pair(run.artifact_dir), inputs)


def _metadata(artifact: dict, schema: re.Pattern[str], invocation: str) -> dict:
    header = _object(artifact.get("metadata"))
    version = header.get("dbt_schema_version")
    if (header.get("invocation_id") != invocation or not isinstance(version, str) or not schema.fullmatch(version)):
        raise ValueError
    return header


def _run_result(run: DbtRun, before: _Before, code: int, events: _Events) -> DbtRunResult:
    result = DbtRunResult(code)
    failure = events.failure() if code else ""
    if run.verb == "compile" and code:
        return replace(result, message=failure)
    if not events.invocation or events.conflict:
        return replace(result, message=failure or "dbt did not identify this invocation; no results can be verified.")
    try:
        project = DbtFiles(run.root)
        first, second = _artifact_pair(run.artifact_dir, parse=True)
        if first is None or second is None:
            raise ValueError
        manifest_stamp, manifest = first
        results_stamp, results = second
        if (manifest is None or results is None
                or before.artifacts[0] and manifest_stamp[-1] == before.artifacts[0][0][-1]
                or before.artifacts[1] and results_stamp[-1] == before.artifacts[1][0][-1]):
            raise ValueError
        header = _metadata(manifest, _MANIFEST_SCHEMA, events.invocation)
        _metadata(results, _RESULT_SCHEMA, events.invocation)
        target, args = run.target, _object(results.get("args"))
        profiles_dir = args.get("profiles_dir")
        if (target is None or header.get("project_name") != project.name
                or target.adapter and header.get("adapter_type") != target.adapter
                or args.get("profile") != target.profile or args.get("target") != target.name
                or not isinstance(profiles_dir, str) or Path(profiles_dir) != target.profiles_dir
                or args.get("which", run.verb) != run.verb):
            raise ValueError
        nodes = manifest.get("nodes")
        if not isinstance(nodes, dict):
            raise ValueError
        rows = _result_rows(results, manifest)
        if run.verb == "compile":
            sql = _compiled_sql(run, project, nodes, rows)
            revision = (manifest_stamp, results_stamp, before.inputs)
            if not dbt_artifact_current(run, revision):
                raise ValueError
            result = replace(result, compiled_sql=sql, artifact_revision=revision,
                             message="Compiled the saved model.")
        else:
            report = _report(project, manifest, rows)
            summary = (f"dbt {run.verb} {'failed' if code else 'completed'} · {len(report):,} node results."
                       if report else f"dbt {run.verb} returned no node results. Check the selection.")
            result = replace(result, nodes=report, message=summary + (f" {failure}" if failure else ""))
        # An external dbt invocation may have overwritten a file during report construction.
        if not all(_unchanged(run.artifact_dir / name, stamp[:-1])
                   for name, stamp in zip(_ARTIFACTS, (manifest_stamp, results_stamp), strict=True)):
            raise ValueError
        return result
    except DbtError as exc:
        return DbtRunResult(code, message=str(exc))
    except (ValueError, OSError, RecursionError):
        message = ("No fresh, matching dbt artifacts for the saved model. Check the project and compile it again."
                   if run.verb == "compile" else "No verified node results for this invocation. Check the dbt target and selection.")
        return DbtRunResult(code, message=f"{failure} {message}" if failure else message)


def _lookup(manifest: dict, key: str) -> dict:
    for collection in ("nodes", "sources", "unit_tests"):
        node = _object(manifest.get(collection)).get(key)
        if isinstance(node, dict) and node.get("unique_id") == key:
            return node
    return {}


def _result_rows(results: dict, manifest: dict) -> dict[str, dict]:
    rows = results.get("results")
    if not isinstance(rows, list):
        raise ValueError
    indexed = {}
    for row in rows:
        row = _object(row)
        key, status, failures = row.get("unique_id"), row.get("status"), row.get("failures")
        if (not isinstance(key, str) or not _NODE_ID.fullmatch(key) or key in indexed
                or not isinstance(status, str) or status not in _STATUSES or not _lookup(manifest, key)
                or failures is not None and (type(failures) is not int or not 0 <= failures < 2**63)):
            raise ValueError
        indexed[key] = row
    return indexed


def _original_path(node: dict) -> str:
    path = node.get("original_file_path")
    return path.replace("\\", "/") if isinstance(path, str) and path.isprintable() else ""


def _compiled_sql(run: DbtRun, project: DbtFiles, nodes: dict, rows: dict[str, dict]) -> str:
    document = _check_source(project, run.model_path, run.source_revision)
    matches = []
    for key, row in rows.items():
        node = _object(nodes.get(key))
        if (row["status"] == "success" and node.get("resource_type") == "model"
                and node.get("package_name") == project.name and _original_path(node) == run.model_path):
            matches.append((node, row))
    if len(matches) != 1:
        raise ValueError
    node, row = matches[0]
    raw = node.get("raw_code")
    compiled = node.get("compiled_code")
    if (node.get("compiled") is not True or not isinstance(raw, str)
            or DbtProject._artifact_input(raw) != DbtProject._artifact_input(document.text)
            or not isinstance(compiled, str) or not compiled.strip()
            or "compiled_code" in row and row["compiled_code"] != compiled
            or len(compiled) > 2 * 1024 * 1024
            or not all(ch.isprintable() or ch in "\n\r\t" for ch in compiled)):
        raise ValueError
    return compiled


def _node_reason(row: dict) -> str:
    status, failures = row["status"], row.get("failures")
    if status in {"fail", "warn"}:
        if failures is not None:
            return f"{failures:,} failures reported. Review the test definition and affected data."
        return "Review the test definition and its configured failure thresholds."
    if status in {"error", "runtime error"}:
        return _classification(row.get("message")) or "Check the node and its dependencies; use dbt debug to verify the target."
    if status == "skipped":
        return "Not executed. Check upstream failures and the selection."
    if status == "partial success":
        return "Only part of the node succeeded. Review the node and its dependencies before rerunning."
    return ""


def _report(project: DbtFiles, manifest: dict, rows: dict[str, dict]) -> tuple[DbtNodeResult, ...]:
    documents: dict[str, yaml.Node | None] = {}
    report = []
    for key, row in rows.items():
        node = _lookup(manifest, key)
        path, anchor = _location(project, manifest, node, documents)
        report.append(DbtNodeResult(key, row["status"], _node_reason(row), path, anchor))
    return tuple(report)


def _location(project: DbtFiles, manifest: dict, node: dict,
              documents: dict[str, yaml.Node | None]) -> tuple[str, int | None]:
    if node.get("package_name") != project.name or not (path := _original_path(node)):
        return "", None
    try:
        local = project.editable_path(path)
        if local.suffix.lower() not in {".sql", ".yml", ".yaml"} or not local.is_file():
            return "", None
    except (DbtError, OSError, ValueError):
        return "", None
    if node.get("resource_type") != "test" or local.suffix.lower() == ".sql":
        return path, None
    if path not in documents:
        try:
            documents[path] = yaml.compose(project.read(path).text, Loader=yaml.BaseLoader)
        except (DbtError, yaml.YAMLError, RecursionError):
            documents[path] = None
    return path, _test_anchor(documents[path], manifest, node)


def _yaml_key(node: yaml.Node | None, key: str) -> yaml.Node | None:
    if isinstance(node, yaml.MappingNode):
        matches = [value for name, value in node.value if isinstance(name, yaml.ScalarNode) and name.value == key]
        if len(matches) == 1:
            return matches[0]
    return None


def _scalar(node: yaml.Node | None) -> str | None:
    return node.value if isinstance(node, yaml.ScalarNode) else None


def _named_item(sequence: yaml.Node | None, name: object) -> yaml.Node | None:
    if isinstance(sequence, yaml.SequenceNode) and isinstance(name, str):
        matches = [item for item in sequence.value if _scalar(_yaml_key(item, "name")) == name]
        if len(matches) == 1:
            return matches[0]
    return None


def _test_anchor(document: yaml.Node | None, manifest: dict, test: dict) -> int | None:
    meta = _object(test.get("test_metadata"))
    name, namespace, attached = meta.get("name"), meta.get("namespace"), test.get("attached_node")
    if not isinstance(name, str) or namespace is not None and not isinstance(namespace, str) or not isinstance(attached, str):
        return None
    node = _lookup(manifest, attached)
    if node.get("package_name") != test.get("package_name"):
        return None
    group = {"model": "models", "seed": "seeds", "snapshot": "snapshots"}.get(attached.split(".", 1)[0])
    if group:
        owner = _named_item(_yaml_key(document, group), node.get("name"))
        if _yaml_key(owner, "versions") is not None:
            return None  # Versioned tests may be inherited or overridden; open the file only.
    elif attached.startswith("source."):
        source = _named_item(_yaml_key(document, "sources"), node.get("source_name"))
        owner = _named_item(_yaml_key(source, "tables"), node.get("name"))
    else:
        return None
    if column := test.get("column_name"):
        owner = _named_item(_yaml_key(owner, "columns"), column)
    if owner is None:
        return None
    spelling = f"{namespace}.{name}" if namespace else name
    matches = []
    for property_name in ("tests", "data_tests"):
        tests = _yaml_key(owner, property_name)
        if not isinstance(tests, yaml.SequenceNode):
            continue
        for item in tests.value:
            if _scalar(item) == spelling or (isinstance(item, yaml.MappingNode) and len(item.value) == 1
                                             and _scalar(item.value[0][0]) == spelling):
                matches.append(item.start_mark.index)
                line = item.start_mark.line
    # Aliases point to their definition's marks, not necessarily to this owner.
    if len(matches) == 1 and owner.start_mark.index <= matches[0] < owner.end_mark.index:
        return line
    return None
