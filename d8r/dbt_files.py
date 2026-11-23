"""Confined, source-agnostic access to the editable files in a dbt project."""
from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

from d8r.dbt import DbtError, _call, _expression_end, _linked, _relative, _safe_root, _yaml

_FILE_LIMIT = 2 * 1024 * 1024
_TREE_LIMIT = 20_000
_EDITABLE = {".sql", ".yml", ".yaml", ".md", ".csv", ".py"}
_CONTROL_FILES = {".gitignore", ".dbtignore"}
_EXCLUDED = {"profiles.yml", "profiles.yaml"}
_REFERENCE = re.compile(r"\{\{(-)?\s*(?:ref|source)\s*\(")


def file_revision(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DbtFile:
    text: str
    revision: str  # Empty only for a file that hasn't been created yet.
    bom: bool = False


@dataclass(frozen=True)
class DbtModel:
    path: str
    name: str
    properties: str
    row: int = 0
    version: str = ""


class DbtFiles:
    """Opening broken YAML must not require a valid conversion/manifest index."""

    def __init__(self, directory: str | Path) -> None:
        self.root = _safe_root(directory)
        try:
            project = self.read("dbt_project.yml")
        except DbtError as exc:
            raise DbtError("Open the directory containing dbt_project.yml. " + str(exc)) from exc
        self.config_error = ""
        try:
            self.config = _yaml(project.text, self.root / "dbt_project.yml")
        except DbtError as exc:
            self.config, self.config_error = {}, str(exc)
        self.name = self.config.get("name", "")
        self.excluded = {Path(".git"), Path("target"), Path("logs"), Path("dbt_packages"), Path("dbt_modules")}
        for key in ("target-path", "log-path", "packages-install-path"):
            if key in self.config:
                try:
                    self.excluded.add(_relative(self.config[key]))
                except DbtError:
                    pass  # Still allow the project file to be repaired in the editor.

    def _path(self, relative: str) -> Path:
        path = self.root
        for part in _relative(relative).parts:
            path /= part
            if _linked(path):
                raise DbtError(f"Symlink/junction paths are refused: {relative}")
        return path

    def _visible(self, path: Path) -> bool:
        relative = path.relative_to(self.root)
        return not (any(part.startswith(".") or part.casefold() in {"__pycache__", "node_modules"}
                        for part in relative.parts[:-1]) or any(
            relative.is_relative_to(excluded) for excluded in self.excluded
        ))

    def editable_path(self, relative: str) -> Path:
        path = self._path(relative)
        if (path.name.casefold() in _EXCLUDED or
                path.name not in _CONTROL_FILES and (path.name.startswith(".") or path.suffix.lower() not in _EDITABLE) or
                not (relative == "dbt_project.yml" or self._visible(path))):
            raise DbtError(f"Not an editable dbt project file: {relative}")
        return path

    def paths(self, *, ignore_directory: Callable[[str], bool] | None = None) -> tuple[str, ...]:
        paths = []
        visited = 0
        try:
            for directory, folders, files in self.root.walk(follow_symlinks=False):
                visited += len(files) + len(folders)
                if visited > _TREE_LIMIT:
                    raise DbtError(f"The project exceeds the explorer's {_TREE_LIMIT:,}-entry limit.")
                folders[:] = sorted(name for name in folders if not name.startswith('.') and
                                    not _linked(directory / name) and self._visible(directory / name / '_') and
                                    not (ignore_directory and ignore_directory(
                                        (directory / name).relative_to(self.root).as_posix() + '/')))
                for name in files:
                    path = directory / name
                    relative = path.relative_to(self.root).as_posix()
                    try:
                        self.editable_path(relative)
                    except DbtError:
                        continue
                    if path.is_file():
                        paths.append(relative)
        except OSError as exc:
            raise DbtError("Cannot list files in the dbt project.") from exc
        return tuple(sorted(paths, key=str.casefold))

    def read(self, relative: str, *, missing: bool = False) -> DbtFile:
        path = self.editable_path(relative)
        try:
            if not stat.S_ISREG(path.stat().st_mode):
                raise DbtError(f"Not a regular file: {relative}")
            with path.open("rb") as handle:
                data = handle.read(_FILE_LIMIT + 1)
            if len(data) > _FILE_LIMIT:
                raise DbtError(f"{relative} exceeds the 2 MiB editor limit.")
            text = data.decode("utf-8-sig")
        except FileNotFoundError as exc:
            if missing:
                return DbtFile("", "")
            raise DbtError(f"File not found: {relative}") from exc
        except (OSError, UnicodeError) as exc:
            raise DbtError(f"Cannot read {relative} as UTF-8 text.") from exc
        if "\0" in text or any(ord(char) < 32 and char not in "\r\n\t" for char in text):
            raise DbtError(f"Not an editable text file: {relative}")
        return DbtFile(text, file_revision(text), data.startswith(b"\xef\xbb\xbf"))

    def save(self, relative: str, text: str, expected: str) -> DbtFile:
        """Replace one file atomically; never clobber a draft's changed disk baseline."""
        path = self.editable_path(relative)
        before = self.read(relative, missing=True)
        if before.revision != expected:
            raise DbtError(f"{relative} changed on disk. Review Git Diff or reload before saving.")
        if before.revision and text == before.text:
            return before
        data = text.encode("utf-8-sig" if before.bom else "utf-8")
        if len(data) > _FILE_LIMIT or any(ord(char) < 32 and char not in "\r\n\t" for char in text):
            raise DbtError("Save requires UTF-8 text of at most 2 MiB without control characters.")
        temp = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pgtui-", delete=False) as handle:
                temp = Path(handle.name)
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            if self.read(relative, missing=True) != before:
                raise DbtError(f"{relative} changed on disk while saving; nothing was overwritten.")
            if before.revision:
                temp.chmod(stat.S_IMODE(path.stat().st_mode))
                temp.replace(path)
            else:
                # Exclusive publication: a concurrent creation cannot be replaced.
                os.link(temp, path)
        except OSError as exc:
            raise DbtError(f"Could not save {relative}: {exc.strerror or type(exc).__name__}.") from exc
        finally:
            if temp is not None:
                temp.unlink(missing_ok=True)
        return DbtFile(text, file_revision(text), before.bom)

    def directories(self, key: str, default: str) -> tuple[Path, ...]:
        if self.config_error:
            raise DbtError(self.config_error)
        paths = self.config.get(key, [default])
        if not isinstance(paths, list) or not paths:
            raise DbtError(f"{key} must be a nonempty list of relative directories.")
        return tuple(self._path(value) for value in paths)

    def test_file(self, name: str) -> str:
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]{0,127}", name):
            raise DbtError("Name a test with letters, digits and underscores, starting with a letter or underscore.")
        roots = self.directories("test-paths", "tests")
        if any(self.read((root / f"{name}.sql").relative_to(self.root).as_posix(), missing=True).revision for root in roots):
            raise DbtError(f"A dbt test named {name!r} already exists at that path.")
        return (roots[0] / f"{name}.sql").relative_to(self.root).as_posix()

    def _models(self) -> tuple[list[str], list[str]]:
        roots = self.directories("model-paths", "models")
        paths = [path for path in self.paths() if any((self.root / path).is_relative_to(root) for root in roots)]
        return ([path for path in paths if Path(path).suffix.lower() in {".sql", ".py"}],
                [path for path in paths if Path(path).suffix.lower() in {".yml", ".yaml"}])

    def models(self) -> tuple[str, ...]:
        return tuple(self._models()[0])

    def _properties(self, paths: list[str], drafts: dict[str, str]):
        for path in sorted(set(paths) | set(drafts)):
            if Path(path).suffix.lower() not in {".yml", ".yaml"}:
                continue
            try:
                text = drafts[path] if path in drafts else self.read(path).text
                data = _yaml(text, Path(path))
                node = yaml.compose(text, Loader=yaml.SafeLoader)
            except (DbtError, yaml.YAMLError, RecursionError):
                # A malformed neighbor must not block navigation in the rest of the project.
                continue
            yield path, data, node

    @staticmethod
    def _entries(value: object, key: str) -> list[dict]:
        items = value.get(key) if isinstance(value, dict) else None
        return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []

    @staticmethod
    def _row(node: yaml.Node | None, group: str, index: int) -> int:
        if isinstance(node, yaml.MappingNode):
            for key, value in node.value:
                if isinstance(key, yaml.ScalarNode) and key.value == group and isinstance(value, yaml.SequenceNode):
                    return value.value[index].start_mark.line if index < len(value.value) else 0
        return 0

    def model(self, path: str, drafts: dict[str, str] | None = None) -> DbtModel | None:
        """Map a model filename (including defined_in/default version names) to its YAML entry."""
        models, files = self._models()
        if path not in models:
            return None
        stem = Path(path).stem
        matches = []
        for properties, data, node in self._properties(files, drafts or {}):
            entries = data.get("models")
            for index, entry in enumerate(entries if isinstance(entries, list) else []):
                name = entry.get("name") if isinstance(entry, dict) else None
                if not isinstance(name, str):
                    continue
                row = self._row(node, "models", index)
                versions = self._entries(entry, "versions")
                if versions:
                    for version in versions:
                        number = version.get("v")
                        defined = version.get("defined_in", f"{name}_v{number}")
                        if number is not None and isinstance(defined, str) and Path(defined).stem == stem:
                            matches.append(DbtModel(path, name, properties, row, str(number)))
                elif name == stem:
                    matches.append(DbtModel(path, name, properties, row))
        if len(matches) > 1:
            raise DbtError(f"Multiple YAML entries describe {path}; resolve the ambiguity first.")
        if matches:
            return matches[0]
        if sum(Path(other).stem == stem for other in models) > 1:
            raise DbtError(f"Multiple models use the name {stem!r}; rename them before adding properties.")
        sibling = Path(path).with_suffix(".yaml").as_posix()
        properties = sibling if sibling in files or sibling in (drafts or {}) else Path(path).with_suffix(".yml").as_posix()
        return DbtModel(path, stem, properties)

    def resource_properties(self, relative: str, document: str, drafts: dict[str, str]) -> tuple[str, int] | None:
        """Look up properties for a non-model SQL resource by dbt's resource name."""
        path = self.editable_path(relative)
        groups = []
        names = {path.stem}
        for key, default, group, tag in (
            ("snapshot-paths", "snapshots", "snapshots", "snapshot"),
            ("macro-paths", "macros", "macros", "macro"),
            ("analysis-paths", "analyses", "analyses", None),
            ("test-paths", "tests", "data_tests", None),
        ):
            roots = self.config.get(key, [default])
            if not roots or not any(path.is_relative_to(root) for root in self.directories(key, default)):
                continue
            groups.append(group)
            if group == "data_tests":
                groups.append("tests")
            if tag:
                names.update(re.findall(r"\{%-?\s*" + tag + r"\s+([A-Za-z_]\w*)", document))
        if not groups:
            return None
        matches = {}
        for candidate, data, node in self._properties(list(self.paths()), drafts):
            for group in groups:
                entries = data.get(group)
                for index, item in enumerate(entries if isinstance(entries, list) else []):
                    name = item.get("name") if isinstance(item, dict) else None
                    if isinstance(name, str) and name in names:
                        matches.setdefault(candidate, self._row(node, group, index))
        if len(matches) > 1:
            raise DbtError(f"Multiple YAML files describe {relative}; choose the file in the explorer.")
        return next(iter(matches.items()), None)

    def definition(self, kind: str, args: list, kwargs: dict,
                   drafts: dict[str, str] | None = None) -> tuple[str, int]:
        models, files = self._models()
        if kind == "source":
            if len(args) != 2 or kwargs or not all(isinstance(value, str) for value in args):
                raise DbtError("source() requires two literal names.")
            matches = []
            for path, data, node in self._properties(files, drafts or {}):
                entries = data.get("sources")
                for index, source in enumerate(entries if isinstance(entries, list) else []):
                    if (isinstance(source, dict) and source.get("name") == args[0] and
                            any(table.get("name") == args[1] for table in self._entries(source, "tables"))):
                        matches.append((path, self._row(node, "sources", index)))
        else:
            if (len(args) not in {1, 2} or not all(isinstance(value, str) for value in args)
                    or len(kwargs) > 1 or set(kwargs) - {"v", "version"}):
                raise DbtError("ref() requires literal model/package names and optional version=.")
            if len(args) == 2 and args[0] != self.name:
                raise DbtError("Open the referenced package as its own dbt project to navigate its files.")
            name = args[-1]
            requested = kwargs.get("version", kwargs.get("v"))
            stems = {name} if requested is None else set()
            for _, data, _ in self._properties(files, drafts or {}):
                for entry in self._entries(data, "models"):
                    versions = self._entries(entry, "versions")
                    if entry.get("name") != name or not versions:
                        continue
                    latest = entry.get("latest_version")
                    if latest is None:
                        numbers = [item.get("v") for item in versions if item.get("v") is not None]
                        try:
                            latest = max(numbers, key=float)
                        except (ValueError, TypeError):
                            pass  # A nonnumeric version needs an explicit latest_version.
                    version = str(requested if requested is not None else latest)
                    stems.discard(name)
                    for item in versions:
                        if str(item.get("v")) == version:
                            defined = item.get("defined_in", f"{name}_v{version}")
                            if isinstance(defined, str):
                                stems.add(defined.removesuffix(".sql").removesuffix(".py"))
            matches = [(path, 0) for path in models if Path(path).stem in stems]
        if len(matches) != 1:
            reason = "Multiple definitions" if matches else "No local definition"
            raise DbtError(f"{reason} for {kind}{tuple(args)!r} in this dbt project.")
        return matches[0]


def reference_at(text: str, offset: int) -> tuple[str, list, dict] | None:
    """Find a literal Jinja ref/source under the cursor without evaluating templates."""
    for match in _REFERENCE.finditer(text):
        if match.start() > offset:
            break
        try:
            start = match.start() + (3 if match[1] else 2)
            end = _expression_end(text, start)
            if offset < end + 2:
                return _call(text[start:end].rstrip().removesuffix("-"))
        except DbtError:
            if offset < match.end():
                raise
    return None
