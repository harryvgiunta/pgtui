"""Offline dbt SQL conversion; never invokes a dbt runtime.

Only literal ref/source/config calls are interpreted. Relation names come from
manifest artifacts or literal source YAML, never profiles or schema guesses.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import json
import ntpath
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

import yaml

from d8r.sql import (
    SqlError, Token, _relation_end, _relation_sites, lex_sql, relation_parts,
    quote_identifier, split_statements,
)


_JINJA_OPEN = re.compile(r"\{[{%#]")
_JINJA_PART = re.compile(r'''"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|[{}'"]''', re.DOTALL)


class DbtError(ValueError):
    """An actionable project or conversion error, safe to show in the form."""


def has_query_jinja(document: str, *, dialect: str = "postgres") -> bool:
    """Recognize template code without treating SQL strings or comments as Jinja."""
    if not _JINJA_OPEN.search(document):
        return False
    try:
        tokens = lex_sql(document, dialect=dialect)
    except SqlError:
        # Raw Jinja literals may use escapes that SQL does not recognize.
        return True
    return any(token.kind == "symbol" and _JINJA_OPEN.match(document, token.start) for token in tokens)


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    # Flatten YAML merges before constructing values, but reject explicit duplicate
    # keys rather than silently discarding project configuration.
    keys = [loader.construct_object(key, deep=deep) for key, _ in node.value
            if key.tag != "tag:yaml.org,2002:merge"]
    try:
        if len(keys) != len(set(keys)):
            raise DbtError("Duplicate YAML keys: resolve them before exchanging this project.")
    except TypeError as exc:
        raise DbtError("YAML mapping keys must be scalar values.") from exc
    loader.flatten_mapping(node)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueLoader.add_constructor("tag:yaml.org,2002:map", _mapping)


def _yaml(text: str, path: Path) -> dict:
    try:
        value = yaml.load(text, Loader=_UniqueLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise DbtError(f"Invalid YAML in {path.name}: {exc}") from exc
    if not isinstance(value, dict):
        raise DbtError(f"{path.name} must contain a YAML mapping.")
    return value


def _literal(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and not any(
        marker in value for marker in ("{{", "{%", "{#", "\x00")
    )


def _linked(path: Path) -> bool:
    return path.is_symlink() or path.is_junction()


def _safe_root(value: str | Path) -> Path:
    if not str(value).strip():
        raise DbtError("Choose a local project directory containing dbt_project.yml.")
    try:
        original = Path(value).expanduser()
    except RuntimeError as exc:
        raise DbtError("Choose an existing local project directory.") from exc
    if ".." in original.parts:
        raise DbtError("Project paths cannot contain '..' traversal.")
    root = original.absolute()
    if any(_linked(item) for item in (root, *root.parents)):
        raise DbtError("Symlink/junction project paths are refused; choose the real directory.")
    if not root.is_dir():
        raise DbtError(f"Project directory does not exist: {root}")
    return root.resolve()


def _relative(value: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise DbtError("Use a nonempty project-relative path with '/' separators.")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise DbtError(f"Absolute paths and '..' traversal are refused: {value}")
    if any(ntpath.isreserved(part) for part in path.parts):
        raise DbtError("Project paths cannot contain reserved names or control characters.")
    if any(part.lower() in {"profiles.yml", "profiles.yaml"} for part in path.parts):
        raise DbtError("Credential profiles are outside dbt exchange and are never accessed.")
    return Path(*path.parts)


@dataclass(frozen=True)
class ImportedModel:
    sql: str
    name: str
    materialized: str
    description: str
    config: dict[str, Any]
    used_compiled: bool = False


@dataclass(frozen=True)
class _Relation:
    kind: str
    package: str
    name: str
    relation: str
    source: str = ""
    version: Any = None
    latest_version: Any = None

    def expression(self) -> str:
        if self.kind == "source":
            call = f"source({self.source!r}, {self.name!r})"
        else:
            args = f"{self.package!r}, {self.name!r}"
            if self.version is not None:
                args += f", version={self.version!r}"
            call = f"ref({args})"
        return "{{ " + call + " }}"


def _name(token: Token, dialect: str) -> str:
    return (relation_parts(token.text, dialect=dialect) or (token.text,))[0]


def _cte_ranges(tokens: tuple[Token, ...], dialect: str) -> list[tuple[str, int, int]]:
    """Record lexical CTE visibility, including nested WITH and recursive bodies."""
    pairs: dict[int, int] = {}
    stack: list[int] = []
    enclosing: dict[int, int | None] = {}
    for i, token in enumerate(tokens):
        enclosing[i] = stack[-1] if stack else None
        if token.kind == "symbol" and token.text == "(":
            stack.append(i)
        elif token.kind == "symbol" and token.text == ")" and stack:
            pairs[stack.pop()] = i
    ranges = []
    for i, token in enumerate(tokens):
        if token.kind != "word" or token.text.upper() != "WITH":
            continue
        end = pairs.get(enclosing[i], len(tokens))
        j = i + 1
        recursive = j < len(tokens) and tokens[j].text.upper() == "RECURSIVE"
        j += int(recursive)
        definitions = []
        while j < end and tokens[j].kind in {"word", "quoted_identifier"}:
            name = _name(tokens[j], dialect)
            j += 1
            if j < end and tokens[j].text == "(":
                j = pairs.get(j, end) + 1
            if j >= end or tokens[j].text.upper() != "AS":
                break
            j += 1
            if j < end and tokens[j].text.upper() == "NOT":
                j += 1
            if j < end and tokens[j].text.upper() == "MATERIALIZED":
                j += 1
            if j >= end or tokens[j].text != "(" or j not in pairs:
                break
            closing = pairs[j]
            definitions.append((name, j if recursive else closing + 1, end))
            j = closing + 1
            if j >= end or tokens[j].text != ",":
                break
            j += 1
        ranges.extend(definitions)
    return ranges


def _call(expression: str) -> tuple[str, list[Any], dict[str, Any]]:
    """Parse literal call syntax, with no eval, attribute access, or Jinja runtime."""
    try:
        node = ast.parse(expression.strip(), mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError
        if node.func.id not in {"ref", "source", "config"}:
            raise ValueError
        if any(keyword.arg is None for keyword in node.keywords):
            raise ValueError
        def literal(item):
            if isinstance(item, ast.Name) and item.id in {"true", "false", "none"}:
                return {"true": True, "false": False, "none": None}[item.id]
            if isinstance(item, ast.List):
                return [literal(child) for child in item.elts]
            if isinstance(item, ast.Tuple):
                return tuple(literal(child) for child in item.elts)
            if isinstance(item, ast.Dict):
                return {literal(key): literal(value) for key, value in zip(item.keys, item.values)}
            value = ast.literal_eval(item)
            if value is not None and not isinstance(value, (str, int, float, bool)):
                raise ValueError
            return value
        args = [literal(item) for item in node.args]
        kwargs = {item.arg: literal(item.value) for item in node.keywords}
        if len(kwargs) != len(node.keywords):
            raise ValueError
        return node.func.id, args, kwargs
    except (SyntaxError, ValueError, TypeError, RecursionError) as exc:
        raise DbtError("Advanced/dynamic Jinja requires an actual compiled artifact. Compile outside PGTUI, then reload.") from exc


def _expression_end(text: str, start: int) -> int:
    """Find the closing }} without mistaking literal strings or dictionaries for it."""
    braces = 0
    for part in _JINJA_PART.finditer(text, start):
        if part[0] == "{":
            braces += 1
        elif part[0] == "}":
            if braces:
                braces -= 1
            elif text.startswith("}}", part.start()):
                return part.start()
            else:
                break
        elif part[0] in {"'", '"'}:
            break  # unterminated quoted literal
    raise DbtError("Unbalanced Jinja expression; compile the model outside PGTUI.")


class DbtProject:
    """A user-opened local project snapshot. Construction performs file reads only."""

    def __init__(self, directory: str | Path, *, dialect: str = "postgres") -> None:
        if dialect not in {"postgres", "sqlite", "duckdb"}:
            raise DbtError("dbt exchange supports PostgreSQL, SQLite and DuckDB SQL only.")
        self.root = _safe_root(directory)
        self.dialect = dialect
        self.project = _yaml(self._read("dbt_project.yml"), self.root / "dbt_project.yml")
        self.name = self.project.get("name")
        if not _literal(self.name):
            raise DbtError("dbt_project.yml needs a literal project name.")
        _relative(self.name)
        if "/" in self.name:
            raise DbtError("The dbt project name cannot contain path separators.")
        paths = self.project.get("model-paths", ["models"])
        if not isinstance(paths, list) or not paths:
            raise DbtError("model-paths must be a nonempty list of relative directories.")
        self.model_paths = tuple(self._path(path) for path in paths)
        target = self.project.get("target-path", "target")
        self.target = self._path(target)
        self.models: tuple[str, ...] = ()
        self.yaml_files: dict[Path, dict] = {}
        models = []
        for directory_path in self.model_paths:
            if not directory_path.exists():
                continue
            if not directory_path.is_dir():
                raise DbtError(f"Model path is not a directory: {directory_path}")
            for current, dirs, files in os.walk(directory_path, followlinks=False):
                for child in dirs + files:
                    candidate = Path(current) / child
                    self._path(candidate.relative_to(self.root).as_posix())
                for filename in files:
                    path = Path(current) / filename
                    if path.suffix.lower() == ".sql":
                        models.append(path.relative_to(self.root).as_posix())
                    elif path.suffix.lower() in {".yml", ".yaml"}:
                        self.yaml_files[path] = _yaml(self._read(path.relative_to(self.root).as_posix()), path)
        self.models = tuple(sorted(set(models)))
        manifest_path = self._path((self.target / "manifest.json").relative_to(self.root).as_posix())
        self.manifest: dict = {}
        if manifest_path.exists():
            try:
                self.manifest = json.loads(self._read(manifest_path.relative_to(self.root).as_posix()))
            except json.JSONDecodeError as exc:
                raise DbtError("Invalid manifest.json; regenerate the artifact outside PGTUI.") from exc
            if not isinstance(self.manifest, dict):
                raise DbtError("manifest.json must contain an object.")
        self.relations: list[_Relation] = []
        self.nodes: list[dict] = []
        self._index()

    def _path(self, value: str) -> Path:
        path = self.root / _relative(value)
        current = self.root
        if _linked(current):
            raise DbtError(f"Symlink/junction paths are refused: {current}")
        for part in path.relative_to(self.root).parts:
            current = current / part
            if _linked(current):
                raise DbtError(f"Symlink/junction paths are refused: {current}")
        return path

    def _read(self, value: str) -> str:
        path = self._path(value)
        try:
            return path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise DbtError(f"Cannot read {path.name}: {exc}") from exc

    def _index(self) -> None:
        source_keys = set()
        for group in ("nodes", "sources"):
            entries = self.manifest.get(group, {})
            if not isinstance(entries, dict):
                raise DbtError(f"manifest.json {group} must be an object.")
            for node in entries.values():
                if not isinstance(node, dict):
                    raise DbtError(f"manifest.json contains an invalid {group} entry.")
                kind = node.get("resource_type", "source" if group == "sources" else "")
                if kind == "source":
                    source_keys.add((node.get("source_name"), node.get("name")))
                config = node.get("config") or {}
                if not isinstance(config, dict):
                    raise DbtError("Manifest node config must be an object.")
                if config.get("enabled") is False:
                    continue
                if kind == "model":
                    self.nodes.append(node)
                if kind not in {"model", "seed", "snapshot", "source"}:
                    continue
                relation = node.get("relation_name")
                if not _literal(relation) or relation_parts(relation, dialect=self.dialect) is None:
                    continue
                if config.get("materialized") == "ephemeral":
                    continue
                package, name = node.get("package_name"), node.get("name")
                if not _literal(package) or not _literal(name):
                    continue
                self.relations.append(_Relation(
                    "source" if kind == "source" else "ref", package, name, relation,
                    str(node.get("source_name", "")), node.get("version"), node.get("latest_version"),
                ))
        for data in self.yaml_files.values():
            for source in self._entries(data, "sources"):
                source_name = source.get("name")
                for table in self._entries(source, "tables"):
                    name = table.get("name")
                    if (source_name, name) in source_keys:
                        continue
                    enabled = True
                    for owner in (source, table):
                        source_config = owner.get("config", {})
                        if not isinstance(source_config, dict):
                            raise DbtError("Source config must be a mapping.")
                        enabled = source_config.get("enabled", owner.get("enabled", enabled))
                    if enabled is not True:
                        continue
                    schema = source.get("schema", source_name)
                    database = source.get("database")
                    identifier = table.get("identifier", name)
                    parts = ([database] if database is not None else []) + [schema, identifier]
                    if not all(_literal(part) for part in [source_name, name, *parts]):
                        continue
                    quoting = {}
                    for owner in (self.project, source, table):
                        policy = owner.get("quoting", {})
                        if not isinstance(policy, dict) or any(not isinstance(value, bool) for value in policy.values()):
                            raise DbtError("Source quoting must be a mapping of literal booleans.")
                        quoting.update(policy)
                    labels = (["database"] if database is not None else []) + ["schema", "identifier"]
                    rendered = [quote_identifier(part) if quoting.get(label, True) else part
                                for label, part in zip(labels, parts)]
                    if any(relation_parts(value, dialect=self.dialect) is None or
                           len(relation_parts(value, dialect=self.dialect)) != 1 for value in rendered):
                        raise DbtError("An unquoted source identifier is not one SQL identifier; enable quoting or supply a manifest.")
                    self.relations.append(_Relation("source", self.name, name, ".".join(rendered), source_name))

    @staticmethod
    def _entries(data: dict, key: str) -> list[dict]:
        entries = data.get(key, [])
        if not isinstance(entries, list) or any(not isinstance(item, dict) for item in entries):
            raise DbtError(f"YAML {key} must be a list of mappings.")
        return entries

    def _model_metadata(self, name: str) -> tuple[Path | None, dict]:
        matches = [(path, item) for path, data in self.yaml_files.items()
                   for item in self._entries(data, "models") if item.get("name") == name]
        if len(matches) > 1:
            raise DbtError(f"Model {name!r} is described in multiple YAML files; resolve the ambiguity first.")
        return matches[0] if matches else (None, {})

    def render_query(self, jinja: str, *, database: str) -> str:
        """Resolve supported Jinja to one native query without executing it."""
        sql = self._simple_import(jinja, {}, database=database)
        self._query_tokens(sql)
        return sql

    def query_views(self, document: str, *, database: str, schema: str = "") -> tuple[str, str]:
        """Render to SQL first, then normalize every convertible relation for dbt."""
        if _JINJA_OPEN.search(document):
            if not has_query_jinja(document, dialect=self.dialect):
                raise DbtError("Jinja markers only inside SQL comments or quoted text are ambiguous; escape literal markers before toggling.")
            templates = []
            sql = self._simple_import(document, {}, database=database, templates=templates)
            jinja = self._convert_to_dbt(sql, database, schema, templates)
        else:
            sql = document
            jinja = self.to_dbt(sql, database=database, schema=schema)
        if jinja != sql:
            # A unique physical mapping may still emit a call shared by two relations.
            self.render_query(jinja, database=database)
        return sql, jinja

    def _query_tokens(self, sql: str) -> tuple[Token, ...]:
        try:
            tokens = tuple(token for token in lex_sql(sql, dialect=self.dialect)
                           if token.kind not in {"whitespace", "comment"})
            statements = split_statements(sql, dialect=self.dialect)
        except SqlError as exc:
            raise DbtError(str(exc)) from exc
        if len(statements) != 1 or not tokens or tokens[0].text.upper() not in {"SELECT", "WITH", "VALUES", "TABLE"}:
            raise DbtError("Use one native SELECT/WITH/VALUES/TABLE query, not a script or DDL statement.")
        return tokens

    def to_dbt(self, sql: str, *, database: str = "", schema: str = "") -> str:
        return self._convert_to_dbt(sql, database, schema, [])

    def _convert_to_dbt(self, sql: str, database: str, schema: str,
                        templates: list[tuple[int, int, str]]) -> str:
        tokens = self._query_tokens(sql)
        if _JINJA_OPEN.search(sql):
            raise DbtError("Cannot normalize template markers inside native SQL. Use literal dbt calls or remove the markers.")
        ctes = _cte_ranges(tokens, self.dialect)
        replacements = []
        sites = set(_relation_sites(tokens))
        authored = {(start, end): template for start, end, template in templates if start != end}
        # PostgreSQL/DuckDB TABLE query shorthand has no FROM keyword.
        for index, token in enumerate(tokens[:-1]):
            if token.kind == "word" and token.text.upper() == "TABLE" and (
                index == 0 or tokens[index - 1].text.upper() in {"(", "UNION", "INTERSECT", "EXCEPT", "ALL"}
            ):
                sites.add(index + 1)

        def in_database(parts: tuple[str, ...]) -> tuple[str, ...]:
            # PostgreSQL's schema.table and current_database.schema.table are the same relation.
            if self.dialect == "postgres" and database and len(parts) == 2:
                return (database, *parts)
            return parts

        for start in sorted(sites):
            end = _relation_end(tokens, start)
            if end < len(tokens) and tokens[end].text == "(":
                continue  # native table-valued functions are not dbt relations
            raw = sql[tokens[start].start:tokens[end - 1].end]
            parts = relation_parts(raw, dialect=self.dialect)
            if parts is None:
                continue
            if (tokens[start].start, tokens[end - 1].end) in authored:
                continue  # Keep the user's exact package/version and call spelling.
            if len(parts) == 1 and any(name == parts[0] and low <= start < high for name, low, high in ctes):
                continue
            if self.dialect == "postgres" and len(parts) == 1 and schema:
                parts = (schema, *parts)
            parts = in_database(parts)
            if self.dialect == "postgres" and database and len(parts) == 3 and parts[0] != database:
                raise DbtError(f"{raw!r} belongs to another database; choose this project's PostgreSQL target.")
            matches = [item for item in self.relations
                       if in_database(relation_parts(item.relation, dialect=self.dialect) or ()) == parts]
            expressions = {item.expression() for item in matches}
            if len(expressions) != 1:
                reason = "Ambiguous" if expressions else "No authoritative"
                raise DbtError(f"{reason} dbt mapping for {raw!r}. Use the exact manifest relation_name or a literal YAML source; refresh artifacts outside PGTUI.")
            # Preserve comments embedded between qualified identifier components.
            comments = [token.text for token in lex_sql(raw, dialect=self.dialect) if token.kind == "comment"]
            suffix = "".join(" " + comment + ("\n" if comment.startswith("--") else " ") for comment in comments)
            replacements.append((tokens[start].start, tokens[end - 1].end, expressions.pop() + suffix))
        if any(low < end and start < high for start, end, _ in replacements for low, high, _ in templates):
            raise DbtError("A template overlaps a native relation name; separate them before converting.")
        if templates:
            # Several config() calls can share one SQL offset. Keep their authored order.
            calls: dict[int, list[str]] = {}
            for start, end, template in templates:
                if start == end:
                    calls.setdefault(start, []).append(template)
                else:
                    replacements.append((start, end, template))
            replacements.extend((start, start, "".join(group)) for start, group in calls.items())
        for start, end, replacement in sorted(replacements, reverse=True):
            sql = sql[:start] + replacement + sql[end:]
        return sql

    def config_calls(self, jinja: str, *, database: str = "") -> tuple[str, ...]:
        templates = []
        self._simple_import(jinja, {}, database=database, templates=templates)
        return tuple(template for start, end, template in templates if start == end)

    def relation_call_at(self, sql: str, offset: int, *, database: str = "", schema: str = "") -> str | None:
        """Map a native relation click using the same scope and authority as conversion."""
        tokens = self._query_tokens(sql)
        ctes = _cte_ranges(tokens, self.dialect)
        sites = set(_relation_sites(tokens))
        if tokens[0].text.upper() == "TABLE" and len(tokens) > 1:
            sites.add(1)
        for start in sites:
            end = _relation_end(tokens, start)
            if not tokens[start].start <= offset < tokens[end - 1].end:
                continue
            if end < len(tokens) and tokens[end].text == "(":
                return None
            relation = sql[tokens[start].start:tokens[end - 1].end]
            parts = relation_parts(relation, dialect=self.dialect)
            if parts and len(parts) == 1 and any(name == parts[0] and low <= start < high for name, low, high in ctes):
                return None
            call = self.to_dbt("SELECT * FROM " + relation, database=database, schema=schema).removeprefix("SELECT * FROM ")
            return call if call.startswith("{{") else None
        return None

    def _resolve(self, kind: str, args: list, kwargs: dict, *, database: str = "") -> str:
        if kind == "source":
            if len(args) != 2 or kwargs or not all(isinstance(arg, str) for arg in args):
                raise DbtError("source() requires two literal names.")
            matches = [item for item in self.relations if item.kind == kind and (item.source, item.name) == tuple(args)]
        else:
            if len(args) not in {1, 2} or not all(isinstance(arg, str) for arg in args) or set(kwargs) - {"v", "version"} or len(kwargs) > 1:
                raise DbtError("ref() requires literal model/package names and optional version=.")
            package, name = (self.name, args[0]) if len(args) == 1 else args
            version = kwargs.get("version", kwargs.get("v"))
            matches = [item for item in self.relations if item.kind == kind and item.package == package and item.name == name
                       and (item.version == version if version is not None else
                            item.version is None or item.version == item.latest_version)]
        values = {item.relation for item in matches}
        if len(values) != 1:
            raise DbtError(f"Cannot resolve {kind}{tuple(args)!r} unambiguously. Supply a current manifest with relation_name, or an actual compiled artifact.")
        relation = values.pop()
        if self.dialect == "postgres" and database:
            parts = relation_parts(relation, dialect=self.dialect)
            if parts is not None and len(parts) == 3 and parts[0] != database:
                raise DbtError(f"{relation!r} belongs to another database; refresh the project's target manifest.")
        return relation

    def _compiled(self, path: str, raw: str, node: dict | None) -> str | None:
        if node is not None:
            recorded = node.get("raw_code", node.get("raw_sql"))
            if recorded is not None and recorded != raw:
                raise DbtError("The manifest is stale for this model. Compile the current model outside PGTUI and reload.")
            compiled = node.get("compiled_code", node.get("compiled_sql"))
            if isinstance(compiled, str) and compiled.strip():
                return compiled
        artifact = self.target / "compiled" / self.name / _relative(path)
        artifact = self._path(artifact.relative_to(self.root).as_posix())
        if artifact.is_file():
            if artifact.stat().st_mtime_ns < self._path(path).stat().st_mtime_ns:
                raise DbtError("The compiled SQL file predates this model. Compile outside PGTUI and reload.")
            return self._read(artifact.relative_to(self.root).as_posix())
        return None

    def _project_config(self, path: str, name: str) -> dict:
        config = {}
        tree = self.project.get("models", {})
        relative = self._path(path)
        directory = next(root for root in self.model_paths if relative.is_relative_to(root))
        levels = [self.name, *relative.relative_to(directory).parts[:-1], name]
        for level in [*levels, None]:
            if not isinstance(tree, dict):
                raise DbtError("Project model configuration must be a mapping.")
            for key, value in tree.items():
                if isinstance(key, str) and key.startswith("+"):
                    config[key[1:]] = value
                elif key in {"materialized", "tags", "enabled", "alias", "schema", "database"} and not isinstance(value, dict):
                    config[key] = value
            if level is not None:
                tree = tree.get(level, {})
        return config

    def import_model(self, path: str) -> ImportedModel:
        candidate = self._path(path)
        if path not in self.models:
            raise DbtError("Select a .sql model within the configured model-paths.")
        raw = self._read(path)
        nodes = [node for node in self.nodes if node.get("original_file_path") == path and node.get("package_name") == self.name]
        if len(nodes) > 1:
            raise DbtError("Multiple manifest nodes use this file; import an unambiguous compiled model.")
        node = nodes[0] if nodes else None
        name = node.get("name", candidate.stem) if node else candidate.stem
        _, metadata = self._model_metadata(name)
        config = dict(node.get("config", {})) if node else self._project_config(path, name)
        yaml_config = metadata.get("config", {})
        if not isinstance(yaml_config, dict):
            raise DbtError("Model YAML config must be a mapping.")
        if node is None:
            config.update(yaml_config)
        try:
            sql = self._simple_import(raw, config)
            used_compiled = False
        except DbtError as exc:
            sql = self._compiled(path, raw, node)
            if sql is None:
                raise DbtError(f"{exc} No compiled artifact found in manifest.json or target/compiled/<project>/{path}.") from exc
            if any(marker in sql for marker in ("{{", "{%", "{#")):
                raise DbtError("The compiled artifact still contains Jinja; regenerate it outside PGTUI.")
            used_compiled = True
        return ImportedModel(sql, name, str(config.get("materialized", "view")),
                             str(metadata.get("description", node.get("description", "") if node else "")),
                             config, used_compiled)

    def _simple_import(self, raw: str, config: dict, *, database: str = "",
                       templates: list[tuple[int, int, str]] | None = None) -> str:
        result = []
        offset = 0
        rendered_size = 0
        while match := _JINJA_OPEN.search(raw, offset):
            start = match.end()
            if match[0] != "{{" or raw.startswith("-", start):
                raise DbtError("Jinja blocks, comments and whitespace controls require compiled SQL.")
            end = _expression_end(raw, start)
            if raw[end - 1] == "-":
                raise DbtError("Jinja whitespace controls require compiled SQL.")
            kind, args, kwargs = _call(raw[start:end])
            if kind == "config":
                if args:
                    if len(args) != 1 or not isinstance(args[0], dict) or kwargs:
                        raise DbtError("config() requires literal keyword arguments or one literal dictionary.")
                    kwargs = args[0]
                if any(not isinstance(key, str) for key in kwargs):
                    raise DbtError("config() keys must be strings.")
                config.update(kwargs)
                replacement = ""
            else:
                replacement = self._resolve(kind, args, kwargs, database=database)
            rendered_size += match.start() - offset
            if templates is not None:
                templates.append((rendered_size, rendered_size + len(replacement), raw[match.start():end + 2]))
            rendered_size += len(replacement)
            result.extend((raw[offset:match.start()], replacement))
            offset = end + 2
        result.append(raw[offset:])
        return "".join(result)

