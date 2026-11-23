"""Bounded offline dbt discovery and literal Jinja spans. No templates are evaluated."""
from __future__ import annotations

import ast
import csv
import io
import json
import re
import stat
from bisect import bisect_left, bisect_right
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from types import MappingProxyType

import yaml

from d8r.dbt import DbtError, _literal, _UniqueLoader
from d8r.dbt_files import DbtFiles
from d8r.dbt_ignore import DbtIgnore
from d8r.sql import ColumnDef, quote_identifier, relation_parts

_MAX_TEXT = 2 * 1024 * 1024
_MAX_TAGS = 8_192
_MAX_TOKENS = 40_000
_MAX_DEPTH = 128
_OPEN = re.compile(r"\{[{%#]")
_RAW_END = re.compile(r"\{%[-+]?\s*endraw\s*[-+]?%\}")
# Match strings before punctuation so a balanced string can contain a Jinja delimiter.
_TOKEN = re.compile(r'''\s+|'[^'\\]*(?:\\[\s\S][^'\\]*)*'|"[^"\\]*(?:\\[\s\S][^"\\]*)*"|[A-Za-z_]\w*|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[^\s]''')
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*\Z")
_OPEN_TO_CLOSE = {"(": ")", "[": "]", "{": "}"}
_CLOSERS = frozenset(_OPEN_TO_CLOSE.values())
_DECLARATIONS = {"macro", "snapshot", "test", "materialization"}


@dataclass(frozen=True, slots=True)
class _Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Tag:
    kind: str
    start: int
    end: int
    body_start: int
    body_end: int
    tokens: tuple[_Token, ...] = ()
    closed: bool = True
    broken: bool = False


def _tags(text: str, *, strict: bool = False) -> Iterator[_Tag]:
    """Leave application SQL opaque; Jinja processes it even inside SQL comments."""
    def fail(message: str) -> None:
        if strict:
            raise DbtError(message)

    if len(text) > _MAX_TEXT:
        fail("Jinja document exceeds the offline analysis limit.")
        return
    pos = total = tags = 0
    while match := _OPEN.search(text, pos):
        tags += 1
        if tags > _MAX_TAGS:
            fail("Jinja tag limit exceeded.")
            return
        start = match.start()
        kind = match[0][1]
        pos = match.end()
        if text[pos:pos + 1] in {"-", "+"}:
            pos += 1
        body_start = pos
        if kind == "#":
            close = text.find("#}", pos)
            end = len(text) if close < 0 else close + 2
            if close < 0:
                fail("Unclosed Jinja comment.")
            yield _Tag(kind, start, end, body_start, end, closed=close >= 0)
            pos = end
            continue
        delimiter = "}}" if kind == "{" else "%}"
        stack, tokens = [], []
        broken = closed = False
        body_end = end = len(text)
        while pos < len(text):
            control = int(text[pos] in "-+")
            if (kind == "%" or not stack or stack[-1] != "}") and text.startswith(delimiter, pos + control):
                body_end, end = pos, pos + len(delimiter) + control
                closed = True
                break
            if _OPEN.match(text, pos):
                # Recover past an incomplete tag without losing later draft edits.
                body_end = end = pos
                broken = True
                break
            token = _TOKEN.match(text, pos)
            assert token is not None
            word = token[0]
            if word in {"'", '"'}:
                # Incomplete quote (while editing): preserve its exact prefix, and keep
                # any existing tag delimiter out of that token for completion edits.
                close = text.find(delimiter, token.end())
                body_end = len(text) if close < 0 else close - int(text[close - 1] in "-+")
                end = len(text) if close < 0 else close + 2
                tokens.append(_Token(text[pos:body_end], pos, body_end))
                broken, closed = True, close >= 0
                break
            if not word.isspace():
                total += 1
                if total > _MAX_TOKENS or len(stack) >= _MAX_DEPTH:
                    fail("Jinja token or nesting limit exceeded.")
                    return
                if word in _OPEN_TO_CLOSE:
                    stack.append(_OPEN_TO_CLOSE[word])
                elif word in _CLOSERS:
                    if not stack or stack[-1] != word:
                        broken = True
                    else:
                        stack.pop()
                tokens.append(_Token(word, pos, token.end()))
            pos = token.end()
        broken |= bool(stack)
        if not closed or broken:
            fail(f"Unclosed or unbalanced Jinja tag at character {start + 1}.")
        if kind == "%" and len(tokens) == 1 and tokens[0].text == "raw" and closed and not broken:
            raw_end = _RAW_END.search(text, end)
            if raw_end is None:
                fail("Unclosed Jinja raw block.")
            raw_limit = raw_end.end() if raw_end else len(text)
            yield _Tag("raw", start, raw_limit, end, raw_end.start() if raw_end else raw_limit,
                       closed=raw_end is not None)
            pos = raw_limit
        else:
            if strict and tokens and tokens[0].text in {"raw", "endraw"} and kind == "%":
                fail("Malformed Jinja raw block.")
            yield _Tag(kind, start, end, body_start, body_end, tuple(tokens), closed, broken)
            pos = end


def _string(token: _Token) -> str | None:
    word = token.text
    if len(word) < 2 or word[0] not in {"'", '"'} or word[-1] != word[0]:
        return None
    if "\\" not in word:
        return word[1:-1]
    try:
        # Use a string literal decoder only, never a Jinja or Python evaluator.
        # Triple quotes allow physical newlines as in Jinja's string tokens.
        value = ast.literal_eval(word[0] * 2 + word + word[0] * 2)
        return value if isinstance(value, str) else None
    except (SyntaxError, ValueError, RecursionError):
        return None


def _version(value: object) -> str:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        text = str(value)
        if re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.+-]{0,127}", text):
            return text
    return ""


def _version_literal(tokens: tuple[_Token, ...]) -> str | int | float | None:
    if len(tokens) != 1:
        return None
    node = tokens[0]
    value = _string(node)
    if value is not None:
        return value if _version(value) else None
    if not re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", node.text) or len(node.text) > 128:
        return None
    value = float(node.text) if any(mark in node.text for mark in ".eE") else int(node.text)
    return value if _version(value) else None


@dataclass(frozen=True, slots=True)
class DbtCall:
    kind: str
    start: int
    end: int
    args: tuple[str, ...]
    kwargs: dict[str, str | int | float]
    dynamic: bool
    name_span: tuple[int, int] | None
    tag_start: int
    tag_end: int
    standalone: bool


def _static_call(kind: str, tokens: tuple[_Token, ...]) -> tuple[tuple[str, ...], dict, tuple[int, int] | None] | None:
    if not 1 <= len(tokens) <= 9:
        return None
    args, kwargs, spans = [], {}, []
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if cursor + 1 < len(tokens) and tokens[cursor + 1].text == "=":
            if kind != "ref" or kwargs or token.text not in {"v", "version"}:
                return None
            value = _version_literal(tokens[cursor + 2:cursor + 3])
            if value is None:
                return None
            kwargs[token.text] = value
            cursor += 3
        else:
            value = _string(token)
            if kwargs or not _resource_name(value) or len(args) >= 2:
                return None
            args.append(value)
            spans.append((token.start, token.end))
            cursor += 1
        if cursor < len(tokens):
            if tokens[cursor].text != ",":
                return None
            cursor += 1
    if not args or kind == "source" and len(args) != 2:
        return None
    return tuple(args), kwargs, spans[-1]


def _calls(tags: Iterable[_Tag], *, strict: bool = False) -> Iterator[DbtCall]:
    for tag in tags:
        tokens = tag.tokens
        pairs, stack = {}, []
        for i, token in enumerate(tokens):
            if token.text in _OPEN_TO_CLOSE:
                stack.append(i)
            elif token.text in _CLOSERS and stack and _OPEN_TO_CLOSE[tokens[stack[-1]].text] == token.text:
                pairs[stack.pop()] = i
        if strict:
            for i, token in enumerate(tokens):
                if token.text.startswith(("'", '"')) and ("{" in token.text or "\\" in token.text):
                    decoded = _string(token)
                    if decoded and _OPEN.search(decoded) and re.search(r"\b(?:ref|source)\b", decoded):
                        raise DbtError(f"Embedded Jinja dependency at character {token.start + 1} is not editable safely.")
                if (token.text == "[" and i and tokens[i - 1].text in {"builtins", "context"}
                        and i + 1 < len(tokens) and _string(tokens[i + 1]) in {"ref", "source"}):
                    raise DbtError(f"Indirect dbt function lookup at character {token.start + 1} is not editable safely.")
        for i, token in enumerate(tokens):
            kind = token.text
            if kind not in {"ref", "source"}:
                continue
            start = i
            if i and tokens[i - 1].text == ".":
                if i < 2 or tokens[i - 2].text != "builtins" or i >= 3 and tokens[i - 3].text == ".":
                    continue
                start = i - 2
            if i + 1 == len(tokens) or tokens[i + 1].text != "(":
                if strict:
                    raise DbtError(f"Indirect use of {kind} at character {token.start + 1} is not editable safely.")
                continue
            if tag.kind == "%" and i == 1 and tokens[0].text in _DECLARATIONS:
                if strict:
                    raise DbtError(f"A local declaration shadows {kind} at character {token.start + 1}.")
                continue  # A declaration, not an invocation.
            close = pairs.get(i + 1)
            if close is None and strict:
                raise DbtError(f"Unclosed {kind}() at character {token.start + 1}.")
            static = (_static_call(kind, tokens[i + 2:close])
                      if close is not None and close - i <= 11 else None)
            args, kwargs, name_span = static if static is not None else ((), {}, None)
            yield DbtCall(kind, tokens[start].start, tokens[close].end if close is not None else tag.body_end,
                          args, kwargs, static is None, name_span, tag.start, tag.end,
                          tag.kind == "{" and tag.closed and not tag.broken and start == 0 and close == len(tokens) - 1)


def template_calls(text: str, *, strict: bool = False) -> tuple[DbtCall, ...]:
    """Find literal and dynamic ref/source calls; spans use Python character offsets.

    Strict mode rejects truncated, unbalanced or over-limit lexing. Unsupported
    argument expressions are marked dynamic even in strict mode; callers making
    edits must not treat them as provably unrelated.
    """
    return tuple(_calls(_tags(text, strict=strict), strict=strict))


@dataclass(frozen=True, slots=True)
class DbtDeclaration:
    kind: str
    name: str
    start: int
    end: int
    name_span: tuple[int, int]
    body_start: int
    body_end: int


def _declarations(tags: Iterable[_Tag], *, strict: bool = False) -> Iterator[DbtDeclaration]:
    pending: list[tuple[str, _Tag, _Token]] = []
    for tag in tags:
        tokens = tag.tokens
        if tag.kind != "%" or not tokens:
            continue
        command = tokens[0].text
        if command in _DECLARATIONS:
            if (tag.broken or not tag.closed or len(tokens) < 2 or not _IDENTIFIER.fullmatch(tokens[1].text)
                    or command == "snapshot" and len(tokens) != 2
                    or command in {"macro", "test"} and (len(tokens) < 4 or tokens[2].text != "(")):
                if strict:
                    raise DbtError(f"Malformed Jinja {command} declaration at character {tag.start + 1}.")
                continue
            pending.append((command, tag, tokens[1]))
        elif command.startswith("end") and command[3:] in _DECLARATIONS:
            if not pending or pending[-1][0] != command[3:] or len(tokens) != 1:
                if strict:
                    raise DbtError(f"Unexpected Jinja {command} at character {tag.start + 1}.")
                continue
            kind, opening, name = pending.pop()
            yield DbtDeclaration(kind, name.text, opening.start, opening.end, (name.start, name.end),
                                 opening.end, tag.start)
    if strict and pending:
        raise DbtError(f"Unclosed Jinja {pending[-1][0]} declaration.")
    for kind, opening, name in pending:
        yield DbtDeclaration(kind, name.text, opening.start, opening.end, (name.start, name.end),
                             opening.end, opening.end)


def template_declarations(text: str, *, strict: bool = False) -> tuple[DbtDeclaration, ...]:
    """Return source-ordered dbt block declarations with exact name and body spans."""
    return tuple(sorted(_declarations(_tags(text, strict=strict), strict=strict), key=lambda item: item.start))


def _resource_name(value: object) -> bool:
    return (_literal(value) and len(value) <= 256 and
            not any(ord(char) < 32 for char in value))


_REF_KINDS = {"model", "seed", "snapshot"}
_NODE_KINDS = _REF_KINDS | {"test", "analysis"}
_PATH_DEFAULTS = {
    "model": ("model-paths", "models"), "seed": ("seed-paths", "seeds"),
    "snapshot": ("snapshot-paths", "snapshots"), "test": ("test-paths", "tests"),
    "analysis": ("analysis-paths", "analyses"), "macro": ("macro-paths", "macros"),
}
_MAX_PROJECT_TEXT = 32 * 1024 * 1024
_MAX_ARTIFACT = 32 * 1024 * 1024
_MAX_RESOURCES = 10_000
_MAX_EDGES = 40_000
_MAX_WARNINGS = 100
_MAX_YAML_NODES = 40_000
_ARTIFACT_SCHEMA = re.compile(r"https://schemas\.getdbt\.com/dbt/manifest/v(?:[4-9]|1\d)\.json\Z")


@dataclass(frozen=True, slots=True)
class DbtResource:
    key: str
    kind: str
    name: str
    path: str = ""
    row: int = 0
    package: str = ""
    version: str = ""
    source: str = ""
    columns: tuple[ColumnDef, ...] = ()
    relation: str = ""
    columns_complete: bool = False
    properties_path: str = ""
    properties_row: int = 0
    latest_version: str = ""

    @property
    def label(self) -> str:
        name = f"{self.source}.{self.name}" if self.source else self.name
        package = f"{self.package}." if self.package else ""
        version = f" (v{self.version})" if self.version else ""
        return f"{package}{name}{version} [{self.kind}]"


@dataclass(frozen=True, slots=True)
class DbtMacro:
    name: str
    package: str
    path: str = ""
    row: int = 0


def _latest(versions: Iterable[str]) -> str:
    def order(value: str) -> tuple[int, Decimal | str, str]:
        try:
            numeric = Decimal(value)
            if numeric.is_finite():
                return 0, numeric, value
        except InvalidOperation:
            pass
        return 1, value, value

    return max(versions, key=order, default="")


def _yaml_document(text: str, path: str) -> tuple[dict, yaml.Node]:
    # Compose first to retain marks. Bound alias expansion as well as node count:
    # merge keys can multiply work in SafeLoader's mapping construction.
    loader = _UniqueLoader(text)
    try:
        node = loader.get_single_node()
        if not isinstance(node, yaml.MappingNode):
            raise DbtError(f"{path} must contain a YAML mapping.")
        pending = [(node, 0)]
        count = 0
        while pending:
            current, depth = pending.pop()
            count += 1
            if count > _MAX_YAML_NODES or depth > _MAX_DEPTH:
                raise DbtError(f"{path} exceeds the YAML analysis limit.")
            if isinstance(current, yaml.MappingNode):
                for key, value in current.value:
                    pending.extend(((key, depth + 1), (value, depth + 1)))
            elif isinstance(current, yaml.SequenceNode):
                pending.extend((item, depth + 1) for item in current.value)
        return loader.construct_document(node), node
    except (yaml.YAMLError, RecursionError, ValueError, TypeError) as exc:
        raise DbtError(f"Invalid YAML in {path}: {str(exc)[:240]}") from exc
    finally:
        loader.dispose()


def _child(node: yaml.Node | None, name: str) -> yaml.Node | None:
    if isinstance(node, yaml.MappingNode):
        return next((value for key, value in node.value if isinstance(key, yaml.ScalarNode) and key.value == name), None)
    return None


def _columns(*owners: Mapping) -> tuple[ColumnDef, ...]:
    result: dict[str, ColumnDef] = {}
    for owner in owners:
        entries = owner.get("columns")
        if isinstance(entries, dict):  # manifest nodes key the mapping by column name
            entries = list(entries.values())
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict) or not _resource_name(entry.get("name")):
                continue
            name = entry["name"]
            before = result.get(name, ColumnDef(name, ""))
            datatype = entry.get("data_type", before.type)
            doc = entry.get("description", before.doc)
            result[name] = ColumnDef(name, datatype[:256] if isinstance(datatype, str) else "",
                                     doc[:2048] if isinstance(doc, str) else "")
    return tuple(result.values())


def _identity(item: DbtResource) -> tuple[str, str, str, str, str]:
    return item.kind, item.package, item.source, item.name, item.version


class DbtIndex:
    """Draft-aware resource snapshot scoped to one DbtFiles project.

    Edges and unresolved call spans are keyed by resource key. External artifact
    nodes have empty paths; all nonempty paths are project-relative and navigable.
    """

    def __init__(self, project: DbtFiles, drafts: Mapping[str, str], *, dialect: str = 'postgres',
                 include_artifact: bool = True) -> None:
        builder = _IndexBuilder(project, drafts, dialect)
        builder.discover()
        artifact_edges = builder.artifact() if include_artifact else {}
        self.project = project
        self.dialect = dialect
        self.package = builder.package
        self.resources = MappingProxyType(builder.resources)
        self.macros = tuple(sorted(builder.macros.values(), key=lambda item: (item.package, item.name)))
        self._texts = MappingProxyType(builder.texts)
        by_path = defaultdict(list)
        refs, sources = defaultdict(lambda: defaultdict(list)), defaultdict(dict)
        for item in self.resources.values():
            for path in dict.fromkeys((item.path, item.properties_path)):
                if path:
                    by_path[path].append(item)
            if item.kind in _REF_KINDS:
                refs[item.name][item.package].append(item)
            elif item.kind == "source":
                owners = sources[item.source, item.name]
                owners[item.package] = None if item.package in owners else item
        self._by_path = {path: tuple(items) for path, items in by_path.items()}
        self._refs = {name: {pkg: self._ref_versions(items) for pkg, items in owners.items()}
                      for name, owners in refs.items()}
        self._sources = dict(sources)
        up: dict[str, tuple[str, ...]] = dict.fromkeys(self.resources, ())
        down: dict[str, list[str]] = defaultdict(list)
        unresolved = {}
        total = 0
        for key in self.resources:
            targets: dict[str, None] = {}
            for call in builder.calls.get(key, ()):
                owner = self.resources[key]
                target = None if call.dynamic else self.resolve(call.kind, call.args, call.kwargs, package=owner.package)
                if target is None:
                    unresolved.setdefault(key, []).append(call)
                else:
                    targets[target.key] = None
            targets.update(dict.fromkeys(artifact_edges.get(key, ())))
            total += len(targets)
            if total > _MAX_EDGES:
                builder.warn("dbt dependency edge limit exceeded; some graph edges are omitted.")
                break
            up[key] = tuple(targets)
            for target in targets:
                down[target].append(key)
        self.upstream = MappingProxyType(up)
        self.downstream = MappingProxyType({key: tuple(down[key]) for key in self.resources})
        self.unresolved = MappingProxyType({key: tuple(calls) for key, calls in unresolved.items()})
        self.warnings = tuple(builder.warnings)

    def for_path(self, path: str) -> tuple[DbtResource, ...]:
        return self._by_path.get(path, ())

    def read(self, path: str) -> str:
        """Return the indexed version, never a new disk read."""
        try:
            return self._texts[path]
        except KeyError:
            raise DbtError(f"No indexed text for {path}.") from None

    def sql_files(self) -> Iterator[tuple[str, str]]:
        return ((path, text) for path, text in self._texts.items() if path.lower().endswith('.sql'))

    def resolve(self, kind: str, args: list | tuple, kwargs: dict, *, package: str | None = None) -> DbtResource | None:
        """Resolve literal names in project scope; return None for missing/ambiguous calls."""
        if not all(_resource_name(arg) for arg in args):
            return None
        owner = self.package if package is None else package
        if kind == "source" and len(args) == 2 and not kwargs:
            candidates = self._sources.get(tuple(args), {})
            for namespace in dict.fromkeys((self.package, owner)):
                if namespace in candidates:
                    return candidates[namespace]
            return next(iter(candidates.values())) if len(candidates) == 1 else None
        if kind != "ref" or len(args) not in {1, 2} or len(kwargs) > 1 or set(kwargs) - {"v", "version"}:
            return None
        requested = _version(next(iter(kwargs.values()))) if kwargs else ""
        if kwargs and not requested:
            return None
        matching = self._refs.get(args[-1], {})
        if len(args) == 2:
            return matching.get(args[0], {}).get(requested)
        for namespace in dict.fromkeys((self.package, owner)):
            versions = matching.get(namespace)
            if versions is not None and requested in versions:
                return versions[requested]
        if len(matching) == 1:
            return next(iter(matching.values())).get(requested)
        return None

    @staticmethod
    def _ref_versions(entries: list[DbtResource]) -> dict[str, DbtResource | None]:
        versions: dict[str, DbtResource | None] = {}
        for item in entries:
            versions[item.version] = None if item.version in versions else item
        configured = {item.latest_version for item in entries if item.latest_version}
        if len(configured) > 1 or "" in versions and len(versions) > 1:
            versions[""] = None
        elif "" not in versions:
            latest = next(iter(configured)) if configured else _latest(versions)
            versions[""] = versions.get(latest)
        return versions

    def dependencies(self, key: str, *, direction: str = "upstream", transitive: bool = False) -> tuple[DbtResource, ...]:
        """Follow a directional graph in breadth-first order, terminating on cycles."""
        if direction not in {"upstream", "downstream"}:
            raise ValueError("direction must be upstream or downstream")
        graph = self.upstream if direction == "upstream" else self.downstream
        if not transitive:
            return tuple(self.resources[item] for item in graph.get(key, ()))
        visited, result = {key}, []
        pending = deque(graph.get(key, ()))
        while pending:
            item = pending.popleft()
            if item not in visited:
                visited.add(item)
                result.append(self.resources[item])
                pending.extend(graph[item])
        return tuple(result)


class _IndexBuilder:
    def __init__(self, project: DbtFiles, drafts: Mapping[str, str], dialect: str) -> None:
        self.project, self.drafts, self.dialect = project, drafts, dialect
        self.warnings: dict[str, None] = {}
        self.resources: dict[str, DbtResource] = {}
        self.macros: dict[tuple[str, str], DbtMacro] = {}
        self.texts: dict[str, str] = {}
        self.calls: dict[str, tuple[DbtCall, ...]] = {}
        self.total_text = self.total_calls = 0
        self.config: dict = {}
        if (text := self.read("dbt_project.yml")) is not None:
            try:
                self.config, _ = _yaml_document(text, "dbt_project.yml")
            except DbtError as exc:
                self.warn(str(exc))
        name = self.config.get("name")
        self.package = name if isinstance(name, str) and _IDENTIFIER.fullmatch(name) else ""
        if not self.package:
            self.warn("dbt_project.yml needs a literal package name; local unqualified references remain available.")
        self.excluded = set(project.excluded)
        for key in ("target-path", "log-path", "packages-install-path"):
            if key in self.config:
                try:
                    self.excluded.add(self.directory(self.config[key]))
                except (DbtError, OSError) as exc:
                    self.warn(f"Invalid {key}: {exc}")

    def warn(self, message: str) -> None:
        if len(self.warnings) < _MAX_WARNINGS:
            self.warnings[message[:500]] = None
        else:
            self.warnings.setdefault("Additional dbt index warnings omitted.", None)

    def read(self, path: str) -> str | None:
        if path in self.texts:
            return self.texts[path]
        if self.total_text >= _MAX_PROJECT_TEXT:
            self.warn("dbt index text limit exceeded; some files were omitted.")
            return None
        try:
            self.project.editable_path(path)
            text = self.drafts[path] if path in self.drafts else self.project.read(path).text
            if len(text) > _MAX_TEXT:
                raise DbtError(f"{path} exceeds the dbt index file limit.")
        except (DbtError, OSError) as exc:
            self.warn(str(exc))
            return None
        if self.total_text + len(text) > _MAX_PROJECT_TEXT:
            self.warn("dbt index text limit exceeded; some files were omitted.")
            return None
        self.total_text += len(text)
        self.texts[path] = text
        return text

    def directory(self, value: str) -> Path:
        path = self.project._path(value)
        if path.exists() and not path.is_dir():
            raise DbtError(f"Not a directory: {value}")
        return path.relative_to(self.project.root)

    def roots(self, key: str, default: str) -> tuple[Path, ...]:
        configured = self.config.get(key, [default])
        if not isinstance(configured, list):
            self.warn(f"{key} must be a list of relative directories; using {default!r}.")
            configured = [default]
        if len(configured) > 128:
            self.warn(f"{key} exceeds the configured directory limit.")
            configured = configured[:128]
        result = []
        for value in configured:
            try:
                path = self.directory(value)
                if not self.project._visible(self.project.root / path / "_") or any(
                        path.is_relative_to(excluded) for excluded in self.excluded):
                    raise DbtError(f"Directory is outside editable project paths: {value}")
            except (DbtError, OSError) as exc:
                self.warn(f"Invalid {key}: {exc}")
                continue
            if path not in result:
                result.append(path)
        if not result and configured and configured != [default]:
            return self.roots_default(key, default)
        return tuple(result)

    def roots_default(self, key: str, default: str) -> tuple[Path, ...]:
        try:
            path = self.directory(default)
            if not any(path.is_relative_to(ignore) for ignore in self.excluded):
                return (path,)
        except (DbtError, OSError) as exc:
            self.warn(f"Invalid default {key}: {exc}")
        return ()

    def entries(self, data: dict, node: yaml.Node | None, group: str, path: str) -> Iterator[tuple[dict, yaml.Node | None]]:
        if group not in data:
            return
        items, marks = data[group], _child(node, group)
        if not isinstance(items, list):
            self.warn(f"{path}: {group} must be a list of mappings.")
            return
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                self.warn(f"{path}: invalid {group} entry #{i + 1}.")
                continue
            yield item, marks.value[i] if isinstance(marks, yaml.SequenceNode) and i < len(marks.value) else None

    def enabled(self, owner: dict, path: str, inherit: bool = True) -> bool:
        config = owner.get("config", {})
        if not isinstance(config, dict):
            self.warn(f"{path}: resource config must be a mapping.")
            config = {}
        enabled = config.get("enabled", owner.get("enabled", inherit))
        if not isinstance(enabled, bool):
            self.warn(f"{path}: dynamic or invalid enabled flag; using inherited value.")
            return inherit
        return enabled

    def add(self, kind: str, name: str, path: str, *, row: int = 0, source: str = "", version: str = "",
            props: tuple[str, int] = ("", 0), columns: tuple[ColumnDef, ...] = (), latest: str = "",
            relation: str = "", columns_complete: bool = False, calls: Iterable[DbtCall] = ()) -> None:
        key = f"{kind}:{path}:{row}:{(source, name, version)!r}"
        if key in self.resources:
            self.warn(f"{path}: duplicate {kind} {name!r}.")
            return
        if len(self.resources) >= _MAX_RESOURCES:
            self.warn("dbt resource limit exceeded; some resources were omitted.")
            return
        self.resources[key] = DbtResource(key, kind, name, path, row, self.package, version, source,
                                          columns, relation, columns_complete, *props, latest)
        remaining = _MAX_EDGES - self.total_calls
        recorded = []
        for call in calls:
            if not remaining:
                self.warn("dbt call limit exceeded; some dependencies were omitted.")
                break
            recorded.append(call)
            remaining -= 1
        if recorded:
            self.total_calls += len(recorded)
            self.calls[key] = tuple(recorded)

    def discover(self) -> None:
        roots = {kind: self.roots(*settings) for kind, settings in _PATH_DEFAULTS.items()}
        root_kinds: dict[Path, set[str]] = defaultdict(set)
        for kind, directories in roots.items():
            for directory in directories:
                root_kinds[directory].add(kind)
        ignored = DbtIgnore(self.project, self.drafts, warn=self.warn)
        try:
            paths = set(self.project.paths(ignore_directory=ignored.git_ignored))
        except DbtError as exc:
            self.warn(str(exc))
            paths = set()
        if len(self.drafts) > _MAX_RESOURCES:
            self.warn("Too many dbt drafts; new draft-only files were omitted.")
        else:
            paths.update(self.drafts)
        files: dict[str, list[str]] = defaultdict(list)
        props = []
        order = ("snapshot", "seed", "macro", "test", "analysis", "model")
        for value in sorted(paths):
            if value == "dbt_project.yml":
                continue
            path = Path(value)
            suffix = path.suffix.lower()
            if suffix not in {".sql", ".py", ".csv", ".yml", ".yaml"} or any(
                    path.is_relative_to(ignore) for ignore in self.excluded):
                continue
            kinds = {kind: len(parent.parts) for parent in reversed(path.parents) for kind in root_kinds.get(parent, ())}
            if not kinds:
                continue
            try:
                if value != self.project.editable_path(value).relative_to(self.project.root).as_posix():
                    raise DbtError(f"Use the canonical project-relative path: {value}")
            except (DbtError, OSError) as exc:
                self.warn(str(exc))
                continue
            if ignored.match_file(value):
                continue
            if suffix in {".yml", ".yaml"}:
                props.append(value)
            elif suffix == ".csv" and "seed" in kinds:
                files["seed"].append(value)
            elif suffix == ".py" and "model" in kinds:
                files["model"].append(value)
            elif suffix == ".sql":
                kind = max((kind for kind in order if kind in kinds and kind != "seed"), key=kinds.get, default=None)
                if kind:
                    files[kind].append(value)

        model_stems: dict[str, list[str]] = defaultdict(list)
        for path in files["model"]:
            model_stems[Path(path).stem].append(path)
        claimed: set[str] = set()
        model_specs = []
        properties: dict[tuple[str, str], tuple[str, int, dict]] = {}
        for path in props:
            if (text := self.read(path)) is None:
                continue
            try:
                data, node = _yaml_document(text, path)
            except DbtError as exc:
                self.warn(str(exc))
                continue
            self.add_sources(path, data, node)
            for item, mark in self.entries(data, node, "models", path):
                name = item.get("name")
                if not _resource_name(name):
                    self.warn(f"{path}: model names must be literal strings.")
                    continue
                row = mark.start_mark.line if mark else 0
                versions = list(self.entries(item, mark, "versions", path))
                if not versions:
                    versions = [(None, mark)]
                latest = _version(item.get("latest_version"))
                declared = {_version(version.get("v")) for version, _ in versions if version is not None} - {""}
                if declared and not latest:
                    if "latest_version" in item:
                        self.warn(f"{path}: invalid latest_version on {name!r}.")
                    latest = _latest(declared)
                if latest and latest not in declared:
                    self.warn(f"{path}: latest_version {latest!r} for {name!r} is not declared.")
                for spec, _ in versions:
                    version = _version(spec.get("v")) if spec is not None else ""
                    if spec is not None and not version:
                        self.warn(f"{path}: invalid version on {name!r}.")
                        continue
                    defined = spec.get("defined_in", f"{name}_v{version}") if spec is not None else name
                    model = self.find_model(defined, path, model_stems)
                    if model is None and spec is not None and "defined_in" not in spec and version == latest:
                        model = self.find_model(name, path, model_stems)
                    if model is None:
                        self.warn(f"{path}: no unique file for model {name!r} (defined in {defined!r}).")
                        continue
                    if model in claimed:
                        self.warn(f"{path}: multiple model declarations claim {model}.")
                    claimed.add(model)
                    if self.enabled(spec or {}, path, self.enabled(item, path)):
                        if len(model_specs) < _MAX_RESOURCES:
                            model_specs.append((model, name, path, row, version, latest, item, spec or {}))
                        else:
                            self.warn("dbt resource limit exceeded; some model versions were omitted.")

            for group, kind in (("seeds", "seed"), ("snapshots", "snapshot"), ("analyses", "analysis"),
                                ("data_tests", "test"), ("tests", "test")):
                for item, mark in self.entries(data, node, group, path):
                    name = item.get("name")
                    if not _resource_name(name):
                        self.warn(f"{path}: {kind} names must be literal strings.")
                        continue
                    row = mark.start_mark.line if mark else 0
                    if kind == "snapshot" and "relation" in item:
                        if not self.enabled(item, path):
                            continue
                        relation = item["relation"]
                        calls = ()
                        if isinstance(relation, str) and len(relation) <= 8192:
                            # YAML-decoded relation expressions are dependencies, not editable SQL spans.
                            calls = template_calls(relation if _OPEN.search(relation) else "{{ " + relation + " }}")
                        else:
                            self.warn(f"{path}: invalid relation on snapshot {name!r}.")
                        self.add(kind, name, path, row=row, columns=_columns(item), calls=calls)
                    elif (kind, name) not in properties:
                        properties[kind, name] = path, row, item
                    else:
                        self.warn(f"{path}: multiple properties entries for {kind} {name!r}.")

        scanned: dict[str, tuple[DbtCall, ...]] = {}
        base_columns: dict[int, tuple[ColumnDef, ...]] = {}
        for model, name, props_path, row, version, latest, item, spec in model_specs:
            if not spec.get("columns"):
                # Versions normally inherit the same declared column metadata.
                columns = base_columns.get(id(item))
                if columns is None:
                    columns = base_columns[id(item)] = _columns(item)
            else:
                columns = _columns(item, spec)
            if model not in scanned:
                scanned[model] = self.file_calls(model)
            self.add("model", name, model, version=version, latest=latest, props=(props_path, row),
                     columns=columns, calls=scanned[model])
        for path in files["model"]:
            if path not in claimed:
                name = Path(path).stem
                if _resource_name(name):
                    self.add("model", name, path, calls=self.file_calls(path))

        for path in files["seed"]:
            name = Path(path).stem
            props_path, row, item = properties.get(("seed", name), ("", 0, {}))
            if _resource_name(name) and self.enabled(item, props_path):
                header = self.seed_header(path, item, roots["seed"])
                described = {col.name: col for col in _columns(item)}
                columns = tuple(described.get(col, ColumnDef(col, "")) for col in header) or tuple(described.values())
                self.add("seed", name, path, props=(props_path, row), columns=columns, columns_complete=bool(header))

        for kind in ("macro", "snapshot", "test", "analysis"):
            for path in files[kind]:
                self.add_sql(kind, path, properties)

    def find_model(self, value: object, props: str, stems: Mapping[str, list[str]]) -> str | None:
        try:
            if not _resource_name(value):
                return None
            requested = self.project._path(value).relative_to(self.project.root)
        except (DbtError, OSError):
            return None
        stem = requested.stem if requested.suffix.lower() in {".sql", ".py"} else requested.name
        matches = stems.get(stem, ())
        if len(matches) == 1 and requested.parent == Path(".") and requested.suffix.lower() not in {".sql", ".py"}:
            return matches[0]
        if requested.parent != Path("."):
            parents = {requested.parent, Path(props).parent / requested.parent}
            matches = [path for path in matches if Path(path).parent in parents]
        if requested.suffix.lower() in {".sql", ".py"}:
            matches = [path for path in matches if Path(path).suffix.lower() == requested.suffix.lower()]
        local = [path for path in matches if Path(path).parent == Path(props).parent]
        return local[0] if len(local) == 1 else matches[0] if len(matches) == 1 else None

    def add_sources(self, path: str, data: dict, node: yaml.Node) -> None:
        for source, mark in self.entries(data, node, "sources", path):
            group = source.get("name")
            if not _resource_name(group):
                self.warn(f"{path}: source names must be literal strings.")
                continue
            source_enabled = self.enabled(source, path)
            for table, table_mark in self.entries(source, mark, "tables", path):
                name = table.get("name")
                if not _resource_name(name):
                    self.warn(f"{path}: source table names must be literal strings.")
                    continue
                if not self.enabled(table, path, source_enabled):
                    continue
                row = table_mark.start_mark.line if table_mark else mark.start_mark.line if mark else 0
                self.add("source", name, path, row=row, source=group, columns=_columns(table),
                         relation=self.source_relation(source, table, path))

    def source_relation(self, source: dict, table: dict, path: str) -> str:
        parts = {"database": source.get("database"), "schema": source.get("schema", source["name"]),
                 "identifier": table.get("identifier", table["name"])}
        quoting = {}
        for owner in (self.config, source, table):
            policy = owner.get("quoting", {})
            if not isinstance(policy, dict) or any(not isinstance(value, bool) for value in policy.values()):
                self.warn(f"{path}: source quoting must use literal booleans.")
                return ""
            quoting.update(policy)
        result = []
        for key, value in parts.items():
            if value is None and key == "database":
                continue
            if not _resource_name(value):
                return ""
            identifier = quote_identifier(value) if quoting.get(key, True) else value
            if len(relation_parts(identifier, dialect=self.dialect) or ()) != 1:
                return ""
            result.append(identifier)
        return ".".join(result)

    def seed_header(self, path: str, item: dict, roots: tuple[Path, ...]) -> tuple[str, ...]:
        text = self.read(path)
        if text is None:
            return ()
        settings = self.config.get("seeds")
        delimiter = ","
        if isinstance(settings, dict):
            relative = Path(path)
            root = max((root for root in roots if relative.is_relative_to(root)), key=lambda root: len(root.parts))
            for part in (self.package, *relative.relative_to(root).with_suffix("").parts):
                delimiter = settings.get("+delimiter", delimiter)
                settings = settings.get(part)
                if not isinstance(settings, dict):
                    break
            if isinstance(settings, dict):
                delimiter = settings.get("+delimiter", delimiter)
        if isinstance(item.get("config"), dict):
            delimiter = item["config"].get("delimiter", delimiter)
        if not isinstance(delimiter, str) or len(delimiter) != 1:
            self.warn(f"{path}: seed delimiter is not a literal character.")
            return ()
        try:
            sample = io.StringIO(text[:65_537], newline="")
            header = next(csv.reader(sample, delimiter=delimiter, strict=True), [])
            if sample.tell() >= 65_536 and len(text) > 65_536:
                raise csv.Error("seed header exceeds the analysis limit")
        except (TypeError, csv.Error):
            self.warn(f"{path}: cannot parse the seed CSV header.")
            return ()
        if (len(header) > 2_000 or not all(_resource_name(name) for name in header)
                or len(set(header)) != len(header)):
            self.warn(f"{path}: invalid or duplicate seed column names.")
            return ()
        return tuple(header)

    def file_calls(self, path: str) -> tuple[DbtCall, ...]:
        if Path(path).suffix.lower() != ".sql" or (text := self.read(path)) is None:
            return ()
        return tuple(_calls(self.sql_tags(path, text)))

    def sql_tags(self, path: str, text: str) -> tuple[_Tag, ...]:
        try:
            return tuple(_tags(text, strict=True))
        except DbtError as exc:
            self.warn(f"{path}: {exc} Dependencies may be incomplete.")
            return tuple(_tags(text))

    def add_sql(self, kind: str, path: str, properties: Mapping[tuple[str, str], tuple[str, int, dict]]) -> None:
        text = self.read(path)
        if text is None:
            return
        tags = self.sql_tags(path, text)
        decls = sorted(_declarations(tags), key=lambda item: item.start)
        newlines = [match.end() for match in re.finditer(r"\r\n?|\n", text)] if decls else []
        if kind == "macro":
            for decl in decls:
                if decl.kind == "macro":
                    self.add_macro(decl.name, self.package, path, bisect_right(newlines, decl.start))
            return
        calls = tuple(_calls(tags))
        starts = [call.start for call in calls]
        matching = [decl for decl in decls if decl.kind == kind]
        for decl in matching:
            props_path, row, item = properties.get((kind, decl.name), ("", 0, {}))
            if self.enabled(item, props_path):
                lo = bisect_left(starts, decl.body_start)
                hi = bisect_left(starts, decl.body_end)
                self.add(kind, decl.name, path, row=bisect_right(newlines, decl.start), props=(props_path, row),
                         columns=_columns(item), calls=calls[lo:hi])
        if not matching:
            name = Path(path).stem
            props_path, row, item = properties.get((kind, name), ("", 0, {}))
            if _resource_name(name) and self.enabled(item, props_path):
                self.add(kind, name, path, props=(props_path, row), columns=_columns(item), calls=calls)

    def add_macro(self, name: str, package: str, path: str = "", row: int = 0) -> None:
        if len(self.macros) < _MAX_RESOURCES:
            self.macros.setdefault((package, name), DbtMacro(name, package, path, row))
        else:
            self.warn("dbt macro limit exceeded; some macros were omitted.")

    def artifact(self) -> dict[str, tuple[str, ...]]:
        if not self.package:
            return {}
        target = self.config.get("target-path", "target")
        try:
            directory = self.directory(target)
            protected = [Path(".git"), Path("dbt_packages"), Path("dbt_modules"), Path("logs")]
            for setting in ("packages-install-path", "log-path"):
                if setting in self.config:
                    try:
                        protected.append(self.directory(self.config[setting]))
                    except (DbtError, OSError):
                        pass  # Already reported during configured path discovery.
            if any(part.startswith(".") for part in directory.parts) or any(
                    directory.is_relative_to(path) for path in protected):
                raise DbtError("target-path cannot point into a hidden, package or log directory.")
            # Artifact files are intentionally not editable. Reuse the same confined
            # path walk, and access exactly this fixed filename, never a manifest path.
            path = self.project._path((directory / "manifest.json").as_posix())
            if not stat.S_ISREG(path.stat().st_mode):
                raise DbtError("manifest.json is not a regular file.")
            with path.open("rb") as handle:
                content = handle.read(_MAX_ARTIFACT + 1)
            if len(content) > _MAX_ARTIFACT:
                raise DbtError("manifest.json exceeds the offline artifact size limit.")
            value = json.loads(content.decode("utf-8-sig"))
            if not isinstance(value, dict):
                raise DbtError("manifest.json must contain an object.")
            metadata = value.get("metadata", {})
            if not isinstance(metadata, dict):
                raise DbtError("manifest.json metadata must contain an object.")
            version = metadata.get("dbt_schema_version")
            if version is not None and (not isinstance(version, str) or not _ARTIFACT_SCHEMA.fullmatch(version)):
                raise DbtError("manifest.json has an unrecognized dbt schema version.")
            manifest_project = metadata.get("project_name")
            if manifest_project is not None and manifest_project != self.package:
                raise DbtError("manifest.json belongs to a different dbt project.")
        except FileNotFoundError:
            return {}
        except (DbtError, OSError, ValueError, RecursionError) as exc:
            self.warn(f"Artifact ignored: {str(exc)[:350]}")
            return {}

        live: dict[tuple[str, str, str, str, str], list[str]] = defaultdict(list)
        for item in self.resources.values():
            live[_identity(item)].append(item.key)
        ids = {}
        external = []
        malformed = False
        for group in ("nodes", "sources"):
            entries = value.get(group, {})
            if not isinstance(entries, dict):
                malformed = True
                continue
            for uid, node in entries.items():
                if not isinstance(node, dict):
                    malformed = True
                    continue
                kind = node.get("resource_type", "source" if group == "sources" else "")
                if not isinstance(kind, str) or kind not in (_NODE_KINDS if group == "nodes" else {"source"}):
                    continue
                package, name = node.get("package_name"), node.get("name")
                source = node.get("source_name") if kind == "source" else ""
                node_version = _version(node.get("version")) if kind == "model" else ""
                latest = _version(node.get("latest_version")) if node_version else ""
                if (not isinstance(package, str) or not _IDENTIFIER.fullmatch(package) or not _resource_name(name) or
                        kind == "source" and not _resource_name(source) or
                        kind == "model" and node.get("version") is not None and not node_version):
                    malformed = True
                    continue
                qualified = f"{kind}.{package}." + (source + "." if source else "") + name
                if node_version:
                    qualified += f".v{node_version}"
                # dbt's tests include a digest in their IDs, unlike relation nodes.
                if node.get("unique_id", uid) != uid or not (uid == qualified or kind == "test" and uid.startswith(qualified + ".")):
                    malformed = True
                    continue
                config = node.get("config", {})
                if not isinstance(config, dict):
                    malformed = True
                    continue
                if config.get("enabled") is False:
                    continue
                if package == self.package:
                    matching = live.get((kind, package, source, name, node_version), ())
                    if len(matching) == 1:
                        ids[uid] = matching[0]
                    continue  # Never reintroduce deleted/renamed local resources or stale edges.
                if len(self.resources) + len(external) >= _MAX_RESOURCES:
                    self.warn("dbt resource limit exceeded; some artifact nodes were omitted.")
                    continue
                relation = node.get("relation_name")
                if (not _resource_name(relation) or not relation_parts(relation, dialect=self.dialect) or
                        config.get("materialized") == "ephemeral"):
                    relation = ""
                item = DbtResource(uid, kind, name, package=package, version=node_version, source=source,
                                   columns=_columns(node), relation=relation, latest_version=latest)
                external.append((item, node.get("depends_on")))

        # Older/minimal artifacts may not name the root project. Only accept them
        # when a node independently matches the live local project namespace.
        if manifest_project is None and not ids:
            self.warn("Artifact ignored: manifest.json has no matching local project metadata.")
            return {}
        for item, _ in external:
            self.resources[item.key] = item
            ids[item.key] = item.key
        edges = {}
        total_edges = 0
        for item, depends in external:
            if depends is None:
                continue
            if not isinstance(depends, dict) or not isinstance(depends.get("nodes", []), list):
                malformed = True
                continue
            targets = {}
            for uid in depends.get("nodes", []):
                if isinstance(uid, str) and uid in ids and ids[uid] not in targets:
                    if total_edges >= _MAX_EDGES:
                        self.warn("dbt dependency edge limit exceeded; some artifact edges were omitted.")
                        break
                    targets[ids[uid]] = None
                    total_edges += 1
            if targets:
                edges[item.key] = tuple(targets)

        macros = value.get("macros", {})
        if isinstance(macros, dict):
            for uid, entry in macros.items():
                if not isinstance(entry, dict):
                    malformed = True
                    continue
                package, name = entry.get("package_name"), entry.get("name")
                if (not isinstance(package, str) or not _IDENTIFIER.fullmatch(package) or not isinstance(name, str)
                        or not _IDENTIFIER.fullmatch(name) or uid != f"macro.{package}.{name}"
                        or entry.get("unique_id", uid) != uid or entry.get("resource_type", "macro") != "macro"):
                    malformed = True
                    continue
                if package != self.package:
                    self.add_macro(name, package)
        else:
            malformed = True
        if malformed:
            self.warn("Some invalid dbt manifest entries were ignored.")
        return edges
