"""Read-only dbt model rename previews and optimistic, rollback-capable publication."""
from __future__ import annotations

import ast
from bisect import bisect_left
from collections.abc import Collection, Iterator
from dataclasses import dataclass
from fnmatch import fnmatchcase
import io
import os
from pathlib import Path
import re
import stat
import tempfile
import tokenize

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode
from yaml.tokens import ScalarToken

from d8r.dbt import DbtError, _linked, _yaml
from d8r.dbt_files import DbtFile, DbtFiles, _FILE_LIMIT, _TREE_LIMIT, file_revision
from d8r.dbt_ignore import DbtIgnore
from d8r.dbt_index import template_calls, template_declarations


_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,127}\Z")
_IMPLICIT = re.compile(r"yes|no|true|false|on|off|null", re.IGNORECASE)
_SELECTOR = re.compile(r"(@?(?:\d*\+)?)(?:([\w.]+):)?(.*?)(\+\d*)?\Z")
_STRING = "tag:yaml.org,2002:str"
_YAML = {".yml", ".yaml"}
_ROOTS = dict(models="models", tests="tests", snapshots="snapshots", analyses="analyses",
              macros="macros", seeds="seeds")
_PATH_KEYS = {"models": "model-paths", "tests": "test-paths", "snapshots": "snapshot-paths",
              "analyses": "analysis-paths", "macros": "macro-paths", "seeds": "seed-paths"}
_CONTROL = {"dbt_project.yml", "selectors.yml", ".dbtignore"}
_SCAN_LIMIT = 64 * 1024 * 1024
_EDIT_LIMIT = 16 * 1024 * 1024
_PATCH_LIMIT = 20_000


@dataclass(frozen=True)
class DbtRenameEdit:
    path: str
    destination: str
    before: DbtFile
    text: str


@dataclass(frozen=True)
class _FileStamp:
    revision: str
    bom: bool
    mode: int
    identity: tuple[int, int, int, int, int]  # Device, inode, mtime, ctime and size.


@dataclass(frozen=True)
class DbtRenamePlan:
    old_name: str
    new_name: str
    old_path: str
    new_path: str
    edits: tuple[DbtRenameEdit, ...]
    _root: Path
    _files: tuple[tuple[str, _FileStamp], ...]
    _column_offset: int | None = None
    _dialect: str = 'postgres'


_Patch = tuple[int, int, str]


def _patch(text: str, patches: Collection[_Patch]) -> str:
    if len(patches) > _PATCH_LIMIT:
        raise DbtError("Too many edits in one file to rename safely.")
    chunks = []
    end = 0
    for start, stop, replacement in sorted(patches):
        if not end <= start < stop <= len(text):
            raise DbtError("Overlapping or unrecognized model reference spans.")
        chunks.extend((text[end:start], replacement))
        end = stop
    chunks.append(text[end:])
    return "".join(chunks)


def _local_ref(args: tuple, kwargs: dict, package: str, old: str, new: str) -> bool:
    if len(args) == 2 and isinstance(args[0], str) and args[0] != package:
        return False
    if len(args) not in (1, 2) or not all(isinstance(arg, str) for arg in args):
        raise DbtError("Cannot resolve a nonliteral or malformed ref() safely.")
    if args[-1] == new:
        raise DbtError(f"An existing ref() to {new!r} could change its binding after the rename.")
    if args[-1] != old:
        return False
    if kwargs:
        if set(kwargs) <= {"version", "v"}:
            raise DbtError(f"Cannot rename a version-qualified reference to {old!r}.")
        raise DbtError(f"Cannot safely resolve arguments of ref() to {old!r}.")
    return True


def _jinja(text: str, package: str, old: str, new: str) -> list[_Patch]:
    result = []
    for call in template_calls(text, strict=True):
        if call.kind != "ref":
            continue
        if call.dynamic:
            raise DbtError("Cannot prove that a dynamic ref() does not refer to this model.")
        if not _local_ref(call.args, call.kwargs, package, old, new):
            continue
        if call.name_span is None:
            raise DbtError("Cannot locate the quoted ref() argument safely.")
        start, stop = call.name_span
        quote = text[start:start + 1]
        if quote not in ("'", '"') or text[stop - 1:stop] != quote:
            raise DbtError("Cannot preserve the ref() argument's quoting.")
        result.append((start, stop, quote + new + quote))
    return result


def _python(text: str, package: str, old: str, new: str) -> list[_Patch]:
    try:
        tree = ast.parse(text)
    except (SyntaxError, RecursionError, ValueError) as exc:
        raise DbtError(f"Cannot parse Python model safely: {exc}.") from exc
    # AST columns are UTF-8 byte offsets, and Python uses universal newlines.
    line_starts = [0, *(match.end() for match in re.finditer(r"\r\n|\r|\n", text)), len(text)]
    lines: dict[int, tuple[list[int], list[int]]] = {}

    def offset(line: int, column: int) -> int:
        start = line_starts[line - 1]
        if line not in lines:
            chars = text[start:line_starts[line]]
            positions, extra = [], [0]
            if not chars.isascii():
                for index, char in enumerate(chars):
                    if not char.isascii():
                        positions.append(index + extra[-1])
                        extra.append(extra[-1] + len(char.encode("utf-8")) - 1)
            lines[line] = positions, extra
        positions, extra = lines[line]
        return start + column - extra[bisect_left(positions, column)]

    result = []
    direct = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (isinstance(func, ast.Name) and func.id == "getattr" and len(node.args) >= 2
                and isinstance(node.args[0], ast.Name) and node.args[0].id == "dbt"
                and not (isinstance(node.args[1], ast.Constant) and node.args[1].value != "ref")):
            raise DbtError("Cannot resolve indirect dbt.ref access safely.")
        if not (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
                and func.value.id == "dbt" and func.attr == "ref"):
            continue
        direct.add(id(func))
        args = tuple(arg.value if isinstance(arg, ast.Constant) else None for arg in node.args)
        if not _local_ref(args, {kw.arg: None for kw in node.keywords}, package, old, new):
            continue
        arg = node.args[-1]
        start = offset(arg.lineno, arg.col_offset)
        stop = offset(arg.end_lineno, arg.end_col_offset)
        literal = text[start:stop]
        try:
            strings = [tok for tok in tokenize.generate_tokens(io.StringIO(literal).readline)
                       if tok.type == tokenize.STRING]
        except (tokenize.TokenError, SyntaxError) as exc:
            raise DbtError("Cannot tokenize the dbt.ref argument safely.") from exc
        delim = re.match(r"([rRuU]?)(\"\"\"|'''|\"|')", literal)
        if len(strings) != 1 or strings[0].string != literal or not delim or "\n" in literal or "\r" in literal:
            raise DbtError("Cannot preserve a split or multiline Python dbt.ref argument.")
        result.append((start, stop, delim[0] + new + delim[2]))
        if len(result) > _PATCH_LIMIT:
            raise DbtError("Too many Python references to rename safely.")
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and id(node) not in direct and node.attr == "ref"
                and isinstance(node.value, ast.Name) and node.value.id == "dbt"):
            raise DbtError("Cannot resolve indirect dbt.ref access safely.")
    return result


def _string(node: Node | None) -> str | None:
    return node.value if isinstance(node, ScalarNode) and node.tag == _STRING else None


def _mapping(node: Node | None) -> dict[str, Node]:
    if not isinstance(node, MappingNode):
        return {}
    return {str(key.value): value for key, value in node.value if _string(key) is not None}


def _nodes(root: Node | None) -> tuple[list[Node], set[int]]:
    """Collect syntax nodes and track shared values (editing one could edit an alias)."""
    visited = set()
    shared = set()
    nodes = []
    pending = [] if root is None else [(root, False)]
    while pending:
        node, alias = pending.pop()
        ident = id(node)
        if ident in shared:
            continue
        if alias or ident in visited:
            shared.add(ident)
        else:
            nodes.append(node)
        visited.add(ident)
        alias = ident in shared
        if isinstance(node, MappingNode):
            keys = set()
            for key, value in reversed(node.value):
                if key.tag == "tag:yaml.org,2002:merge":
                    raise DbtError("YAML merge keys can hide model bindings; expand the merge before renaming.")
                if not isinstance(key, ScalarNode):
                    raise DbtError("Unsupported YAML mapping key.")
                signature = (key.tag, key.value)
                if signature in keys:
                    raise DbtError(f"Duplicate YAML key {key.value!r}.")
                keys.add(signature)
                pending.extend(((value, alias), (key, alias)))
        elif isinstance(node, SequenceNode):
            pending.extend((item, alias) for item in reversed(node.value))
    return nodes, shared


def _compose(text: str) -> Node | None:
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
    except (yaml.YAMLError, RecursionError) as exc:
        raise DbtError(f"Cannot parse YAML safely: {exc}") from exc
    if root is not None and not isinstance(root, MappingNode):
        raise DbtError("Expected a YAML mapping.")
    return root


def _yaml_entries(root: Node | None, group: str) -> Iterator[dict[str, Node]]:
    sequence = _mapping(root).get(group)
    if sequence is None:
        return
    if not isinstance(sequence, SequenceNode):
        raise DbtError(f"Expected a sequence in the YAML {group!r} section.")
    for entry in sequence.value:
        fields = _mapping(entry)
        name = _string(fields.get("name"))
        if not name or any(marker in name for marker in ("{{", "{%", "{#")):
            raise DbtError(f"Expected a literal resource name in the YAML {group!r} section.")
        yield fields


def _scalar_offsets(raw: str, cooked: str, style: str | None, patches: list[_Patch]) -> list[_Patch]:
    """Project recognized cooked spans onto authored text, without reserializing YAML."""
    prefix = 1 if style in ("'", '"') else 0
    body = raw[prefix:len(raw) - prefix]
    if body == cooked:
        if not style and len(patches) == 1 and patches[0][:2] == (0, len(cooked)):
            replacement = patches[0][2]
            if _IMPLICIT.fullmatch(replacement):
                return [(0, len(raw), "'" + replacement + "'")]
        return [(start + prefix, stop + prefix, value) for start, stop, value in patches]
    if prefix and len(patches) == 1 and patches[0][:2] == (0, len(cooked)) and not re.search(r"[\r\n]", body):
        replacement = patches[0][2]
        if re.fullmatch(r"[\w./+@:-]+", replacement):
            return [(prefix, len(raw) - prefix, replacement)]
    # For multiline/escaped YAML, match ordered occurrences of literal spans in the
    # scalar only, then verify decoded values for the entire document after surgery.
    cache: dict[str, tuple[list[int], list[int], str]] = {}
    result = []
    for start, stop, value in patches:
        needle = cooked[start:stop]
        if needle not in cache:
            if len(cache) >= 64:
                raise DbtError("Too many distinct escaped YAML references to relocate safely.")
            wanted = [m.start() for m in re.finditer(re.escape(needle), cooked)]
            forms = [needle]
            if style == "'":
                forms = [needle.replace("'", "''")]
            elif style == '"':
                escaped = needle.replace("\\", "\\\\")
                forms = [escaped.replace('"', '\\"'), escaped.replace('"', '\\x22'),
                         escaped.replace('"', '\\u0022')]
            for form in dict.fromkeys(forms):
                found = [m.start() for m in re.finditer(re.escape(form), body)]
                if len(wanted) == len(found):
                    cache[needle] = (wanted, found, form)
                    break
            else:
                raise DbtError("Cannot locate a reference in escaped or folded YAML without changing formatting.")
        wanted, found, form = cache[needle]
        index = bisect_left(wanted, start)
        if index == len(wanted) or wanted[index] != start:
            raise DbtError("Cannot disambiguate a reference in a YAML scalar.")
        if style == "'":
            value = value.replace("'", "''")
        elif style == '"':
            value = value.replace("\\", "\\\\")
            if '"' in value:
                form_quote = next((q for q in ('\\"', '\\x22', '\\u0022') if q in form), '\\"')
                value = value.replace('"', form_quote)
        where = found[index] + prefix
        result.append((where, where + len(form), value))
    return result


def _yaml_replace(text: str, nodes: list[Node], shared: set[int], edits: dict[int, list[_Patch]]) -> str:
    if not edits:
        return text
    if len(edits) > _PATCH_LIMIT or sum(map(len, edits.values())) > _PATCH_LIMIT:
        raise DbtError("Too many YAML references to rename safely.")
    try:
        tokens = {tok.end_mark.index: tok for tok in yaml.scan(text, Loader=yaml.SafeLoader)
                  if isinstance(tok, ScalarToken)}
    except (yaml.YAMLError, RecursionError) as exc:
        raise DbtError("Cannot map edited YAML values to their authored text.") from exc
    patches = []
    expected = []
    for node in nodes:
        if not isinstance(node, ScalarNode):
            continue
        changes = edits.get(id(node))
        expected.append((node.tag, _patch(node.value, changes) if changes else node.value))
        if not changes:
            continue
        if id(node) in shared:
            raise DbtError("An affected YAML value is shared by an anchor/alias; expand it before renaming.")
        token = tokens.get(node.end_mark.index)
        if token is None or token.value != node.value:
            raise DbtError("Cannot find the source span of an affected YAML value.")
        start, stop = token.start_mark.index, token.end_mark.index
        for a, b, value in _scalar_offsets(text[start:stop], node.value, node.style, changes):
            patches.append((start + a, start + b, value))
    result = _patch(text, patches)
    checked, _ = _nodes(_compose(result))
    if [(node.tag, node.value) for node in checked if isinstance(node, ScalarNode)] != expected:
        raise DbtError("The affected YAML cannot be rewritten without changing unrelated values.")
    return result


@dataclass(frozen=True)
class _ModelMove:
    package: str
    old: str
    new: str
    path: str
    destination: str
    folders: tuple[str, ...]

    def selector(self, value: str, method: str | None) -> str:
        if any(marker in value for marker in ("{{", "{%", "{#")):
            raise DbtError("Cannot resolve a templated model selector offline.")
        if method is None:
            method = ("path" if "/" in value or "\\" in value else "file"
                      if Path(value).suffix.lower() in {".sql", ".py", ".csv"} else "fqn")
        if method == "fqn":
            candidates = [(self.old, self.new)]
            for prefix in ((self.package, *self.folders), self.folders):
                candidates.append((".".join((*prefix, self.old)), ".".join((*prefix, self.new))))
        elif method == "path":
            slash = "\\" if "\\" in value else "/"
            candidates = [(self.path.replace("/", slash), self.destination.replace("/", slash)),
                          ("." + slash + self.path.replace("/", slash),
                           "." + slash + self.destination.replace("/", slash))]
        elif method == "file":
            candidates = [(Path(self.path).name, Path(self.destination).name), (self.old, self.new)]
        else:
            return value
        for old, new in candidates:
            if value == old:
                return new
        if set(value) & set("*?[") and (any(fnmatchcase(old, value) for old, _ in candidates) !=
                                         any(fnmatchcase(new, value) for _, new in candidates)):
            raise DbtError(f"Wildcard selector {value!r} changes membership; review it before renaming.")
        return value


def _selector_edits(root: Node | None, move: _ModelMove, replace) -> None:
    visited = set()

    def visit(node: Node | None) -> None:
        if id(node) in visited:
            return
        visited.add(id(node))
        value = _string(node)
        if value is not None:
            match = _SELECTOR.fullmatch(value)
            if not match or not match[3]:
                raise DbtError("Invalid YAML selector expression.")
            new = move.selector(match[3], match[2])
            if new != match[3]:
                replace(node, value[:match.start(3)] + new + value[match.end(3):])
            return
        fields = _mapping(node)
        groups = [key for key in ("union", "intersection", "exclude") if key in fields]
        for group in groups:
            items = fields.pop(group)
            if not isinstance(items, SequenceNode):
                raise DbtError(f"YAML selector {group!r} must contain a list of selectors.")
            for item in items.value:
                visit(item)
        if not fields and groups:
            return
        if "method" in fields and "value" in fields:
            method, scalar = _string(fields["method"]), fields["value"]
        elif len(fields) == 1:
            method, scalar = next(iter(fields.items()))
        else:
            raise DbtError("Cannot interpret a YAML selector definition safely.")
        value = _string(scalar)
        if method is None or value is None:
            # Other methods, such as config.*, may use non-string values.
            if method is None or method in {"fqn", "file", "path"}:
                raise DbtError("Expected a literal model selector value.")
        else:
            replace(scalar, move.selector(value, method))

    try:
        for entry in _yaml_entries(root, "selectors"):
            if "definition" not in entry:
                raise DbtError("A YAML selector is missing its definition.")
            visit(entry["definition"])
    except RecursionError as exc:
        raise DbtError("YAML selector nesting is too deep to rename safely.") from exc


def _project_config(root: Node | None, move: _ModelMove, paths: tuple[str, ...], replace) -> None:
    node = root
    for part in ("models", move.package, *move.folders):
        node = _mapping(node).get(part)
        if node is None:
            return
        if not isinstance(node, MappingNode):
            raise DbtError(f"Invalid project model configuration at {part!r}.")
    fields = _mapping(node)
    if move.new in fields:
        raise DbtError(f"Project model configuration already contains {move.new!r}.")
    if move.old not in fields:
        return
    # A model's FQN can also be a folder prefix in this same configuration.
    # Moving that key would unexpectedly move all configuration for that folder.
    directory = Path(move.path).parent / move.old
    if any(Path(path).is_relative_to(directory) for path in paths):
        raise DbtError(f"Project model configuration {move.old!r} is shared with a model directory.")
    if not isinstance(fields[move.old], MappingNode):
        raise DbtError(f"Cannot safely interpret project model config {move.old!r}.")
    for key, _ in node.value:
        if _string(key) == move.old:
            replace(key, move.new)


def _version_binding(fields: dict[str, Node], logical: str, names: set[str]) -> bool:
    versions = fields.get("versions")
    if versions is None:
        if "latest_version" in fields:
            raise DbtError(f"Cannot interpret latest_version without versions for {logical!r}.")
        return False
    if not isinstance(versions, SequenceNode):
        raise DbtError(f"Malformed versioned model properties for {logical!r}.")
    # Latest version can fall back to the unsuffixed logical name. Refuse all
    # logical/default/defined_in bindings that could involve either move endpoint.
    found = logical.casefold() in names
    for version in versions.value:
        data = _mapping(version)
        number = data.get("v")
        defined = _string(data.get("defined_in"))
        if (not isinstance(number, ScalarNode) or number.tag not in
                {_STRING, "tag:yaml.org,2002:int", "tag:yaml.org,2002:float"}
                or ("defined_in" in data and not defined)):
            raise DbtError(f"Malformed versioned model properties for {logical!r}.")
        if defined:
            stem = re.split(r"[/\\]", defined)[-1]
            if Path(stem).suffix.lower() in {".sql", ".py"}:
                stem = Path(stem).stem
        else:
            number_value = number.value
            if number.tag != _STRING:
                try:
                    number_value = yaml.safe_load(number_value)
                except (yaml.YAMLError, ValueError) as exc:
                    raise DbtError(f"Malformed version number for {logical!r}.") from exc
            stem = f"{logical}_v{number_value}"
        found |= stem.casefold() in names
    if "latest_version" in fields and not versions.value:
        raise DbtError(f"Cannot interpret latest_version without versions for {logical!r}.")
    return bool(versions.value) and found


def _yaml_edits(text: str, path: str, move: _ModelMove, models: tuple[str, ...]) -> tuple[str, int]:
    root = _compose(text)
    nodes, shared = _nodes(root)
    changes: dict[int, list[_Patch]] = {}
    bare: set[int] = set()
    literal_data: set[int] = set()
    patches = 0

    def replace(node: ScalarNode, value: str) -> None:
        if node.value != value:
            changes[id(node)] = [(0, len(node.value), value)]

    def expression(node: Node | None) -> None:
        if node is None:
            return
        if not isinstance(node, ScalarNode) or node.tag != _STRING:
            raise DbtError("Expected a literal reference expression in YAML properties.")
        bare.add(id(node))

    def mark_values(node: Node | None, target: set[int]) -> None:
        pending = [node] if node is not None else []
        while pending:
            item = pending.pop()
            if id(item) in target:
                continue
            target.add(id(item))
            if isinstance(item, MappingNode):
                pending.extend(value for _, value in item.value)
            elif isinstance(item, SequenceNode):
                pending.extend(item.value)

    if path == "dbt_project.yml":
        _project_config(root, move, models, replace)
    elif path == "selectors.yml":
        _selector_edits(root, move, replace)
        return _yaml_replace(text, nodes, shared, changes), patches
    else:
        ends = {move.old.casefold(), move.new.casefold()}
        for fields in _yaml_entries(root, "models"):
            name = fields["name"]
            logical = name.value
            if _version_binding(fields, logical, ends):
                raise DbtError(f"Cannot rename a versioned model binding ({logical!r}); review its versions manually.")
            if logical == move.old:
                patches += 1
                replace(name, move.new)
            elif logical.casefold() in ends:
                raise DbtError(f"YAML model properties for {logical!r} conflict with this rename.")
        for group in ("snapshots", "seeds", "exposures", "semantic_models", "unit_tests"):
            for fields in _yaml_entries(root, group):
                name = fields["name"].value
                if group in {"snapshots", "seeds"} and name.casefold() in ends:
                    raise DbtError(f"{group.title()} resource/properties {name!r} conflict with this rename.")
                if group == "snapshots":
                    expression(fields.get("relation"))
                elif group == "semantic_models":
                    expression(fields.get("model"))
                elif group == "exposures":
                    deps = fields.get("depends_on")
                    if deps is not None:
                        if not isinstance(deps, SequenceNode):
                            raise DbtError("Expected a list for an exposure's depends_on.")
                        for dep in deps.value:
                            expression(dep)
                elif group == "unit_tests":
                    model = fields.get("model")
                    value = _string(model)
                    if value == move.old:
                        replace(model, move.new)
                    elif value is None:
                        raise DbtError("Expected a literal model name for a unit test.")
                    inputs = fields.get("given")
                    if inputs is not None:
                        if not isinstance(inputs, SequenceNode):
                            raise DbtError("Expected a list of unit test inputs.")
                        for item in inputs.value:
                            input_fields = _mapping(item)
                            expression(input_fields.get("input"))
                            mark_values(input_fields.get("rows"), literal_data)
                    mark_values(_mapping(fields.get("expect")).get("rows"), literal_data)

        # dbt wraps bare ref(...) in generic-test argument values. Tags and
        # fixture rows are literal data; ordinary prose isn't an expression.
        for node in nodes:
            fields = _mapping(node)
            for group in ("tests", "data_tests"):
                tests = fields.get(group)
                if isinstance(tests, SequenceNode):
                    for test in tests.value:
                        if isinstance(test, MappingNode):
                            for _, arguments in test.value:
                                mark_values(arguments, bare)
            if _string(fields.get("type")) == "foreign_key":
                expression(fields.get("to"))
                expression(fields.get("expression"))
    keys = {id(key) for node in nodes if isinstance(node, MappingNode) for key, _ in node.value}
    for node in nodes:
        value = _string(node)
        if value is None or id(node) in keys or (id(node) in literal_data and id(node) not in shared):
            continue
        if any(marker in value for marker in ("{{", "{%", "{#")):
            found = _jinja(value, move.package, move.old, move.new)
        elif id(node) in bare and re.match(r"\s*(?:builtins\.ref|ref|var|env_var|source|doc)\s*\(", value):
            found = [(start - 3, stop - 3, new) for start, stop, new in
                     _jinja("{{ " + value + " }}", move.package, move.old, move.new)]
        else:
            continue
        if found:
            if id(node) in changes:
                raise DbtError("Ambiguous model binding and template in the same YAML value.")
            changes[id(node)] = found
    return _yaml_replace(text, nodes, shared, changes), patches


def _roots(project: DbtFiles) -> dict[str, tuple[Path, ...]]:
    roots = {}
    for group, key in _PATH_KEYS.items():
        configured = project.config.get(key, [_ROOTS[group]])
        if not isinstance(configured, list):
            raise DbtError(f"Expected a literal list of directories in {key!r}.")
        paths = tuple(dict.fromkeys(project._path(value) for value in configured))
        for root in paths:
            if not project._visible(root / "_") or (root.exists() and not root.is_dir()):
                raise DbtError(f"Not a visible dbt resource directory: {root}")
        roots[group] = paths
    return roots


def _file_kind(project: DbtFiles, relative: str, roots: dict[str, tuple[Path, ...]]) -> tuple[str, ...]:
    if relative in _CONTROL:
        return (relative,)
    absolute = project.root / relative
    suffix = absolute.suffix.lower()
    if (absolute.name.startswith((".", "!", "#", "~")) or not project._visible(absolute)
            or absolute.name.casefold() in {"profiles.yml", "profiles.yaml"}):
        return ()
    return tuple(group for group, directories in roots.items() if
                 (suffix in _YAML or suffix == ".sql" and group != "seeds" or
                  suffix == ".py" and group == "models" or suffix == ".csv" and group == "seeds")
                 and any(absolute.is_relative_to(folder) for folder in directories))


def _inventory(project: DbtFiles, roots: dict[str, tuple[Path, ...]], ignore: DbtIgnore) -> tuple[str, ...]:
    directories = {path for group in roots.values() for path in group}
    count = 0
    candidates = []

    def relevant(child: Path) -> bool:
        return project._visible(child / "_") and any(
            child.is_relative_to(root) or root.is_relative_to(child) for root in directories)

    def fail(error: OSError) -> None:
        raise error

    try:
        for folder, subfolders, files in project.root.walk(on_error=fail, follow_symlinks=False):
            count += len(subfolders) + len(files)
            if count > _TREE_LIMIT:
                raise DbtError('Too many project entries to safely preview the rename.')
            keep = []
            for name in subfolders:
                child = folder / name
                if not relevant(child):
                    continue
                relative = child.relative_to(project.root).as_posix()
                if ignore.git_ignored(relative + '/'):
                    continue
                if _linked(child):
                    if not ignore.match_file(relative + "/"):
                        raise DbtError(f"A dbt resource directory is a symlink/junction: {relative}")
                else:
                    keep.append(name)
            subfolders[:] = keep
            for name in files:
                child = folder / name
                relative = child.relative_to(project.root).as_posix()
                if name.startswith('.') or ignore.match_file(relative):
                    continue
                if _file_kind(project, relative, roots):
                    if relative == 'dbt_project.yml' and ignore.match_file(relative):
                        raise DbtError('Cannot rename when dbt_project.yml is ignored.')
                    _stat(project.editable_path(relative))
                    candidates.append(relative)
                elif relevant(child) and name.casefold() not in {"profiles.yml", "profiles.yaml"} and _linked(child):
                    # Path.walk lists symlinked directories as file entries.
                    if not ignore.match_file(relative + "/"):
                        raise DbtError(f"A possible dbt resource directory is a symlink/junction: {relative}")
    except OSError as exc:
        raise DbtError(f"Cannot inspect dbt resource directories safely: {exc}") from exc
    if 'dbt_project.yml' not in candidates:
        raise DbtError('Cannot rename when dbt_project.yml is ignored or missing.')
    return tuple(sorted(set(candidates) | ignore.controls, key=lambda path: (path.casefold(), path)))


def _stat(path: Path) -> tuple[int, tuple[int, int, int, int, int]]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise DbtError(f"Not a regular dbt resource: {path.name}")
    return stat.S_IMODE(info.st_mode), (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_ctime_ns, info.st_size)


def _read(project: DbtFiles, relative: str) -> tuple[DbtFile | None, _FileStamp]:
    try:
        path = project.editable_path(relative)
        mode, identity = _stat(path)
        before = None if path.suffix.lower() == ".csv" else project.read(relative)
        if (mode, identity) != _stat(path):
            raise DbtError(f"{relative} changed on disk while being read.")
    except OSError as exc:
        raise DbtError(f"Cannot inspect {relative} on disk.") from exc
    return before, _FileStamp(before.revision if before else "", bool(before and before.bom), mode, identity)


def _ignore(project: DbtFiles) -> DbtIgnore:
    return DbtIgnore(project)


def _check_dirty(project: DbtFiles, roots: dict[str, tuple[Path, ...]], ignored: DbtIgnore,
                 dirty: Collection[str]) -> None:
    for draft in sorted(dirty):
        if (draft in _CONTROL or Path(draft).name == '.gitignore' or
                _file_kind(project, draft, roots) and not ignored.match_file(draft)):
            raise DbtError(f'Save or discard the unsaved dbt buffer {draft} before renaming.')


def _unoccupied(project: DbtFiles, relative: str) -> None:
    path = project.editable_path(relative)
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise DbtError(f"Cannot inspect rename destination: {relative}") from exc
    raise DbtError(f"Rename destination already exists: {relative}")


def plan_rename(project: DbtFiles, path: str, name: str, *, dirty: Collection[str] = ()) -> DbtRenamePlan:
    """Preview a local model rename. No disk writes or evaluation of project code."""
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise DbtError("Use a model name with 1–128 ASCII letters, digits or underscores, starting with a letter or underscore.")
    # Refresh path configuration and exclusions, even when the editor object was
    # opened before an external dbt_project.yml change.
    project = DbtFiles(project.root)
    if project.config_error:
        raise DbtError(project.config_error)
    if not isinstance(project.name, str) or not _NAME.fullmatch(project.name):
        raise DbtError("The dbt project must have a literal project name before renaming a model.")
    source = project.editable_path(path)
    path = source.relative_to(project.root).as_posix()
    old = source.stem
    if old.casefold() == name.casefold():
        raise DbtError("Choose a different model name; case-only renames are not supported.")
    destination = source.with_name(name + source.suffix).relative_to(project.root).as_posix()
    roots = _roots(project)
    model_roots = [folder for folder in roots["models"] if source.is_relative_to(folder)]
    if len(model_roots) != 1 or _file_kind(project, path, roots) != ("models",):
        raise DbtError("Select an unambiguous SQL or Python model in a configured model directory.")
    if source.suffix.lower() not in {".sql", ".py"}:
        raise DbtError("Select a SQL or Python model file.")
    _unoccupied(project, destination)
    ignored = _ignore(project)
    if ignored.match_file(path) or ignored.match_file(destination):
        raise DbtError('Cannot rename a model from or into a path excluded by .dbtignore or .gitignore.')
    _check_dirty(project, roots, ignored, dirty)
    candidates = _inventory(project, roots, ignored)
    models = tuple(candidate for candidate in candidates if Path(candidate).suffix.lower() in {".sql", ".py"}
                   and "models" in _file_kind(project, candidate, roots))
    if path not in models:
        raise DbtError(f"No editable local model at {path}.")
    endpoints = {old.casefold(), name.casefold()}
    for candidate in models:
        if candidate != path and Path(candidate).stem.casefold() in endpoints:
            raise DbtError(f"Another model conflicts with this rename: {candidate}.")
    move = _ModelMove(project.name, old, name, path, destination, source.parent.relative_to(model_roots[0]).parts)
    edits = []
    files = []
    scan_size = edit_size = property_count = 0
    for candidate in candidates:
        suffix = Path(candidate).suffix.lower()
        try:
            before, stamp = _read(project, candidate)
            files.append((candidate, stamp))
            if before is None:
                if Path(candidate).stem.casefold() in endpoints:
                    raise DbtError("Seed name conflicts with this rename.")
                continue
            original = before.text
            scan_size += stamp.identity[-1]
            if scan_size > _SCAN_LIMIT:
                raise DbtError("Too much dbt source text to preview this rename safely.")
            if candidate == "dbt_project.yml" and _yaml(original, Path(candidate)) != project.config:
                raise DbtError("Project configuration changed while planning the rename.")
            changed = original
            if suffix == ".sql":
                kinds = _file_kind(project, candidate, roots)
                if "snapshots" in kinds or "macros" in kinds:
                    for declaration in template_declarations(original, strict=True):
                        if declaration.kind == "snapshot" and declaration.name.casefold() in endpoints:
                            raise DbtError(f"Snapshot {declaration.name!r} conflicts with this rename.")
                        if declaration.kind == "macro" and declaration.name == "ref":
                            raise DbtError("The project overrides ref(); its model bindings cannot be verified offline.")
                changed = _patch(original, _jinja(original, move.package, old, name))
            elif suffix == ".py":
                changed = _patch(original, _python(original, move.package, old, name))
            elif suffix in _YAML:
                changed, count = _yaml_edits(original, candidate, move, models)
                property_count += count
                if property_count > 1:
                    raise DbtError(f"Multiple YAML model properties describe {old!r}.")
            if changed != original or candidate == path:
                size = len(changed.encode("utf-8")) + (3 if before.bom else 0)
                if size > _FILE_LIMIT:
                    raise DbtError("The edited file would exceed the 2 MiB editor limit.")
                edit_size += size + stamp.identity[-1]
                if edit_size > _EDIT_LIMIT:
                    raise DbtError("The rename preview would be too large.")
                edits.append(DbtRenameEdit(candidate, destination if candidate == path else candidate, before, changed))
        except DbtError as exc:
            raise DbtError(f"Cannot rename model: {candidate}: {exc}") from exc
    plan = DbtRenamePlan(old, name, path, destination, tuple(edits), project.root, tuple(files))
    _check_inventory(project, plan)
    return plan


def _check_inventory(project: DbtFiles, plan: DbtRenamePlan, *, published: dict[str, _FileStamp] | None = None) -> None:
    expected = dict(plan._files)
    moved = plan.old_path != plan.new_path
    if published is None and moved:
        _unoccupied(project, plan.new_path)
    if published is not None:
        if moved:
            del expected[plan.old_path]
        expected.update(published)
    ignored = _ignore(project)
    if ignored.match_file(plan.old_path) or ignored.match_file(plan.new_path):
        raise DbtError('A rename endpoint is now excluded by .dbtignore or .gitignore.')
    current = _inventory(project, _roots(project), ignored)
    if set(current) != expected.keys():
        raise DbtError("The set of dbt inputs changed on disk. Create a new rename preview.")
    for path in current:
        if _read(project, path)[1] != expected[path]:
            raise DbtError(f"{path} changed on disk. Create a new rename preview.")


def _stage(project: DbtFiles, relative: str, file: DbtFile, mode: int) -> Path:
    path = project.editable_path(relative)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pgtui-rename-", delete=False) as handle:
            temp = Path(handle.name)
            handle.write(file.text.encode("utf-8-sig" if file.bom else "utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        temp.chmod(mode)
        return temp
    except OSError:
        if temp is not None:
            temp.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class _Published:
    path: str
    before: _FileStamp
    backup: Path | None
    after: DbtFile | None
    identity: tuple[int, int] | None


def _check_previous(project: DbtFiles, path: str, stamp: _FileStamp) -> Path:
    if _read(project, path)[1] != stamp:
        raise DbtError(f"{path} changed during rename; its contents were not overwritten.")
    return project.editable_path(path)


def _check_published(project: DbtFiles, change: _Published) -> _FileStamp:
    current, stamp = _read(project, change.path)
    if (current != change.after or stamp.mode != change.before.mode
            or stamp.identity[:2] != change.identity):
        raise DbtError(f"{change.path} changed after publication.")
    return stamp


def _rollback(project: DbtFiles, changes: list[_Published], retained: set[Path]) -> list[str]:
    errors = []
    for change in reversed(changes):
        try:
            path = project.editable_path(change.path)
            if change.after is None:
                # Restore the removed source without overriding a concurrent create.
                os.link(change.backup, path)
            else:
                _check_published(project, change)
                if change.backup is None:
                    path.unlink()
                else:
                    os.replace(change.backup, path)
        except (DbtError, OSError) as exc:
            message = f"{change.path}: {exc}"
            if change.backup is not None:
                retained.add(change.backup)
                message += f" (original retained at {change.backup.relative_to(project.root).as_posix()})"
            errors.append(message)
    return errors


def apply_rename(project: DbtFiles, plan: DbtRenamePlan) -> dict[str, DbtFile]:
    """Publish a reviewed preview, or refuse stale disk state and roll back writes.

    Callers must keep their buffers clean for the lifetime of the preview. This
    is an optimistic multi-file operation, not a crash-proof filesystem transaction.
    """
    if project.root != plan._root:
        raise DbtError("This rename preview belongs to another dbt project.")
    project = DbtFiles(project.root)
    try:
        _check_inventory(project, plan)
        if plan._column_offset is None:
            current = plan_rename(project, plan.old_path, plan.new_name)
        else:
            from d8r.dbt_column_rename import plan_column_rename
            current = plan_column_rename(project, plan.old_path, plan._column_offset, plan.new_name,
                                         dialect=plan._dialect)
    except DbtError as exc:
        raise DbtError(f"The reviewed rename is no longer safe: {exc}") from exc
    if current != plan:
        raise DbtError("The proposed edits changed. Create and review a new rename preview.")

    originals = dict(plan._files)
    outputs = {}
    temporary = set()
    retained = set()
    staged = {}
    journal = []
    try:
        for edit in plan.edits:
            before = originals[edit.path]
            after = DbtFile(edit.text, file_revision(edit.text), edit.before.bom)
            backup = _stage(project, edit.path, edit.before, before.mode)
            temporary.add(backup)
            output = _stage(project, edit.destination, after, before.mode)
            temporary.add(output)
            staged[edit.path] = (backup, output, _stat(output)[1][:2])
            outputs[edit.destination] = after

        # Validate *all* potential inputs after staging and before the first write.
        _check_inventory(project, plan)
        for edit in sorted(plan.edits, key=lambda item: item.path != item.destination):
            before = originals[edit.path]
            backup, output, identity = staged[edit.path]
            source = _check_previous(project, edit.path, before)
            if edit.path == edit.destination:
                os.replace(output, source)
                journal.append(_Published(edit.path, before, backup, outputs[edit.destination], identity))
            else:
                destination = project.editable_path(edit.destination)
                os.link(output, destination)  # Exclusive; never overwrite a late collision.
                journal.append(_Published(edit.destination, before, None, outputs[edit.destination], identity))
                _check_previous(project, edit.path, before).unlink()
                journal.append(_Published(edit.path, before, backup, None, None))

        published = {entry.path: _check_published(project, entry) for entry in journal if entry.after is not None}
        _check_inventory(project, plan, published=published)
    except (DbtError, OSError) as exc:
        errors = _rollback(project, journal, retained)
        if errors:
            raise DbtError(f"Rename failed: {exc} Rollback incomplete; " + "; ".join(errors)) from exc
        status = "All published changes were rolled back." if journal else "No project files were changed."
        raise DbtError(f"Rename failed: {exc} {status}") from exc
    finally:
        for temp in temporary - retained:
            try:
                temp.unlink(missing_ok=True)
            except OSError:
                # A hidden staging artifact isn't a partially renamed dbt input.
                pass
    return outputs
