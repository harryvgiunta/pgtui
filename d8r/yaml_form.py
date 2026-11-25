"""Immutable form views of ordinary dbt YAML, with source-local, checked edits.

Use () for the root. Mapping path components are strings; list components are
zero-based integers. An empty stream is a virtual empty mapping until edited.
Only concrete source ranges are edited: an untouched value never passes through
an emitter. PyYAML's SafeLoader supplies the same implicit scalar types as dbt
project validation; neither tags, Jinja nor aliases are evaluated.
"""
from __future__ import annotations

import json
import math
import re
from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import yaml
from yaml import tokens

YamlPath = tuple[str | int, ...]
_MAX_TEXT = 512 * 1024
_MAX_NODES = 20_000  # Includes mapping keys.
_MAX_DEPTH = 48
_MAX_NUMERIC = 1024
_CONTAINERS = {"list", "mapping"}
_KINDS = {"str": "text", "int": "integer", "float": "number", "bool": "boolean", "null": "null"}


class YamlFormError(ValueError):
    """An actionable error safe to display without exposing file contents."""


def _raw(reason: str) -> YamlFormError:
    return YamlFormError(f"{reason} Use RAW mode to edit this YAML.")


@dataclass(frozen=True, slots=True)
class _Node:
    kind: str
    value: str
    children: tuple[YamlPath, ...]
    start: int
    end: int  # End of concrete syntax, not a block-end mark at the next sibling.
    column: int
    entry: int  # First character of the mapping key or sequence dash/item.
    key_end: int
    block: bool = False


_COMMENT = re.compile(r"#[^\n]*")


class _Loader(yaml.SafeLoader):
    """Bound composition before descending; keep comment and dash source marks."""

    def __init__(self, text: str):
        super().__init__(text)
        self.source = text
        self.concrete_end = 0
        self.comments: list[tuple[int, str]] = []
        self.dashes: list[int] = []
        self.depth = self.count = 0

    def get_token(self):
        token = super().get_token()
        if token is None:
            return token
        if isinstance(token, (tokens.AnchorToken, tokens.AliasToken, tokens.TagToken, tokens.DirectiveToken)):
            raise _raw("Anchors, aliases, tags and directives are not supported in the form.")
        a, b = token.start_mark.index, token.end_mark.index
        if isinstance(token, tokens.KeyToken) and a != b:
            raise _raw("Explicit mapping keys are not supported in the form.")
        if isinstance(token, tokens.BlockEntryToken):
            self.dashes.append(a)
        if a >= self.concrete_end:
            # Comments live between tokens, except on literal/folded scalar headers.
            self.comments.extend((m.start(), m[0]) for m in _COMMENT.finditer(self.source, self.concrete_end, a))
            self.concrete_end = b
        if isinstance(token, tokens.ScalarToken) and token.style in ("|", ">"):
            end = self.source.find("\n", a, b)
            match = _COMMENT.search(self.source, a, b if end < 0 else end)
            if match:
                self.comments.append((match.start(), match[0]))
        return token

    def compose_node(self, parent, index):
        self.depth += 1
        self.count += 1
        if self.depth > _MAX_DEPTH or self.count > _MAX_NODES:
            raise _raw("This document exceeds the form's nesting or node limit.")
        try:
            return super().compose_node(parent, index)
        finally:
            self.depth -= 1


def _scalar(loader: _Loader, node: yaml.ScalarNode) -> tuple[str, str]:
    kind = _KINDS.get(node.tag.removeprefix("tag:yaml.org,2002:"))
    if kind is None:
        raise _raw("Only ordinary text, integers, finite numbers, booleans and nulls are supported.")
    if kind in ("integer", "number") and len(node.value) > _MAX_NUMERIC:
        raise _raw("This numeric literal is too large for the form.")
    value = loader.construct_object(node)
    if kind == "number" and not math.isfinite(value):
        raise _raw("Non-finite numbers are not supported in the form.")
    if kind == "text" and re.search(r"[\ud800-\udfff]", value):
        raise _raw("An invalid Unicode scalar is not editable in the form.")
    return kind, ("null" if value is None else str(value).lower() if kind == "boolean" else str(value))


def _parse(source: str) -> tuple[Mapping[YamlPath, _Node], tuple[tuple[int, str], ...]]:
    nodes: dict[YamlPath, _Node] = {}
    loader = _Loader(source)

    def visit(path, node, entry, key_end=-1):
        start = node.start_mark.index
        end = node.end_mark.index
        children = []
        block = False
        value = ""
        if isinstance(node, yaml.ScalarNode):
            kind, value = _scalar(loader, node)
        else:
            block = not node.flow_style
            if isinstance(node, yaml.MappingNode):
                kind = "mapping"
                if not block and source[start:start + 1] != "{":
                    raise _raw("Implicit flow mappings are not supported in the form.")
                for key, child in node.value:
                    if not isinstance(key, yaml.ScalarNode) or key.tag != "tag:yaml.org,2002:str":
                        raise _raw("Mapping keys must be strings. Quote numeric, boolean or other special keys.")
                    if re.search(r"[\ud800-\udfff]", key.value):
                        raise _raw("An invalid Unicode mapping key is not editable in the form.")
                    if not re.match(r"[ \t]*:", source[key.end_mark.index:key.end_mark.index + 1025]):
                        raise _raw("Mapping entries need an ordinary key and colon.")
                    child_path = path + (key.value,)
                    if child_path in nodes:
                        raise _raw("The document has duplicate mapping keys.")
                    visit(child_path, child, key.start_mark.index, key.end_mark.index)
                    children.append(child_path)
            else:
                kind = "list"
                for i, child in enumerate(node.value):
                    pos = child.start_mark.index
                    if block:
                        # A compact nested list can start at another dash. The
                        # owning entry dash is strictly BEFORE the child's start.
                        pos = loader.dashes[bisect_left(loader.dashes, pos) - 1]
                    child_path = path + (i,)
                    visit(child_path, child, pos)
                    children.append(child_path)
            if block:
                end = nodes[children[-1]].end
        nodes[path] = _Node(kind, value, tuple(children), start, end, node.start_mark.column, entry, key_end, block)

    try:
        root = loader.get_single_node()
        if root is None:
            nodes[()] = _Node("mapping", "", (), len(source), len(source), 0, len(source), -1, True)
        else:
            visit((), root, root.start_mark.index)
        return MappingProxyType(nodes), tuple(loader.comments)
    finally:
        loader.dispose()


def _string(text: str) -> str:
    # Conservative plain spellings that stay strings under both YAML 1.1 and 1.2.
    if (re.fullmatch(r"[A-Za-z_][A-Za-z0-9_./ -]*", text) and not text.endswith(" ")
            and text.lower() not in {"true", "false", "yes", "no", "on", "off", "y", "n", "null"}):
        return text
    quoted = json.dumps(text, ensure_ascii=False)
    return re.sub(r"[\x7f-\x9f\u2028\u2029\ufeff\ufffe\uffff]", lambda m: f"\\u{ord(m[0]):04x}", quoted)


def _typed(kind: str, value: str) -> tuple[str, str]:
    """Return canonical form-field value and an unambiguous YAML spelling."""
    if not isinstance(value, str) or len(value) > _MAX_TEXT or re.search(r"[\ud800-\udfff]", value):
        raise YamlFormError("Enter a bounded Unicode value.")
    if kind == "text":
        return value, _string(value)
    s = value.strip()
    if kind in _CONTAINERS and not s:
        return "", "{}" if kind == "mapping" else "[]"
    if kind == "null" and s.lower() in ("", "null", "~"):
        return "null", "null"
    if kind == "boolean" and s.lower() in ("true", "false"):
        return s.lower(), s.lower()
    if kind == "integer" and len(s) <= _MAX_NUMERIC and re.fullmatch(r"[+-]?[0-9]+", s):
        s = str(int(s, 10))
        return s, s
    if (kind == "number" and len(s) <= _MAX_NUMERIC
            and re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?", s)):
        number = float(s)
        if math.isfinite(number):
            s = repr(number)
            mantissa, e, exponent = s.partition("e")
            return s, (mantissa + ".0" if e and "." not in mantissa else mantissa) + e + exponent
    raise YamlFormError("Enter a value matching the selected type (true/false, decimal integer or finite number); leave containers blank.")


@dataclass(frozen=True, slots=True)
class YamlFormDocument:
    """A snapshot, never a writer. Publish change()'s returned text only on success."""

    text: str = field(repr=False)
    _source: str = field(init=False, repr=False, compare=False)
    _eol: str = field(init=False, repr=False, compare=False)
    _nodes: Mapping[YamlPath, _Node] = field(init=False, repr=False, compare=False)
    _comments: tuple[tuple[int, str], ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.text, str) or len(self.text) > _MAX_TEXT:
            raise _raw("The form accepts up to 512 Ki characters of YAML text.")
        eol = "\r\n" if "\r\n" in self.text else "\r" if "\r" in self.text else "\n"
        remainder = self.text.replace(eol, "")
        if re.search(r"[\r\n\x85\u2028\u2029]", remainder):
            raise _raw("Mixed or nonstandard line endings cannot be preserved by the form.")
        source = self.text.replace(eol, "\n")
        try:
            nodes, comments = _parse(source)
        except YamlFormError:
            raise
        except (yaml.YAMLError, ValueError, OverflowError, RecursionError, IndexError) as exc:
            mark = getattr(exc, "problem_mark", None)
            where = f" at line {mark.line + 1}" if mark else ""
            raise _raw(f"Invalid or unsupported single-document YAML{where}.") from None
        for name, val in (("_source", source), ("_eol", eol), ("_nodes", nodes), ("_comments", comments)):
            object.__setattr__(self, name, val)

    def _node(self, path: YamlPath) -> _Node:
        if (not isinstance(path, tuple) or len(path) > _MAX_DEPTH or any(type(p) not in (int, str) for p in path)
                or path not in self._nodes):
            raise YamlFormError("Select an existing YAML entry.")
        return self._nodes[path]

    def kind(self, path: YamlPath) -> str:
        return self._node(path).kind

    def value(self, path: YamlPath) -> str:
        return self._node(path).value

    def children(self, path: YamlPath) -> tuple[YamlPath, ...]:
        return self._node(path).children

    def first_line(self, path: YamlPath, limit: int) -> str:
        """A bounded excerpt of the authored syntax for an otherwise unnamed group."""
        node = self._node(path)
        return self._source[node.start:min(node.end, node.start + limit)].partition("\n")[0]

    def _model(self, path=()):
        """Detached, order- and type-sensitive semantics for the post-edit check."""
        node = self._nodes[path]
        if node.kind == "mapping":
            return OrderedDict((p[-1], self._model(p)) for p in node.children)
        if node.kind == "list":
            return [self._model(p) for p in node.children]
        return node.kind, node.value

    def _edit_model(self, path):
        """Return the detached root and the container to edit within it."""
        root = owner = self._model()
        for part in path:
            owner = owner[part]
        return root, owner

    def _append(self, node, item):
        s = self._source
        if not node.block:
            pos = node.end - 1
            # A trailing comma can be separated from the closing bracket by comments.
            between = s[self._nodes[node.children[-1]].end:pos] if node.children else ""
            comma = ", " if node.children and not _COMMENT.sub("", between).strip() else ""
            return pos, pos, comma + item
        pos = node.end
        if pos and s[pos - 1] != "\n":
            newline = s.find("\n", pos)
            pos = len(s) if newline < 0 else newline + 1
        prefix = "\n" if pos and s[pos - 1] != "\n" else ""
        suffix = "\n" if pos < len(s) or s.endswith("\n") else ""
        return pos, pos, prefix + " " * node.column + item + suffix

    def _rewrite(self, start, end, replacement):
        """Retain real comments inside a changed span (including | # headers).

        Hashes inside scalar bodies are data, not comments. Preserve a separator
        after a rewritten block scalar when the scanner included its last newline.
        """
        s = self._source
        for pos, comment in self._comments:
            if start <= pos < end:
                if replacement.endswith("\n"):
                    line = s.rfind("\n", 0, pos) + 1
                    prefix = s[line:pos]
                    replacement += prefix[:len(prefix) - len(prefix.lstrip(" \t"))]
                elif replacement:
                    replacement += " "
                replacement += comment + "\n"
        if start < end and s[end - 1] == "\n" and replacement and not replacement.endswith("\n"):
            replacement += "\n"
        if replacement.endswith("\n") and end == len(s) and not s.endswith("\n"):
            replacement = replacement[:-1]
        return start, end, replacement

    def _remove(self, parent, node):
        s = self._source
        start, end = node.entry, node.end
        replacement = ""
        if not parent.block:
            # Removing the last item may leave a legal trailing comma. Otherwise
            # consume the next comma (but keep comments before it).
            rest = re.match(r"(?:\s|#[^\n]*(?:\n|$))*,", s[end:parent.end - 1])
            if rest:
                end += rest.end()
        elif len(parent.children) == 1:
            replacement = "{}" if parent.kind == "mapping" else "[]"
            if parent.kind == "list" and parent.key_end >= 0:
                key_column = parent.entry - (s.rfind("\n", 0, parent.entry) + 1)
                if parent.column == key_column:  # indentless sequence under a key
                    replacement = "  " + replacement
        else:
            line = s.rfind("\n", 0, start) + 1
            # Do not eat the containing dash in a compact list of mappings.
            if not s[line:start].strip() and (end == len(s) or s[end - 1:end] == "\n" or s[end:end + 1] == "\n"):
                start = line
                if s[end - 1:end] != "\n" and s[end:end + 1] == "\n":
                    end += 1
            elif end > start and s[end - 1] == "\n":
                replacement = "\n"  # Keep the outer dash and next sibling on separate lines.
        return self._rewrite(start, end, replacement)

    def _first_named_row(self, parent, key, node, entry):
        """Insert the field with its first row, or fill an existing placeholder."""
        flow = "[{" + entry + "}]"
        block_row = " " * (parent.column + 2) + "- " + entry
        if node is None:
            field = _string(key) + ":" + ("\n" + block_row if parent.block else " " + flow)
            return self._append(parent, field)

        # Keep a simple block field's end-of-line comment on the key, before
        # its new list. For more complex syntax, only replace the scalar itself.
        s = self._source
        after_colon = s.index(":", node.key_end) + 1
        if (parent.block and s.find("\n", after_colon, node.end) < 0
                and s[node.start:node.start + 1] not in ("|", ">")):
            end = s.find("\n", node.end)
            if end < 0:
                end = len(s)
            tail = s[node.end:end]
            if tail.startswith("#"):
                tail = " " + tail
            return after_colon, end, tail + "\n" + block_row
        return self._rewrite(node.start, node.end, (" " if node.start == node.end else "") + flow)

    def append_named_item(self, collection: YamlPath, name: str) -> tuple[str, YamlPath]:
        """Append {name: <literal text>} and return its checked source and item path.

        Only the final mapping field may be created. Null and the old form's
        empty text field are placeholders, not list members to retain.
        """
        _, spelling = _typed("text", name)
        if not name.strip() or re.search(r"[\r\n\v\f\x1c-\x1e\x85\u2028\u2029]", name):
            raise YamlFormError("Enter a nonempty, single-line name.")

        parent = None
        if isinstance(collection, tuple) and collection and type(collection[-1]) is str:
            parent = self._node(collection[:-1])
            if parent.kind != "mapping":
                raise YamlFormError("Select a list or a list field of a mapping.")
            node = self._nodes.get(collection)
        else:
            node = self._node(collection)

        literal = ("text", name)
        item = OrderedDict(name=literal)
        entry = "name: " + spelling
        if node is not None and node.kind == "list":
            expected, owner = self._edit_model(collection)
            if any((child.get("name") if isinstance(child, OrderedDict) else child) == literal for child in owner):
                raise YamlFormError("This list already contains that name.")
            index = len(owner)
            owner.append(item)
            patch = self._append(node, "- " + entry if node.block else "{" + entry + "}")
        else:
            if parent is None or (node is not None and (node.kind, node.value) not in {("null", "null"), ("text", "")}):
                raise YamlFormError("Select a list or an empty list field. Existing values cannot be replaced.")
            key = collection[-1]
            if node is None and (len(key) > 1024 or re.search(r"[\ud800-\udfff]", key)):
                raise YamlFormError("Enter a Unicode mapping key of up to 1024 characters.")
            expected, owner = self._edit_model(collection[:-1])
            owner[key] = [item]
            index = 0
            patch = self._first_named_row(parent, key, node, entry)

        return self._checked(expected, [patch]).text, collection + (index,)

    def as_named_mapping(self, path: YamlPath) -> YamlFormDocument:
        """Promote one text list item, preserving its literal value; mappings are already usable."""
        node = self._node(path)
        if node.kind == "mapping":
            return self
        if not path or self._nodes[path[:-1]].kind != "list" or node.kind != "text":
            raise YamlFormError("Select a mapping or a text list item to add fields.")
        expected, owner = self._edit_model(path[:-1])
        owner[path[-1]] = OrderedDict(name=("text", node.value))
        _, spelling = _typed("text", node.value)
        entry = "name: " + spelling
        replacement = entry if self._nodes[path[:-1]].block else "{" + entry + "}"
        return self._checked(expected, [self._rewrite(node.start, node.end, replacement)])

    def change(self, operation: str, path: YamlPath, *, key: str | None = None,
               kind: str = "text", value: str = "") -> str:
        """Edit/rename an entry, append to a container, or remove a nonroot entry.

        Editing a container as its current kind retains its children. To replace
        a populated container with a different type, explicitly remove it first.
        All failures leave both the source and navigable snapshot unchanged.
        """
        node = self._node(path)
        if operation not in ("add", "edit", "remove"):
            raise YamlFormError("Choose add, edit or remove.")
        parent = self._nodes[path[:-1]] if path else None
        target = node if operation == "add" else parent
        if key is not None and (not target or target.kind != "mapping" or operation == "remove"
                                or not isinstance(key, str) or len(key) > 1024 or re.search(r"[\ud800-\udfff]", key)):
            raise YamlFormError("Only mapping entries accept a Unicode key of up to 1024 characters.")
        if operation == "add" and (node.kind not in _CONTAINERS or node.kind == "mapping" and key is None):
            raise YamlFormError("Add to a list or supply a key for a mapping.")
        if key is not None and (path if operation == "add" else path[:-1]) + (key,) in self._nodes:
            if operation == "add" or key != path[-1]:
                raise YamlFormError("This mapping already contains that key.")
        if operation == "remove" and not path:
            raise YamlFormError("The document root cannot be removed. Edit its type instead.")

        expected, owner = self._edit_model(path if operation == "add" else path[:-1])
        patches = []
        if operation == "remove":
            del owner[path[-1]]
            patches.append(self._remove(parent, node))
        else:
            if not isinstance(kind, str):
                raise YamlFormError("Choose a YAML value type.")
            canonical, spelling = _typed(kind, value)
            new = OrderedDict() if kind == "mapping" else [] if kind == "list" else (kind, canonical)
            if operation == "add":
                if key is not None:
                    owner[key] = new
                    item = _string(key) + ": " + spelling
                else:
                    owner.append(new)
                    item = ("- " if node.block else "") + spelling
                patches.append(self._append(node, item))
            else:
                if (kind, canonical) != (node.kind, node.value):
                    if node.children:
                        raise YamlFormError("Remove the children before changing this container's type.")
                    if path:
                        owner[path[-1]] = new
                    else:
                        expected = new
                    if node.start == node.end:
                        if node.block:  # virtual root after an empty or comment-only stream
                            patches.append(self._append(node, spelling))
                        else:  # implicit null after a colon, dash or document marker
                            patches.append((node.start, node.end, " " + spelling))
                    else:
                        patches.append(self._rewrite(node.start, node.end, spelling))
                if key is not None and key != path[-1]:
                    items = [(key if k == path[-1] else k, v) for k, v in owner.items()]
                    owner.clear()
                    owner.update(items)
                    patches.append((node.entry, node.key_end, _string(key)))
        return self._checked(expected, patches).text

    def _checked(self, expected, patches):
        """Publish only a snapshot with exactly the requested semantics and original comments."""
        if not patches:
            return self
        source = self._source
        for a, b, replacement in sorted(patches, reverse=True):
            source = source[:a] + replacement + source[b:]
        # Deleting the final row exposes the previous line's terminator. Drop
        # it only if that does not also chomp a surviving literal scalar.
        candidates = (source[:-1], source) if not self._source.endswith("\n") and source.endswith("\n") else (source,)
        comments = tuple(c for _, c in self._comments)
        for candidate_source in candidates:
            candidate = YamlFormDocument(candidate_source.replace("\n", self._eol))
            if candidate._model() == expected and tuple(c for _, c in candidate._comments) == comments:
                return candidate
        raise _raw("This edit would change other YAML values, key order or comments.")
