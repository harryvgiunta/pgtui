"""In-memory completion for dbt Jinja in the SQL editor."""
from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from d8r.completion.context import _MAX_CHARS, _scan
from d8r.completion.models import Call, Completion, Context, Suggestion
from d8r.dbt_index import (
    DbtIndex, DbtResource, _OPEN_TO_CLOSE, _REF_KINDS, _Tag, _Token, _string, _tags,
)

_LIMIT = 100
_BARE_NAME = re.compile(r"[\w.-]+\Z")
_NAME_TAIL = re.compile(r"[\w.-]*")
_SPACE = re.compile(r"\s*")
_MACRO = re.compile(r"(?:[A-Za-z_]\w*\.)?(?:[A-Za-z_]\w*)?\Z")
_MASK = re.compile(r"[^\r\n]")
_COMMON = ("config", "doc", "env_var", "is_incremental", "log", "ref", "return", "run_query", "source", "var")


def _active(document: str, offset: int, dialect: str) -> _Tag | None:
    chunks, previous = [], 0
    for tag in _tags(document):
        if tag.start >= offset:
            break
        if offset < tag.end or offset == tag.end and not tag.closed:
            if offset < tag.body_start and tag.kind != "raw":
                return None
            chunks.append(document[previous:tag.start])
            # SQL lexers can consume Jinja syntax as a literal. Check the host
            # position at the opener, not the SQL token at the caret inside it.
            host = "".join(chunks) + "{"
            tokens = _scan(host, dialect)
            if not tokens or tokens[-1].kind != "symbol" or tokens[-1].start != len(host) - 1:
                return None
            return tag
        chunks.append(document[previous:tag.start])
        if tag.kind == "raw":
            chunks.append(_MASK.sub(" ", document[tag.start:tag.body_start]))
            chunks.append(document[tag.body_start:tag.body_end])
            chunks.append(_MASK.sub(" ", document[tag.body_end:tag.end]))
        else:
            chunks.append(_MASK.sub(" ", document[tag.start:tag.end]))
        previous = tag.end
    return None


@dataclass(frozen=True, slots=True)
class _Arguments:
    name: str
    start: int
    index: int
    spans: tuple[tuple[int, int], ...]
    closing: bool


def _arguments(tag: _Tag, document: str, offset: int) -> _Arguments | None:
    tokens = tag.tokens
    stack: list[int] = []
    for i, token in enumerate(tokens):
        if token.end > offset:
            break
        if token.text in _OPEN_TO_CLOSE:
            stack.append(i)
        elif stack and token.text == _OPEN_TO_CLOSE[tokens[stack[-1]].text]:
            stack.pop()
    if not stack:
        return None
    i = stack[-1]
    if not i or tokens[i].text != "(" or tokens[i - 1].text not in {"ref", "source"}:
        return None
    start = tokens[i - 1].start
    if i >= 2 and tokens[i - 2].text == ".":
        if i < 3 or tokens[i - 3].text != "builtins" or i >= 4 and tokens[i - 4].text == ".":
            return None
        start = tokens[i - 3].start
    if i == 2 and tag.kind == "%" and tokens[0].text in {"macro", "test"}:
        return None
    spans, left, end, closing = [], tokens[i].end, tag.body_end, False
    brackets = []
    for token in tokens[i + 1:]:
        if token.text in _OPEN_TO_CLOSE:
            brackets.append(_OPEN_TO_CLOSE[token.text])
        elif brackets and token.text == brackets[-1]:
            brackets.pop()
        elif not brackets and token.text in {",", ")"}:
            spans.append((left, token.start))
            left = token.end
            if token.text == ")":
                closing = True
                break
        elif not brackets and token.text.startswith(("'", '"')) and _string(token) is None:
            # An unclosed quote can consume a pre-existing closing paren.
            # Leave that paren and any surrounding whitespace outside the edit.
            paren = document.find(")", max(token.start, offset), token.end)
            if paren >= 0 and not document[paren + 1:token.end].strip():
                end, closing = paren, True
                spans.append((left, end))
                break
    else:
        spans.append((left, end))
    for index, (lo, hi) in enumerate(spans):
        if lo <= offset <= hi:
            return _Arguments(tokens[i - 1].text, start, index, tuple(spans), closing)
    return None


def _trim(document: str, lo: int, hi: int) -> tuple[int, int]:
    while lo < hi and document[lo].isspace():
        lo += 1
    while hi > lo and document[hi - 1].isspace():
        hi -= 1
    return lo, hi


@dataclass(frozen=True, slots=True)
class _Slot:
    start: int
    end: int
    prefix: str
    quote: str = ""


def _slot(document: str, offset: int, span: tuple[int, int]) -> _Slot | None:
    lo, hi = _trim(document, *span)
    if lo == hi:
        return _Slot(offset, offset, "")
    text = document[lo:hi]
    if text[0] in "'\"":
        quote = text[0]
        match = re.match(r"(?:\\[\s\S]|[^\\" + quote + r"])*", text[1:])
        assert match is not None
        stop = 1 + match.end()
        if stop < len(text) and text[stop] == quote:
            if stop + 1 != len(text):
                return None  # Not a simple literal argument, e.g. a concatenation.
        elif stop < len(text) and text[stop:] != "\\":
            return None
        if offset < lo + 1:
            return _Slot(lo, hi, "", quote)
        prefix = text[1:max(1, min(stop, offset - lo))]
        decoded = _string(_Token(quote + prefix + quote, 0, len(prefix) + 2))
        return _Slot(lo, hi, prefix if decoded is None else decoded, quote)
    if _BARE_NAME.fullmatch(text):
        return _Slot(lo, hi, document[lo:max(lo, min(offset, hi))])
    return None


def _value(document: str, span: tuple[int, int]) -> tuple[str, _Slot] | None:
    start, end = _trim(document, *span)
    if start == end:
        return None
    slot = _slot(document, end, (start, end))
    if slot is None:
        return None
    if slot.quote:
        result = _string(_Token(document[start:end], start, end))
        return (result, slot) if result is not None else None
    return slot.prefix, slot


def _draft_source_arguments(tag: _Tag, document: str, offset: int, args: _Arguments) -> _Arguments:
    """An unfinished call owns the name at the caret, not the SQL after it."""
    if tag.closed and not tag.broken:
        return args
    span = args.spans[args.index]
    delimiter = "}}" if tag.kind == "{" else "%}"
    complete = _value(document, span)
    if complete is not None and all(document.find(mark, offset, complete[1].end) < 0 for mark in ("\n", "\r", delimiter)):
        return args  # An intact literal may contain spaces or punctuation; replace it as a unit.

    before = (span[0], offset)
    slot = _slot(document, offset, before)
    if slot is None or (slot.quote and _value(document, before) is not None):
        return args  # Never turn a computed argument into a literal.
    end = _NAME_TAIL.match(document, offset, tag.body_end).end()
    if end > offset and document.startswith("-" + delimiter, end - 1):
        end -= 1
    closed_quote = bool(slot.quote and document.startswith(slot.quote, end))
    end += int(closed_quote)
    next_pos = _SPACE.match(document, end).end()
    following = document[next_pos:next_pos + 1]
    if next_pos < tag.body_end and following not in {",", ")"} and not document.startswith(
            (delimiter, "-" + delimiter, "+" + delimiter), next_pos):
        # After a closed quote or a bare name, operators may continue a Jinja expression.
        # Inside an unfinished quote, never absorb following punctuation or host SQL.
        if closed_quote or (not slot.quote and (next_pos == end or not following.isalpha())):
            return args
    spans = (*args.spans[:args.index], (span[0], end))
    if following == ",":
        rest = args.spans[args.index + 1:] if next_pos == span[1] else ()
        spans += rest or ((next_pos + 1, tag.body_end),)
    return _Arguments(args.name, args.start, args.index, spans, following == ")")


def _quote(name: str, quote: str = "'") -> str:
    return quote + name.replace("\\", "\\\\").replace(quote, "\\" + quote) + quote


def _matching(prefix: str, names: Iterable[str]) -> list[str]:
    prefix = prefix.casefold()
    return sorted({name for name in names if name.casefold().startswith(prefix)}, key=lambda s: (s.casefold(), s))[:_LIMIT]


def _item(slot: _Slot, name: str, detail: str, kind: str, args: _Arguments, *,
          next_slot: bool = False, version: bool = False, key: str = "") -> Suggestion:
    # Numeric versions can be bare; non-numeric ones must stay literal strings.
    quote = slot.quote or "'"
    literal = name if version and not slot.quote and re.fullmatch(r"\d+(?:\.\d+)?(?:[eE][+-]?\d+)?", name) else _quote(name, quote)
    insert = key + literal
    cursor_back = 0
    if args.index == len(args.spans) - 1:
        if next_slot:
            insert += ", "
        if not args.closing:
            insert += ")"
            cursor_back = int(next_slot)
    return Suggestion(key + name, insert, detail, kind, cursor_back=cursor_back)


def _literalize_before(item: Suggestion, document: str, slot: _Slot,
                       before: Iterable[tuple[str, _Slot]]) -> Suggestion:
    edits = [(name, value) for name, value in before if not value.quote]
    if not edits:
        return item
    parts, pos = [], edits[0][1].start
    for name, value in edits:
        parts.extend((document[pos:value.start], _quote(name)))
        pos = value.end
    parts.extend((document[pos:slot.start], item.insert))
    return Suggestion(item.label, "".join(parts), item.detail, item.kind, cursor_back=item.cursor_back,
                      start=edits[0][1].start, end=slot.end)


def _result(slot: _Slot, items: Iterable[Suggestion], args: _Arguments | None = None) -> Completion:
    proposals = tuple(items)[:_LIMIT]
    call = Call(args.name, args.index, args.start) if args else None
    return Completion(Context(slot.start, slot.end, token=slot.prefix, quote=slot.quote,
                              clause="jinja", blocked=not proposals, call=call, literal=bool(slot.quote)), proposals)


def _complete_source(index: DbtIndex, document: str, offset: int, args: _Arguments) -> Completion:
    slot = _slot(document, offset, args.spans[args.index])
    if slot is None or len(args.spans) > 2:
        return _result(_Slot(offset, offset, ""), (), args)
    sources = [item for item in index.resources.values() if item.kind == "source"]
    if args.index == 0:
        groups = {item.source for item in sources if index.resolve("source", (item.source, item.name), {}) is not None}
        return _result(slot, (_item(slot, name, "source name", "source", args, next_slot=True)
                              for name in _matching(slot.prefix, groups)), args)
    if args.index == 1 and (first := _value(document, args.spans[0])):
        group = first[0]
        names = {item.name for item in sources if item.source == group and index.resolve("source", (group, item.name), {})}
        return _result(slot, (_literalize_before(_item(slot, name, group, "source", args), document, slot, (first,))
                              for name in _matching(slot.prefix, names)), args)
    return _result(slot, (), args)


def _versions(index: DbtIndex, refs: list[DbtResource], names: tuple[str, ...]) -> set[str]:
    return {item.version for item in refs if item.version and item.name == names[-1] and
            (resolved := index.resolve("ref", names, {"v": item.version})) is not None and resolved.key == item.key}


def _version_suffix(document: str, spans: tuple[tuple[int, int], ...]) -> tuple[bool, dict]:
    if spans:
        lo, hi = spans[0]
        keyword = re.match(r"\s*(?:version|v)\s*=", document[lo:hi])
        if keyword:
            existing = _value(document, (lo + keyword.end(), hi))
            return True, {"v": existing[0]} if existing else {}
    return False, {}


def _complete_ref(index: DbtIndex, document: str, offset: int, args: _Arguments) -> Completion:
    refs = [item for item in index.resources.values() if item.kind in _REF_KINDS]
    lo, hi = args.spans[args.index]
    keyword = re.match(r"\s*(?:version|v)\s*=", document[lo:hi])
    if keyword and offset >= lo + keyword.end() and args.index in {1, 2} and len(args.spans) == args.index + 1:
        slot = _slot(document, offset, (lo + keyword.end(), hi))
        before = tuple(_value(document, span) for span in args.spans[:args.index])
        if slot is not None and all(before):
            versions = _versions(index, refs, tuple(value[0] for value in before))
            return _result(slot, (_literalize_before(_item(slot, version, "model version", "version", args, version=True),
                                                    document, slot, before)
                                  for version in _matching(slot.prefix, versions)), args)
    slot = _slot(document, offset, (lo, hi))
    if slot is None or len(args.spans) > 3:
        return _result(_Slot(offset, offset, ""), (), args)
    options = []
    if args.index == 0:
        version_keyword, kwargs = _version_suffix(document, args.spans[1:])
        if not version_keyword:
            model = _value(document, args.spans[1]) if len(args.spans) > 1 else None
            known_model = model and any(item.name == model[0] for item in refs)
            _, package_kwargs = _version_suffix(document, args.spans[2:])
            packages = {item.package for item in refs if item.package and
                        (not known_model or index.resolve("ref", (item.package, model[0]), package_kwargs) is not None)}
            options.extend(_item(slot, package, "dbt package", "package", args, next_slot=True)
                           for package in _matching(slot.prefix, packages))
        if len(args.spans) == 1 or version_keyword and len(args.spans) == 2:
            targets = {name: index.resolve("ref", (name,), kwargs) for name in {item.name for item in refs}}
            for name in _matching(slot.prefix, (name for name, target in targets.items() if target is not None)):
                item = targets[name]
                options.append(_item(slot, name, item.label, item.kind, args))
    elif args.index == 1 and (first := _value(document, args.spans[0])) is not None:
        package = first[0]
        has_version, kwargs = _version_suffix(document, args.spans[2:])
        if len(args.spans) == 2 or has_version:
            names = {item.name for item in refs if item.package == package}
            targets = {name: index.resolve("ref", (package, name), kwargs) for name in names}
            for name in _matching(slot.prefix, (name for name, target in targets.items() if target is not None)):
                item = targets[name]
                options.append(_literalize_before(_item(slot, name, item.label, item.kind, args), document, slot, (first,)))
    if args.index in {1, 2} and len(args.spans) == args.index + 1 and not slot.quote and "version".startswith(slot.prefix):
        before = tuple(_value(document, span) for span in args.spans[:args.index])
        if all(before):
            versions = _versions(index, refs, tuple(value[0] for value in before))
            options.extend(_literalize_before(_item(slot, version, "model version", "version", args, version=True,
                                                    key="version="), document, slot, before)
                           for version in _matching("", versions))
    # Rank exact model-name matches ahead of an equal-named package constructor.
    options.sort(key=lambda item: (item.label.casefold() != slot.prefix.casefold(), item.kind == "package", item.label.casefold()))
    return _result(slot, options, args)


def _complete_macro(index: DbtIndex | None, document: str, offset: int, tag: _Tag, manual: bool) -> Completion:
    left, right = offset, offset
    while left > tag.body_start and (document[left - 1].isalnum() or document[left - 1] in "_."):
        left -= 1
    while right < tag.body_end and (document[right].isalnum() or document[right] in "_."):
        right += 1
    slot = _Slot(left, right, document[left:offset])
    tokens = tag.tokens
    if any(token.text.startswith(("'", '"')) and token.start < offset and
           (offset < token.end or offset == token.end and _string(token) is None) for token in tokens):
        return _result(slot, ())
    if not _MACRO.fullmatch(document[left:right]) or not slot.prefix and not manual:
        return _result(slot, ())
    before = [token.text for token in tokens if token.end <= left]
    if tag.kind == "%" and (not before or before[0] not in {"do", "elif", "for", "if", "set"} or
                            before[0] == "for" and "in" not in before or before[0] == "set" and "=" not in before):
        return _result(slot, ())
    if before and before[-1] not in {"do", "elif", "if", "in", "not", "and", "or", "(", "[", "{", ",", "=", "+", "-", "~", ":"}:
        return _result(slot, ())
    suffix = document[right:tag.body_end].lstrip()
    tail, back = ("", 0) if suffix.startswith("(") else ("(", 0) if suffix.startswith(")") else ("()", 1)
    names = dict.fromkeys(_COMMON, "dbt function")
    names.update({"builtins.ref": "dbt builtin", "builtins.source": "dbt builtin"})
    if index is not None:
        for macro in index.macros:
            if macro.package:
                names[f"{macro.package}.{macro.name}"] = macro.package
            if macro.path:
                names[macro.name] = macro.package or "local macro"
    prefix = slot.prefix.casefold()
    matches = [name for name in names if name.casefold().startswith(prefix) or
               prefix and "." not in prefix and name.rsplit(".", 1)[-1].casefold().startswith(prefix)]
    matches.sort(key=lambda name: (not name.casefold().startswith(prefix), "." in name, name.casefold(), name))
    return _result(slot, (Suggestion(name, name + tail, names[name], "macro", cursor_back=back)
                          for name in matches[:_LIMIT]))


def complete_dbt(index: DbtIndex | None, document: str, offset: int, manual: bool) -> Completion | None:
    """Complete dbt Jinja only; None delegates to the editor's native mode."""
    if not 0 <= offset <= len(document) or len(document) > _MAX_CHARS:
        return None
    tag = _active(document, offset, index.dialect if index is not None else "postgres")
    if tag is None:
        return None
    if tag.kind in {"#", "raw"} or offset > tag.body_end:
        return _result(_Slot(offset, offset, ""), ())
    args = _arguments(tag, document, offset)
    if args is not None:
        if index is None:
            return _result(_Slot(offset, offset, ""), (), args)
        if args.name == "source":
            return _complete_source(index, document, offset, _draft_source_arguments(tag, document, offset, args))
        return _complete_ref(index, document, offset, args)
    return _complete_macro(index, document, offset, tag, manual)
