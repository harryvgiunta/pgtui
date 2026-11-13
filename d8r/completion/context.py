"""Bounded, tolerant SQL scope discovery for editor completion only.

This module deliberately does not validate, compile, rewrite, or execute SQL.
Unknown expression names/types stay unknown rather than becoming schema facts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import re

from d8r.sql import SchemaContext, SqlError, Token, lex_sql
from .models import Call, Context, Field, Relation

_MAX_CHARS = 200_000
_MAX_TOKENS = 24_000
_MAX_DEPTH = 24
_IDS = frozenset({"word", "quoted_identifier", "parameter"})
_STOP = frozenset("SELECT FROM WHERE JOIN LEFT RIGHT FULL INNER OUTER CROSS NATURAL ON USING GROUP ORDER HAVING LIMIT OFFSET FETCH UNION INTERSECT EXCEPT RETURNING SET VALUES WINDOW QUALIFY INSERT UPDATE DELETE INTO AS WITH RECURSIVE LATERAL TABLESAMPLE FOR WHEN THEN ELSE END AND OR ASC DESC NULLS BY OVER PARTITION ROWS RANGE GROUPS CALL".split())
_WORD = re.compile(r"(?:[^\W\d]|_)[\w$]*", re.UNICODE)
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_DOLLAR = re.compile(r"\$(?:[A-Za-z_][\w]*)?\$")


def _scan(text: str, dialect: str) -> tuple[Token, ...]:
    """Use the native lexer for complete text, retaining incomplete lexical tails."""
    try:
        return lex_sql(text, dialect=dialect)
    except SqlError:
        pass
    result: list[Token] = []
    i, size = 0, len(text)
    sqlite = dialect == "sqlite"
    while i < size and len(result) <= _MAX_TOKENS:
        start, char = i, text[i]
        if char.isspace():
            i += 1
            while i < size and text[i].isspace():
                i += 1
            kind = "whitespace"
        elif text.startswith("--", i):
            stop = text.find("\n", i + 2)
            i = size if stop < 0 else stop
            kind = "comment"
        elif text.startswith("/*", i):
            i, depth = i + 2, 1
            while i < size and depth:
                if not sqlite and text.startswith("/*", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("*/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            kind = "comment"
        elif char in "'\"" or (sqlite and char in "`[") or (
            not sqlite and char in "eE" and text[i:i + 2].lower() == "e'"
        ):
            escaped = char in "eE"
            if escaped:
                i += 1
            quote = text[i]
            close = "]" if quote == "[" else quote
            kind = "string" if quote == "'" else "quoted_identifier"
            i += 1
            while i < size:
                if escaped and text[i] == "\\":
                    i = min(size, i + 2)
                elif text[i] == close:
                    i += 1
                    if quote != "[" and i < size and text[i] == close:
                        i += 1
                    else:
                        break
                else:
                    i += 1
        elif not sqlite and char == "$" and (match := _DOLLAR.match(text, i)):
            delimiter = match.group()
            stop = text.find(delimiter, match.end())
            i = size if stop < 0 else stop + len(delimiter)
            kind = "dollar_string"
        elif char == "@" and (match := _WORD.match(text, i + 1)):
            i, kind = match.end(), "parameter"
        elif match := _WORD.match(text, i):
            i, kind = match.end(), "word"
        elif match := _NUMBER.match(text, i):
            i, kind = match.end(), "number"
        else:
            i, kind = i + 1, "symbol"
            if text[start:i + 1] in {"::", ">=", "<=", "!=", "<>", "||", "->"}:
                i += 1
        result.append(Token(kind, text[start:i], start, i))
    return tuple(result)


def _name(token: Token) -> str:
    value = token.text
    if token.kind == "quoted_identifier":
        quote = value[0]
        close = "]" if quote == "[" else quote
        value = value[1:]
        if value.endswith(close):
            value = value[:-1]
        return value.replace(close * 2, close)
    return value.lower()


def _key(text: str, dialect: str) -> tuple[str, ...]:
    parts = tuple(_name(t) for t in _scan(text, dialect) if t.kind in _IDS)
    return parts if dialect == "postgres" else tuple(part.lower() for part in parts)


def _merge(local: tuple[Relation, ...], outer: tuple[Relation, ...], dialect: str) -> tuple[Relation, ...]:
    names = {_key(r.reference, dialect) for r in local}
    return local + tuple(replace(r, outer=True) for r in outer if _key(r.reference, dialect) not in names)


@dataclass
class _Scope:
    lo: int
    hi: int
    start: int
    end: int
    relations: tuple[Relation, ...] = ()
    ctes: tuple[Relation, ...] = ()
    outputs: tuple[Field, ...] = ()
    children: list[_Scope] = field(default_factory=list)
    # Relation source positions make JOIN ON/USING visibility left-to-right.
    sources: list[tuple[int, Relation]] = field(default_factory=list)
    insert_open: int | None = None


class _Model:
    def __init__(self, text: str, tokens: tuple[Token, ...], schema: SchemaContext, dialect: str,
                 depth: int = 0, bindings: dict[str, Field] | None = None,
                 budget: list[int] | None = None):
        self.text, self.tokens, self.schema, self.dialect = text, tokens, schema, dialect
        self.depth, self.bindings = depth, bindings or {}
        self.pairs: dict[int, int] = {}
        stack: list[int] = []
        for i, token in enumerate(tokens):
            if token.text == "(":
                stack.append(i)
            elif token.text == ")" and stack:
                opening = stack.pop()
                self.pairs[opening] = i
        for opening in stack:
            self.pairs[opening] = len(tokens)
        self.budget = budget if budget is not None else [_MAX_TOKENS * 4]

    def upper(self, i: int) -> str:
        return self.tokens[i].text.upper() if 0 <= i < len(self.tokens) and self.tokens[i].kind == "word" else ""

    def top(self, lo: int, hi: int):
        while lo < hi:
            yield lo
            lo = self.pairs.get(lo, lo) + 1

    def split(self, lo: int, hi: int) -> list[tuple[int, int]]:
        parts, start = [], lo
        for i in self.top(lo, hi):
            if self.tokens[i].text == ",":
                parts.append((start, i))
                start = i + 1
        parts.append((start, hi))
        return parts

    def identifier(self, i: int, hi: int) -> tuple[str, int]:
        if i >= hi or self.tokens[i].kind not in _IDS:
            return "", i
        start = i
        i += 1
        while i + 1 < hi and self.tokens[i].text == "." and self.tokens[i + 1].kind in _IDS:
            i += 2
        return self.text[self.tokens[start].start:self.tokens[i - 1].end], i

    def resolve(self, lo: int, hi: int, relations: tuple[Relation, ...]) -> Field | None:
        while lo < hi and self.tokens[lo].text == "(" and self.pairs.get(lo) == hi - 1:
            lo, hi = lo + 1, hi - 1
        if hi == lo + 1 and self.tokens[lo].kind == "parameter":
            return self.bindings.get(self.tokens[lo].text[1:].lower())
        name, end = self.identifier(lo, hi)
        if not name or end != hi:
            return None
        parts = _key(name, self.dialect)
        candidates = []
        for relation in relations:
            if len(parts) > 1 and _key(relation.reference, self.dialect) != parts[:-1]:
                continue
            for column in relation.fields:
                # PostgreSQL folds the reference, never a quoted catalog name.
                actual = column.name if self.dialect == "postgres" else column.name.lower()
                expected = parts[-1] if self.dialect == "postgres" else parts[-1].lower()
                if actual == expected:
                    candidates.append((relation.outer, column))
        local = [column for outer, column in candidates if not outer]
        matches = local or [column for _, column in candidates]
        return matches[0] if len(matches) == 1 else None

    def expression(self, lo: int, hi: int, relations: tuple[Relation, ...]) -> Field | None:
        if lo >= hi:
            return None
        direct = self.resolve(lo, hi, relations)
        if direct is not None:
            return direct
        top = list(self.top(lo, hi))
        casts = [i for i in top if self.tokens[i].text == "::" or (
            self.tokens[i].text == ":" and i + 1 < hi and self.tokens[i + 1].text == ":")]
        if casts:
            at = casts[-1]
            begin = at + (2 if self.tokens[at].text == ":" else 1)
            original = self.expression(lo, at, relations)
            typ = self.text[self.tokens[begin].start:self.tokens[hi - 1].end] if begin < hi else ""
            return replace(original, type=typ) if original else Field("", typ)
        first = self.tokens[lo]
        if self.upper(lo) == "CAST" and lo + 1 < hi and self.tokens[lo + 1].text == "(":
            close = min(self.pairs.get(lo + 1, hi), hi)
            as_at = next((i for i in self.top(lo + 2, close) if self.upper(i) == "AS"), None)
            if as_at is not None and as_at + 1 < close:
                original = self.expression(lo + 2, as_at, relations)
                typ = self.text[self.tokens[as_at + 1].start:self.tokens[close - 1].end]
                return replace(original, type=typ) if original else Field("", typ)
        if hi == lo + 1:
            if first.kind in {"string", "dollar_string"}:
                return Field("", "text")
            if first.kind == "number":
                return Field("", "numeric" if any(c in first.text.lower() for c in ".e") else "integer")
            if self.upper(lo) in {"TRUE", "FALSE"}:
                return Field("", "boolean")
        name, end = self.identifier(lo, hi)
        if name and end < hi and self.tokens[end].text == "(":
            routine = next((r for r in self.schema.routines if _key(r.name, self.dialect) == _key(name, self.dialect)), None)
            if routine:
                return Field("", routine.return_type)
            if name.lower() == "count":
                return Field("", "bigint")
            if name.lower() in {"min", "max", "sum", "avg", "coalesce", "nullif"}:
                args = self.split(end + 1, min(self.pairs.get(end, hi), hi))
                original = self.expression(*args[0], relations) if args else None
                if original:
                    return Field("", original.type)
        return None

    def projection(self, lo: int, hi: int, relations: tuple[Relation, ...]) -> tuple[Field, ...]:
        result: list[Field] = []
        for start, end in self.split(lo, hi):
            if start >= end:
                continue
            alias = None
            top = list(self.top(start, end))
            as_at = next((i for i in reversed(top) if self.upper(i) == "AS"), None)
            if as_at is not None and as_at + 1 < end and self.tokens[as_at + 1].kind in _IDS:
                alias, end = _name(self.tokens[as_at + 1]), as_at
            elif end - start > 1 and self.tokens[end - 1].kind in _IDS and self.upper(end - 1) not in _STOP:
                previous = self.tokens[end - 2]
                if previous.text not in {".", "::", ":", "+", "-", "*", "/", "=", "||"} and previous.end < self.tokens[end - 1].start:
                    alias, end = _name(self.tokens[end - 1]), end - 1
            if end <= start:
                continue
            if self.tokens[end - 1].text == "*":
                if end == start + 1:
                    result.extend(c for r in relations if not r.outer for c in r.fields)
                    continue
                if self.tokens[end - 2].text == ".":
                    qualifier = self.text[self.tokens[start].start:self.tokens[end - 2].start]
                    result.extend(c for r in relations if _key(r.reference, self.dialect) == _key(qualifier, self.dialect) for c in r.fields)
                    continue
            value = self.expression(start, end, relations)
            if alias is not None:
                result.append(replace(value, name=alias) if value else Field(alias))
            elif value is not None and value.name:
                result.append(value)
        return tuple(result)

    def renamed(self, relation: Relation, opening: int, hi: int) -> tuple[Relation, int]:
        close = min(self.pairs.get(opening, hi), hi)
        names = [_name(self.tokens[a]) for a, b in self.split(opening + 1, close)
                 if b == a + 1 and self.tokens[a].kind in _IDS]
        fields = tuple(replace(relation.fields[i], name=name) if i < len(relation.fields) else Field(name)
                       for i, name in enumerate(names))
        if len(relation.fields) > len(names):
            fields += relation.fields[len(names):]
        return replace(relation, fields=fields), min(close + 1, hi)

    def query(self, lo: int, hi: int, ctes: tuple[Relation, ...] = (),
              outer: tuple[Relation, ...] = (), depth: int = 0) -> _Scope:
        start = self.tokens[lo].start if lo < hi else (self.tokens[lo - 1].end if lo else 0)
        end = self.tokens[hi].start if hi < len(self.tokens) else len(self.text)
        scope = _Scope(lo, hi, start, end, ctes=ctes)
        self.budget[0] -= hi - lo
        if depth + self.depth >= _MAX_DEPTH or self.budget[0] < 0:
            return scope
        consumed: set[int] = set()
        main = lo
        if self.upper(main) == "WITH":
            main += 1
            recursive = self.upper(main) == "RECURSIVE"
            main += int(recursive)
            while main < hi:
                name, pos = self.identifier(main, hi)
                if not name:
                    break
                explicit = pos if pos < hi and self.tokens[pos].text == "(" else None
                if explicit is not None:
                    pos = min(self.pairs.get(pos, hi) + 1, hi)
                if self.upper(pos) != "AS":
                    break
                pos += 1
                if self.upper(pos) == "NOT":
                    pos += 1
                if self.upper(pos) == "MATERIALIZED":
                    pos += 1
                if pos >= hi or self.tokens[pos].text != "(":
                    break
                close = min(self.pairs.get(pos, hi), hi)
                relation = Relation(name, name)
                if explicit is not None:
                    relation, _ = self.renamed(relation, explicit, hi)
                visible = tuple(r for r in ctes if _key(r.reference, self.dialect) != _key(name, self.dialect))
                body_ctes = visible + (relation,) if recursive else ctes
                child = self.query(pos + 1, close, body_ctes, outer, depth + 1)
                child.start = self.tokens[pos].end
                scope.children.append(child)
                consumed.add(pos)
                relation = replace(relation, fields=child.outputs)
                if explicit is not None:
                    relation, _ = self.renamed(relation, explicit, hi)
                ctes = visible + (relation,)
                main = close + 1
                if main < hi and self.tokens[main].text == ",":
                    main += 1
                else:
                    break
            scope.ctes = ctes
        # Set-operation branches are sibling scopes, never alias providers.
        sets = [i for i in self.top(main, hi) if self.upper(i) in {"UNION", "INTERSECT", "EXCEPT"}]
        if sets:
            begin = main
            for stop in sets + [hi]:
                child = self.query(begin, stop, ctes, outer, depth + 1)
                scope.children.append(child)
                if not scope.outputs:
                    scope.outputs = child.outputs
                begin = stop + 1
                if self.upper(begin) in {"ALL", "DISTINCT"}:
                    begin += 1
            return scope
        local: list[Relation] = []
        i, in_from, expect = main, False, False
        while i < hi:
            word = self.upper(i)
            if word in {"FROM", "JOIN", "UPDATE", "INTO"} and not (word == "FROM" and self.upper(i - 1) == "DISTINCT"):
                in_from, expect = True, True
                i += 1
                continue
            if word in {"WHERE", "GROUP", "ORDER", "HAVING", "SET", "VALUES", "RETURNING", "LIMIT", "WINDOW", "QUALIFY"}:
                in_from, expect = False, False
            if in_from and self.tokens[i].text == ",":
                expect = True
                i += 1
                continue
            if not expect:
                i = self.pairs.get(i, i) + 1
                continue
            lateral = word == "LATERAL" and self.dialect != "sqlite"
            if lateral:
                i += 1
                if i >= hi:
                    break
            source = i
            insert_target = self.upper(main) == "INSERT" and not local
            relation: Relation | None = None
            if self.tokens[i].text == "(":
                close = min(self.pairs.get(i, hi), hi)
                if self.upper(i + 1) in {"SELECT", "WITH", "VALUES"}:
                    child = self.query(i + 1, close, ctes, _merge(tuple(local), outer, self.dialect) if lateral else (), depth + 1)
                    child.start = self.tokens[i].end
                    scope.children.append(child)
                    consumed.add(i)
                    relation = Relation("", "", child.outputs)
                i = min(close + 1, hi)
            else:
                name, after = self.identifier(i, hi)
                if not name:
                    expect = False
                    i += 1
                    continue
                relation = next((r for r in reversed(ctes) if _key(r.reference, self.dialect) == _key(name, self.dialect)), None)
                if relation is not None:
                    relation = replace(relation, reference=name)
                table = self.schema.table_by_name(name, dialect=self.dialect) if relation is None else None
                if table is not None:
                    relation = Relation(name, name, tuple(Field(c.name, c.type, table.name, c.name, c.doc, c.values) for c in table.columns), table.name, table.doc, table.foreign_keys)
                i = after
                if i < hi and self.tokens[i].text == "(" and not insert_target:
                    close = min(self.pairs.get(i, hi), hi)
                    relation = self.routine(name, i + 1, close, _merge(tuple(local), outer, self.dialect), depth)
                    i = min(close + 1, hi)
                if relation is None:
                    relation = Relation(name, name)
            if relation is None:
                expect = False
                continue
            if self.upper(i) == "AS":
                i += 1
            if i < hi and self.tokens[i].kind in _IDS and self.upper(i) not in _STOP:
                reference = self.tokens[i].text
                relation = replace(relation, name=relation.name or reference, reference=reference)
                i += 1
                if i < hi and self.tokens[i].text == "(" and not insert_target:
                    relation, i = self.renamed(relation, i, hi)
            if insert_target and i < hi and self.tokens[i].text == "(":
                scope.insert_open = i
                i = min(self.pairs.get(i, hi) + 1, hi)
            if relation.reference or relation.fields:
                local.append(relation)
                scope.sources.append((source, relation))
            expect = False
        scope.relations = _merge(tuple(local), outer, self.dialect)
        top = list(self.top(main, hi))
        select = next((i for i in top if self.upper(i) == "SELECT"), None)
        returning = next((i for i in top if self.upper(i) == "RETURNING"), None)
        projection = returning if returning is not None else select
        insert_select = self.upper(main) == "INSERT" and select is not None
        if insert_select:
            child = self.query(select, returning if returning is not None else hi, ctes, (), depth + 1)
            scope.children.append(child)
            scope.relations = tuple(local[:1])
            scope.sources = scope.sources[:1]
            scope.outputs = child.outputs
            projection = returning
        if projection is not None:
            begin = projection + 1
            if self.upper(begin) in {"ALL", "DISTINCT"}:
                begin += 1
                if self.upper(begin) == "ON" and begin + 1 < hi and self.tokens[begin + 1].text == "(":
                    begin = self.pairs.get(begin + 1, hi) + 1
            stop = next((i for i in top if i > projection and self.upper(i) in {"FROM", "WHERE", "GROUP", "HAVING", "ORDER", "LIMIT", "WINDOW", "QUALIFY", "INTO"}), hi)
            scope.outputs = self.projection(begin, stop, scope.relations)
        # Scalar/EXISTS subqueries correlate; ordinary expression parentheses do not make scopes.
        def children(a: int, b: int, level: int) -> None:
            if level >= _MAX_DEPTH:
                return
            for opening in self.top(a, b):
                if self.tokens[opening].text != "(" or opening in consumed:
                    continue
                close = min(self.pairs.get(opening, b), b)
                if self.upper(opening + 1) in {"SELECT", "WITH", "VALUES"}:
                    available = scope.relations
                    preceding = [r for at, r in scope.sources if at < opening]
                    # ON subqueries cannot see relations introduced by later joins.
                    last_clause = self.clause(scope, self.tokens[opening].start)[0]
                    if last_clause in {"on", "using"}:
                        available = _merge(tuple(preceding), outer, self.dialect)
                    child = self.query(opening + 1, close, ctes, available, depth + 1)
                    child.start = self.tokens[opening].end
                    scope.children.append(child)
                else:
                    children(opening + 1, close, level + 1)
        children(main, select if insert_select else hi, depth)
        if insert_select and returning is not None:
            children(returning, hi, depth)
        return scope

    def routine(self, name: str, lo: int, hi: int, relations: tuple[Relation, ...], depth: int) -> Relation:
        fn = self.schema.fn_by_name(name)
        if fn is not None and self.depth + depth + 1 < _MAX_DEPTH and self.budget[0] > 0 and len(fn.body) <= _MAX_CHARS:
            arguments = self.split(lo, hi)
            bindings = {param.lower().lstrip("@"): value for param, span in zip(fn.params, arguments)
                        if (value := self.expression(*span, relations)) is not None}
            tokens = tuple(t for t in _scan(fn.body, self.dialect) if t.kind not in {"whitespace", "comment"})
            if len(tokens) <= _MAX_TOKENS:
                model = _Model(fn.body, tokens, self.schema, self.dialect, self.depth + depth + 1, bindings, self.budget)
                scope = model.query(0, len(tokens))
                return Relation(name, name, scope.outputs, doc=fn.doc)
        native = next((r for r in self.schema.routines if _key(r.name, self.dialect) == _key(name, self.dialect) and r.columns), None)
        if native:
            return Relation(name, name, tuple(Field(c.name, c.type, doc=c.doc, values=c.values) for c in native.columns), doc=native.doc)
        return Relation(name, name)

    def clause(self, scope: _Scope, offset: int) -> tuple[str, int]:
        clause, anchor = "statement", scope.lo
        names = {"SELECT": "select", "FROM": "from", "JOIN": "join", "ON": "on", "USING": "using",
                 "WHERE": "where", "GROUP": "group", "ORDER": "order", "HAVING": "having", "SET": "update_set",
                 "INSERT": "insert", "UPDATE": "update", "DELETE": "delete", "INTO": "insert",
                 "VALUES": "values", "RETURNING": "returning", "LIMIT": "limit", "WINDOW": "window", "QUALIFY": "where"}
        if self.dialect in {"postgres", "duckdb"}:
            names["CALL"] = "call"
        for i in self.top(scope.lo, scope.hi):
            if self.tokens[i].end > offset:
                break
            word = self.upper(i)
            if word in names and not (word in {"ON", "FROM"} and self.upper(i - 1) == "DISTINCT"):
                clause, anchor = names[word], i
        if scope.insert_open is not None:
            opening = scope.insert_open
            close = self.pairs.get(opening, scope.hi)
            end = self.tokens[close].start if close < len(self.tokens) else len(self.text)
            if self.tokens[opening].end <= offset <= end:
                return "insert_columns", opening
        return clause, anchor


def _active(scope: _Scope, offset: int) -> _Scope:
    for child in scope.children:
        if child.start <= offset <= child.end:
            return _active(child, offset)
    return scope


def _closed_identifier(text: str) -> bool:
    close = "]" if text[0] == "[" else text[0]
    i = 1
    while i < len(text):
        if text[i] != close:
            i += 1
        elif close != "]" and i + 1 < len(text) and text[i + 1] == close:
            i += 2
        else:
            return True
    return False


def _recover_identifier(document: str, offset: int, scanned: tuple[Token, ...],
                        current: Token, dialect: str) -> tuple[tuple[Token, ...], Token]:
    """Separate an unfinished identifier from a recognizable same-line clause.

    Recovery is caret-local and only applies when no closing delimiter exists.
    Complete identifiers (including spaces, keywords and newlines) are never
    reinterpreted. The replacement keeps the stale identifier tail but excludes
    the suffix's whitespace; all returned offsets still address the original SQL.
    """
    if current.kind != "quoted_identifier" or _closed_identifier(current.text):
        return scanned, current
    suffix = re.search(r"[ \t]+(?=(?:FROM|JOIN|WHERE|HAVING|LIMIT|RETURNING|UNION|INTERSECT|EXCEPT)\b|(?:GROUP|ORDER)[ \t]+BY\b)",
                       document[offset:current.end], re.IGNORECASE)
    if suffix is None:
        return scanned, current
    boundary = offset + suffix.start()
    identifier = document[current.start:boundary]
    if "\n" in identifier or "\r" in identifier:
        return scanned, current
    tail = _scan(document[boundary:], dialect)
    significant = [t for t in tail if t.kind not in {"whitespace", "comment"}]
    if len(significant) < 2:
        return scanned, current
    if significant[0].text.upper() in {"FROM", "JOIN"} and significant[1].kind not in _IDS and significant[1].text != "(":
        return scanned, current
    recovered = Token("quoted_identifier", identifier, current.start, boundary)
    prefix = tuple(t for t in scanned if t.start < current.start)
    shifted = tuple(Token(t.kind, t.text, t.start + boundary, t.end + boundary) for t in tail)
    return prefix + (recovered,) + shifted, recovered


def analyze(document: str, offset: int, schema: SchemaContext, *, dialect: str = "postgres",
            parameters: tuple[str, ...] = ()) -> Context:
    """Return completion facts from the caret's statement and legal query scope.

    All offsets refer to the original document. Identifier and literal replacement
    spans include their delimiters; token/value_prefix contain only decoded text
    before the caret. Oversized input is deliberately blocked, not partially parsed.
    """
    offset = max(0, min(offset, len(document)))
    dialect = "sqlite" if dialect.lower() in {"sqlite", "d1"} else dialect.lower()
    if len(document) > _MAX_CHARS:
        return Context(offset, offset, blocked=True)
    scanned = _scan(document, dialect)
    if len(scanned) > _MAX_TOKENS:
        return Context(offset, offset, blocked=True)
    current = next((t for t in scanned if t.start < offset <= t.end and (
        t.kind in _IDS or t.kind in {"string", "dollar_string", "comment"} or t.text == "*")), None)
    if current is None:
        current = next((t for t in scanned if t.start == offset and (t.kind in _IDS or t.text == "*")), None)
    if current is not None:
        scanned, current = _recover_identifier(document, offset, scanned, current, dialect)
        if len(scanned) > _MAX_TOKENS:
            return Context(offset, offset, blocked=True)
    start, end, token, quote = offset, offset, "", ""
    literal, blocked = False, False
    if current is not None:
        start, end = current.start, current.end
        token = document[start:offset]
        if current.kind == "quoted_identifier":
            quote = current.text[0]
            close = "]" if quote == "[" else quote
            token = token[1:]
            if offset == current.end and _closed_identifier(current.text):
                token = token[:-1]
            token = token.replace(close * 2, close)
        elif current.kind == "string":
            literal, quote = True, "'"
            token = token[2:] if current.text[:2].lower() == "e'" else token[1:]
            if offset == current.end and current.text.endswith("'") and len(current.text) > 1:
                token = token[:-1]
            token = token.replace("''", "'")
            blocked = True
        elif current.kind in {"comment", "dollar_string"}:
            blocked = True
    tokens = tuple(t for t in scanned if t.kind not in {"whitespace", "comment"})
    lo, hi, statement_start, statement_end = 0, len(tokens), 0, len(document)
    for i, item in enumerate(tokens):
        if item.text == ";":
            if item.end <= offset:
                lo, statement_start = i + 1, item.end
            elif item.start >= offset:
                hi, statement_end = i, item.start
                break
    model = _Model(document, tokens, schema, dialect)
    root = model.query(lo, hi)
    root.start, root.end = statement_start, statement_end
    scope = _active(root, offset)
    clause, anchor = model.clause(scope, offset)
    before = [i for i in range(scope.lo, scope.hi) if tokens[i].end <= start]
    qualifier = ""
    if before and tokens[before[-1]].text == ".":
        stop, at = before[-1], before[-1] - 1
        while at >= scope.lo and tokens[at].kind in _IDS:
            if at - 1 >= scope.lo and tokens[at - 1].text == ".":
                at -= 2
            else:
                break
        if at >= scope.lo:
            qualifier = document[tokens[at].start:tokens[stop].start]
    relations = scope.relations
    if clause in {"insert_columns", "update_set"}:
        relations = tuple(r for r in relations if not r.outer)[:1]
    join_target, join_left = None, ()
    visible_sources = [(at, r) for at, r in scope.sources if tokens[at].start < offset]
    if clause in {"on", "using"} and visible_sources:
        join_target = visible_sources[-1][1]
        join_left = tuple(r for _, r in visible_sources[:-1])
        relations = _merge(join_left + (join_target,), tuple(r for r in relations if r.outer), dialect)
    elif clause == "join" and visible_sources:
        join_left = tuple(r for at, r in visible_sources if at < anchor)
        if current is None:
            joined = [(at, r) for at, r in visible_sources if at > anchor]
            if joined:
                at, candidate = joined[-1]
                _, after = model.identifier(at, scope.hi)
                if after < scope.hi and tokens[after].text == "(":
                    after = min(model.pairs.get(after, scope.hi) + 1, scope.hi)
                if model.upper(after) == "AS":
                    after += 1
                if after < scope.hi and tokens[after].kind in _IDS and model.upper(after) not in _STOP:
                    after += 1
                if after > at and tokens[after - 1].end <= offset:
                    join_target = candidate
    # The innermost open call supplies the active argument, not commas in children.
    call = None
    containing = [i for i in range(scope.lo, scope.hi) if tokens[i].text == "(" and tokens[i].end <= offset
                  and (model.pairs.get(i, scope.hi) >= len(tokens) or tokens[model.pairs[i]].start >= offset)]
    if any(model.upper(i - 1) == "OVER" for i in containing):
        clause = "window"
    for opening in reversed(containing):
        if opening and tokens[opening - 1].kind in _IDS:
            word = model.upper(opening - 1)
            if word in {"OVER", "FILTER", "WITHIN"}:
                if word == "OVER":
                    clause = "window"
                continue
            if word in _STOP or word in {"IN", "EXISTS", "NOT"} or opening == scope.insert_open:
                continue
            begin = opening - 1
            while begin >= scope.lo + 2 and tokens[begin - 1].text == "." and tokens[begin - 2].kind in _IDS:
                begin -= 2
            name = document[tokens[begin].start:tokens[opening - 1].end]
            argument = sum(tokens[i].text == "," and tokens[i].start < offset
                           for i in model.top(opening + 1, min(model.pairs.get(opening, scope.hi), scope.hi)))
            separator = {"CAST": "AS", "EXTRACT": "FROM"}.get(word)
            if separator and any(model.upper(i) == separator and tokens[i].end <= offset
                                 for i in model.top(opening + 1, min(model.pairs.get(opening, scope.hi), scope.hi))):
                argument = 1
                if word == "CAST":
                    clause = "type"
            call = Call(name, argument, tokens[begin].start)
            break
    # A cast type follows :: even when the type token is only partly typed.
    if before and (tokens[before[-1]].text == "::" or (
        len(before) > 1 and tokens[before[-1]].text == tokens[before[-2]].text == ":")):
        clause = "type"
    value_field = None
    rhs = before[-1] if before else -1
    # IN lists retain their field across commas; nested calls retain their own context.
    in_open = next((i for i in reversed(containing) if model.upper(i - 1) == "IN"), None)
    if in_open is not None and (rhs == in_open or (rhs >= 0 and tokens[rhs].text == ",")):
        rhs = in_open - 1
    if rhs >= scope.lo and model.upper(rhs) == "NOT" and model.upper(rhs - 1) == "IS":
        rhs -= 1
    if rhs >= scope.lo and (tokens[rhs].text in {"=", "<", ">", "<=", ">=", "<>", "!="} or model.upper(rhs) in {"LIKE", "ILIKE", "IN", "IS", "BETWEEN"}):
        field_end = rhs
        if model.upper(rhs - 1) == "NOT":
            field_end -= 1
        field_start = field_end - 1
        while field_start >= scope.lo + 2 and tokens[field_start - 1].text == "." and tokens[field_start - 2].kind in _IDS:
            field_start -= 2
        if field_start >= scope.lo:
            value_field = model.resolve(field_start, field_end, relations)
    if literal and value_field is not None:
        blocked = False
    used: list[str] = []
    if clause in {"insert_columns", "update_set", "using"}:
        begin = anchor + 1
        finish = next((i for i in model.top(begin, scope.hi) if model.upper(i) in {"WHERE", "RETURNING", "VALUES", "FROM"}), scope.hi)
        if clause in {"insert_columns", "using"}:
            if clause == "using" and begin < scope.hi and tokens[begin].text == "(":
                begin += 1
            finish = min(finish, model.pairs.get(anchor if clause == "insert_columns" else anchor + 1, finish))
        for a, b in model.split(begin, finish):
            if a < b and tokens[a].kind in _IDS and not (tokens[a].start <= offset <= tokens[a].end):
                used.append(_name(tokens[a]))
    identifiers = tuple(dict.fromkeys([r.reference for r in relations] + [r.reference for r in scope.ctes] + list(parameters)))
    return Context(start, end, token, qualifier, quote, clause, blocked, relations, scope.ctes, scope.outputs,
                   join_target, join_left, call, value_field, token if value_field else "", literal,
                   tuple(used), statement_start, statement_end, identifiers)
