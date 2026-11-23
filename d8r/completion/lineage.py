"""Conservative projection lineage for already executed SQLite and DuckDB SQL."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from d8r.sql import SqlError, Token, _identifier, lex_sql

from .catalog import dialect_name, keywords_for

Origin = tuple[str, str]

_MAX_SQL = 100_000
_MAX_TOKENS = 8_192
_MAX_DATASETS = 4_096
_MAX_CATALOG_TEXT = 300_000
_MAX_FIELDS = 2_048
_MAX_DEPTH = 12
_MAX_SOURCES = 16
_ASCII_FOLD = str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz")
_IDS = {"word", "quoted_identifier"}
_SET_OPS = {"UNION", "INTERSECT", "EXCEPT"}
_CLAUSES = {"WHERE", "GROUP", "HAVING", "WINDOW", "QUALIFY", "ORDER", "LIMIT", "OFFSET", "FETCH"}
_JOIN = {"JOIN", "INNER", "LEFT", "RIGHT", "FULL", "CROSS"}
_UNSUPPORTED_JOIN = {"USING", "NATURAL", "POSITIONAL", "ASOF", "SEMI", "ANTI"}
_SOURCE_BOUNDARY = _JOIN | _UNSUPPORTED_JOIN | {"ON", "INDEXED", "NOT", "LATERAL", "PIVOT", "UNPIVOT",
                                               "SAMPLE", "TABLESAMPLE"}
_SPECIAL_VALUES = {"NULL", "TRUE", "FALSE", "DEFAULT", "CURRENT_DATE", "CURRENT_TIME", "CURRENT_TIMESTAMP",
                   "CURRENT_USER", "CURRENT_ROLE", "CURRENT_CATALOG", "CURRENT_SCHEMA", "SESSION_USER",
                   "USER", "CURRENT_LOCALTIME", "CURRENT_LOCALTIMESTAMP", "LOCALTIME", "LOCALTIMESTAMP"}


def _fold(name: str) -> str:
    # Both engines fold ASCII, including inside quoted identifiers.
    return name.translate(_ASCII_FOLD)


def _name(token: Token) -> str:
    # The shared decoder lowercases unquoted names for PostgreSQL. Retain their
    # spelling here so only the engine's ASCII fold is applied during lookup.
    return _identifier(token) if token.kind == "quoted_identifier" else token.text


class _Unproven(Exception):
    """No proof for this scope; the SQL itself may be perfectly valid."""


@dataclass(frozen=True)
class _Slot:
    name: str | None
    origin: Origin | None = None


@dataclass(frozen=True)
class _Shape:
    slots: tuple[_Slot, ...]
    # Ordering/width came from a catalog star rather than fixed SQL items.
    catalog_order: bool = False


@dataclass(frozen=True)
class _Source:
    qualifiers: tuple[tuple[str, ...], ...]
    shape: _Shape
    by_name: dict[str, _Slot]


class _Binder:
    def __init__(self, tokens: tuple[Token, ...], dialect: str, datasets: Mapping[str, dict]):
        self.tokens = tokens
        self.dialect = dialect
        self.sqlite = dialect == "sqlite"
        self.keywords = frozenset(keywords_for(dialect)) | {"ISNULL", "NOTNULL"}
        self.pairs: dict[int, int] = {}
        stack: list[int] = []
        closing = {"(": ")", "[": "]", "{": "}"}
        for i, token in enumerate(tokens):
            if token.kind != "symbol":
                continue
            if token.text in closing:
                stack.append(i)
            elif token.text in closing.values():
                if not stack or closing[tokens[stack[-1]].text] != token.text:
                    raise _Unproven
                self.pairs[stack.pop()] = i
            elif token.text == ";":
                raise _Unproven
        if stack:
            raise _Unproven

        self.catalog: dict[tuple[str, ...], list[tuple[tuple[str, ...], str, dict]]] = {}
        remaining = _MAX_CATALOG_TEXT
        for dataset, entry in datasets.items():
            remaining -= len(dataset)
            if len(dataset) > _MAX_SQL or remaining < 0:
                raise _Unproven
            tokens = _tokens(dataset, dialect)
            parts, stop = self.path_from(tokens, 0, len(tokens))
            if stop != len(tokens) or not parts:
                continue
            key = self.base_key(parts)
            self.catalog.setdefault(key[-1:], []).append((key, dataset, entry))

    def word(self, i: int) -> str:
        if i < len(self.tokens) and self.tokens[i].kind == "word":
            return self.tokens[i].text.upper()
        return ""

    def symbol(self, i: int, text: str) -> bool:
        return i < len(self.tokens) and self.tokens[i].kind == "symbol" and self.tokens[i].text == text

    def top(self, lo: int, hi: int):
        while lo < hi:
            yield lo
            lo = self.pairs.get(lo, lo) + 1

    def items(self, lo: int, hi: int):
        for i in self.top(lo, hi):
            if self.symbol(i, ","):
                if lo == i:
                    raise _Unproven
                yield lo, i
                lo = i + 1
        if lo >= hi:
            raise _Unproven
        yield lo, hi

    @staticmethod
    def path_from(tokens: tuple[Token, ...], lo: int, hi: int) -> tuple[tuple[str, ...], int]:
        parts = []
        while lo < hi and tokens[lo].kind in _IDS:
            name = _fold(_name(tokens[lo]))
            if not name or "\x00" in name:
                raise _Unproven
            parts.append(name)
            lo += 1
            if lo + 1 >= hi or tokens[lo].text != "." or tokens[lo + 1].kind not in _IDS:
                break
            lo += 1
        return tuple(parts), lo

    def path(self, lo: int, hi: int) -> tuple[tuple[str, ...], int]:
        return self.path_from(self.tokens, lo, hi)

    @staticmethod
    def base_key(path: tuple[str, ...]) -> tuple[str, ...]:
        # D1 publishes unqualified names (main only); Parquet publishes "data"
        # for main.data and schema-qualified identifiers for other relations.
        return ("main", *path) if len(path) == 1 else path

    def base(self, path: tuple[str, ...]) -> tuple[_Shape, tuple[tuple[str, ...], ...]]:
        key = self.base_key(path)
        candidates = self.catalog.get(path[-1:], ())
        matches = [(name, dataset, entry) for name, dataset, entry in candidates
                   if len(path) == 1 or key == name]
        if len(matches) != 1:
            raise _Unproven
        name, dataset, entry = matches[0]
        columns = entry.get("columns", ())
        if not columns or len(columns) > _MAX_FIELDS:
            raise _Unproven
        return (_Shape(tuple(_Slot(c.name, (dataset, c.name)) for c in columns), catalog_order=True),
                (name[-1:], name))

    def alias(self, i: int, hi: int) -> str | None:
        if i >= hi:
            return None
        token = self.tokens[i]
        if token.kind in _IDS:
            return _name(token) or None
        if self.sqlite and token.kind == "string":
            return token.text[1:-1].replace("''", "'") or None
        return None

    def names(self, opening: int) -> tuple[tuple[str, ...], int]:
        close = self.pairs[opening]
        names = []
        for lo, hi in self.items(opening + 1, close):
            if hi != lo + 1 or (name := self.alias(lo, hi)) is None:
                raise _Unproven
            names.append(name)
        return tuple(names), close + 1

    def rename(self, shape: _Shape, names: tuple[str, ...]) -> _Shape:
        # Positional alias lists can hide reordered/stale catalog columns.
        if shape.catalog_order or len(names) > len(shape.slots) or (self.sqlite and len(names) != len(shape.slots)):
            raise _Unproven
        if len(names) < len(shape.slots):
            # For partial lists, don't guess the engine's implicit dedup names.
            self.named(shape)
        return _Shape(tuple(_Slot(name, slot.origin) for name, slot in zip(names, shape.slots))
                      + shape.slots[len(names):])

    @staticmethod
    def named(shape: _Shape) -> dict[str, _Slot]:
        fields = {_fold(slot.name): slot for slot in shape.slots if slot.name is not None}
        if len(fields) != len(shape.slots):
            raise _Unproven
        return fields

    def source(self, lo: int, hi: int, ctes: dict[str, _Shape | None], depth: int) -> tuple[_Source, int]:
        if self.symbol(lo, "("):
            close = self.pairs[lo]
            shape = self.query(lo + 1, close, ctes, depth + 1)
            qualifiers, i = (), close + 1
        else:
            path, i = self.path(lo, hi)
            if not path or self.symbol(i, "(") or self.word(lo) == "LATERAL":
                raise _Unproven
            if len(path) == 1 and path[0] in ctes:
                shape = ctes[path[0]]
                if shape is None:
                    raise _Unproven
                qualifiers = (path,)
            else:
                shape, qualifiers = self.base(path)

        explicit = i < hi and self.word(i) == "AS"
        if explicit:
            i += 1
        name = self.alias(i, hi)
        if name is not None and (explicit or self.word(i) not in _SOURCE_BOUNDARY):
            qualifiers = ((_fold(name),),)
            i += 1
            if i < hi and self.symbol(i, "("):
                if self.sqlite:
                    raise _Unproven
                names, i = self.names(i)
                shape = self.rename(shape, names)
        elif explicit:
            raise _Unproven
        return _Source(qualifiers, shape, self.named(shape)), i

    def sources(self, lo: int, hi: int, ctes: dict[str, _Shape | None], depth: int) -> tuple[_Source, ...]:
        sources = []
        qualifiers: set[tuple[str, ...]] = set()
        joined = False
        while lo < hi:
            source, i = self.source(lo, hi, ctes, depth)
            if len(sources) == _MAX_SOURCES or qualifiers.intersection(source.qualifiers):
                raise _Unproven
            qualifiers.update(source.qualifiers)
            sources.append(source)
            if joined and i < hi and self.word(i) == "ON":
                end = next((j for j in self.top(i + 1, hi) if self.symbol(j, ",")
                            or self.word(j) in _JOIN | _UNSUPPORTED_JOIN), hi)
                if end == i + 1:
                    raise _Unproven
                i = end
            if i == hi:
                return tuple(sources)
            if self.symbol(i, ","):
                lo, joined = i + 1, False
                continue
            # USING/NATURAL may coalesce fields and change wildcard order.
            word = self.word(i)
            if word not in _JOIN:
                raise _Unproven
            if word != "JOIN":
                i += 1
                if word in {"LEFT", "RIGHT", "FULL"} and self.word(i) == "OUTER":
                    i += 1
                if self.word(i) != "JOIN":
                    raise _Unproven
            lo, joined = i + 1, True
        raise _Unproven

    def direct(self, lo: int, hi: int, sources: tuple[_Source, ...]) -> _Slot | None:
        while self.pairs.get(lo) == hi - 1 and self.symbol(lo, "("):
            lo, hi = lo + 1, hi - 1
        path, end = self.path(lo, hi)
        if not path or end != hi:
            return None
        if len(path) == 1 and self.word(lo) in _SPECIAL_VALUES:
            return None
        if not self.sqlite and len(path) > 1 and any(path[0] in s.by_name for s in sources):
            return None  # The same spelling may begin a struct path.
        matches = [s.by_name[path[-1]] for s in sources if (len(path) == 1 or path[:-1] in s.qualifiers)
                   and path[-1] in s.by_name]
        return matches[0] if len(matches) == 1 else None

    def scalar_width(self, lo: int, hi: int) -> None:
        if self.sqlite:
            return
        for i in range(lo, hi):
            t = self.tokens[i]
            if t.kind in _IDS and _fold(_name(t)) in {"columns", "unnest"} and i + 1 < hi and self.symbol(i + 1, "("):
                raise _Unproven
            if t.kind == "symbol" and "*" in t.text:
                # COUNT(*) has fixed width. A star in a wildcard position,
                # including wrapped or prefixed stars, is not a scalar slot.
                if (t.text == "*" and i >= lo + 2 and self.word(i - 2) == "COUNT"
                        and self.symbol(i - 1, "(") and i + 1 < hi and self.symbol(i + 1, ")")):
                    continue
                if (i == lo or i + 1 == hi or self.tokens[i - 1].kind == "symbol"
                        and self.tokens[i - 1].text not in {")", "]", "}"}):
                    raise _Unproven

    def projection(self, lo: int, hi: int, sources: tuple[_Source, ...], *, final: bool) -> _Shape:
        items = tuple(self.items(lo, hi))
        slots = []
        stars = 0
        catalog_order = False
        for a, b in items:
            if self.symbol(b - 1, "*") and (b == a + 1 or self.symbol(b - 2, ".")):
                stars += 1
                if b == a + 1:
                    matches = sources
                else:
                    path, end = self.path(a, b - 2)
                    if not path or end != b - 2:
                        raise _Unproven
                    matches = tuple(s for s in sources if path in s.qualifiers)
                    if not self.sqlite and any(path[0] in s.by_name for s in sources):
                        raise _Unproven  # Could be a struct star, not a table star.
                if stars != 1 or len(matches) != 1:
                    raise _Unproven
                shape = matches[0].shape
                catalog_order = shape.catalog_order
                if catalog_order and len(items) != 1 and not final:
                    raise _Unproven  # An outer projection could hide a stale star's boundary.
                slots.extend(shape.slots)
                continue

            alias = None
            as_at = next((i for i in self.top(a, b) if self.word(i) == "AS"), None)
            if as_at is not None:
                alias = self.alias(as_at + 1, b)
                if as_at == a or as_at + 2 != b or alias is None:
                    raise _Unproven
                b = as_at
            slot = self.direct(a, b, sources)
            if (slot is None and alias is None and b > a + 1 and self.word(b - 1) not in self.keywords
                    and self.tokens[b - 2].end < self.tokens[b - 1].start):
                implicit = self.alias(b - 1, b)
                if implicit is not None:
                    slot = self.direct(a, b - 1, sources)
                    if slot is not None:
                        alias = implicit
            if slot is None:
                self.scalar_width(a, b)
                slots.append(_Slot(alias))
            else:
                slots.append(_Slot(alias if alias is not None else slot.name, slot.origin))
        if len(slots) > _MAX_FIELDS:
            raise _Unproven
        return _Shape(tuple(slots), catalog_order)

    def with_clause(self, lo: int, hi: int, outer: dict[str, _Shape | None], depth: int):
        if self.word(lo) == "RECURSIVE":
            raise _Unproven
        ctes = outer.copy()
        declarations = []
        local = set()
        while lo < hi and self.tokens[lo].kind in _IDS:
            name = _fold(_name(self.tokens[lo]))
            if name in local:
                raise _Unproven
            local.add(name)
            ctes[name] = None  # Reserve forward references too; never fall back to a base relation.
            lo += 1
            names = ()
            if self.symbol(lo, "("):
                names, lo = self.names(lo)
            if self.word(lo) != "AS":
                raise _Unproven
            lo += 1
            if self.word(lo) == "NOT" and self.word(lo + 1) == "MATERIALIZED":
                lo += 2
            elif self.word(lo) == "MATERIALIZED":
                lo += 1
            if not self.symbol(lo, "("):
                raise _Unproven
            close = self.pairs[lo]
            declarations.append((name, names, lo + 1, close))
            lo = close + 1
            if not self.symbol(lo, ","):
                break
            lo += 1
        if not declarations:
            raise _Unproven
        for name, names, a, b in declarations:
            try:
                shape = self.query(a, b, ctes, depth + 1)
                ctes[name] = self.rename(shape, names) if names else shape
            except _Unproven:
                pass  # An unused CTE doesn't establish any names in the main FROM.
        return lo, ctes

    def insert(self, lo: int, hi: int) -> _Shape:
        if self.word(lo) != "INTO":
            raise _Unproven
        path, i = self.path(lo + 1, hi)
        if not path:
            raise _Unproven
        shape, qualifiers = self.base(path)
        if self.word(i) == "AS":
            alias = self.alias(i + 1, hi)
            if alias is None:
                raise _Unproven
            qualifiers, i = ((_fold(alias),),), i + 2
        if self.symbol(i, "("):
            _, i = self.names(i)
        if self.word(i) == "DEFAULT" and self.word(i + 1) == "VALUES":
            i += 2
        elif self.word(i) == "VALUES":
            i += 1
            while self.symbol(i, "("):
                i = self.pairs[i] + 1
                if not self.symbol(i, ","):
                    break
                i += 1
        if self.word(i) != "RETURNING":
            raise _Unproven
        target = _Source(qualifiers, shape, self.named(shape))
        return self.projection(i + 1, hi, (target,), final=True)

    def query(self, lo: int, hi: int, ctes: dict[str, _Shape | None], depth: int = 0) -> _Shape:
        if lo >= hi or depth >= _MAX_DEPTH:
            raise _Unproven
        if self.word(lo) == "WITH":
            lo, ctes = self.with_clause(lo + 1, hi, ctes, depth)
        word = self.word(lo)
        if word == "INSERT" and not depth:
            return self.insert(lo + 1, hi)
        if word != "SELECT":
            raise _Unproven
        lo += 1
        if self.word(lo) in {"ALL", "DISTINCT"}:
            if self.word(lo) == "DISTINCT" and self.word(lo + 1) == "ON" and self.symbol(lo + 2, "("):
                lo = self.pairs[lo + 2] + 1
            else:
                lo += 1
        from_at = stop = hi
        for i in self.top(lo, hi):
            word = self.word(i)
            if word in _SET_OPS | {"INTO", "PIVOT", "UNPIVOT"}:
                raise _Unproven
            if stop != hi:
                continue
            if from_at == hi and word == "FROM" and self.word(i - 1) != "DISTINCT":
                from_at = i
            elif word in _CLAUSES:
                stop = i
        sources = () if from_at == hi else self.sources(from_at + 1, stop, ctes, depth)
        return self.projection(lo, min(from_at, stop), sources, final=not depth)


def _tokens(sql: str, dialect: str) -> tuple[Token, ...]:
    result = []
    for token in lex_sql(sql, dialect=dialect):
        if token.kind == "comment":
            # The shared DuckDB lexer ends line comments on LF. Don't hide SQL
            # after a bare CR if the database ended the comment there instead.
            if dialect == "duckdb" and token.text.startswith("--") and "\r" in token.text.rstrip("\r"):
                raise _Unproven
        elif token.kind == "whitespace":
            if not token.text.isascii():
                raise _Unproven
        else:
            result.append(token)
    return tuple(result)


def projection_origins(sql: str, dialect: str, datasets: Mapping[str, dict],
                       columns: Sequence[str]) -> tuple[Origin | None, ...]:
    """Return one canonical (dataset, column) or ``None`` per executed result slot.

    This is a proof against the caller's captured catalog, not SQL validation.
    Unknown scalar values retain their slots. If the order or width cannot be
    established, all slots are unknown; result labels only cross-check the proof.
    """
    unknown = (None,) * len(columns)
    dialect = dialect_name(dialect)
    if (dialect not in {"sqlite", "duckdb"} or not columns or len(columns) > _MAX_FIELDS
            or len(sql) > _MAX_SQL or not datasets or len(datasets) > _MAX_DATASETS):
        return unknown
    try:
        tokens = _tokens(sql, dialect)
        if tokens and tokens[-1].text == ";":
            tokens = tokens[:-1]
        if not tokens or len(tokens) > _MAX_TOKENS:
            return unknown
        shape = _Binder(tokens, dialect, datasets).query(0, len(tokens), {})
    except (SqlError, _Unproven):
        return unknown
    if len(shape.slots) != len(columns) or any(slot.name is not None and _fold(slot.name) != _fold(header)
                                              for slot, header in zip(shape.slots, columns)):
        return unknown
    return tuple(slot.origin for slot in shape.slots)
