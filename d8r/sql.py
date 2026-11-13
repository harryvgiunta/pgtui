"""Native SQL text and immutable completion metadata.

This is deliberately not a SQL parser. The database owns SQL grammar and name
resolution. Tokens preserve every byte, including trivia; their kinds are word,
quoted_identifier, string, dollar_string, number, parameter, symbol, whitespace,
and comment. Only saved relation calls and their expression parameters expand.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re
from types import MappingProxyType


@dataclass(frozen=True)
class FnDef:
    name: str
    params: tuple[str, ...] = ()
    body: str = ""
    doc: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "params", tuple(self.params))


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str
    doc: str = ""
    values: tuple[object, ...] | None = None

    def __post_init__(self) -> None:
        if self.values is not None:
            object.__setattr__(self, "values", tuple(self.values))


@dataclass(frozen=True)
class RoutineDef:
    name: str
    parameters: tuple[str, ...] = ()
    return_type: str = ""
    doc: str = ""
    kind: str = "scalar"
    variadic: bool = False
    parameter_types: tuple[str, ...] = ()
    columns: tuple[ColumnDef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "parameter_types", tuple(self.parameter_types))
        object.__setattr__(self, "columns", tuple(self.columns))


@dataclass(frozen=True)
class ForeignKey:
    columns: tuple[str, ...]
    target_table: str
    target_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "target_columns", tuple(self.target_columns))


@dataclass(frozen=True)
class TableDef:
    name: str
    doc: str = ""
    columns: tuple[ColumnDef, ...] = ()
    foreign_keys: tuple[ForeignKey, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "columns", tuple(self.columns))
        object.__setattr__(self, "foreign_keys", tuple(self.foreign_keys))


@dataclass(frozen=True)
class PoolColumn(ColumnDef):
    tables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "tables", tuple(self.tables))


@dataclass(frozen=True)
class OpenTable:
    dataset: str
    identifier: str


@dataclass(frozen=True)
class SchemaContext:
    tables: tuple[TableDef, ...] = ()
    fns: tuple[FnDef, ...] = ()
    default_schema: str | None = None
    routines: tuple[RoutineDef, ...] = ()
    pool: tuple[PoolColumn, ...] = field(init=False)
    _fns_by_name: Mapping[str, FnDef] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tables", tuple(self.tables))
        object.__setattr__(self, "fns", tuple(self.fns))
        object.__setattr__(self, "routines", tuple(self.routines))
        object.__setattr__(self, "_fns_by_name", MappingProxyType(_registry(self.fns)))
        columns: dict[str, tuple[ColumnDef, list[str]]] = {}
        for table in self.tables:
            for column in table.columns:
                if column.name in columns:
                    columns[column.name][1].append(table.name)
                else:
                    columns[column.name] = (column, [table.name])
        object.__setattr__(self, "pool", tuple(
            PoolColumn(column.name, column.type, column.doc, column.values, tuple(tables))
            for column, tables in columns.values()
        ))

    def table_by_name(self, name: str, *, dialect: str = "postgres") -> TableDef | None:
        exact = next((table for table in self.tables if table.name == name), None)
        if exact is not None:
            return exact
        parts = relation_parts(name, dialect=dialect)
        if parts is None:
            return None
        equivalent = next((table for table in self.tables if relation_parts(table.name, dialect=dialect) == parts), None)
        if equivalent is not None:
            return equivalent
        if len(parts) == 1 and self.default_schema is not None:
            preferred = next((table for table in self.tables
                              if relation_parts(table.name, dialect=dialect) == (self.default_schema, parts[0])), None)
            if preferred is not None:
                return preferred
        matches = [table for table in self.tables
                   if (candidate := relation_parts(table.name, dialect=dialect)) is not None
                   and len(parts) == 1 and candidate[-1] == parts[0]]
        return matches[0] if len(matches) == 1 else None

    def column_by_name(self, name: str) -> PoolColumn | None:
        return next((column for column in self.pool if column.name == name), None)

    def fn_by_name(self, name: str) -> FnDef | None:
        return self._fns_by_name.get(name.lower())


class SqlError(ValueError):
    def __init__(self, message: str, *, line: int = 1) -> None:
        super().__init__(message)
        self.line = line


@dataclass(frozen=True)
class SQLIssue:
    line: int
    message: str


@dataclass(frozen=True)
class SQLDocument:
    sql: str
    statements: tuple[str, ...]
    errors: tuple[SQLIssue, ...]


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    start: int
    end: int


_WORD = re.compile(r"(?:[^\W\d]|_)[\w$]*", re.UNICODE)
_DOLLAR = re.compile(r"\$(?:[^\W\d][\w]*|_[\w]*)?\$", re.UNICODE)
_NUMBER = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_OPERATORS = frozenset("+-*/<>=~!@#%^&|`?")
_TRIVIA = frozenset({"whitespace", "comment"})
_IDENTIFIERS = frozenset({"word", "quoted_identifier"})


def is_identifier(text: str) -> bool:
    """Whether text is one unquoted SQL identifier (not a qualified name)."""
    return bool(_WORD.fullmatch(text))


def quote_identifier(name: str) -> str:
    if not name or "\x00" in name:
        raise SqlError("An SQL identifier must be nonempty and cannot contain NUL.")
    return '"' + name.replace('"', '""') + '"'


def lex_sql(text: str, *, dialect: str = "postgres") -> tuple[Token, ...]:
    """Lex SQL without interpreting grammar or modifying spelling.

    Ordinary literals use doubled quotes; PostgreSQL E'...' additionally uses
    backslash escapes, nested block comments and tagged dollar bodies. SQLite
    uses nonnested comments, backtick/bracket identifiers and ordinary strings.
    Unterminated lexical regions are errors; incomplete SQL grammar is not.
    """
    tokens: list[Token] = []
    i = 0
    size = len(text)
    sqlite = dialect == "sqlite"
    while i < size:
        start = i
        char = text[i]
        if char.isspace():
            i += 1
            while i < size and text[i].isspace():
                i += 1
            kind = "whitespace"
        elif text.startswith("--", i):
            end = text.find("\n", i + 2)
            i = size if end < 0 else end
            kind = "comment"
        elif text.startswith("/*", i):
            i += 2
            depth = 1
            while i < size and depth:
                if not sqlite and text.startswith("/*", i):
                    depth += 1
                    i += 2
                elif text.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    i += 1
            if depth:
                raise SqlError("Unterminated block comment.", line=text.count("\n", 0, start) + 1)
            kind = "comment"
        elif char in "'\"" or (sqlite and char in "`[") or (
            not sqlite and char in "eE" and i + 1 < size and text[i + 1] == "'"
            and (i == 0 or not _word_char(text[i - 1]))
        ):
            escaped = not sqlite and char in "eE"
            if escaped:
                i += 1
            quote = text[i]
            closing = "]" if quote == "[" else quote
            kind = "string" if quote == "'" else "quoted_identifier"
            i += 1
            while i < size:
                if escaped and text[i] == "\\":
                    i += 2
                elif text[i] == closing:
                    i += 1
                    if quote != "[" and i < size and text[i] == closing:
                        i += 1
                    else:
                        break
                else:
                    i += 1
            else:
                raise SqlError(f"Unterminated {kind.replace('_', ' ')}.", line=text.count("\n", 0, start) + 1)
        elif not sqlite and char == "$" and (match := _DOLLAR.match(text, i)):
            delimiter = match.group()
            end = text.find(delimiter, match.end())
            if end < 0:
                raise SqlError("Unterminated dollar-quoted body.", line=text.count("\n", 0, start) + 1)
            i = end + len(delimiter)
            kind = "dollar_string"
        elif (char == "@" and (i == 0 or not _word_char(text[i - 1]))
              and (match := _WORD.match(text, i + 1))):
            i = match.end()
            kind = "parameter"
        elif match := _WORD.match(text, i):
            i = match.end()
            kind = "word"
        elif match := _NUMBER.match(text, i):
            i = match.end()
            kind = "number"
        elif char in _OPERATORS:
            i += 1
            while i < size and text[i] in _OPERATORS:
                if sqlite and text[i] == "`":
                    break
                if text.startswith("--", i) or text.startswith("/*", i):
                    break
                if text[i] == "@" and _WORD.match(text, i + 1) and text[start:i] not in {"<", "@"}:
                    break
                i += 1
            kind = "symbol"
        else:
            i += 1
            kind = "symbol"
        tokens.append(Token(kind, text[start:i], start, i))
    return tuple(tokens)


def _word_char(char: str) -> bool:
    return char.isalnum() or char in "_$"


def _significant(text: str, *, dialect: str = "postgres") -> tuple[Token, ...]:
    return tuple(token for token in lex_sql(text, dialect=dialect) if token.kind not in _TRIVIA)


def _identifier(token: Token) -> str:
    if token.kind != "quoted_identifier":
        return token.text.lower()
    quote = token.text[0]
    value = token.text[1:-1]
    return value if quote == "[" else value.replace(quote * 2, quote)


def relation_parts(name: str, *, dialect: str = "postgres") -> tuple[str, ...] | None:
    try:
        tokens = _significant(name, dialect=dialect)
    except SqlError:
        return None
    if not tokens or len(tokens) % 2 == 0:
        return None
    parts: list[str] = []
    for index, token in enumerate(tokens):
        if index % 2:
            if token.text != ".":
                return None
        elif token.kind in _IDENTIFIERS:
            part = _identifier(token)
            if not part or "\x00" in part:
                return None
            parts.append(part)
        else:
            return None
    return tuple(parts)


def relation_name(*parts: str) -> str:
    if not parts:
        raise SqlError("A relation name requires an identifier.")
    return ".".join(quote_identifier(part) for part in parts)


def split_statements(text: str, *, dialect: str = "postgres") -> tuple[str, ...]:
    """Split top-level semicolons, preserving native SQL routine/rule bodies.

    BEGIN ATOMIC is a SQL-standard routine body, unlike the ordinary BEGIN
    transaction command. SQLite CREATE TRIGGER owns its BEGIN ... END body.
    CASE expressions own their END inside either kind of body.
    Delimiter tracking finds boundaries only; the server validates their grammar.
    """
    statements: list[str] = []
    start = 0
    significant = False
    parentheses = 0
    blocks: list[str] = []
    previous_word = ""
    prefix: list[str] = []
    trigger_body = False
    previous_text = ""
    for token in lex_sql(text, dialect=dialect):
        if token.kind in _TRIVIA:
            continue
        word = token.text.upper() if token.kind == "word" else ""
        if len(prefix) < 3:
            prefix.append(word or token.text)
        trigger = (prefix[:2] == ["CREATE", "TRIGGER"]
                   or prefix == ["CREATE", "TEMP", "TRIGGER"]
                   or prefix == ["CREATE", "TEMPORARY", "TRIGGER"])
        if dialect == "sqlite" and trigger and word == "BEGIN" and not trigger_body:
            blocks.append("trigger")
            trigger_body = True
        elif dialect != "sqlite" and word == "ATOMIC" and previous_word == "BEGIN":
            blocks.append("atomic")
        elif word == "CASE" and blocks:
            blocks.append("case")
        elif word == "END" and blocks:
            if blocks[-1] != "trigger" or previous_text in {";", "BEGIN"}:
                blocks.pop()
        if token.kind == "symbol":
            if token.text == "(":
                parentheses += 1
            elif token.text == ")":
                parentheses = max(0, parentheses - 1)
            elif token.text == ";" and not parentheses and not blocks:
                if significant:
                    statements.append(text[start:token.start].strip())
                start = token.end
                significant = False
                previous_word = ""
                prefix.clear()
                trigger_body = False
                previous_text = ""
                continue
        significant = True
        previous_word = word
        previous_text = word or token.text
    if significant:
        statements.append(text[start:].strip())
    return tuple(statements)


# These are lexical relation boundaries, not a grammar/keyword whitelist. A
# database may accept other syntax; only known saved calls at these sites expand.
_FROM_END = frozenset({
    "WHERE", "GROUP", "HAVING", "WINDOW", "QUALIFY", "ORDER", "LIMIT", "OFFSET",
    "FETCH", "FOR", "UNION", "INTERSECT", "EXCEPT", "RETURNING", "SET",
})
_ALIAS_BOUNDARY = _FROM_END | frozenset({
    "JOIN", "LEFT", "RIGHT", "FULL", "INNER", "OUTER", "CROSS", "NATURAL",
    "ON", "USING", "WITH", "TABLESAMPLE", "AS", "PIVOT", "UNPIVOT",
})


@dataclass
class _RelationFrame:
    query: bool = False
    in_from: bool = False
    expect: bool = False


def _relation_sites(tokens: tuple[Token, ...]) -> Iterable[int]:
    """Find relation heads, including comma lists and parenthesized joins.

    Each expression parenthesis gets a fresh frame, so FROM in extract/trim and
    commas in arrays, calls, windows or subqueries never act on an outer FROM.
    """
    frames = [_RelationFrame()]
    for index, token in enumerate(tokens):
        frame = frames[-1]
        text = token.text
        word = text.upper() if token.kind == "word" else ""
        if text in {"(", "[", "{"} and token.kind == "symbol":
            relation_group = frame.expect and text == "("
            frame.expect = False
            frames.append(_RelationFrame(in_from=relation_group, expect=relation_group))
        elif text in {")", "]", "}"} and token.kind == "symbol":
            if len(frames) > 1:
                frames.pop()
        elif text == ";" and token.kind == "symbol":
            frames = [_RelationFrame()]
        elif word in {"SELECT", "DELETE", "WITH"} and not (word == "WITH" and frame.in_from):
            frame.query = True
            frame.in_from = False
            frame.expect = False
        elif (word == "FROM" and frame.query
              and not (index and tokens[index - 1].kind == "word" and tokens[index - 1].text.upper() == "DISTINCT")):
            frame.in_from = True
            frame.expect = True
        elif word in _FROM_END:
            frame.in_from = False
            frame.expect = False
        elif word == "JOIN" and frame.in_from:
            frame.expect = True
        elif text == "," and token.kind == "symbol" and frame.in_from:
            frame.expect = True
        elif frame.expect:
            if word in {"LATERAL", "ONLY"}:
                continue
            if token.kind in _IDENTIFIERS or token.kind == "parameter":
                yield index
            frame.expect = False


def _relation_end(tokens: tuple[Token, ...], start: int) -> int:
    end = start + 1
    while end + 1 < len(tokens) and tokens[end].text == "." and tokens[end + 1].kind in _IDENTIFIERS:
        end += 2
    return end


def _alias(tokens: tuple[Token, ...], index: int) -> Token | None:
    if index < len(tokens) and tokens[index].kind == "word" and tokens[index].text.upper() == "AS":
        index += 1
    if index < len(tokens):
        token = tokens[index]
        if token.kind == "quoted_identifier" or (token.kind == "word" and token.text.upper() not in _ALIAS_BOUNDARY):
            return token
    return None


def open_tables(text: str, *, dialect: str = "postgres") -> tuple[OpenTable, ...]:
    """Best-effort lexical completion hints, never SQL validation or execution."""
    try:
        tokens = _significant(text, dialect=dialect)
    except SqlError:
        return ()
    tables: list[OpenTable] = []
    for index in _relation_sites(tokens):
        if tokens[index].kind not in _IDENTIFIERS:
            continue
        end = _relation_end(tokens, index)
        dataset = text[tokens[index].start:tokens[end - 1].end]
        if end < len(tokens) and tokens[end].text == "(":
            try:
                end, _ = _arguments(text, tokens, end, dataset)
            except SqlError:
                continue
            end += 1
        alias = _alias(tokens, end)
        tables.append(OpenTable(dataset, _identifier(alias) if alias else _identifier(tokens[_relation_end(tokens, index) - 1])))
    return tuple(tables)


MAX_FUNCTION_DEPTH = 16
MAX_FUNCTION_EXPANSIONS = 256
MAX_EXPANDED_SQL = 1_000_000
Functions = Mapping[str, FnDef] | Iterable[FnDef]


def _registry(functions: Functions) -> dict[str, FnDef]:
    registry: dict[str, FnDef] = {}
    for fn in functions.values() if isinstance(functions, Mapping) else functions:
        key = fn.name.lower()
        if key in registry:
            raise SqlError(f'Saved function names are case-insensitive: "{fn.name}" conflicts with "{registry[key].name}".')
        registry[key] = fn
    return registry


def _with_function(fn: FnDef, functions: Functions) -> dict[str, FnDef]:
    registry = _registry(functions)
    key = fn.name.lower()
    existing = registry.get(key)
    if existing is not None and existing.name != fn.name:
        raise SqlError(f'Saved function names are case-insensitive: "{fn.name}" conflicts with "{existing.name}".')
    registry[key] = fn
    return registry


def _function_body(fn: FnDef, *, dialect: str = "postgres") -> str:
    if not is_identifier(fn.name):
        raise SqlError("A saved function name must be one bare SQL identifier.")
    if any(not is_identifier(param) for param in fn.params):
        raise SqlError(f"{fn.name}() parameters must be bare SQL identifiers.")
    if len(set(fn.params)) != len(fn.params):
        raise SqlError(f"{fn.name}() has duplicate parameters.")
    statements = split_statements(fn.body, dialect=dialect)
    if len(statements) != 1:
        raise SqlError(f"{fn.name}() must contain exactly one SELECT, WITH, VALUES or TABLE query.")
    body = statements[0]
    tokens = _significant(body, dialect=dialect)
    if tokens[0].kind != "word" or tokens[0].text.upper() not in {"SELECT", "WITH", "VALUES", "TABLE"}:
        raise SqlError(f"{fn.name}() body must be a SELECT, WITH, VALUES or TABLE query.")
    if tokens[0].text.upper() == "WITH":
        depth = 0
        for token in tokens[1:]:
            if token.kind == "symbol" and token.text == "(":
                depth += 1
            elif token.kind == "symbol" and token.text == ")":
                depth -= 1
            elif depth == 0 and token.kind == "word":
                if token.text.upper() in {"SELECT", "VALUES", "TABLE"}:
                    break
                if token.text.upper() in {"INSERT", "UPDATE", "DELETE", "MERGE"}:
                    raise SqlError(f"{fn.name}() must return a query, not a data-modifying statement.")
    relation_sites = set(_relation_sites(tokens))
    for index, token in enumerate(tokens):
        if token.kind != "parameter":
            continue
        line = body.count("\n", 0, token.start) + 1
        if token.text[1:] not in fn.params:
            raise SqlError(f"{fn.name}() has undeclared parameter {token.text}.", line=line)
        previous = tokens[index - 1] if index else None
        following = tokens[index + 1] if index + 1 < len(tokens) else None
        identifier_slot = (
            index in relation_sites
            or previous is not None and previous.kind == "word" and previous.text.upper() in {"AS", "TABLE", "WITH", "RECURSIVE"}
            or previous is not None and previous.text == "."
            or following is not None and following.text == "("
        )
        if identifier_slot:
            raise SqlError(f"{token.text} is an expression parameter, not an SQL identifier.", line=line)
    return body


def _arguments(text: str, tokens: tuple[Token, ...], opening: int, name: str) -> tuple[int, list[str]]:
    closing = {"(": ")", "[": "]", "{": "}"}
    stack = [")"]
    args: list[str] = []
    start = tokens[opening].end
    has_value = False
    for index in range(opening + 1, len(tokens)):
        token = tokens[index]
        value = token.text
        if token.kind == "symbol":
            if value == ";":
                raise SqlError(f"An argument to {name}() cannot contain a statement separator.")
            if value in closing:
                stack.append(closing[value])
            elif value in {")", "]", "}"}:
                if value != stack[-1]:
                    raise SqlError(f"Mismatched argument delimiter in {name}().")
                stack.pop()
                if not stack:
                    if has_value:
                        args.append(text[start:token.start].strip())
                    elif args:
                        raise SqlError(f"{name}() has an empty argument.")
                    return index, args
            elif value == "," and len(stack) == 1:
                if not has_value:
                    raise SqlError(f"{name}() has an empty argument.")
                args.append(text[start:token.start].strip())
                start = token.end
                has_value = False
                continue
        has_value = True
    raise SqlError(f"A call to {name}() needs a closing ')'.")


def _substitute(body: str, arguments: Mapping[str, str], *, dialect: str = "postgres") -> str:
    pieces: list[str] = []
    start = 0
    size = len(body)
    for token in lex_sql(body, dialect=dialect):
        if token.kind == "parameter":
            # A newline keeps an argument's final -- comment from swallowing the
            # closing expression parenthesis. Parentheses preserve precedence.
            argument = "(" + arguments[token.text[1:]] + "\n)"
            size += len(argument) - len(token.text)
            if size > MAX_EXPANDED_SQL:
                raise SqlError("Saved function expansion is too large.")
            pieces.extend((body[start:token.start], argument))
            start = token.end
    pieces.append(body[start:])
    return "".join(pieces)


@dataclass
class _Expansion:
    functions: dict[str, FnDef]
    dialect: str = "postgres"
    count: int = 0

    def expand(self, text: str, stack: tuple[str, ...] = (), depth: int = 0) -> str:
        tokens = _significant(text, dialect=self.dialect)
        pieces: list[str] = []
        start = 0
        size = len(text)
        for index in _relation_sites(tokens):
            token = tokens[index]
            if token.start < start or index + 1 >= len(tokens) or tokens[index + 1].text != "(":
                continue
            key = _identifier(token)
            if self.dialect == "sqlite":
                key = key.lower()
            fn = self.functions.get(key)
            if fn is None:
                continue
            line = text.count("\n", 0, token.start) + 1
            try:
                if key in stack:
                    raise SqlError("Saved function cycle: " + " -> ".join((*stack, key)))
                if depth >= MAX_FUNCTION_DEPTH:
                    raise SqlError(f"Saved functions exceed {MAX_FUNCTION_DEPTH} nested calls.")
                self.count += 1
                if self.count > MAX_FUNCTION_EXPANSIONS:
                    raise SqlError(f"Saved functions exceed {MAX_FUNCTION_EXPANSIONS} expansions.")
                end, args = _arguments(text, tokens, index + 1, fn.name)
                if len(args) != len(fn.params):
                    raise SqlError(f"{fn.name}() takes {len(fn.params)} argument(s); got {len(args)}.")
                body = _function_body(fn, dialect=self.dialect)
                bound = {param: self.expand(arg, stack, depth + 1) for param, arg in zip(fn.params, args)}
                body = self.expand(_substitute(body, bound, dialect=self.dialect), (*stack, key), depth + 1)
                after = end + 1
                if after < len(tokens) and tokens[after].kind == "word" and tokens[after].text.upper() == "WITH":
                    raise SqlError("WITH ORDINALITY is not supported on saved query functions.")
                lateral = (index > 0 and tokens[index - 1].kind == "word"
                           and tokens[index - 1].text.upper() == "LATERAL")
                if lateral and self.dialect == "sqlite":
                    raise SqlError("LATERAL is not supported on SQLite saved query functions; correlated arguments are not rewritten.")
                replacement = ("" if lateral or self.dialect == "sqlite" else "LATERAL ") + "(" + body + "\n)"
                if _alias(tokens, after) is None:
                    replacement += " AS " + token.text
                size += len(replacement) - (tokens[end].end - token.start)
                if size > MAX_EXPANDED_SQL:
                    raise SqlError("Saved function expansion is too large.")
            except SqlError as exc:
                raise SqlError(str(exc), line=line) from None
            pieces.extend((text[start:token.start], replacement))
            start = tokens[end].end
        if not pieces:
            return text
        pieces.append(text[start:])
        return "".join(pieces)


def expand_functions(text: str, functions: Functions = (), *, dialect: str = "postgres") -> str:
    """Expand saved relations in their native dialect, without SQL translation."""
    return _Expansion(_registry(functions), dialect=dialect).expand(text)


def parse_sql(text: str, functions: Functions = (), *, dialect: str = "postgres") -> SQLDocument:
    """Report lexical/expansion errors only; success is not server validation."""
    try:
        sql = expand_functions(text, functions, dialect=dialect)
        return SQLDocument(sql, split_statements(sql, dialect=dialect), ())
    except SqlError as exc:
        return SQLDocument(text, (), (SQLIssue(exc.line, str(exc)),))


def validate_function(fn: FnDef, functions: Functions = (), *, dialect: str = "postgres") -> None:
    registry = _with_function(fn, functions)
    _function_body(fn, dialect=dialect)
    # NULL is a real expression placeholder, not an attempt to type-check SQL.
    function_query(fn, ", ".join("NULL" for _ in fn.params), registry, dialect=dialect)


def function_query(fn: FnDef, arguments: str, functions: Functions = (), *, dialect: str = "postgres") -> str:
    """Produce the same expanded native relation query as a workspace call."""
    registry = _with_function(fn, functions)
    _function_body(fn, dialect=dialect)
    # Check the argument boundary before embedding it in a document. A supplied
    # closing parenthesis must not smuggle an additional statement into preview.
    wrapped = "(" + arguments + "\n)"
    tokens = _significant(wrapped, dialect=dialect)
    end, _ = _arguments(wrapped, tokens, 0, fn.name)
    if end != len(tokens) - 1:
        raise SqlError(f"Unexpected SQL after the arguments to {fn.name}().")
    return expand_functions("SELECT * FROM " + fn.name + wrapped, registry, dialect=dialect)
