"""Bounded, snapshot-only native SQL completion proposals.

This module never executes SQL or refreshes metadata. The optional values callback
must read an already successful cache, not issue a database query.
"""
from __future__ import annotations

from bisect import insort
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from itertools import chain
import math
import re

from d8r.sql import RoutineDef, SchemaContext, SqlError, lex_sql, relation_parts

from .catalog import dialect_name, expressions_for, functions_for, keywords_for, types_for
from .context import analyze
from .models import Completion, Context, Field, Relation, Signature, Suggestion

_LIMIT = 80
_RELATION_CLAUSES = frozenset({"from", "join", "into", "update", "table", "relation", "insert"})
_COLUMN_CLAUSES = frozenset({"insert_columns", "update_set", "set", "using"})
Usage = Mapping[tuple[str, str, str], int]
Values = Callable[[str, str], tuple[list[object], bool] | None]


def _decode_parts(name: str, dialect: str) -> tuple[str, ...]:
    return relation_parts(name, dialect=dialect) or (name,)


_cached_parts = lru_cache(maxsize=8192)(_decode_parts)


def _parts(name: str, dialect: str) -> tuple[str, ...]:
    return (_cached_parts if len(name) <= 512 else _decode_parts)(name, dialect)


def _identifier(name: str, dialect: str, quote: str = "") -> str:
    if quote == "[":
        return "[" + name.replace("]", "]]") + "]"
    if quote == "`":
        return "`" + name.replace("`", "``") + "`"
    if quote or not re.fullmatch(r"[a-z_][a-z_0-9$]*", name) or name.upper() in keywords_for(dialect):
        return '"' + name.replace('"', '""') + '"'
    return name


def _qualified(name: str, dialect: str) -> str:
    return ".".join(_identifier(part, dialect) for part in _parts(name, dialect))


def _same_name(left: str, right: str, dialect: str) -> bool:
    return _parts(left, dialect) == _parts(right, dialect)


def _near(left: str, right: str) -> bool:
    """One edit or adjacent transposition; no unbounded edit-distance matrix."""
    if abs(len(left) - len(right)) > 1:
        return False
    if len(left) == len(right):
        differences = [i for i, (a, b) in enumerate(zip(left, right)) if a != b]
        return len(differences) <= 1 or (
            len(differences) == 2 and differences[1] == differences[0] + 1
            and left[differences[0]] == right[differences[1]]
            and left[differences[1]] == right[differences[0]])
    shorter, longer = (left, right) if len(left) < len(right) else (right, left)
    index = next((i for i, (a, b) in enumerate(zip(shorter, longer)) if a != b), len(shorter))
    return shorter[index:] == longer[index + 1:]


def _match(token: str, name: str) -> tuple[int, int] | None:
    token, folded = token.casefold(), name.casefold()
    if not token:
        return (1, 0)
    if folded == token:
        return (0, 0)
    if folded.startswith(token):
        return (1, 0)
    words = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+", name)
    words = [word.casefold() for word in words] or folded.split("_")
    if any(word.startswith(token) for word in words[1:]):
        return (2, 0)
    initials = "".join(word[0] for word in words if word)
    if len(token) >= 2 and initials.startswith(token):
        return (3, len(initials) - len(token))
    if len(token) >= 3:
        # A typo in a partial identifier may still be a prefix of its full name.
        for length in (len(token), len(token) + 1, len(token) - 1):
            if length > 0 and _near(token, folded[:length]):
                return (4, abs(length - len(token)))
        position = -1
        gaps = 0
        for char in token:
            following = folded.find(char, position + 1)
            if following < 0:
                return None
            gaps += following - position - 1
            position = following
        if gaps <= max(3, len(token)):
            return (5, gaps)
    return None


class _Offers:
    """Keep only the best 80 rows while streaming even very large catalogs."""

    def __init__(self, token: str, usage: Usage) -> None:
        self.token = token
        self.usage = usage
        self.rows: list[tuple[tuple, int, Suggestion]] = []
        self.identities: set[tuple] = set()
        self.serial = 0

    def add(self, item: Suggestion, *, name: str = "", relevance: int = 2,
            category: str = "", table: str = "", usage_name: str = "", force: bool = False) -> None:
        match = (0, 0) if force else _match(self.token, name or item.label)
        if match is None:
            return
        identity = (item.label, item.insert, item.start, item.end)
        if identity in self.identities:
            return
        recent = self.usage.get((category, table, usage_name or name or item.label), 1 << 60)
        key = (*match, relevance, recent, item.label.casefold(), item.insert)
        if len(self.rows) == _LIMIT and key >= self.rows[-1][0]:
            return
        self.serial += 1
        insort(self.rows, (key, self.serial, item))
        self.identities.add(identity)
        if len(self.rows) > _LIMIT:
            removed = self.rows.pop()[2]
            self.identities.remove((removed.label, removed.insert, removed.start, removed.end))

    def result(self) -> tuple[Suggestion, ...]:
        # Typo/subsequence candidates are a fallback, never noise alongside a
        # literal prefix, word-boundary, or deliberate abbreviation match.
        strong = any(row[0][0] < 4 for row in self.rows)
        return tuple(row[2] for row in self.rows if not strong or row[0][0] < 4)


def _routines(schema: SchemaContext, dialect: str) -> Iterable[RoutineDef]:
    # A populated native catalog is authoritative, including extension absence.
    # Offline core signatures are a fallback before discovery (and for D1).
    available = schema.routines or functions_for(dialect)
    yield from available
    names = {_parts(routine.name, dialect)[-1].casefold() for routine in available}
    yield from (routine for routine in expressions_for(dialect) if routine.name not in names)
    for fn in schema.fns:
        yield RoutineDef(fn.name, tuple(fn.params), "table", fn.doc, "table")


def _call_routines(context: Context, schema: SchemaContext, dialect: str) -> list[RoutineDef]:
    if context.call is None:
        return []
    name = _parts(context.call.name, dialect)
    return [routine for routine in _routines(schema, dialect)
            if _parts(routine.name, dialect) == name
            or (len(name) == 1 and _parts(routine.name, dialect)[-1].casefold() == name[0].casefold())]


def _signature(context: Context, routines: list[RoutineDef]) -> tuple[Signature | None, str]:
    if not routines or context.call is None:
        return None, ""
    argument = context.call.argument
    routine = min(routines, key=lambda item: (
        not (argument < len(item.parameters) or item.variadic),
        abs(len(item.parameters) - argument - 1), item.name, item.parameters))
    active = min(argument, max(0, len(routine.parameters) - 1)) if routine.variadic else argument
    expected = routine.parameter_types[active] if active < len(routine.parameter_types) else ""
    separator = {"cast": " AS ", "extract": " FROM "}.get(routine.name.casefold(), ", ")
    return Signature(routine.name, routine.parameters, active, routine.return_type,
                     routine.doc, separator), expected


def _type_family(value: str) -> str:
    value = value.casefold()
    if re.search(r"\b(?:u?(?:tiny|small|big|huge)?int(?:eger|[248])?|numeric|decimal|double|float[48]?|real|number)\b", value):
        return "number"
    if any(part in value for part in ("text", "char", "string", "clob")):
        return "text"
    if "bool" in value:
        return "boolean"
    if any(part in value for part in ("date", "time", "interval")):
        return "time"
    if "json" in value:
        return "json"
    return ""


def _field_relevance(field: Field, expected: str, outer: bool = False) -> int:
    target = _type_family(expected)
    return int(outer) * 2 + (int(_type_family(field.type) != target) if target else 0)


def _columns(context: Context, offers: _Offers, dialect: str, expected: str, following_dot: bool = False) -> None:
    relations = context.relations
    if context.qualifier:
        qualifier = _parts(context.qualifier, dialect)
        relations = tuple(relation for relation in relations
                          if _parts(relation.reference, dialect) == qualifier)
    names: dict[str, int] = {}
    for relation in relations:
        for field in relation.fields:
            names[field.name] = names.get(field.name, 0) + 1
    used = set(context.used_columns)
    intersection: set[str] | None = None
    if context.clause == "using":
        if context.join_target is None or not context.join_left:
            return
        intersection = {field.name for field in context.join_target.fields}
        intersection &= {field.name for relation in context.join_left for field in relation.fields}
    for relation in relations:
        alias_match = (not context.qualifier and bool(context.token)
                       and context.clause not in _COLUMN_CLAUSES
                       and _parts(relation.reference, dialect)[-1].casefold().startswith(context.token.casefold()))
        if alias_match:
            reference = (_identifier(_parts(relation.reference, dialect)[-1], dialect, context.quote)
                         if context.quote else relation.reference)
            offers.add(Suggestion(reference, reference, "relation alias", "alias"),
                       name=_parts(reference, dialect)[-1], relevance=-1)
        for field in relation.fields:
            if context.clause in _COLUMN_CLAUSES and field.name in used:
                continue
            if intersection is not None and field.name not in intersection:
                continue
            component = _identifier(field.name, dialect, context.quote)
            ambiguous = names[field.name] > 1 and context.clause not in _COLUMN_CLAUSES
            qualified = bool(context.qualifier) or ambiguous
            label = relation.reference + "." + field.name if qualified else field.name
            insert = relation.reference + "." + component if ambiguous and not context.qualifier else component
            detail = " · ".join(part for part in (field.type, relation.reference,
                                                  "outer scope" if relation.outer else "") if part)
            item = Suggestion(label, insert, detail, "column", field.doc)
            offers.add(item, name=field.name, relevance=_field_relevance(field, expected, relation.outer),
                       category="column", table=field.table or relation.table or "",
                       usage_name=field.column or field.name)
            if alias_match and not following_dot:
                reference = relation.reference
                # The current quote belongs to the alias component, not its fields.
                if context.quote:
                    reference = _identifier(_parts(reference, dialect)[-1], dialect, context.quote)
                qualified_insert = reference + "." + _identifier(field.name, dialect)
                offers.add(Suggestion(relation.reference + "." + field.name, qualified_insert,
                                      detail, "column", field.doc),
                           name=_parts(relation.reference, dialect)[-1],
                           relevance=_field_relevance(field, expected, relation.outer),
                           category="column", table=field.table or relation.table or "",
                           usage_name=field.column or field.name)
            if context.clause in {"update_set", "set"}:
                offers.add(Suggestion(label + " =", insert + " = ", detail, "assignment", field.doc),
                           name=field.name, relevance=1, category="column",
                           table=field.table or relation.table or "", usage_name=field.column or field.name)
    if context.clause in {"group", "group_by", "order", "order_by", "qualify"} and not context.qualifier:
        outputs = (context.group_outputs if context.clause in {"group", "group_by"}
                   and context.group_outputs is not None else context.outputs)
        for field in outputs:
            offers.add(Suggestion(field.name, _identifier(field.name, dialect, context.quote),
                                  field.type or "output alias", "alias", field.doc),
                       name=field.name, relevance=0)


def _alias(name: str, context: Context) -> str:
    words = re.findall(r"[A-Za-z0-9]+", name)
    stem = "".join(word[0] for word in words).lower() or "t"
    if not stem[0].isalpha():
        stem = "t" + stem
    occupied = {name.strip('"`[]').casefold() for name in context.identifiers}
    occupied.update(relation.reference.strip('"`[]').casefold() for relation in context.relations)
    candidate = stem
    suffix = 2
    while candidate in occupied:
        candidate = stem + str(suffix)
        suffix += 1
    return candidate


def _following(document: str, end: int) -> str:
    """Skip trivia without needing to lex an incomplete trailing statement."""
    position = end
    while position < len(document):
        if document[position].isspace():
            position += 1
        elif document.startswith("--", position):
            newline = document.find("\n", position + 2)
            position = len(document) if newline < 0 else newline + 1
        elif document.startswith("/*", position):
            position += 2
            depth = 1
            while position < len(document) and depth:
                if document.startswith("/*", position):
                    depth += 1
                    position += 2
                elif document.startswith("*/", position):
                    depth -= 1
                    position += 2
                else:
                    position += 1
        else:
            break
    return document[position:]


def _has_following_alias(document: str, end: int) -> bool:
    suffix = _following(document, end)
    if suffix == document[end:]:
        return False
    if re.match(r"AS\b", suffix, re.I):
        return True
    match = re.match(r'([A-Za-z_][A-Za-z_0-9$]*|"(?:[^"]|"")+"|`[^`]+`|\[[^\]]+\])', suffix)
    return bool(match and match.group(1).upper() not in keywords_for("postgres")
                and match.group(1).upper() not in keywords_for("duckdb"))


def _following_alias(document: str, end: int) -> tuple[str, int] | None:
    if not _has_following_alias(document, end):
        return None
    suffix = _following(document, end)
    if re.match(r"AS\b", suffix, re.I):
        suffix = _following(document, len(document) - len(suffix) + 2)
    match = re.match(r'([A-Za-z_][A-Za-z_0-9$]*|"(?:[^"]|"")+"|`(?:[^`]|``)+`|\[[^\]]+\])', suffix)
    if match:
        return match.group(), len(document) - len(suffix) + match.end()
    return None


def _tables(document: str, context: Context, schema: SchemaContext, offers: _Offers, dialect: str) -> None:
    qualifier = _parts(context.qualifier, dialect) if context.qualifier else ()
    following_dot = _following(document, context.end).startswith(".")
    following_alias = _following_alias(document, context.end)
    alias_allowed = (context.clause in {"from", "join"}
                     and not _has_following_alias(document, context.end)
                     and not _following(document, context.end).startswith((".", "(")))
    for relation in context.ctes:
        if qualifier or following_dot:
            continue
        name = _parts(relation.name, dialect)[-1]
        insert = _identifier(name, dialect, context.quote)
        if alias_allowed:
            insert += " AS " + _identifier(_alias(name, context), dialect)
        offers.add(Suggestion(relation.name, insert, relation.doc or "common table expression", "cte"),
                   name=name, relevance=0)
    for table in schema.tables:
        parts = _parts(table.name, dialect)
        if following_dot or (qualifier and parts[:-1] != qualifier):
            continue
        if _match(context.token, parts[-1]) is None:
            continue
        insert = (_identifier(parts[-1], dialect, context.quote) if qualifier or len(parts) == 1
                  else _qualified(table.name, dialect))
        label = parts[-1] if qualifier else table.name
        alias = _identifier(_alias(parts[-1], context), dialect) if alias_allowed else ""
        if alias:
            insert += " AS " + alias
        offers.add(Suggestion(label, insert, table.doc or "table", "table", table.doc, table=table.name),
                   name=parts[-1], relevance=1, category="dataset", usage_name=table.name)
        reference = alias or (following_alias[0] if following_alias else "")
        join_end = following_alias[1] if following_alias else context.end
        if (context.clause == "join" and reference
                and not re.match(r"(?:ON|USING)\b", _following(document, join_end), re.I)):
            target = Relation(table.name, reference,
                              tuple(Field(c.name, c.type, table.name, c.name) for c in table.columns),
                              table.name, foreign_keys=table.foreign_keys)
            for left in context.join_left:
                for source, other in ((left, target), (target, left)):
                    for condition in _fk_conditions(source, other, dialect):
                        join_insert = insert + (document[context.end:join_end] if following_alias else "")
                        offers.add(Suggestion(label + " ON " + condition, join_insert + " ON " + condition,
                                              "table and foreign-key join", "join", end=join_end, table=table.name),
                                   name=parts[-1], relevance=0)
    if not qualifier:
        schemas = {_parts(table.name, dialect)[0] for table in schema.tables
                   if len(_parts(table.name, dialect)) > 1}
        for name in schemas:
            offers.add(Suggestion(name + ".", _identifier(name, dialect, context.quote) + ("" if following_dot else "."),
                                  "schema", "schema"), name=name, relevance=3)


def _functions(document: str, context: Context, schema: SchemaContext,
               offers: _Offers, dialect: str, relation_only: bool) -> None:
    qualifier = _parts(context.qualifier, dialect) if context.qualifier else ()
    following = _following(document, context.end)
    has_paren = following.startswith("(")
    call_end = len(document) - len(following) + 1 if has_paren else None
    call_suffix = document[context.end:call_end] if has_paren else "()"
    available = _routines(schema, dialect)
    entries = [(routine, _parts(routine.name, dialect)) for routine in available]
    namespaces: dict[str, set[tuple[str, ...]]] = {}
    if not qualifier:
        for routine, parts in entries:
            namespaces.setdefault(parts[-1], set()).add(parts)
    for routine, parts in entries:
        if routine.kind == "syntax":
            continue
        if context.clause == "call" and context.call is None:
            allowed = "table" if dialect_name(dialect) == "duckdb" else "procedure"
            if dialect_name(dialect) == "sqlite" or routine.kind != allowed:
                continue
        elif relation_only:
            if routine.kind != "table":
                continue
        elif routine.kind in {"table", "procedure"}:
            continue
        if qualifier and parts[:-1] != qualifier:
            continue
        if _match(context.token, parts[-1]) is None:
            continue
        implicit = (len(parts) > 1 and parts[0] in {"pg_catalog", schema.default_schema}
                    and len(namespaces.get(parts[-1], ())) <= 1)
        if qualifier or len(parts) == 1 or implicit:
            insert = _identifier(parts[-1], dialect, context.quote)
        else:
            insert = _qualified(routine.name, dialect)
        label = parts[-1] if qualifier or implicit else routine.name
        detail = routine.kind + " (" + ", ".join(routine.parameters) + ")"
        if routine.return_type:
            detail += " → " + routine.return_type
        if len(parts) > 1:
            detail += " · " + routine.name
        offers.add(Suggestion(label, insert + call_suffix, detail,
                              "function", routine.doc, 0 if has_paren else 1, end=call_end),
                   name=parts[-1], relevance=2, category="function", usage_name=parts[-1])


def _fk_conditions(source: Relation, target: Relation, dialect: str) -> Iterable[str]:
    if not target.table:
        return
    for key in source.foreign_keys:
        if not _same_name(key.target_table, target.table, dialect):
            continue
        if not key.columns or len(key.columns) != len(key.target_columns):
            continue
        pairs: list[str] = []
        for source_column, target_column in zip(key.columns, key.target_columns):
            left = next((field for field in source.fields
                         if (field.column or field.name) == source_column), None)
            right = next((field for field in target.fields
                          if (field.column or field.name) == target_column), None)
            if left is None or right is None:
                break
            pairs.append(source.reference + "." + _identifier(left.name, dialect) + " = "
                         + target.reference + "." + _identifier(right.name, dialect))
        else:
            yield " AND ".join(pairs)


def _joins(document: str, context: Context, offers: _Offers, dialect: str) -> None:
    if context.join_target is None or context.clause not in {"on", "join_condition", "join"}:
        return
    if context.qualifier and not any(_same_name(relation.reference, context.qualifier, dialect)
                                     for relation in (*context.join_left, context.join_target)):
        return
    for left in context.join_left:
        for source, target in ((left, context.join_target), (context.join_target, left)):
            for condition in _fk_conditions(source, target, dialect):
                insert = ("ON " if context.clause == "join" else "") + condition
                # The whole condition replaces the typed component; qualified
                # prefixes must be replaced too or they would be duplicated.
                start = context.start
                if context.qualifier:
                    prefix = document[context.statement_start:context.start]
                    match = re.search(re.escape(context.qualifier) + r"\s*\.\s*$", prefix)
                    start = context.statement_start + match.start() if match else start - len(context.qualifier) - 1
                offers.add(Suggestion(insert, insert, "foreign-key join", "join",
                                      start=max(context.statement_start, start), end=context.end),
                           name=insert, relevance=-1,
                           force=not context.token or (context.clause == "join" and "on".startswith(context.token.casefold())))


def _star(document: str, context: Context, offers: _Offers, dialect: str) -> None:
    if context.token != "*" or context.clause not in {"select", "returning"} or context.call is not None:
        return
    start = context.start
    if context.qualifier:
        prefix = document[context.statement_start:context.start]
        match = re.search(re.escape(context.qualifier) + r"\s*\.\s*$", prefix)
        if match:
            start = context.statement_start + match.start()
    try:
        before = [token for token in lex_sql(document[context.statement_start:start], dialect=dialect)
                  if token.kind not in {"whitespace", "comment"}]
    except SqlError:
        return
    if not before:
        return
    last = before[-1]
    projection = (last.kind == "word" and last.text.upper() in {"SELECT", "DISTINCT", "ALL", "RETURNING"}
                  or last.text == ",")
    if not projection and last.text == ")":
        # PostgreSQL DISTINCT ON (...) precedes the projection, not a multiply.
        depth = 0
        for index in range(len(before) - 1, -1, -1):
            if before[index].kind != "symbol":
                continue
            if before[index].text == ")":
                depth += 1
            elif before[index].text == "(":
                depth -= 1
                if depth == 0:
                    projection = (index >= 2 and before[index - 1].text.upper() == "ON"
                                  and before[index - 2].text.upper() == "DISTINCT")
                    break
    if not projection or re.match(r"(?:EXCLUDE|EXCEPT|REPLACE|RENAME)\b", _following(document, context.end), re.I):
        return
    relations = tuple(relation for relation in context.relations if not relation.outer)
    if context.qualifier:
        relations = tuple(relation for relation in relations
                          if _same_name(relation.reference, context.qualifier, dialect))
    fields = [(relation, field) for relation in relations for field in relation.fields]
    if not fields:
        return
    counts: dict[str, int] = {}
    for _, field in fields:
        counts[field.name] = counts.get(field.name, 0) + 1
    parts = [(relation.reference + "." if context.qualifier or counts[field.name] > 1 else "")
             + _identifier(field.name, dialect) for relation, field in fields]
    offers.add(Suggestion("Expand " + (context.qualifier + "." if context.qualifier else "") + "*",
                          ", ".join(parts), f"{len(parts)} visible columns", "expansion",
                          start=start, end=context.end), force=True, relevance=0)


def _literal(value: object, dialect: str) -> str | None:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Decimal):
        return str(value) if value.is_finite() else None
    if isinstance(value, float):
        return str(value) if math.isfinite(value) else None
    if isinstance(value, (datetime, date, time)):
        value = value.isoformat()
    if isinstance(value, str):
        if "\x00" in value:
            return None
        escaped = value.replace("'", "''")
        if dialect_name(dialect) == "postgres" and "\\" in escaped:
            return "E'" + escaped.replace("\\", "\\\\") + "'"
        return "'" + escaped + "'"
    if isinstance(value, bytes):
        if dialect_name(dialect) == "postgres":
            return "decode('" + value.hex() + "', 'hex')"
        if dialect_name(dialect) == "duckdb":
            return "from_hex('" + value.hex() + "')"
        return "X'" + value.hex() + "'"
    return None


def _values(context: Context, offers: _Offers, dialect: str, values: Values | None) -> None:
    field = context.value_field
    if field is None:
        return
    available: Iterable[object] = field.values or ()
    if values is not None and field.table and field.column:
        cached = values(field.table, field.column)
        if cached is not None:
            available = chain(available, cached[0])
    prefix = context.value_prefix if context.literal else context.token
    for value in available:
        literal = _literal(value, dialect)
        if literal is None or _match(prefix, str(value) if value is not None else "NULL") is None:
            continue
        offers.add(Suggestion(literal, literal, field.name + " · cached value", "value", field.doc),
                   name=str(value) if value is not None else "NULL", relevance=0)


_CLAUSE_KEYWORDS = {
    "statement": "SELECT WITH INSERT UPDATE DELETE CREATE ALTER DROP EXPLAIN BEGIN COMMIT ROLLBACK CALL",
    "select": "DISTINCT CASE NULL TRUE FALSE FROM AS OVER FILTER",
    "from": "LATERAL WHERE JOIN LEFT RIGHT INNER FULL CROSS GROUP ORDER LIMIT OFFSET",
    "join": "LATERAL ON USING",
    "on": "AND OR NOT IS NULL IN EXISTS TRUE FALSE",
    "where": "AND OR NOT IS NULL IN EXISTS BETWEEN LIKE ILIKE TRUE FALSE",
    "having": "AND OR NOT IS NULL IN EXISTS BETWEEN TRUE FALSE",
    "group": "BY HAVING ORDER",
    "group_by": "HAVING ORDER",
    "order": "BY ASC DESC NULLS LIMIT OFFSET",
    "order_by": "ASC DESC NULLS LIMIT OFFSET",
    "returning": "AS NULL CASE",
    "values": "NULL DEFAULT TRUE FALSE",
    "update_set": "NULL DEFAULT CASE WHERE RETURNING",
    "set": "NULL DEFAULT CASE WHERE RETURNING",
    "window": "PARTITION BY ORDER ROWS RANGE GROUPS BETWEEN UNBOUNDED PRECEDING FOLLOWING CURRENT ROW",
    "qualify": "AND OR NOT NULL",
    "using": "",
    "call": "",
    "insert_columns": "",
    "into": "",
    "insert": "INTO",
    "update": "",
    "delete": "FROM",
    "limit": "OFFSET",
}


def _syntax(document: str, context: Context, offers: _Offers, dialect: str) -> None:
    native = keywords_for(dialect)
    words = _CLAUSE_KEYWORDS.get(context.clause, "AND OR NOT NULL CASE").split()
    if context.clause == "from" and not context.relation_transition:
        words = ["LATERAL"]
    for word in words:
        if word in native and word not in {"LEFT", "RIGHT", "INNER", "FULL", "CROSS"}:
            offers.add(Suggestion(word, word, "native SQL keyword", "keyword"), relevance=4)
    if context.relation_transition and context.clause in {"from", "on"}:
        for word in ("INNER", "LEFT", "RIGHT", "FULL", "CROSS", "NATURAL"):
            if word in native:
                label = word + " JOIN"
                offers.add(Suggestion(label, label + " ", "native join clause", "snippet"), relevance=-2)
        for label in ("WHERE", "GROUP BY", "ORDER BY", "LIMIT"):
            offers.add(Suggestion(label, label + " ", "native query clause", "keyword"), relevance=3)
    if context.group_expressions:
        in_group = context.clause in {"group", "group_by"}
        prefix = context.group_prefix if in_group else "GROUP BY "
        eligible = (context.group_slot if in_group else not context.token or "group by".startswith(context.token.casefold()))
        if eligible:
            offers.add(Suggestion("GROUP BY all nonaggregate SELECT expressions",
                                  prefix + ", ".join(context.group_expressions),
                                  "complete nonaggregate projection; known stars expanded", "snippet"),
                       force=True, relevance=-2)
    snippets: list[tuple[str, str, int, str]] = []
    if context.clause in {"select", "where", "having", "on", "update_set", "set", "returning"}:
        snippets.extend((("CASE WHEN", "CASE WHEN  THEN  ELSE  END", 16, "conditional expression"),
                         ("CAST", "CAST( AS )", 5, "native cast expression")))
        if dialect_name(dialect) != "sqlite":
            snippets.append(("EXTRACT", "EXTRACT( FROM )", 7, "native date/time field extraction"))
    if context.clause == "select":
        snippets.append(("COUNT(*)", "COUNT(*)", 0, "count rows"))
    if context.clause in {"where", "having", "on"}:
        snippets.extend((("EXISTS", "EXISTS (SELECT 1 FROM )", 1, "existence predicate"),
                         ("IS NULL", "IS NULL", 0, "NULL predicate"),
                         ("IS NOT NULL", "IS NOT NULL", 0, "non-NULL predicate"),
                         ("BETWEEN", "BETWEEN  AND ", 5, "inclusive range predicate"),
                         ("IN", "IN ()", 1, "membership predicate")))
    if context.clause in {"select", "window"}:
        snippets.append(("OVER", "OVER (PARTITION BY  ORDER BY )", 11, "window specification"))
    if context.clause in {"from", "where", "on", "having"}:
        snippets.append(("GROUP BY", "GROUP BY ", 0, "group rows"))
    if context.clause in {"from", "where", "on", "group", "group_by", "having"}:
        snippets.append(("ORDER BY", "ORDER BY ", 0, "sort rows"))
    if context.clause == "select" and not context.token:
        for relation in context.relations:
            if relation.outer or not relation.fields:
                continue
            projection = ", ".join(relation.reference + "." + _identifier(field.name, dialect)
                                   for field in relation.fields)
            offers.add(Suggestion(relation.reference + ".* columns", projection,
                                  "explicit projection list", "expansion"), relevance=2)
    if context.clause == "statement":
        snippets.extend((("SELECT FROM", "SELECT  FROM ", 6, "projection and source"),
                         ("WITH", "WITH  AS (SELECT ) SELECT ", 21, "common table expression")))
    for label, insert, back, detail in snippets:
        if _following(document, context.end).startswith("(") and "(" in insert:
            continue
        offers.add(Suggestion(label, insert, detail, "snippet", cursor_back=back),
                   name=label, relevance=5)
    if context.clause == "insert_columns" and not context.token:
        used = set(context.used_columns)
        fields = [field for relation in context.relations for field in relation.fields if field.name not in used]
        if fields:
            offers.add(Suggestion("All missing columns", ", ".join(_identifier(field.name, dialect) for field in fields),
                                  "insert column list", "expansion"), force=True, relevance=1)


def complete(document: str, offset: int, schema: SchemaContext, *, dialect: str = "postgres",
             parameters: Iterable[str] = (), usage: Usage | None = None, values: Values | None = None,
             manual: bool = False) -> Completion:
    """Propose native SQL edits from immutable scope and cached metadata only."""
    parameters = tuple(parameters)
    context = analyze(document, offset, schema, dialect=dialect, parameters=parameters)
    call_routines = _call_routines(context, schema, dialect)
    signature, expected = _signature(context, call_routines)
    offers = _Offers(context.token, usage or {})
    if context.blocked and not context.literal:
        return Completion(context)
    if context.literal:
        _values(context, offers, dialect, values)
        return Completion(context, offers.result(), signature)
    if context.token.startswith("@"):
        for parameter in parameters:
            offers.add(Suggestion("@" + parameter, "@" + parameter, "function parameter", "parameter"), relevance=0)
        return Completion(context, offers.result(), signature)
    if context.clause == "statement" and context.call is None:
        if not context.qualifier and not context.quote:
            _syntax(document, context, offers, dialect)
        return Completion(context, offers.result(), signature)
    if context.clause in {"type", "cast", "cast_type"}:
        for name in types_for(dialect):
            offers.add(Suggestion(name, name, "native data type", "type"), relevance=0)
        return Completion(context, offers.result(), signature)
    if (context.call and context.call.name.casefold() == "extract" and context.call.argument == 0
            and dialect_name(dialect) != "sqlite"):
        fields = "YEAR MONTH DAY HOUR MINUTE SECOND QUARTER WEEK DOW DOY EPOCH"
        for name in fields.split():
            offers.add(Suggestion(name, name, "native date/time field", "keyword"), relevance=0)
        return Completion(context, offers.result(), signature)
    _star(document, context, offers, dialect)
    if context.token == "*":
        return Completion(context, offers.result(), signature)
    relation_only = context.clause in _RELATION_CLAUSES and context.call is None
    transition = context.relation_transition and context.call is None
    if relation_only:
        if not transition:
            _tables(document, context, schema, offers, dialect)
    elif context.clause != "call" or context.call is not None:
        _columns(context, offers, dialect, expected, _following(document, context.end).startswith("."))
        if not context.relations and not context.qualifier and context.clause in {"select", "statement"}:
            # SchemaContext owns this precomputed pool; never build global columns
            # for a query whose visible relations already determine its fields.
            for field in schema.pool:
                table = min(field.tables, key=lambda name: offers.usage.get(
                    ("column", name, field.name), 1 << 60), default="")
                offers.add(Suggestion(field.name, _identifier(field.name, dialect, context.quote),
                                      field.type + " · " + ", ".join(field.tables), "column", field.doc),
                           name=field.name, relevance=3, category="column",
                           table=table)
    if context.clause not in _COLUMN_CLAUSES and not transition:
        # A qualifier belonging to a relation resolves columns only, not global
        # functions. Unknown qualifiers can resolve a real native routine schema.
        relation_qualifier = context.qualifier and any(
            _same_name(relation.reference, context.qualifier, dialect) for relation in context.relations)
        if not relation_qualifier:
            _functions(document, context, schema, offers, dialect, relation_only)
    _joins(document, context, offers, dialect)
    _values(context, offers, dialect, values)
    if not context.qualifier and not context.quote:
        syntax_context = (replace(context, clause="select")
                          if context.call and (context.clause in _RELATION_CLAUSES or context.clause == "call") else context)
        _syntax(document, syntax_context, offers, dialect)
    hint = "Cached metadata only" if manual and not offers.rows else ""
    return Completion(context, offers.result(), signature, hint)
