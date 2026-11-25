"""Completion-only SQL projections for the dbt model and manual test editors."""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from itertools import count
from pathlib import Path

from d8r.completion.catalog import dialect_name, keywords_for
from d8r.completion.context import _MAX_CHARS, _MAX_TOKENS, _Model, _scan, command_span
from d8r.completion.engine import complete
from d8r.completion.models import Completion, Context
from d8r.dbt import DbtError, _call, _JINJA_OPEN, _yaml
from d8r.dbt_files import DbtFiles, DbtModel
from d8r.dbt_index import DbtIndex, DbtResource, _Tag, _calls, _tags
from d8r.sql import ColumnDef, SchemaContext, TableDef, relation_parts


def _pad(tag: str, insertion: str) -> str | None:
    """Keep every offset, including the line endings *inside* multiline tags."""
    blank = re.sub(r"[^\r\n]+", lambda m: " " * len(m[0]), tag)
    if not insertion:
        return blank
    run = re.search(r" {%d}" % len(insertion), blank)
    if run is None:
        return None
    return blank[:run.start()] + insertion + blank[run.end():]


def _shadow(document: str, dialect: str, *,
            bind: Callable[[_Tag], str] | None = None,
            selected_ref: Callable[[list, dict], bool] | None = None,
            names: tuple[str, str] = ("_dbr0_", "_dbr1_")) -> tuple[str, tuple[tuple[int, int], ...]] | None:
    """Map literal refs to temporary relation identifiers, preserving all offsets.

    The result is only for the completion parser. In particular it is not a dbt
    compilation, is never executed, and never replaces the editor's document.
    """
    if len(document) > _MAX_CHARS:
        return None
    if not _JINJA_OPEN.search(document):
        return document, ()
    chunks, spans = [], []
    previous = 0
    # The manual test editor recognizes tags only in native SQL, whereas the
    # dbt model editor owns template spans even in lexically incomplete SQL
    # (notably an unfinished quoted identifier before a templated FROM).
    active = None if bind else {t.start for t in _scan(document, dialect)
                                if t.kind == "symbol" and _JINJA_OPEN.match(document, t.start)}
    last_tag = 0
    for tag in _tags(document):
        last_tag = tag.end
        if active is not None and tag.start not in active:
            continue
        if not tag.closed or tag.broken or tag.kind not in {"#", "{"}:
            return None  # Never flatten alternative branches into a single scope.
        if tag.kind == "#":
            insertion = ""
        elif bind:
            insertion = bind(tag)
        else:
            try:
                kind, args, kwargs = _call(document[tag.body_start:tag.body_end])
            except DbtError:
                return None
            if kind == "config":
                insertion = ""
            elif kind in {"ref", "source"}:
                insertion = names[0] if kind == "ref" and selected_ref and selected_ref(args, kwargs) else names[1]
            else:
                return None
        start, end = tag.start, tag.end
        padded = _pad(document[start:end], insertion)
        if padded is None:
            return None
        chunks.extend((document[previous:start], padded))
        spans.append((start, end))
        previous = end
    if _JINJA_OPEN.search(document, last_tag):
        return None  # The bounded tag scanner stopped before EOF.
    chunks.append(document[previous:])
    return "".join(chunks), tuple(spans)


@lru_cache(maxsize=32)
def _projection(document: str, dialect: str) -> tuple[str, ...]:
    shadow = _shadow(document, dialect)
    if shadow is None:
        return ()
    text, spans = shadow
    tokens = tuple(token for token in _scan(text, dialect) if token.kind not in {"comment", "whitespace"})
    if not tokens or len(tokens) > _MAX_TOKENS or tokens[0].text.upper() not in {"SELECT", "WITH"}:
        return ()
    parser = _Model(text, tokens, SchemaContext(), dialect)
    end = next((index for index in parser.top(0, len(tokens)) if tokens[index].text == ";"), len(tokens))
    span = parser.query(0, end).projection_span
    if span is None:
        return ()
    names = []
    keywords = keywords_for(dialect)
    for lo, hi in parser.split(*span):
        if lo >= hi:
            continue
        alias = (hi - 2 if hi - lo >= 3 and parser.upper(hi - 2) == "AS"
                 and hi - 2 in parser.top(lo, hi) else None)
        start = alias + 1 if alias is not None else lo
        if any(a <= tokens[start].start < b for a, b in spans):
            continue
        written = tokens[start:hi]
        if any(token.kind != "quoted_identifier" and (
                token.kind != "word" or token.text.upper() in keywords) for token in written[::2]):
            continue
        parts = relation_parts(text[written[0].start:written[-1].end], dialect=dialect)
        if parts is not None:
            names.append(parts[-1])
    return tuple(names)


def model_columns(project: DbtFiles, model: DbtModel, dialect: str,
                  drafts: dict[str, str]) -> tuple[ColumnDef, ...]:
    """Use selected-model properties and explicitly named SQL outputs, never guesses."""
    def read(path: str) -> str:
        return drafts[path] if path in drafts else project.read(path, missing=True).text

    entries = DbtFiles._entries
    try:
        data = _yaml(read(model.properties), Path(model.properties))
    except (DbtError, OSError):
        data = {}
    columns: dict[str, ColumnDef] = {}
    for entry in entries(data, "models"):
        if entry.get("name") != model.name:
            continue
        base = entries(entry, "columns")
        items = base
        if model.version:
            version = next((item for item in entries(entry, "versions") if str(item.get("v")) == model.version), None)
            if version is None:
                continue
            items = entries(version, "columns")
            if "columns" not in version:
                items = base
            else:
                inherited = []
                for item in items:
                    include, exclude = item.get("include"), item.get("exclude")
                    for column in base:
                        name = column.get("name")
                        if (include == "all" or isinstance(include, list) and name in include) and not (
                                isinstance(exclude, list) and name in exclude):
                            inherited.append(column)
                items = inherited + items  # A version's explicit entry wins.
        elif entries(entry, "versions"):
            continue
        for item in items:
            name, dtype, doc = item.get("name"), item.get("data_type"), item.get("description")
            if isinstance(name, str) and name and name.isprintable():
                columns[name] = ColumnDef(name, dtype[:256] if isinstance(dtype, str) else "",
                                          doc[:2048] if isinstance(doc, str) else "")
        break
    if Path(model.path).suffix.lower() == ".sql":
        try:
            for name in _projection(read(model.path), dialect):
                columns.setdefault(name, ColumnDef(name, ""))
        except (DbtError, OSError):
            pass  # Unreadable/unsupported SQL must not hide available YAML columns.
    return tuple(columns.values())


class DbtTestCompletion:
    def __init__(self, model: DbtModel, package: str, dialect: str, columns: tuple[ColumnDef, ...]) -> None:
        self.model, self.package, self.dialect, self.columns = model, package, dialect, columns
        self._column_names = "\n".join(column.name.casefold() for column in columns)
        self._schema_marker = ""
        self._schema = SchemaContext()

    def _selected_ref(self, args: list, kwargs: dict) -> bool:
        return (args in ([self.model.name], [self.package, self.model.name])
                and not (kwargs.keys() - {"v", "version"}) and len(kwargs) <= 1
                and str(kwargs.get("v", kwargs.get("version", ""))) == self.model.version)

    def __call__(self, document: str, offset: int, manual: bool) -> Completion:
        blocked = Completion(Context(offset, offset, blocked=True))
        if len(document) > _MAX_CHARS or command_span(document, offset, dialect=self.dialect) is not None:
            return blocked
        folded = document.casefold()
        names = (f"_dbr{index:x}_" for index in count())
        markers = (next(name for name in names if name not in folded and name not in self._column_names),
                   next(name for name in names if name not in folded and name not in self._column_names))
        projected = _shadow(document, self.dialect, selected_ref=self._selected_ref, names=markers)
        if projected is None:
            return blocked
        text, spans = projected
        if any(a <= offset < b for a, b in spans):
            return blocked
        marker, unknown = markers
        if marker != self._schema_marker:
            self._schema_marker = marker
            self._schema = SchemaContext((TableDef(marker, columns=self.columns),))
        schema = self._schema if marker in text else SchemaContext()
        result = complete(text, offset, schema, dialect=self.dialect, manual=manual)
        scoped = any(relation.table == marker and relation.reference == marker for relation in result.context.relations)
        proposals = []
        for item in result.items:
            start = result.context.start if item.start is None else item.start
            end = result.context.end if item.end is None else item.end
            if any(start < b and end > a for a, b in spans):
                continue
            insert = item.insert
            if (scoped and item.kind == "column" and insert.startswith(marker + ".")
                    and not item.label.startswith(marker + ".")):
                # An unaliased ref has no known SQL qualifier. The native engine
                # only gives this entry an unqualified label if it is unambiguous.
                insert = insert.removeprefix(marker + ".")
            if any(name in value.casefold() for name in markers for value in (item.label, insert)):
                continue
            detail = item.detail.replace(marker, self.model.name).replace(unknown, "dbt relation")
            proposals.append(replace(item, insert=insert, detail=detail, table=None))
        return replace(result, items=tuple(proposals))


@dataclass(frozen=True, kw_only=True)
class _BoundSchema(SchemaContext):
    """Lookup-only dbt relations on top of an existing immutable native catalog."""

    native: SchemaContext
    bindings: dict[str, TableDef]

    def __post_init__(self) -> None:
        # Share the catalog, indexes and routines; never rebuild the column pool
        # per keystroke. Nor should an unbound SELECT offer arbitrary pool fields.
        for field in fields(SchemaContext):
            value = () if field.name == "pool" else getattr(self.native, field.name)
            object.__setattr__(self, field.name, value)

    def table_by_name(self, name: str, *, dialect: str = "postgres") -> TableDef | None:
        return self.bindings.get(name) or self.native.table_by_name(name, dialect=dialect)


# Cache one immutable snapshot by identity: hashing SchemaContext walks its
# columns. A target switch replaces the lookup instead of accumulating catalogs.
_physical_cache: tuple[SchemaContext, str, dict[tuple[str, ...], TableDef | None]] | None = None


def _physical_table(schema: SchemaContext, relation: str, dialect: str) -> TableDef | None:
    if not relation:
        return None
    default = schema.default_schema or ("main" if dialect == "sqlite" else None)

    def identity(name: str) -> tuple[str, ...]:
        parts = relation_parts(name, dialect=dialect) or ()
        # DuckDB Parquet and D1 cache default-namespace tables unqualified.
        if len(parts) == 1 and default:
            parts = (default, *parts)
        return parts if dialect == "postgres" else tuple(part.lower() for part in parts)

    global _physical_cache
    cache = _physical_cache
    if cache is None or cache[0] is not schema or cache[1] != dialect:
        tables: dict[tuple[str, ...], TableDef | None] = {}
        for table in schema.tables:
            if key := identity(table.name):
                tables[key] = None if key in tables else table
        cache = _physical_cache = (schema, dialect, tables)
    return cache[2].get(identity(relation))


def _resource_table(index: DbtIndex, resource: DbtResource, schema: SchemaContext, dialect: str) -> TableDef:
    physical = _physical_table(schema, resource.relation, dialect)
    key = (lambda name: name) if dialect == "postgres" else str.lower
    columns = {key(col.name): col for col in physical.columns} if physical else {}
    for declared in resource.columns:
        name = key(declared.name)
        cached = columns.get(name)
        columns[name] = (ColumnDef(cached.name, declared.type or cached.type,
                                   declared.doc or cached.doc, cached.values) if cached else declared)
    if resource.path.lower().endswith(".sql"):
        try:
            sql = index.read(resource.path)  # DbtIndex.read is snapshot-only.
        except DbtError:
            pass
        else:
            if len(sql) <= _MAX_CHARS:
                for name in _projection(sql, dialect):
                    columns.setdefault(key(name), ColumnDef(name, ""))
    # Use the real table identity for FK/lineage metadata, never the shadow
    # identifier. A logical dbt name alone is not a physical SQL table name.
    return TableDef(physical.name if physical else resource.relation or "", resource.label, tuple(columns.values()))


def complete_dbt_sql(index: DbtIndex | None, document: str, offset: int, schema: SchemaContext,
                     *, dialect: str, manual: bool) -> Completion | None:
    """Complete the SQL outside template tags using only the two supplied snapshots."""
    if not 0 <= offset <= len(document) or len(document) > _MAX_CHARS or not _JINJA_OPEN.search(document):
        return None
    dialect = dialect_name(dialect)
    if command_span(document, offset, dialect=dialect) is not None:
        return None

    # A fresh family absent from the authored SQL cannot collide with an alias
    # or CTE, even one deliberately named like one of our parser identifiers.
    occupied = set(re.findall(r"_d([0-9a-f]+)_", document.casefold()))
    family = next(f"{i:x}" for i in count() if f"{i:x}" not in occupied)
    prefix = f"_d{family}_"
    bindings: dict[str, TableDef] = {}
    markers: dict[str | None, str] = {}

    def bind(tag: _Tag) -> str:
        if tag.tokens and tag.tokens[0].text == "config":
            try:
                if _call(document[tag.body_start:tag.body_end])[0] == "config":
                    return ""
            except DbtError:
                pass  # A compound/dynamic expression is not a known empty config tag.
        resource = None
        if index is not None:
            call = next((c for c in _calls((tag,)) if c.standalone and not c.dynamic), None)
            if call:
                resource = index.resolve(call.kind, call.args, call.kwargs)
        key = resource.key if resource else None
        if key not in markers:
            marker = markers[key] = prefix + f"{len(markers):x}"
            bindings[marker] = (_resource_table(index, resource, schema, dialect)
                                if resource and index is not None else TableDef(""))
        return markers[key]

    projected = _shadow(document, dialect, bind=bind)
    if projected is None:
        return None
    text, spans = projected
    blocked = Completion(Context(offset, offset, blocked=True))
    if any(a <= offset < b for a, b in spans):
        return blocked
    result = complete(text, offset, _BoundSchema(native=schema, bindings=bindings), dialect=dialect, manual=manual)
    context = result.context
    if (context.blocked or context.literal or prefix in context.qualifier.casefold()
            or any(context.start < b and a < context.end for a, b in spans)):
        return blocked
    items = []
    for item in result.items:
        start = context.start if item.start is None else item.start
        end = context.end if item.end is None else item.end
        if any(start < b and a < end for a, b in spans):
            continue
        if item.kind == "expansion" or (context.group_expressions and item.kind == "snippet"
                                         and item.label.startswith("GROUP BY all")):
            # Native relations don't track completeness through stars/CTEs;
            # a dbt properties column list is not an exhaustive projection.
            continue
        insert, detail = item.insert, item.detail
        if item.kind == "column" and insert.startswith(prefix) and prefix not in item.label.casefold():
            marker, dot, component = insert.partition(".")
            if dot and marker in bindings:
                # Native labels are unqualified only for unambiguous columns.
                insert = component
                detail = detail.replace(marker, bindings[marker].doc)
        if any(prefix in value.casefold() for value in (item.label, insert, detail, item.documentation)):
            continue
        if (insert, detail) != (item.insert, item.detail):
            item = replace(item, insert=insert, detail=detail)
        items.append(item)
    # Parser relation/lineage objects may contain private marker names. Only
    # publish facts about the authored caret and its edit, not the shadow scope.
    editor = Context(context.start, context.end, context.token, context.qualifier, context.quote, context.clause,
                     statement_start=context.statement_start, statement_end=context.statement_end)
    return Completion(editor, tuple(items), result.signature, result.hint)
