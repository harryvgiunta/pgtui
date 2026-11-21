"""Local, completion-only metadata and SQL projection for the manual dbt test editor."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from itertools import count
from pathlib import Path

from d8r.completion.catalog import keywords_for
from d8r.completion.context import _MAX_CHARS, _MAX_TOKENS, _Model, _scan, command_span
from d8r.completion.engine import complete
from d8r.completion.models import Completion, Context
from d8r.dbt import DbtError, _call, _expression_end, _JINJA_OPEN, _yaml
from d8r.dbt_files import DbtFiles, DbtModel
from d8r.sql import ColumnDef, SchemaContext, TableDef, relation_parts


def _shadow(document: str, dialect: str, *,
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
    for token in _scan(document, dialect):
        if token.start < previous or token.kind != "symbol" or not _JINJA_OPEN.match(document, token.start):
            continue
        start = token.start
        opening = document[start:start + 2]
        if opening == "{#":
            close = document.find("#}", start + 2)
            if close < 0:
                return None
            insertion = ""
        elif opening == "{{":
            try:
                close = _expression_end(document, start + 2)
                expression = document[start + 2:close].strip().removeprefix("-").removesuffix("-")
                kind, args, kwargs = _call(expression)
            except DbtError:
                return None
            if kind == "config":
                insertion = ""
            elif kind in {"ref", "source"}:
                insertion = names[0] if kind == "ref" and selected_ref and selected_ref(args, kwargs) else names[1]
            else:
                return None
        else:
            return None  # Control flow can change the shape and names of the projection.
        end = close + 2
        if len(insertion) > end - start:
            return None
        blank = "".join(char if char in "\r\n" else " " for char in document[start:end])
        chunks.extend((document[previous:start], insertion + blank[len(insertion):]))
        spans.append((start, end))
        previous = end
    chunks.append(document[previous:])
    return "".join(chunks), tuple(spans)


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
