"""Bounded editor intelligence for dbt drafts. Nothing here renders or runs SQL."""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import PurePosixPath
import re

from jinja2 import Environment, TemplateSyntaxError, nodes
from jinja2.ext import Extension
from jinja2.parser import Parser
from sqlglot import Dialect, exp
from sqlglot.errors import OptimizeError, ParseError, TokenError
from sqlglot.optimizer.scope import Scope, traverse_scope, walk_in_scope
from sqlglot.tokens import Token, TokenType
import yaml

from d8r.dbt import DbtError, _UniqueLoader
from d8r.dbt_index import DbtIndex, DbtResource, template_calls
from d8r.dbt_lint import call_warnings, select_warnings, types_incompatible
from d8r.sql import ColumnDef, SchemaContext, SqlError, lex_sql, quote_identifier, relation_parts


@dataclass(frozen=True)
class Diagnostic:
    start: int
    end: int
    message: str
    source: str
    severity: str = "error"


@dataclass(frozen=True)
class Insight:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class StarExpansion:
    start: int
    end: int
    columns: tuple[str, ...]
    replacement: str = ""


@dataclass(frozen=True)
class ColumnOrigin:
    key: str
    relation: str
    column: str


@dataclass(frozen=True)
class ColumnDefinition:
    start: int
    end: int
    alias_at: int | None = None  # Inherited names can be renamed by introducing an alias.


@dataclass(frozen=True)
class ColumnUse:
    start: int
    end: int
    source: ColumnOrigin
    implicit: bool = False


@dataclass(frozen=True)
class ColumnLineage:
    column: ColumnDef
    origins: tuple[ColumnOrigin, ...]
    # Value lineage includes transformations and explicit aliases. Only this
    # edge says the output's *name* belongs to an upstream column.
    name_from: ColumnOrigin | None = None
    definition: ColumnDefinition | None = None


@dataclass(frozen=True)
class DbtAnalysis:
    diagnostics: tuple[Diagnostic, ...] = ()
    hovers: tuple[Insight, ...] = ()
    stars: tuple[StarExpansion, ...] = ()
    columns: tuple[ColumnLineage, ...] = ()
    uses: tuple[ColumnUse, ...] = ()
    complete: bool = False

    def output_at(self, offset: int) -> ColumnLineage | None:
        for point in (offset, offset - 1):
            for field in self.columns:
                site = field.definition
                if site is not None and site.start <= point < site.end:
                    return field
        return None


_MAX_CHARS = 128_000
_MAX_TOKENS = 12_000
_MAX_FIELDS = 512
_MAX_RELATIONS = 32
_MAX_SCOPES = 128
_MAX_DEPTH = 10
_MAX_UPSTREAM = 24
_MAX_UPSTREAM_CHARS = 384_000
_NEWLINE = re.compile(r"\r\n?|\n")
_ABSENT = ColumnDef("", "")
_IMPLICIT_COLUMNS = {
    'postgres': {'tableoid', 'xmin', 'cmin', 'xmax', 'cmax', 'ctid'},
    'duckdb': {'rowid'},
    'sqlite': {'rowid', '_rowid_', 'oid'},
}


class _DbtBlock(Extension):
    tags = {"test", "snapshot", "materialization", "docs"}

    def parse(self, parser: Parser) -> nodes.Macro:
        tag = next(parser.stream)
        name = parser.parse_assign_target(name_only=True).name
        node = nodes.Macro(name, [], [], [], lineno=tag.lineno)
        if tag.value == "test":
            parser.parse_signature(node)
        elif tag.value == "materialization":
            while parser.stream.skip_if("comma"):
                option = parser.stream.expect("name")
                if option.value in {"adapter", "supported_languages"}:
                    parser.stream.expect("assign")
                    parser.parse_expression()
                elif option.value != "default":
                    parser.fail(f"Unknown materialization option {option.value!r}", option.lineno)
        node.body = parser.parse_statements((f"name:end{tag.value}",), drop_needle=True)
        return node


class _DbtParser(Parser):
    """dbt extends the otherwise standard Jinja signature grammar with types."""

    def parse_signature(self, node: nodes.Macro | nodes.CallBlock) -> None:
        node.args, node.defaults = [], []
        self.stream.expect("lparen")
        while self.stream.current.type != "rparen":
            if node.args:
                self.stream.expect("comma")
            arg = self.parse_assign_target(name_only=True)
            arg.set_ctx("param")
            if self.stream.skip_if("colon"):
                self._type_name()
            if self.stream.skip_if("assign"):
                node.defaults.append(self.parse_expression())
            elif node.defaults:
                self.fail("non-default argument follows default argument")
            node.args.append(arg)
        self.stream.expect("rparen")

    def _type_name(self) -> None:
        self.stream.expect("name")
        if self.stream.skip_if("lbracket"):
            if self.stream.current.type != "rbracket":
                self._type_name()
                while self.stream.skip_if("comma"):
                    self._type_name()
            self.stream.expect("rbracket")


_JINJA = Environment(extensions=[_DbtBlock, "jinja2.ext.do", "jinja2.ext.loopcontrols"],
                     keep_trailing_newline=True)


class _MarkedUniqueLoader(_UniqueLoader):
    """Use the project loader's duplicate policy, attaching the offending mark."""

    def construct_object(self, node: yaml.Node, deep: bool = False) -> object:
        try:
            return super().construct_object(node, deep=deep)
        except DbtError as exc:
            mark = node.start_mark
            if isinstance(node, yaml.MappingNode):
                seen = set()
                for key, _ in node.value:
                    if key.tag == "tag:yaml.org,2002:merge":
                        continue
                    value = self.construct_object(key)
                    try:
                        if value in seen:
                            mark = key.start_mark
                            break
                        seen.add(value)
                    except TypeError:
                        mark = key.start_mark
                        break
            raise yaml.MarkedYAMLError(problem=str(exc), problem_mark=mark) from exc
        except (TypeError, ValueError) as exc:
            raise yaml.MarkedYAMLError(problem="Invalid YAML scalar value.", problem_mark=node.start_mark) from exc


def _problem(text: str, start: int, end: int, message: str, source: str, severity: str = 'error') -> Diagnostic:
    end = max(0, min(len(text), end))
    start = max(0, min(start, end))
    if start == end and text:
        if start == len(text):
            start -= 1
        else:
            end += 1
    return Diagnostic(start, end, message, source, severity)


class _Warnings:
    def __init__(self, text: str) -> None:
        self.text = text
        self.items: set[Diagnostic] = set()

    def add(self, start: int, end: int, message: str, source: str = 'SQL') -> None:
        self.items.add(_problem(self.text, start, end, message, source, 'warning'))


def _yaml_analysis(text: str) -> DbtAnalysis:
    if not text.strip():
        return DbtAnalysis()
    loader = None
    try:
        loader = _MarkedUniqueLoader(text)
        node = loader.get_single_node()
        # Count expanded alias occurrences before construction: merge keys can
        # otherwise multiply SafeLoader's work on a small document.
        todo = [(node, 0)] if node is not None else []
        count = 0
        while todo:
            item, depth = todo.pop()
            count += 1
            if depth > 100 or count > _MAX_TOKENS * 2:
                return DbtAnalysis()
            if isinstance(item, yaml.MappingNode):
                todo.extend((child, depth + 1) for pair in item.value for child in pair)
            elif isinstance(item, yaml.SequenceNode):
                todo.extend((child, depth + 1) for child in item.value)
        value = loader.construct_document(node) if node is not None else None
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None) or getattr(exc, "context_mark", None)
        offset = mark.index if mark else getattr(exc, "position", 0)
        message = getattr(exc, "problem", None) or "Invalid YAML syntax."
    except RecursionError:
        return DbtAnalysis()
    else:
        if node is None or isinstance(value, dict):
            return DbtAnalysis()
        offset = node.start_mark.index
        message = "A dbt YAML file must contain a mapping."
    finally:
        if loader is not None:
            loader.dispose()
    return DbtAnalysis(diagnostics=(_problem(text, offset, offset + 1, message, "YAML"),))


@dataclass(frozen=True)
class _Piece:
    start: int
    original_start: int
    original_end: int
    synthetic: bool


class _Shadow:
    """Translate half-open parser spans without losing Jinja or CRLF offsets."""

    def __init__(self, original: str) -> None:
        self.original = original
        self.parts: list[str] = []
        self.pieces: list[_Piece] = []
        self.starts: list[int] = []
        self.length = 0
        self.refs: dict[str, tuple[DbtResource | None, Insight]] = {}
        self.configs: list[tuple[int, int]] = []
        self.synthetic_prefix = ""

    def append(self, start: int, end: int, substitute: str | None = None) -> None:
        part = self.original[start:end] if substitute is None else substitute
        if part:
            self.parts.append(part)
            self.starts.append(self.length)
            self.pieces.append(_Piece(self.length, start, end, substitute is not None))
            self.length += len(part)

    @property
    def text(self) -> str:
        return "".join(self.parts)

    def span(self, start: int, end: int) -> tuple[int, int]:
        if not self.pieces:
            return 0, 0
        if start >= self.length:
            return self.pieces[-1].original_end, self.pieces[-1].original_end
        first = self.pieces[bisect_right(self.starts, max(start, 0)) - 1]
        last = self.pieces[bisect_right(self.starts, min(max(start, end - 1), self.length - 1)) - 1]
        return (first.original_start if first.synthetic else first.original_start + start - first.start,
                last.original_end if last.synthetic else last.original_start + end - last.start)


def _column_text(column: ColumnDef) -> str:
    return f"{column.name}: {column.type}" if column.type else column.name


def _fields_text(title: str, fields: tuple[ColumnDef, ...]) -> str:
    labels = map(_column_text, fields[:24])
    shown = ", ".join(label if len(label) <= 120 else label[:119] + "…" for label in labels)
    if len(fields) > 24:
        shown += f", … (+{len(fields) - 24})"
    return f"{title}\n{shown}" if shown else title


def _tag_call(tokens: list[str], name: str) -> bool:
    if tokens[:2] != [name, "("] or tokens[-1:] != [")"]:
        return False
    depth = 0
    for i, token in enumerate(tokens[1:], 1):
        if token == "(":
            depth += 1
        elif token == ")":
            depth -= 1
            if depth == 0:
                return i == len(tokens) - 1
    return False


def _crlf_offsets(text: str) -> list[int]:
    return [m.start() - i for i, m in enumerate(re.finditer(r"\r\n", text))]


def _jinja_problem(text: str, exc: TemplateSyntaxError) -> Diagnostic:
    if (precise := re.fullmatch(r"unexpected char .* at (\d+)", exc.message, re.DOTALL)) is not None:
        normalized = int(precise[1])
        pos = normalized + bisect_right(_crlf_offsets(text), normalized - 1)
        return _problem(text, pos, pos + 1, exc.message, "Jinja")
    starts = [0, *(m.end() for m in _NEWLINE.finditer(text))]
    line = min(max(0, exc.lineno - 1), len(starts) - 1)
    start = starts[line]
    end = starts[line + 1] if line + 1 < len(starts) else len(text)
    line_text = text[start:end].rstrip("\r\n")
    # Jinja's parser exposes a line, but not a token offset. Prefer a tag on
    # that line over an unrelated part of the SQL when it is unambiguous.
    tags = list(re.finditer(r"\{[{%#]", line_text))
    if len(tags) == 1:
        start += tags[0].start()
    else:
        start += len(line_text) - len(line_text.lstrip())
    return _problem(text, start, starts[line] + len(line_text), exc.message, "Jinja")


def _shadow_sql(
    index: DbtIndex | None, text: str, package: str | None, warnings: _Warnings | None,
) -> tuple[_Shadow | None, list[Insight], Diagnostic | None]:
    shadow = _Shadow(text)
    if not re.search(r"\{[{%#]", text):
        shadow.append(0, len(text))
        return shadow, [], None

    try:
        _DbtParser(_JINJA, text).parse()
    except TemplateSyntaxError as exc:
        return None, [], _jinja_problem(text, exc)
    except (RecursionError, ValueError):
        return None, [], None

    hovers: list[Insight] = []
    calls: dict[int, tuple[DbtResource | None, Insight]] = {}
    for call in template_calls(text):
        if call.kind not in {"ref", "source"}:
            continue
        resource = index.resolve(call.kind, call.args, call.kwargs, package=package) if index and not call.dynamic else None
        authored = text[call.start:call.end]
        if resource is not None:
            detail = _fields_text(resource.label, resource.columns)
        elif call.dynamic:
            detail = f"Dynamic dbt {call.kind} — not resolved offline"
        else:
            detail = f"{authored}\nNot resolved in the local dbt project"
            if index is not None and warnings is not None:
                warnings.add(call.start, call.end, f'Cannot resolve this literal {call.kind}() offline.', 'dbt')
        insight = Insight(call.start, call.end, detail)
        hovers.append(insight)
        if call.standalone and not call.dynamic:
            calls[call.tag_start] = (resource, insight)

    # The raw lexer gives exact boundaries and models whitespace control. It
    # normalizes newlines, so map CRLF offsets back before copying any bytes.
    crlf = _crlf_offsets(text)

    def original(offset: int) -> int:
        return offset + bisect_right(crlf, offset - 1)

    normalized = _NEWLINE.sub("\n", text)
    occupied = set(re.findall(r"__pgtui_relation_([0-9]+)_", text.lower()))
    family = 0
    while str(family) in occupied:
        family += 1
    prefix = f"__pgtui_relation_{family}_"
    shadow.synthetic_prefix = prefix
    cursor = start = 0
    words: list[str] = []
    dynamic = False
    for _, kind, value in _JINJA.lex(normalized):
        offset = normalized.find(value, cursor)
        if offset < 0:
            return None, hovers, None
        cursor = offset + len(value)
        if kind == "data":
            shadow.append(original(offset), original(cursor))
        elif kind in {"variable_begin", "block_begin"}:
            start, words = original(offset), []
        elif kind in {"variable_end", "block_end"}:
            if kind == "variable_end" and start in calls:
                symbol = f"{prefix}{len(shadow.refs)}"
                shadow.refs[symbol] = calls[start]
                shadow.append(start, original(offset + len(value.rstrip())), symbol)
            elif kind == "variable_end":
                if _tag_call(words, 'config'):
                    shadow.configs.append((start, original(cursor)))
                else:
                    dynamic = True
            else:
                # A snapshot wrapper emits its body. Other blocks, macro calls
                # and templated expressions can change the grammar or fields.
                dynamic |= not words or words[0] not in {"snapshot", "endsnapshot"}
        elif kind in {"name", "operator", "string", "integer", "float"}:
            words.append(value)
    return (None if dynamic else shadow), hovers, None


@dataclass(frozen=True)
class _Shape:
    fields: tuple[ColumnLineage, ...] = ()
    complete: bool = False


@dataclass(frozen=True)
class _Relation:
    parts: tuple[str, ...]
    qualifier: str
    shape: _Shape
    bindings: dict[str, ColumnLineage | None]


class _ColumnRewrite:
    def __init__(self, incoming: Mapping[tuple[str, str], str],
                 target: tuple[ColumnDefinition, str, str] | None) -> None:
        self.incoming = incoming
        self.target = target
        self.sources: dict[str, set[str]] = {}
        for key, name in incoming:
            self.sources.setdefault(key, set()).add(name)
        self.names = set(incoming.values()) | {name for _, name in incoming}
        self.edits: list[tuple[int, int, str]] = []

    def new_name(self, field: ColumnLineage) -> str:
        source = field.name_from
        return self.incoming.get((source.key, source.column), '') if source is not None else ''

    def check_output(self, shape: _Shape, dialect: str) -> None:
        if not self.incoming:
            return
        names: set[str] = set()
        changed: set[str] = set()
        duplicates: set[str] = set()
        for field in shape.fields:
            new = self.new_name(field)
            name = new or field.column.name
            if dialect != 'postgres':
                name = name.lower()
            if name in names:
                duplicates.add(name)
            names.add(name)
            if new:
                changed.add(name)
        if changed and (not shape.complete or changed & duplicates):
            raise DbtError('The new name could collide with another output column. Use explicit, unique projections.')


def _shape(columns: tuple[ColumnDef, ...], label: str, *, key: str = "",
           complete: bool = False) -> _Shape:
    if len(columns) > _MAX_FIELDS:
        return _Shape()
    fields = []
    for column in columns:
        origin = ColumnOrigin(key, label, column.name)
        fields.append(ColumnLineage(column, (origin,), name_from=origin if key else None))
    return _Shape(tuple(fields), complete)


def _deduplicate(origins: Iterable[ColumnOrigin]) -> tuple[ColumnOrigin, ...]:
    seen: dict[ColumnOrigin, None] = {}
    for origin in origins:
        seen[origin] = None
        if len(seen) > 128:
            return ()
    return tuple(seen)


class _Analysis:
    def __init__(self, index: DbtIndex | None, dialect: str, schema: SchemaContext) -> None:
        self.index, self.schema = index, schema
        dialect = dialect.lower()
        self.dialect = {"d1": "sqlite", "postgresql": "postgres"}.get(dialect, dialect)
        self.parser = Dialect.get_or_raise(self.dialect)
        self.resources: dict[str, _Shape] = {}
        self.active: set[str] = set()
        self.physical_resources: dict[tuple[str, ...], list[DbtResource]] | None = None
        self.physical_shapes: dict[tuple[str, ...], tuple[_Shape, str]] = {}
        self.upstream_count = self.upstream_chars = 0

    def key(self, name: str) -> str:
        return name if self.dialect == "postgres" else name.lower()

    def relation_parts(self, name: str) -> tuple[str, ...] | None:
        parts = relation_parts(name, dialect=self.dialect)
        return tuple(map(self.key, parts)) if parts else None

    def physical(self, sql_name: str, parts: tuple[str, ...], label: str) -> tuple[_Shape, str]:
        if parts in self.physical_shapes:
            return self.physical_shapes[parts]
        table = self.schema.table_by_name(sql_name, dialect=self.dialect)
        canonical = self.relation_parts(table.name) if table is not None else parts
        if self.physical_resources is None:
            self.physical_resources = {}
            if self.index is not None:
                for resource in self.index.resources.values():
                    if resource.relation and (identity := self.relation_parts(resource.relation)):
                        self.physical_resources.setdefault(identity, []).append(resource)
        matches = self.physical_resources.get(canonical or (), ())
        active_match = len(matches) == 1 and matches[0].key in self.active
        if len(matches) == 1 and not active_match:
            resource = matches[0]
            result = self.resource(resource), resource.label
        elif table is not None:
            result = _shape(table.columns, table.name, complete=bool(table.columns)), table.name
        else:
            result = _Shape(), label
        if not active_match:
            self.physical_shapes[parts] = result
        return result

    def enrich(self, shape: _Shape, declarations: tuple[ColumnDef, ...]) -> _Shape:
        by_name = {self.key(col.name): col for col in declarations}
        fields = []
        for field in shape.fields:
            column = field.column
            declared = by_name.get(self.key(column.name), _ABSENT)
            fields.append(replace(field, column=ColumnDef(column.name, column.type or declared.type,
                                                          declared.doc or column.doc)))
        return replace(shape, fields=tuple(fields))

    def resource(self, resource: DbtResource) -> _Shape:
        if resource.key in self.resources:
            return self.resources[resource.key]
        if resource.key in self.active:
            return _Shape()
        declared = resource.columns
        projected = _Shape()
        if (self.index is not None and resource.path.lower().endswith(".sql")
                and len(self.active) < _MAX_DEPTH and self.upstream_count < _MAX_UPSTREAM
                and self.upstream_chars < _MAX_UPSTREAM_CHARS):
            try:
                text = self.index.read(resource.path)
            except DbtError:
                pass
            else:
                self.upstream_count += 1
                self.upstream_chars += len(text)
                if len(text) <= _MAX_CHARS and self.upstream_chars <= _MAX_UPSTREAM_CHARS:
                    self.active.add(resource.key)
                    try:
                        projected, _ = self.sql(text, package=resource.package)
                    finally:
                        self.active.remove(resource.key)
        projected = self.enrich(projected, declared)
        catalog = (self.schema.table_by_name(resource.relation, dialect=self.dialect)
                   if resource.relation and not resource.path.lower().endswith(".sql") else None)
        if catalog and catalog.columns and not resource.columns_complete:
            projected = self.enrich(_shape(catalog.columns, resource.label, complete=True), declared)
        columns = tuple(field.column for field in projected.fields)
        if not projected.complete:
            names = {self.key(col.name) for col in columns}
            columns += tuple(col for col in declared if self.key(col.name) not in names)
        complete = projected.complete or resource.columns_complete
        result = _shape(columns, resource.label, key=resource.key, complete=complete)
        self.resources[resource.key] = result
        return result

    def sql(self, text: str, *, package: str | None = None, warnings: _Warnings | None = None,
            rewrite: _ColumnRewrite | None = None) -> tuple[_Shape, DbtAnalysis]:
        present = warnings is not None
        shadow, hovers, error = _shadow_sql(self.index, text, package, warnings)
        if error or shadow is None:
            if rewrite is not None:
                raise DbtError('Cannot verify column bindings in dynamic or invalid Jinja.')
            return _Shape(), DbtAnalysis(diagnostics=(error,) if error and present else (),
                                         hovers=tuple(hovers) if present else ())
        if rewrite is not None and shadow.configs:
            names = {name.lower() for name in rewrite.names}
            if rewrite.target:
                names.update(name.lower() for name in rewrite.target[1:])
            if any(names.intersection(re.findall(r'\w+', text[start:end].lower())) for start, end in shadow.configs):
                raise DbtError('An inline dbt config may name this column; move it to properties or update it manually.')
        sql = shadow.text
        if not sql.strip():
            if rewrite is not None:
                raise DbtError('Expected one authored SELECT query.')
            return _Shape(), DbtAnalysis(hovers=tuple(hovers) if present else ())
        try:
            tokens = self.parser.tokenize(sql)
            if not tokens or len(tokens) > _MAX_TOKENS:
                if rewrite is not None:
                    raise DbtError('SQL exceeds the column analysis limit.')
                return _Shape(), DbtAnalysis(hovers=tuple(hovers) if present else ())
            statements: list[exp.Expr | None] = []
            # Query analysis does not validate native DDL or routines. Keeping
            # unsupported commands out also avoids SQLGlot logging their text.
            for chunk in _query_chunks(tokens):
                statements.extend(self.parser.parser().parse(chunk, sql))
            if rewrite is not None and (len(statements) != 1 or not isinstance(statements[0], exp.Query)
                                        or any(token.token_type == TokenType.SEMICOLON for token in tokens[:-1])):
                raise DbtError('Expected one authored SELECT query.')
        except TokenError:
            if rewrite is not None:
                raise DbtError('Cannot verify column bindings in invalid SQL.') from None
            if present:
                try:
                    lex_sql(sql, dialect=self.dialect)
                except SqlError as exc:
                    lines = [0, *(m.end() for m in _NEWLINE.finditer(sql))]
                    start = lines[min(exc.line - 1, len(lines) - 1)]
                    end = next((m.start() for m in _NEWLINE.finditer(sql, start)), len(sql))
                    error = _problem(text, *shadow.span(start, end), str(exc), "SQL")
            return _Shape(), DbtAnalysis(diagnostics=(error,) if error else (),
                                         hovers=tuple(hovers) if present else ())
        except ParseError as exc:
            if rewrite is not None:
                raise DbtError('Cannot verify column bindings in invalid SQL.') from None
            problems = _sql_problems(text, shadow, sql, chunk, exc) if present else ()
            return _Shape(), DbtAnalysis(diagnostics=problems, hovers=tuple(hovers) if present else ())
        except RecursionError:
            if rewrite is not None:
                raise DbtError('SQL exceeds the column analysis limit.') from None
            return _Shape(), DbtAnalysis(hovers=tuple(hovers) if present else ())

        results: list[_Shape] = []
        stars: list[StarExpansion] = []
        uses: list[ColumnUse] = []
        call_details: dict[Insight, str] = {}
        if warnings is not None:
            for warning in call_warnings(statements, tokens, dialect=self.dialect, routines=self.schema.routines):
                warnings.add(*shadow.span(warning.start, warning.end), warning.message)
        for statement in statements:
            if not isinstance(statement, exp.Query):
                continue
            query = _Projection(self, shadow, hovers if present else None, stars if present else None,
                                call_details, uses if present else None, rewrite, warnings)
            try:
                results.append(query.analyze(statement))
            except (OptimizeError, RecursionError):
                if rewrite is not None:
                    raise DbtError('Could not resolve SQL scopes safely.') from None
                results.append(_Shape())
        if call_details:
            hovers = [replace(hover, text=call_details[hover]) if hover in call_details else hover for hover in hovers]
        return (results[0] if len(results) == 1 else _Shape(),
                DbtAnalysis(hovers=tuple(sorted(set(hovers), key=lambda item: (item.start, item.end, item.text)))
                            if present else (), stars=tuple(sorted(stars, key=lambda item: item.start)),
                            uses=tuple(sorted(set(uses), key=lambda item: (item.start, item.end, item.source.key,
                                                                         item.source.column)))))


def _query_chunks(tokens: list[Token]) -> Iterator[list[Token]]:
    starts = {TokenType.SELECT, TokenType.WITH, TokenType.VALUES, TokenType.L_PAREN, TokenType.TABLE}
    start = depth = 0
    for i, token in enumerate(tokens):
        kind = token.token_type
        if kind == TokenType.L_PAREN:
            depth += 1
        elif kind == TokenType.R_PAREN:
            depth = max(0, depth - 1)
        elif kind == TokenType.SEMICOLON and depth == 0:
            if start < i:
                if tokens[start].token_type not in starts:
                    return
                yield tokens[start:i]
            start = i + 1
    if start < len(tokens) and tokens[start].token_type in starts:
        yield tokens[start:]


def _sql_problems(text: str, shadow: _Shadow, sql: str, tokens: list[Token],
                  error: ParseError) -> tuple[Diagnostic, ...]:
    positions = {(token.line, token.col): (token.start, token.end + 1) for token in tokens}
    lines = [0, *(m.end() for m in _NEWLINE.finditer(sql))]
    problems: list[Diagnostic] = []
    names = {symbol: text[call.start:call.end] for symbol, (_, call) in shadow.refs.items()}
    synthetic = re.compile(re.escape(shadow.synthetic_prefix) + r'\d+') if names else None
    for entry in error.errors:
        line = min(max(0, (entry.get("line") or 1) - 1), len(lines) - 1)
        end = min(len(sql), lines[line] + (entry.get("col") or 0))
        highlight = entry.get("highlight") or ""
        start = max(0, end - len(highlight))
        start, end = positions.get((entry.get("line"), entry.get("col")), (start, end))
        description = entry.get("description") or "Invalid SQL syntax."
        if synthetic is not None:
            description = synthetic.sub(lambda match: names.get(match[0], match[0]), description)
        problems.append(_problem(text, *shadow.span(start, end), description, "SQL"))
    return tuple(dict.fromkeys(problems))


def _span(node: exp.Expr) -> tuple[int, int] | None:
    start, end = node.meta_get("start"), node.meta_get("end")
    return (start, end + 1) if isinstance(start, int) and isinstance(end, int) else None


def _renamed(shape: _Shape, names: list[str]) -> _Shape:
    if not names:
        return shape
    if not shape.complete or len(names) > len(shape.fields):
        return _Shape()
    return replace(shape, fields=tuple(
        replace(field, column=replace(field.column, name=names[i]), name_from=None, definition=None)
        if i < len(names) else field
        for i, field in enumerate(shape.fields)
    ))


class _Projection:
    def __init__(self, analysis: _Analysis, shadow: _Shadow,
                 hovers: list[Insight] | None, stars: list[StarExpansion] | None,
                 call_details: dict[Insight, str], uses: list[ColumnUse] | None,
                 rewrite: _ColumnRewrite | None, warnings: _Warnings | None) -> None:
        self.analysis, self.shadow = analysis, shadow
        self.hovers, self.stars = hovers, stars
        self.call_details = call_details
        self.uses, self.rewrite = uses, rewrite
        self.warnings = warnings
        self.outputs: dict[int, _Shape] = {}
        self.original_names: dict[int, str] = {}

    def unsupported(self) -> _Shape:
        if self.rewrite is not None:
            raise DbtError('Unsupported or incomplete SQL scope; column bindings cannot be verified.')
        return _Shape()

    def analyze(self, statement: exp.Query) -> _Shape:
        for i, node in enumerate(statement.walk()):
            if i >= _MAX_TOKENS * 2 or isinstance(node, exp.With) and node.recursive:
                return self.unsupported()
            if self.rewrite is not None and isinstance(node, exp.Column) and any(
                not isinstance(part, (exp.Identifier, exp.Star)) for part in node.parts
            ):
                return self.unsupported()
            if isinstance(node, exp.Identifier):
                if (self.shadow.refs and self.shadow.synthetic_prefix in node.name.lower()
                        and not (isinstance(node.parent, exp.Table) and node.parent.this is node
                                 and len(node.parent.parts) == 1 and node.name in self.shadow.refs)):
                    return self.unsupported()
                self.original_names[id(node)] = node.name
                self.analysis.parser.normalize_identifier(node)

        statement = statement.unnest() if isinstance(statement, exp.Subquery) else statement
        scopes = traverse_scope(statement)
        if len(scopes) > _MAX_SCOPES:
            return self.unsupported()
        for scope in scopes:
            expr = scope.expression
            outer_names = scope.outer_columns
            alias = expr.parent.args.get("alias") if isinstance(expr.parent, (exp.CTE, exp.Subquery)) else None
            if isinstance(alias, exp.TableAlias) and all(isinstance(p, exp.Identifier) for p in alias.columns):
                outer_names = [self.name(identifier) for identifier in alias.columns]
            if isinstance(expr, exp.Select):
                result = self.select(scope, expr, outer_names)
            elif isinstance(expr, exp.Union) and not expr.args.get("by_name"):
                result = self.union(scope)
                if self.rewrite is not None:
                    names = self.rewrite.names | set(self.rewrite.target[1:] if self.rewrite.target else ())
                    if any(isinstance(node, exp.Column) and node.name in names for node in scope.walk()):
                        raise DbtError('Cannot verify a named reference in a set operation clause; use its column position.')
            elif isinstance(expr, exp.Subquery):
                result = self.outputs.get(id(expr.this), _Shape())
            else:
                result = self.unsupported()
            self.outputs[id(expr)] = _renamed(result, outer_names)
            if self.rewrite is not None:
                self.rewrite.check_output(self.outputs[id(expr)], self.analysis.dialect)
        return self.outputs.get(id(statement), _Shape())

    def name(self, node: exp.Identifier) -> str:
        if self.analysis.dialect == "postgres":
            return node.name
        return self.original_names.get(id(node), node.name)

    def hover(self, span: tuple[int, int] | None, label: str) -> None:
        if span is not None and self.hovers is not None and len(self.hovers) < _MAX_TOKENS:
            self.hovers.append(Insight(*self.shadow.span(*span), label))

    def use(self, span: tuple[int, int] | None, field: ColumnLineage, *, implicit: bool = False) -> None:
        if self.uses is not None and len(self.uses) < _MAX_TOKENS and span is not None and field.name_from:
            self.uses.append(ColumnUse(*self.shadow.span(*span), field.name_from, implicit))

    def sources(self, scope: Scope, select: exp.Select) -> tuple[_Relation, ...] | None:
        from_ = select.args.get("from_")
        joins = select.args.get("joins") or ()
        authored = [from_.this] if from_ else []
        authored.extend(join.this for join in joins)
        if (len(authored) > _MAX_RELATIONS or select.args.get("laterals")
                or any(join.side in {"SEMI", "ANTI"} or join.kind in {"SEMI", "ANTI"}
                       or join.args.get("pivots") for join in joins)):
            return None
        selected = scope.selected_sources
        if len(authored) != len(selected):
            return None
        relations = []
        for relation in authored:
            if not isinstance(relation, (exp.Table, exp.Subquery)) or any(
                relation.args.get(arg) for arg in ("joins", "pivots", "sample", "ordinality", "rows_from")
            ):
                return None
            alias = relation.args.get("alias")
            alias_id = alias.this if isinstance(alias, exp.TableAlias) else None
            entry = selected.get(relation.alias_or_name)
            if entry is None:
                return None
            _, source = entry
            shape = _Shape()
            parts: tuple[str, ...] = ()
            label = "Derived query"
            qualifier = ""
            if (isinstance(relation, exp.Table) and (identifiers := relation.parts)
                    and all(isinstance(p, exp.Identifier) for p in identifiers)):
                parts = tuple(p.name for p in identifiers)
                qualifier = ".".join(quote_identifier(self.name(p)) for p in identifiers)
                first, last = _span(identifiers[0]), _span(identifiers[-1])
                table_span = (first[0], last[1]) if first and last else None
                label = ".".join(self.name(p) for p in identifiers)
                if isinstance(source, Scope):
                    shape = self.outputs.get(id(source.expression), _Shape())
                elif len(parts) == 1 and parts[0] in self.shadow.refs:
                    resource, call = self.shadow.refs[parts[0]]
                    label = call.text
                    parts, qualifier = (), ""  # Never expose a synthetic SQL identifier.
                    if resource is not None:
                        shape = self.analysis.resource(resource)
                        if self.rewrite is not None and (names := self.rewrite.sources.get(resource.key)):
                            known = {field.column.name for field in shape.fields}
                            if not shape.complete or not names <= known:
                                raise DbtError(f'Column shape of {resource.label} could not be verified.')
                        label = resource.label
                        if self.hovers is not None:
                            self.call_details[call] = _fields_text(label, tuple(f.column for f in shape.fields))
                        if resource.relation:
                            physical = self.analysis.relation_parts(resource.relation)
                            if physical:
                                parts = physical
                                qualifier = resource.relation
                else:
                    shape, label = self.analysis.physical(qualifier, parts, label)
                if isinstance(alias, exp.TableAlias) and alias.columns:
                    shape = (_renamed(shape, [self.name(p) for p in alias.columns])
                             if all(isinstance(p, exp.Identifier) for p in alias.columns) else _Shape())
                self.hover(table_span, _fields_text(label, tuple(f.column for f in shape.fields)))
            elif isinstance(source, Scope) and isinstance(relation, exp.Subquery):
                shape = self.outputs.get(id(source.expression), _Shape())
            if isinstance(alias_id, exp.Identifier):
                parts, qualifier = (alias_id.name,), quote_identifier(self.name(alias_id))
                self.hover(_span(alias_id), _fields_text(f"{self.name(alias_id)} → {label}",
                                                        tuple(f.column for f in shape.fields)))
            bindings: dict[str, ColumnLineage | None] = {}
            for field in shape.fields:
                key = self.analysis.key(field.column.name)
                bindings[key] = field if key not in bindings else None
            relations.append(_Relation(parts, qualifier, shape, bindings))
        return tuple(relations)

    @staticmethod
    def qualified(relations: tuple[_Relation, ...], qualifier: tuple[str, ...]) -> tuple[_Relation, ...]:
        return tuple(r for r in relations if qualifier and r.parts[-len(qualifier):] == qualifier)

    def lookup(self, node: exp.Column, relations: tuple[_Relation, ...], merged: bool,
               *, name: str | None = None) -> ColumnLineage | None:
        column = node.this
        if not isinstance(column, exp.Identifier):
            return None
        if node.table:
            relations = self.qualified(relations, tuple(p.name for p in node.parts[:-1]))
            if len(relations) != 1:
                return None
        elif len(relations) > 1 and (merged or any(not r.shape.complete for r in relations)):
            return None
        name = self.analysis.key(column.name if name is None else name)
        hits = [relation.bindings[name] for relation in relations if name in relation.bindings]
        return hits[0] if len(hits) == 1 else None

    def select(self, scope: Scope, select: exp.Select, outer_names: list[str]) -> _Shape:
        relations = self.sources(scope, select)
        if relations is None:
            return self.unsupported()
        joins = select.args.get("joins") or ()
        merged = any(join.args.get("using") or join.method == "NATURAL" for join in joins)
        aliases = {self.analysis.key(projection.alias) for projection in select.expressions if isinstance(projection, exp.Alias)}
        resolved: dict[int, ColumnLineage] = {}
        columns = []
        for node in scope.walk():
            if not isinstance(node, exp.Column) or node.is_star:
                continue
            columns.append(node)
            # Some dialects resolve output aliases before FROM fields in these
            # clauses. Never assume the input wins a collision.
            if not node.table and self.analysis.key(node.name) in aliases and self.alias_sensitive(node, select):
                continue
            field = self.lookup(node, relations, merged)
            if field is not None:
                resolved[id(node)] = field
                label = _column_text(field.column)
                if field.column.doc:
                    label += f"\n{field.column.doc}"
                if field.origins:
                    label += "\nFrom " + ", ".join(f"{o.relation}.{o.column}" for o in field.origins[:8])
                first, last = _span(node.parts[0]), _span(node.this)
                self.hover((first[0], last[1]) if first and last else last, label)
                self.use(last, field)
            elif self.warnings is not None:
                self.check_column(node, scope, relations, merged)

        if self.warnings is not None:
            self.join_warnings(joins, relations, resolved)

            def column_type(node: exp.Column) -> str:
                field = resolved.get(id(node))
                return field.column.type if field else ''

            for warning in select_warnings(select, dialect=self.analysis.dialect, column_type=column_type,
                                           correlated=scope.can_be_correlated):
                self.warn((warning.start, warning.end), warning.message)

        fields: list[ColumnLineage] = []
        complete = bool(select.expressions)
        for projection in select.expressions:
            if isinstance(projection, exp.Star) or isinstance(projection, exp.Column) and projection.is_star:
                expanded = self.star(projection, relations, merged)
                fields.extend(expanded.fields)
                complete &= expanded.complete
            else:
                position = len(fields)
                fallback = outer_names[position] if complete and position < len(outer_names) else ""
                field = self.field(projection, resolved, fallback)
                if field is not None:
                    fields.append(field)
                    if isinstance(projection, exp.Alias):
                        self.hover(_span(projection.args["alias"]), _column_text(field.column))
                else:
                    complete = False
            if len(fields) > _MAX_FIELDS:
                return self.unsupported()
        if self.rewrite is not None:
            self.rewrite_select(scope, select, joins, relations, merged, columns, resolved, fields)
        return _Shape(tuple(fields), complete)

    def warn(self, span: tuple[int, int] | None, message: str) -> None:
        if self.warnings is not None and span is not None:
            self.warnings.add(*self.shadow.span(*span), message)

    def join_warnings(self, joins: Iterable[exp.Join], relations: tuple[_Relation, ...],
                      resolved: dict[int, ColumnLineage]) -> None:
        def on_key(node: exp.Expr) -> tuple[_Relation, str, exp.Identifier] | None:
            value = column = node.unnest()
            while isinstance(column, exp.Cast):
                column = column.this.unnest()
            if not isinstance(column, exp.Column) or not isinstance(column.this, exp.Identifier):
                return None
            field = resolved.get(id(column))
            name = self.analysis.key(column.name)
            if column.table:
                matches = self.qualified(relations, tuple(part.name for part in column.parts[:-1]))
            elif field is not None:
                matches = tuple(r for r in relations if r.bindings.get(name) is field)
            else:
                return None  # No safe owner for an unresolved unqualified key.
            if len(matches) != 1:
                return None
            owner = matches[0]
            if field is None and (owner.shape.complete or name in owner.bindings):
                return None  # Missing or ambiguous in known inputs, not merely untyped.
            typ = self.type_of(value) if isinstance(value, exp.Cast) else field.column.type if field else ''
            return owner, typ, column.this

        def using_type(name: str, relation: _Relation) -> str | None:
            if name not in relation.bindings:
                return None if relation.shape.complete else ''
            field = relation.bindings[name]
            return field.column.type if field is not None else None

        def compare(a: str, b: str, left_key: exp.Identifier, right_key: exp.Identifier) -> None:
            if types_incompatible(a, b, dialect=self.analysis.dialect):
                self.warn(_span(right_key), f'JOIN keys have different known type families ({a} and {b}). '
                          'Verify the comparison or cast the keys explicitly.')
            elif bool(a) != bool(b):
                self.warn(_span(left_key if not a else right_key),
                          f'Cannot verify this JOIN key type offline; the other key has known type {a or b}.')

        merged_left = False
        for i, join in enumerate(joins, 1):
            left, right = relations[:i], relations[i]
            if (on := join.args.get('on')) is not None:
                for node in walk_in_scope(on):
                    if not isinstance(node, (exp.EQ, exp.NullSafeEQ)):
                        continue
                    a, b = on_key(node.this), on_key(node.expression)
                    if a is None or b is None:
                        continue
                    if a[0] is right:
                        a, b = b, a
                    if b[0] is right and any(a[0] is relation for relation in left):
                        compare(a[1], b[1], a[2], b[2])
            # A previous NATURAL/USING join can change which left-hand key is exposed.
            using = join.args.get('using') or ()
            if using and not merged_left and (len(left) == 1 or all(r.shape.complete for r in left)):
                for key in using:
                    if not isinstance(key, exp.Identifier):
                        continue
                    name = self.analysis.key(key.name)
                    owners = left if len(left) == 1 else tuple(r for r in left if name in r.bindings)
                    if len(owners) == 1 and (first := using_type(name, owners[0])) is not None and (
                        other := using_type(name, right)
                    ) is not None:
                        compare(first, other, key, key)
            merged_left |= bool(using) or join.method == 'NATURAL'

    def check_column(self, node: exp.Column, scope: Scope, relations: tuple[_Relation, ...], merged: bool) -> None:
        dialect = self.analysis.dialect
        if (scope.can_be_correlated or dialect == 'duckdb' and scope.is_derived_table or
                not isinstance(node.this, exp.Identifier) or node.name in _IMPLICIT_COLUMNS.get(dialect, ())):
            return
        span = _span(node.this)
        if node.table:
            qualifier = tuple(part.name for part in node.parts[:-1])
            if dialect == 'duckdb' and any(not r.shape.complete or qualifier[0] in r.bindings for r in relations):
                return  # Dotted names can resolve to struct fields instead.
            relations = self.qualified(relations, qualifier)
            if len(relations) != 1:
                return
        elif merged or any(node.name in relation.parts[-1:] for relation in relations):
            return
        elif dialect == 'sqlite' and node.this.quoted and span is not None and (
            self.shadow.original.startswith('"', self.shadow.span(*span)[0])
        ):
            return  # An unresolved double-quoted name can be a SQLite string.
        if relations and all(relation.shape.complete for relation in relations):
            name = self.analysis.key(node.name)
            count = sum(name in relation.bindings for relation in relations)
            if not count:
                self.warn(span, f'Column {node.name!r} is not in the known input schema.')
            elif count > 1 or any(name in relation.bindings and relation.bindings[name] is None for relation in relations):
                self.warn(span, f'Column {node.name!r} is ambiguous in the known input schema.')

    def alias_sensitive(self, node: exp.Column, select: exp.Select) -> bool:
        parent = node
        while parent is not None and parent is not select:
            if self.analysis.dialect == 'duckdb' and parent.parent is select and parent.arg_key == 'expressions':
                return True
            if isinstance(parent, (exp.Order, exp.Group, exp.Having, exp.Qualify, exp.Distinct)) or (
                self.analysis.dialect != 'postgres' and isinstance(parent, (exp.Where, exp.Join))
            ):
                return True
            parent = parent.parent
        return False

    def authored_span(self, identifier: exp.Identifier) -> tuple[int, int]:
        span = _span(identifier)
        if span is None:
            raise DbtError('Cannot find the authored column token.')
        start, end = self.shadow.span(*span)
        if start == end:
            raise DbtError('Cannot find the authored column token.')
        return start, end

    def rewrite_select(self, scope: Scope, select: exp.Select, joins: tuple[exp.Join, ...],
                       relations: tuple[_Relation, ...], merged: bool, columns: list[exp.Column],
                       resolved: dict[int, ColumnLineage], fields: list[ColumnLineage]) -> None:
        rewrite = self.rewrite
        assert rewrite is not None
        key = self.analysis.key
        target = rewrite.target if rewrite.target and any(
            field.definition == rewrite.target[0] for field in fields
        ) else None
        old, new = map(key, target[1:]) if target else ('', '')
        aliases = {key(field.column.name) for field in fields
                   if field.definition is not None and field.definition.alias_at is None}
        changes = [{key(field.column.name): key(next_name) for field in relation.shape.fields
                    if (next_name := rewrite.new_name(field))} for relation in relations]
        changed = any(changes)
        if changed:
            names = {name for renamed in changes for pair in renamed.items() for name in pair}
            for join in joins:
                if join.method == 'NATURAL' or any(key(item.name) in names for item in join.args.get('using') or ()):
                    raise DbtError('A NATURAL/USING join could change its keys. Write an explicit ON condition first.')
            for projection in select.expressions:
                if isinstance(projection, (exp.Star, exp.Column)) and projection.is_star:
                    if not self.star(projection, relations, merged).complete:
                        raise DbtError('Cannot verify a star projection after renaming this column.')
            if any((isinstance(node, exp.Dot) and isinstance(node.expression, exp.Star)) or
                   (isinstance(node, exp.Column) and node.is_star and node.parent is not select)
                   for node in scope.walk()):
                raise DbtError('Cannot verify a whole-row reference after renaming this column.')
        after = []
        for relation, renamed in zip(relations, changes):
            if not renamed:
                after.append(relation)
                continue
            rewrite.check_output(relation.shape, self.analysis.dialect)
            bindings = {renamed.get(name, name): field for name, field in relation.bindings.items()}
            after.append(replace(relation, bindings=bindings))
        after = tuple(after)
        incoming_names = set(map(key, rewrite.names))
        for node in columns:
            name = key(node.name)
            qualifier = tuple(part.name for part in node.parts[:-1])
            relevant = [renamed for relation, renamed in zip(relations, changes)
                        if not qualifier or self.qualified((relation,), qualifier)]
            before = resolved.get(id(node))
            affected = any(name in names or name in names.values() for names in relevant)
            if before is None and name in incoming_names and (scope.can_be_correlated or
                    qualifier and not relevant and changed):
                raise DbtError(f'Cannot verify an outer or implicit table binding for {node.name!r}.')
            if before is None and not qualifier and any(
                name in relation.parts[-1:] and renamed for relation, renamed in zip(relations, changes)
            ):
                raise DbtError('Cannot verify a whole-row reference after renaming this column.')
            alias_sensitive = not qualifier and self.alias_sensitive(node, select)
            order = (isinstance(node.parent, exp.Ordered) and node.parent.this is node
                     and node.parent.parent is select.args.get('order'))
            if alias_sensitive and target and name in {old, new}:
                if name == old and order:
                    rewrite.edits.append((*self.authored_span(node.this), quote_identifier(target[2])))
                    continue
                if name != old or target[0].alias_at is None:
                    raise DbtError(f'Cannot verify the output alias binding of {node.name!r}.')
            if alias_sensitive and order and name in aliases and self.analysis.dialect == 'postgres':
                continue  # A bare ORDER BY name resolves to the explicit output alias.
            replacement = rewrite.new_name(before) if before is not None else ''
            if affected or replacement:
                if (alias_sensitive and (name in aliases or key(replacement) in aliases) or before is None or
                        self.lookup(node, after, merged, name=replacement or name) is not before):
                    raise DbtError(f'The binding of {node.name!r} could change. Qualify columns or use explicit aliases.')
                if replacement:
                    rewrite.edits.append((*self.authored_span(node.this), quote_identifier(replacement)))

    def star(self, projection: exp.Star | exp.Column, relations: tuple[_Relation, ...], merged: bool) -> _Shape:
        qualified = isinstance(projection, exp.Column)
        star = projection.this if qualified else projection
        if not isinstance(star, exp.Star) or any(star.args.values()):
            return _Shape()
        if qualified:
            relations = self.qualified(relations, tuple(p.name for p in projection.parts[:-1]))
            if len(relations) != 1:
                return _Shape()
        elif merged:
            return _Shape()
        if not relations or any(not relation.shape.fields for relation in relations):
            return _Shape()
        if sum(len(relation.shape.fields) for relation in relations) > _MAX_FIELDS:
            return _Shape()
        fields = tuple(replace(field, definition=None) for relation in relations for field in relation.shape.fields)
        if self.uses is not None and (span := _span(star)) is not None:
            for field in fields:
                self.use(span, field, implicit=True)
        complete = all(relation.shape.complete for relation in relations)
        if self.stars is not None and (span := _span(star)) is not None:
            first = _span(projection.parts[0]) if qualified else span
            if first is not None:
                start, end = self.shadow.span(first[0], span[1])
                replacement = self.star_replacement(relations, projection, start, end) if complete else ""
                self.stars.append(StarExpansion(start, end, tuple(_column_text(f.column) for f in fields), replacement))
                title = "Star projection" if complete else "Star projection — known columns only (partial schema)"
                self.hover((first[0], span[1]), _fields_text(title, tuple(field.column for field in fields)))
        return _Shape(fields, complete)

    def star_replacement(self, relations: tuple[_Relation, ...], projection: exp.Star | exp.Column,
                         start: int, end: int) -> str:
        qualifier = ""
        written = self.shadow.original[start:end]
        if not written.endswith("*") or re.search(r"--|/\*|\{[{%#]", written):
            return ""
        if isinstance(projection, exp.Column):
            # Do not delete comments or introduce synthetic names. Keep the
            # exact spelling of the qualifier the author already wrote.
            qualifier = written[:-1].rstrip()
        result = []
        for relation in relations:
            names = [f.column.name for f in relation.shape.fields]
            if len(relation.bindings) != len(names):
                return ""
            prefix = qualifier
            if len(relations) > 1:
                prefix = relation.qualifier + "." if relation.qualifier else ""
                if prefix and len(self.qualified(relations, relation.parts)) != 1:
                    return ""
                if not prefix and any(
                    self.analysis.key(name) in other.bindings
                    for name in names for other in relations if other is not relation
                ):
                    return ""
            result.extend(prefix + quote_identifier(name) for name in names)
        return ", ".join(result)

    def field(self, projection: exp.Expr, resolved: dict[int, ColumnLineage], fallback: str) -> ColumnLineage | None:
        value = projection.this if isinstance(projection, exp.Alias) else projection
        value = value.unnest() if isinstance(value, exp.Paren) else value
        if isinstance(value, (exp.Columns, exp.Unnest, exp.Explode, exp.Aliases)):
            return None
        if isinstance(projection, exp.Alias) and isinstance(projection.args.get("alias"), exp.Identifier):
            name = self.name(projection.args["alias"])
        elif isinstance(value, exp.Column) and isinstance(value.this, exp.Identifier):
            name = self.name(value.this)
        elif (self.analysis.dialect == "postgres" and isinstance(value, exp.Cast)
              and isinstance(value.this, exp.Column) and isinstance(value.this.this, exp.Identifier)):
            name = self.name(value.this.this)
        elif fallback:
            name = fallback
        else:
            # Keep only provable names; default expression labels vary by dialect.
            return None
        definition = None
        if isinstance(projection, exp.Alias) and (span := _span(projection.args['alias'])):
            definition = ColumnDefinition(*self.shadow.span(*span))
        elif projection is value and isinstance(value, exp.Column) and (span := _span(value.this)) is not None:
            start, end = self.shadow.span(*span)
            definition = ColumnDefinition(start, end, alias_at=end)
        direct = resolved.get(id(value))
        if direct is not None:
            return replace(direct, column=replace(direct.column, name=name), definition=definition,
                           name_from=None if isinstance(projection, exp.Alias) else direct.name_from)
        inputs = []
        for node in walk_in_scope(value):
            if isinstance(node, exp.Query):
                child = self.outputs.get(id(node.unnest()), _Shape()) if isinstance(node, exp.Subquery) else _Shape()
                if child.complete and len(child.fields) == 1:
                    inputs.extend(child.fields[0].origins)
            elif (field := resolved.get(id(node))) is not None:
                inputs.extend(field.origins)
            if len(inputs) > _MAX_TOKENS:
                inputs = []
                break
        implicit = resolved.get(id(value.this)) if isinstance(value, exp.Cast) and not isinstance(projection, exp.Alias) else None
        return ColumnLineage(ColumnDef(name, self.type_of(value)), _deduplicate(inputs),
                             implicit.name_from if implicit is not None else None, definition)

    def type_of(self, value: exp.Expr) -> str:
        if isinstance(value, exp.Cast) and isinstance(value.to, exp.DataType):
            return value.to.sql(dialect=self.analysis.dialect)
        if isinstance(value, exp.Count):
            return "integer" if self.analysis.dialect == "sqlite" else "bigint"
        if isinstance(value, exp.Boolean):
            return "integer" if self.analysis.dialect == "sqlite" else "boolean"
        if isinstance(value, exp.CurrentDate):
            return "text" if self.analysis.dialect == "sqlite" else "date"
        if isinstance(value, exp.Literal):
            if value.is_string:
                return "varchar" if self.analysis.dialect == "duckdb" else "text"
            if value.is_int and len(value.this) <= 19:
                number = int(value.this)
                if self.analysis.dialect == "sqlite":
                    return "integer" if number < 2**63 else "real"
                if number < 2**31:
                    return "integer"
                if number < 2**63:
                    return "bigint"
            if self.analysis.dialect == "postgres":
                return "numeric"
            if self.analysis.dialect == "sqlite":
                return "real"
        return ""

    def union(self, scope: Scope) -> _Shape:
        branches = [self.outputs.get(id(branch.expression), _Shape()) for branch in scope.set_operation_scopes]
        if len(branches) != 2:
            return self.unsupported()
        left, right = branches
        if not (left.complete and right.complete and len(left.fields) == len(right.fields)):
            return self.unsupported()
        fields = tuple(replace(a, column=replace(a.column, type=a.column.type if a.column.type == b.column.type else ''),
                               origins=_deduplicate((*a.origins, *b.origins))) for a, b in zip(left.fields, right.fields))
        return _Shape(fields, True)


def analyze_file(index: DbtIndex | None, path: str, text: str, dialect: str, schema: SchemaContext) -> DbtAnalysis:
    """Analyze this exact draft, with offline graph and catalog metadata only."""
    suffix = PurePosixPath(path).suffix.lower()
    if len(text) > _MAX_CHARS:
        return DbtAnalysis()
    if suffix in {".yaml", ".yml"}:
        return _yaml_analysis(text)
    if suffix != ".sql":
        return DbtAnalysis()
    analysis = _Analysis(index, dialect, schema)
    own = index.for_path(path) if index else ()
    analysis.active.update(resource.key for resource in own)
    packages = {resource.package for resource in own}
    warnings = _Warnings(text)
    shape, result = analysis.sql(text, package=next(iter(packages)) if len(packages) == 1 else None,
                                 warnings=warnings)
    if len(own) == 1:
        expected = {analysis.key(column.name): column.type for column in own[0].columns if column.type}
        for field in shape.fields:
            site, column = field.definition, field.column
            if site is not None and (declared := expected.get(analysis.key(column.name))):
                if types_incompatible(column.type, declared, dialect=analysis.dialect):
                    warnings.add(site.start, site.end,
                                 f'Output {column.name!r} has known type {column.type}; properties declare {declared}.',
                                 'dbt')
        shape = analysis.enrich(shape, own[0].columns)
    if not result.diagnostics:
        result = replace(result, diagnostics=tuple(sorted(warnings.items, key=lambda d: (d.start, d.end, d.message))))
    return replace(result, columns=shape.fields, complete=shape.complete)


def column_rename_sql(index: DbtIndex, path: str, text: str, changes: Mapping[tuple[str, str], str], *,
                      target: tuple[ColumnDefinition, str, str] | None = None
                      ) -> tuple[tuple[ColumnLineage, ...], tuple[tuple[int, int, str], ...]]:
    if len(text) > _MAX_CHARS:
        raise DbtError('SQL exceeds the column analysis limit.')
    analysis = _Analysis(index, index.dialect, SchemaContext())
    analysis.active.update(resource.key for resource in index.for_path(path) if resource.path == path)
    rewrite = _ColumnRewrite(changes, target)
    shape, _ = analysis.sql(text, package=index.package, rewrite=rewrite)
    if target is not None:
        site, old, new = target
        names = [analysis.key(field.column.name) for field in shape.fields]
        if not shape.complete or names.count(analysis.key(old)) != 1 or not any(
            field.definition == site and field.column.name == old for field in shape.fields
        ):
            raise DbtError('Select a uniquely named output column in a fully known projection.')
        if analysis.key(new) in names:
            raise DbtError(f'The model already outputs a column named {new!r}.')
        identifier = quote_identifier(new)
        if site.alias_at is None:
            rewrite.edits.append((site.start, site.end, identifier))
        else:
            rewrite.edits.append((site.alias_at, site.alias_at, ' AS ' + identifier))
    return shape.fields, tuple(rewrite.edits)
