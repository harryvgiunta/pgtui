"""Caret-based references against one immutable dbt index snapshot."""
from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass

from d8r.dbt import DbtError
from d8r.dbt_analysis import DbtAnalysis, analyze_file
from d8r.dbt_files import file_revision
from d8r.dbt_index import DbtIndex, DbtResource, template_calls
from d8r.sql import SchemaContext

_LIMIT = 300
_COLUMN_FILES = 64
_NEWLINE = re.compile(r'\r\n?|\n')


@dataclass(frozen=True)
class DbtReference:
    path: str
    row: int
    column: int
    revision: str
    detail: str = ''


@dataclass(frozen=True)
class DbtReferences:
    label: str
    locations: tuple[DbtReference, ...]
    limited: bool


def _at(span: tuple[int, int], offset: int) -> bool:
    return span[0] <= offset < span[1] or offset == span[1] and span[1] > span[0]


def _symbol(index: DbtIndex, path: str, text: str, offset: int,
            analysis: DbtAnalysis) -> tuple[DbtResource, str | None]:
    owners = index.for_path(path)
    authored = tuple(resource for resource in owners if resource.path == path)
    if path.lower().endswith('.sql'):
        calls = [call for call in template_calls(text) if _at((call.tag_start, call.tag_end), offset)]
        exact = [call for call in calls if _at((call.start, call.end), offset)]
        if len(exact or calls) == 1:
            call = (exact or calls)[0]
            target = None if call.dynamic else index.resolve(call.kind, call.args, call.kwargs)
            if target is not None:
                return target, None
            raise DbtError(f'This {call.kind}() is not statically resolved.')
        if calls:
            raise DbtError('Put the caret inside one ref() or source() call.')
        output = analysis.output_at(offset)
        if output is not None and len(authored) == 1 and authored[0].kind in {'model', 'snapshot'}:
            return authored[0], output.column.name
        uses = {(use.source.key, use.source.column) for use in analysis.uses
                if not use.implicit and _at((use.start, use.end), offset)}
        if len(uses) == 1:
            key, column = next(iter(uses))
            return index.resources[key], column
    if len(authored or owners) == 1:
        return (authored or owners)[0], None
    raise DbtError('Put the caret on a resolved ref(), source(), or column, or open a single dbt resource.')


def find_references(index: DbtIndex, path: str, text: str, offset: int,
                    analysis: DbtAnalysis, schema: SchemaContext) -> DbtReferences:
    resource, column = _symbol(index, path, text, offset, analysis)
    found: dict[tuple[str, int, int], DbtReference] = {}
    revisions = {}
    limited = bool(index.warnings)

    def add(file, authored, start, detail=''):
        nonlocal limited
        if len(found) >= _LIMIT:
            limited = True
            return
        row = base = 0
        for match in _NEWLINE.finditer(authored, 0, start):
            row += 1
            base = match.end()
        where = (file, row, start - base)
        if where not in found:
            if file not in revisions:
                revisions[file] = file_revision(authored)
            found[where] = DbtReference(*where, revisions[file], detail)

    if column is None:
        matched = set()
        for file, authored in index.sql_files():
            for call in template_calls(authored):
                if call.dynamic:
                    limited = True
                elif index.resolve(call.kind, call.args, call.kwargs) == resource:
                    add(file, authored, call.name_span[0] if call.name_span else call.start)
                    matched.add(file)
            if len(found) >= _LIMIT:
                limited = True
                break
        for consumer in index.dependencies(resource.key, direction='downstream'):
            if not consumer.path:
                limited = True
            elif consumer.path not in matched:
                try:
                    authored = index.read(consumer.path)
                except DbtError:
                    limited = True
                else:
                    start = next((match.end() for row, match in enumerate(_NEWLINE.finditer(authored), 1)
                                  if row == consumer.row), 0)
                    add(consumer.path, authored, start, 'Resource dependency')
    else:
        if index.unresolved:
            limited = True
        pending = deque([(resource.key, column)])
        visited = set(pending)
        parsed = {}
        while pending and len(found) < _LIMIT:
            key, name = pending.popleft()
            for consumer in index.dependencies(key, direction='downstream'):
                file = consumer.path
                if file not in parsed:
                    if not file.lower().endswith('.sql') or len(parsed) >= _COLUMN_FILES:
                        limited = True
                        continue
                    try:
                        authored = index.read(file)
                    except DbtError:
                        limited = True
                        continue
                    result = analyze_file(index, file, authored, index.dialect, schema)
                    parsed[file] = authored, result
                    limited |= not result.complete
                authored, result = parsed[file]
                for use in result.uses:
                    if (use.source.key, use.source.column) == (key, name):
                        add(file, authored, use.start, 'Implicit SELECT * use' if use.implicit else '')
                if consumer.kind in {'model', 'snapshot'}:
                    for field in result.columns:
                        source = field.name_from
                        pair = consumer.key, field.column.name
                        if source is not None and (source.key, source.column) == (key, name) and pair not in visited:
                            visited.add(pair)
                            pending.append(pair)
        limited |= bool(pending)
    label = resource.label if column is None else f'{resource.label} · {column}'
    return DbtReferences(label, tuple(found[key] for key in sorted(found)), limited)
