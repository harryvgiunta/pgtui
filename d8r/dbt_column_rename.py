"""Lineage-driven output-column rename, published by the model rename transaction."""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Collection
from graphlib import CycleError, TopologicalSorter
from pathlib import Path

from yaml.nodes import MappingNode, ScalarNode, SequenceNode

from d8r.dbt import DbtError, _yaml
from d8r.dbt_analysis import analyze_file, column_rename_sql
from d8r.dbt_files import DbtFile, DbtFiles, _FILE_LIMIT
from d8r.dbt_index import DbtIndex, DbtResource, template_calls
from d8r.dbt_rename import (
    DbtRenameEdit, DbtRenamePlan, _EDIT_LIMIT, _NAME, _SCAN_LIMIT, _YAML, _check_dirty,
    _check_inventory, _compose, _file_kind, _ignore, _inventory, _mapping, _nodes, _patch,
    _python, _read, _roots, _string, _yaml_entries, _yaml_replace,
)
from d8r.sql import SchemaContext

_MAX_CONSUMERS = 128
_COLUMN_CONFIG = {'unique_key', 'cluster_by', 'partition_by', 'sort', 'dist', 'check_cols', 'merge_update_columns',
                  'merge_exclude_columns', 'incremental_predicates', 'updated_at', 'persist_docs'}


def _reference_targets(index: DbtIndex, text: str) -> set[DbtResource]:
    targets = set()
    for call in template_calls(text, strict=True):
        if call.kind != 'ref':
            continue
        if call.dynamic:
            raise DbtError('A dynamic ref() may hide a consumer of the renamed column.')
        target = index.resolve(call.kind, call.args, call.kwargs)
        if target is None:
            if len(call.args) == 1 or call.args[0] == index.package:
                raise DbtError(f'Cannot safely resolve ref() to {call.args[-1]!r}.')
        else:
            targets.add(target)
    return targets


def _properties(index: DbtIndex, path: str, text: str, columns: dict[str, dict[str, str]]) -> str:
    root = _compose(text)
    nodes, shared = _nodes(root)
    edits = {}
    safe_refs = set()
    by_kind = {(resource.kind, resource.name): columns[key]
               for key in columns if (resource := index.resources[key]).package == index.package}

    def rename(node, names):
        value = _string(node)
        if value in names:
            edits[id(node)] = [(0, len(value), names[value])]

    def changes_for(node):
        value = _string(node)
        if not value or not (value.lstrip().startswith(('ref', 'builtins.ref', '{{'))):
            return {}
        expression = value if '{{' in value else '{{ ' + value + ' }}'
        return {old: new for target in _reference_targets(index, expression)
                for old, new in columns.get(target.key, {}).items()}

    def check_config(fields, names):
        for key, value in fields.items():
            if isinstance(value, MappingNode):
                check_config(_mapping(value), names)
            if key.lstrip('+') in _COLUMN_CONFIG:
                snippet = text[value.start_mark.index:value.end_mark.index]
                if any(name.casefold() in re.findall(r'\w+', snippet.casefold()) for name in names):
                    raise DbtError(f'Column-bearing {key!r} configuration must be updated manually.')

    if path == 'dbt_project.yml':
        check_config(_mapping(root), {old for names in columns.values() for old in names})
    elif path != 'selectors.yml':
        for group, kind in (('models', 'model'), ('snapshots', 'snapshot')):
            for entry in _yaml_entries(root, group):
                names = by_kind.get((kind, entry['name'].value))
                if not names:
                    continue
                if entry.get('versions') is not None:
                    raise DbtError('Column properties on versioned models cannot be changed safely.')
                check_config(_mapping(entry.get('config')), names)
                sequence = entry.get('columns')
                if sequence is None:
                    continue
                if not isinstance(sequence, SequenceNode):
                    raise DbtError('Expected literal column properties.')
                fields = [_mapping(item).get('name') for item in sequence.value]
                names_after = []
                for field in fields:
                    value = _string(field)
                    if not value or '{{' in value:
                        raise DbtError('Expected a literal column name in properties.')
                    names_after.append(names.get(value, value).casefold())
                    rename(field, names)
                if len(set(names_after)) != len(names_after):
                    raise DbtError('The new name would collide with a column in properties.')
        for entry in _yaml_entries(root, 'unit_tests'):
            if ('model', _string(entry.get('model'))) in by_kind:
                raise DbtError('A unit test fixture may name this column; update its fixtures manually.')
        for entry in _yaml_entries(root, 'exposures'):
            depends = entry.get('depends_on')
            if isinstance(depends, SequenceNode):
                safe_refs.update(map(id, depends.value))

    keys = {id(key) for node in nodes if isinstance(node, MappingNode) for key, _ in node.value}
    for node in nodes:
        fields = _mapping(node)
        relationship = fields.get('relationships')
        if isinstance(relationship, MappingNode):
            arguments = _mapping(relationship)
            arguments = _mapping(arguments['arguments']) if 'arguments' in arguments else arguments
            target = arguments.get('to')
            if (names := changes_for(target)):
                field = arguments.get('field')
                if _string(field) is None or not re.fullmatch(r'[A-Za-z_][A-Za-z_0-9]*', field.value):
                    raise DbtError('Cannot verify a relationships test with a nonliteral field.')
                rename(field, names)
                safe_refs.add(id(target))
        for key in ('model', 'relation', 'input', 'to'):
            value = fields.get(key)
            if value is not None and id(value) not in safe_refs and key != 'to' and changes_for(value):
                raise DbtError(f'Cannot verify column bindings in YAML {key!r} metadata.')
    for node in nodes:
        if id(node) in keys or id(node) in safe_refs:
            continue
        if changes_for(node):
            raise DbtError('A YAML reference may hide a column use in a test, snapshot, or semantic model.')
    return _yaml_replace(text, nodes, shared, edits)


def plan_column_rename(project: DbtFiles, path: str, offset: int, name: str, *, dialect: str = 'postgres',
                       dirty: Collection[str] = (), revision: str | None = None) -> DbtRenamePlan:
    """Rename an authored output at the caret, never a same-spelled input or alias."""
    if not isinstance(name, str) or not _NAME.fullmatch(name) or len(name) > 63:
        raise DbtError('Use 1–63 ASCII letters, digits or underscores, starting with a letter or underscore.')
    project = DbtFiles(project.root)
    if project.config_error:
        raise DbtError(project.config_error)
    path = project.editable_path(path).relative_to(project.root).as_posix()
    roots, ignored = _roots(project), _ignore(project)
    if Path(path).suffix.lower() != '.sql' or _file_kind(project, path, roots) != ('models',) or ignored.match_file(path):
        raise DbtError('Select an output column in a local, non-ignored SQL model.')
    _check_dirty(project, roots, ignored, dirty)
    files, sources = [], {}
    size = 0
    for candidate in _inventory(project, roots, ignored):
        before, stamp = _read(project, candidate)
        files.append((candidate, stamp))
        if before is not None:
            size += stamp.identity[-1]
            if size > _SCAN_LIMIT:
                raise DbtError('Too much source text to preview this column rename safely.')
            sources[candidate] = before
    if path not in sources:
        raise DbtError('The model is no longer in the rename inventory.')
    if revision is not None and revision != sources[path].revision:
        raise DbtError('The model changed on disk; reload it before renaming a column.')
    if _yaml(sources['dbt_project.yml'].text, Path('dbt_project.yml')) != project.config:
        raise DbtError('Project configuration changed while planning the rename.')
    index = DbtIndex(project, {file: before.text for file, before in sources.items()}, dialect=dialect,
                     include_artifact=False)
    if index.warnings:
        raise DbtError('Cannot verify project bindings: ' + index.warnings[0])
    if any(item.name == 'ref' for item in index.macros):
        raise DbtError('The project overrides ref(); bindings cannot be verified offline.')

    def binding(file):
        entries = tuple(item for item in index.for_path(file) if item.path == file)
        if len(entries) != 1 or entries[0].version or entries[0].kind not in {'model', 'snapshot', 'analysis', 'test'}:
            raise DbtError(f'Cannot verify a unique, unversioned SQL resource for {file}.')
        item = entries[0]
        if item.kind in {'model', 'snapshot'} and index.resolve('ref', (item.name,), {}) != item:
            raise DbtError(f'The dbt name for {file} is ambiguous.')
        return item

    origin = binding(path)
    authored = sources[path].text
    selected = analyze_file(index, path, authored, dialect, SchemaContext()).output_at(offset)
    if selected is None or selected.definition is None:
        raise DbtError('Put the caret on an authored output column name, not a SELECT * or an input-only column.')
    old = selected.column.name
    replacements = {path: _patch(authored, column_rename_sql(
        index, path, authored, {}, target=(selected.definition, old, name))[1])}
    changes = {(origin.key, old): name}
    columns = {origin.key: {old: name}}

    # Use the authored SQL inventory as a backstop for macro/disabled files
    # that have no resource vertex in the normal editor graph.
    inputs, downstream = {}, defaultdict(set)
    for file, before in sources.items():
        if Path(file).suffix.lower() != '.sql':
            continue
        try:
            inputs[file] = {item.path for item in _reference_targets(index, before.text)}
        except DbtError as exc:
            raise DbtError(f'{file}: {exc}') from exc
        for dependency in inputs[file]:
            downstream[dependency].add(file)
    visited, pending = {path}, [path]
    while pending:
        for file in sorted(downstream[pending.pop()] - visited):
            visited.add(file)
            pending.append(file)
            if len(visited) > _MAX_CONSUMERS:
                raise DbtError('Too many downstream resources to verify a column rename safely.')
    try:
        order = tuple(TopologicalSorter({file: sorted(inputs[file] & visited) for file in sorted(visited)}).static_order())
    except CycleError as exc:
        raise DbtError('The downstream dbt bindings contain a cycle.') from exc
    propagating = {path}
    for file in order:
        if file == path or not inputs[file] & propagating:
            continue
        try:
            item = binding(file)
            before = sources[file].text
            fields, edits = column_rename_sql(index, file, before, changes)
            if edits:
                replacements[file] = _patch(before, edits)
            inherited = {field.column.name: new for field in fields
                         if (source := field.name_from) is not None and
                         (new := changes.get((source.key, source.column))) is not None}
            if inherited and item.kind in {'model', 'snapshot'}:
                changes.update({(item.key, old_name): new_name for old_name, new_name in inherited.items()})
                columns[item.key] = inherited
                propagating.add(file)
        except DbtError as exc:
            raise DbtError(f'{file}: {exc}') from exc

    for file, before in sources.items():
        suffix = Path(file).suffix.lower()
        try:
            if suffix in _YAML:
                updated = _properties(index, file, before.text, columns)
                if updated != before.text:
                    replacements[file] = updated
            elif suffix == '.py' and any(_python(before.text, index.package, index.resources[key].name, '')
                                         for key in columns):
                raise DbtError('A Python consumer may name this output column; review its DataFrame operations.')
        except DbtError as exc:
            raise DbtError(f'{file}: {exc}') from exc

    edits = []
    edit_size = 0
    for file, text in sorted(replacements.items()):
        before: DbtFile = sources[file]
        size = len(text.encode('utf-8')) + (3 if before.bom else 0)
        edit_size += size + len(before.text.encode('utf-8'))
        if size > _FILE_LIMIT or edit_size > _EDIT_LIMIT:
            raise DbtError('The column rename preview would be too large.')
        edits.append(DbtRenameEdit(file, file, before, text))
    plan = DbtRenamePlan(old, name, path, path, tuple(edits), project.root, tuple(files), offset, dialect)
    _check_inventory(project, plan)
    return plan
