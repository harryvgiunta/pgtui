"""Live, offline dbt editor feedback. One bounded worker owns project indexing."""
from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import OptionList, Static, TabbedContent, Tree
from textual.widgets.option_list import Option
from textual.widgets.tree import TreeNode

from d8r.dbt import DbtError
from d8r.dbt_analysis import DbtAnalysis, Diagnostic, analyze_file
from d8r.dbt_files import DbtFiles, file_revision
from d8r.dbt_index import DbtIndex
from d8r.dbt_profiles import DbtTargetSettings
from d8r.dbt_references import DbtReference, find_references
from .dbt_model_completion import complete_dbt

if TYPE_CHECKING:
    from d8r.sql import SchemaContext
    from .app import D8RApp


_DELAY = 0.32
_TRACE_FILES = 16
_TREE_ROWS = 300
_TREE_DEPTH = 14
_DIAGNOSTICS_PAGE = 100


@dataclass(frozen=True)
class _Site:
    workspace: str
    page: str
    home: str
    target: DbtTargetSettings
    path: str
    text: str
    dialect: str
    schema_id: int
    linked: bool
    trace: bool
    epoch: int


@dataclass
class _Feedback:
    analysis: DbtAnalysis
    related: dict[str, DbtAnalysis]
    trace_limited: bool = False
    index_notes: tuple[str, ...] = ()


def _analyze(index: DbtIndex | None, site: _Site, schema: SchemaContext) -> _Feedback:
    analysis = (analyze_file(index, site.path, site.text, site.dialect, schema)
                if site.path else DbtAnalysis())
    related = {}
    limited = False
    if site.trace and site.linked and index is not None:
        roots = index.for_path(site.path)
        try:
            authored = index.read(site.path)
        except (DbtError, OSError):
            authored = site.text
        source_analysis = (analysis if authored == site.text else
                           analyze_file(index, site.path, authored, site.dialect, schema))
        related = {resource.key: source_analysis for resource in roots if resource.path == site.path}
        pending = deque((resource.key, edges) for resource in roots
                        for edges in (index.upstream, index.downstream))
        visited: set[tuple[str, bool]] = set()
        parsed: dict[str, DbtAnalysis] = {site.path: source_analysis}
        while pending:
            key, edges = pending.popleft()
            direction = edges is index.upstream
            if (key, direction) in visited:
                continue
            visited.add((key, direction))
            node = index.resources[key]
            path = node.path
            if path and Path(path).suffix.lower() == ".sql" and key not in related:
                if path not in parsed:
                    if len(parsed) >= _TRACE_FILES:
                        limited = True
                        continue
                    try:
                        text = index.read(path)
                    except (DbtError, OSError):
                        continue
                    parsed[path] = analyze_file(index, path, text, site.dialect, schema)
                related[key] = parsed[path]
            pending.extend((child, edges) for child in edges.get(key, ()))
    return _Feedback(analysis, related, limited, index.warnings if index is not None else ())


class DbtAuthoring:
    """Debounce edits and publish only to the captured workspace, page and text."""

    def __init__(self, ide: D8RApp) -> None:
        self.ide = ide
        self._epoch = 0
        self._index: DbtIndex | None = None
        self._index_key: tuple | None = None
        self._requested: _Site | None = None
        self._published: _Site | None = None
        self._feedback = _Feedback(DbtAnalysis(), {})
        self._timer = None
        self._running = False

    def _site(self) -> _Site | None:
        ide = self.ide
        if not ide._workspace_ready or ide._closing or not ide.session.dbt_enabled:
            return None
        editor = ide.editor
        link = ide._dbt_file if not ide.function_mode else None
        path = (link['path'] if link else 'query.sql'
                if not ide.function_mode and editor.dbt_template else '')
        trace = (ide.pane_visible('schema')
                 and ide.query_one('#explorer-tabs', TabbedContent).active == 'tab-dbt'
                 and ide.query_one('#dbt-panels', TabbedContent).active == 'dbt-panel-lineage')
        home = ide._dbt_home
        return _Site(ide._workspace_key, ide._document_identity, home, ide.session.dbt_target_settings(home), path,
                     editor.text if path else '', editor.dialect, id(ide.session.schema),
                     bool(link), trace, self._epoch)

    def invalidate(self) -> None:
        self._epoch += 1
        self.request()

    def opened(self, home: str, path: str, text: str) -> None:
        """A newly read disk baseline can supersede a cached graph input."""
        if self._index is None or self._index_key is None or home != self._index_key[1]:
            return
        try:
            previous = self._index.read(path)
        except DbtError:
            return  # Not an indexed source file.
        if previous != text:
            self.invalidate()

    def request(self) -> None:
        site = self._site()
        previous = self._requested
        if site == previous:
            return
        self._requested = site
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if site is None or previous is None or (site.workspace, site.home, site.target, site.dialect) != (
                previous.workspace, previous.home, previous.target, previous.dialect):
            self._index = self._index_key = None
        self.ide.editor.set_dbt_feedback()
        problems = self.ide.query_one(DbtProblems)
        problems.clear("Checking this file…" if site and site.path else "Open a dbt file to check it.")
        if site is None or previous is None or (site.workspace, site.home, site.path, site.page) != (
                previous.workspace, previous.home, previous.path, previous.page):
            self.ide.query_one(DbtLineage).clear("Indexing lineage…" if site and site.path else "Open a dbt resource.")
        if site is None:
            self._published = None
            self._feedback = _Feedback(DbtAnalysis(), {})
            return
        self._timer = self.ide.set_timer(_DELAY, self._launch)

    def _launch(self) -> None:
        self._timer = None
        if self._running or self._requested is None or self.ide._closing:
            return
        site = self._requested
        drafts = self.ide._dbt_project_drafts(site.home) if site.home else {}
        key = (site.workspace, site.home, site.dialect, site.target, site.epoch, drafts)
        self._running = True
        self.ide.run_worker(self._work(site, key, drafts, self.ide.session.schema),
                            group="dbt-authoring", exit_on_error=False)

    async def _work(self, site: _Site, key: tuple, drafts: dict[str, str], schema: SchemaContext) -> None:
        def build() -> tuple[DbtIndex | None, _Feedback]:
            index = cached
            index_error = ""
            if site.home and index is None:
                try:
                    index = DbtIndex(DbtFiles(site.home), drafts, dialect=site.dialect, target_settings=site.target)
                except (DbtError, OSError, RecursionError) as exc:
                    index_error = str(exc)
            feedback = _analyze(index, site, schema)
            if index_error:
                feedback.index_notes = (index_error,)
            return index, feedback

        cached = self._index if key == self._index_key else None
        try:
            index, feedback = await asyncio.to_thread(build)
            current = self._site()
            if current is not None and (current.workspace, current.home, current.dialect, current.target) == (
                    site.workspace, site.home, site.dialect, site.target) and current.epoch == site.epoch:
                self._index_key, self._index = key, index
            if site != current or site != self._requested:
                return
            self._feedback = feedback
            self._published = site
            self.ide.editor.set_dbt_feedback(feedback.analysis.diagnostics, feedback.analysis.hovers)
            self.ide.query_one(DbtProblems).show(site, feedback)
            if site.trace:
                self.ide.query_one(DbtLineage).show(index, site.path if site.linked else '', feedback)
            if self.ide.palette.is_open or self.ide.focused is self.ide.editor and self.ide.editor.dbt_template:
                self.ide.palette.queue_sync()
        finally:
            self._running = False
            if (self._requested is not None and self._requested != site and self._timer is None
                    and not self.ide._closing):
                self._launch()

    @property
    def current(self) -> bool:
        return self._published is not None and self._published == self._site()

    def jump_to_diagnostic(self, problem: Diagnostic) -> None:
        if self.current:
            editor = self.ide.editor
            editor.move_cursor(editor.dbt_diagnostic_location(problem.start))
            editor.focus()

    def complete(self, document: str, offset: int, manual: bool):
        if not self.ide.session.dbt_enabled or self.ide.function_mode:
            return None
        if "{{" not in document and "{%" not in document and "{#" not in document:
            return None
        return complete_dbt(self._index, document, offset, manual)

    def references(self) -> None:
        ide, site, index = self.ide, self._published, self._index
        if not self.current or site is None or index is None or not site.path:
            ide._set_status('Open a dbt project and wait for analysis before finding references')
            return
        views = ide._active_page().get('dbt_views', {})
        if site.linked and site.text == views.get('sql') and views.get('sql') != views.get('jinja'):
            ide._set_status('Switch back to the dbt draft before finding references')
            return
        editor = ide.editor
        offset = editor.document.get_index_from_location(editor.cursor_location)
        schema, analysis = ide.session.schema, self._feedback.analysis

        def current() -> bool:
            return (not ide._modal_open() and self._site() == site and index is self._index
                    and editor.document.get_index_from_location(editor.cursor_location) == offset)

        def jump(location: DbtReference | None) -> None:
            if location is None or ide._workspace_key != site.workspace or ide._dbt_home != site.home:
                return
            ide.open_dbt_file(site.home, location.path)
            if not ide._dbt_file or ide._dbt_file['path'] != location.path:
                return
            if file_revision(editor.text) != location.revision:
                ide._set_status('This file differs from the indexed draft. Refresh and find references again.')
                return
            editor.move_cursor((location.row, location.column))

        async def search() -> None:
            from .dbt import DbtReferencesScreen
            try:
                results = await asyncio.to_thread(find_references, index, site.path, site.text, offset, analysis, schema)
            except (DbtError, RecursionError) as exc:
                if current():
                    ide._set_status(str(exc) or 'Reference search exceeded the analysis limit')
            else:
                if current():
                    ide.push_screen(DbtReferencesScreen(results), jump)

        ide.run_worker(search(), group='dbt-references', exclusive=True, exit_on_error=False)

    def inspect(self) -> None:
        if not self.ide.session.dbt_enabled or self.ide.function_mode or self.ide._modal_open():
            return
        editor = self.ide.editor
        editor.focus()
        if not self.current:
            self.ide._set_status("Wait for dbt analysis to finish")
            return
        offset = editor.document.get_index_from_location(editor.cursor_location)
        detail = editor.dbt_detail(offset, at_caret=True)
        if detail:
            self.ide.notify(detail, title="dbt schema & diagnostics", timeout=12)
        else:
            self.ide._set_status("No schema information at the caret")

    def expand_star(self) -> None:
        if (not self.ide.session.dbt_enabled or self.ide.function_mode or self.ide._modal_open()
                or not self.current):
            return
        editor = self.ide.editor
        offset = editor.document.get_index_from_location(editor.cursor_location)
        stars = self._feedback.analysis.stars
        star = next((span for span in stars if span.start <= offset <= span.end), None)
        if star is None or not star.replacement:
            self.ide._set_status("No fully known SELECT * expansion at the caret")
            return
        views = self.ide._active_page().get('dbt_views', {})
        if self.ide._dbt_file and editor.text == views.get('sql') and views.get('sql') != views.get('jinja'):
            self.ide._set_status("Switch back to dbt before expanding the projection")
            return
        editor.history.checkpoint()
        editor.replace(star.replacement, editor.document.get_location_from_index(star.start),
                       editor.document.get_location_from_index(star.end),
                       maintain_selection_offset=False)
        editor.history.checkpoint()
        editor.focus()


def _diagnostic_label(ide: D8RApp, problem: Diagnostic) -> Text:
    row, col = ide.editor.dbt_diagnostic_location(problem.start)
    prefix = f"{row + 1}:{col + 1} {problem.source} {problem.severity}"
    return Text.assemble((prefix, 'bold'), f"  {problem.message}")


class DbtProblems(Vertical):
    def __init__(self, ide: D8RApp) -> None:
        super().__init__()
        self.ide = ide
        self.diagnostics: tuple[Diagnostic, ...] = ()
        self._notes: tuple[str, ...] = ()
        self._first = 0
        self._has_file = False

    def compose(self) -> ComposeResult:
        yield Static("Open a dbt file to check it.", classes="dbt-panel-status", markup=False)
        yield OptionList(id="dbt-problems-list", markup=False)

    def clear(self, message: str) -> None:
        self.diagnostics = ()
        self._notes = ()
        self.query_one(OptionList).clear_options()
        self.query_one(Static).update(message)
        self.ide.query_one('#dbt-panels', TabbedContent).get_tab('dbt-panel-problems').label = 'Problems'

    def show(self, site: _Site, feedback: _Feedback) -> None:
        self.diagnostics = feedback.analysis.diagnostics
        self._notes = feedback.index_notes
        self._has_file = bool(site.path)
        total = len(self.diagnostics) + len(self._notes)
        tab = self.ide.query_one('#dbt-panels', TabbedContent).get_tab('dbt-panel-problems')
        tab.label = f'Problems ({total})' if total else 'Problems'
        self._show_page(0)

    def _show_page(self, first: int) -> None:
        self._first = first
        total = len(self.diagnostics) + len(self._notes)
        stop = min(first + _DIAGNOSTICS_PAGE, total)
        options = [Option('‹ Previous page', id='previous')] if first else []
        for index in range(first, stop):
            if index < len(self.diagnostics):
                options.append(Option(_diagnostic_label(self.ide, self.diagnostics[index]), id=str(index)))
            else:
                options.append(Option(f'dbt index  {self._notes[index - len(self.diagnostics)]}'))
        if stop < total:
            options.append(Option('Next page ›', id='next'))
        listing = self.query_one(OptionList)
        listing.set_options(options)
        if options:
            listing.highlighted = int(first > 0)
        if total > _DIAGNOSTICS_PAGE:
            status = f'{first + 1}–{stop} of {total} · Enter opens a diagnostic or page'
        elif self.diagnostics:
            status = 'Enter jumps to the problem'
        else:
            status = 'No file diagnostics' if self._has_file else 'Open a dbt file to check it.'
        self.query_one(Static).update(status)

    @on(OptionList.OptionSelected)
    def _jump(self, event: OptionList.OptionSelected) -> None:
        if not self.ide.dbt_authoring.current:
            return
        if event.option_id == 'previous':
            self._show_page(self._first - _DIAGNOSTICS_PAGE)
        elif event.option_id == 'next':
            self._show_page(self._first + _DIAGNOSTICS_PAGE)
        elif event.option_id is not None:
            self.ide.dbt_authoring.jump_to_diagnostic(self.diagnostics[int(event.option_id)])


class DbtLineage(Vertical):
    """Resource edges and traced output-column edges, bounded for terminal display."""

    def __init__(self, ide: D8RApp) -> None:
        super().__init__()
        self.ide = ide
        self._diagnostics: tuple[Diagnostic, ...] = ()

    def compose(self) -> ComposeResult:
        yield Tree("Lineage", id="dbt-lineage-tree")
        yield Static("Open a dbt resource.", classes="dbt-panel-status", markup=False)

    def clear(self, label: str) -> None:
        self._diagnostics = ()
        tree = self.query_one(Tree)
        tree.clear()
        tree.root.set_label(label)
        self.query_one(Static).update('')

    def show(self, index: DbtIndex | None, path: str, feedback: _Feedback) -> None:
        self.clear("Lineage")
        tree = self.query_one(Tree)
        roots = index.for_path(path) if index is not None and path else ()
        tree.root.expand()
        self._diagnostics = diagnostics = feedback.analysis.diagnostics
        if diagnostics:
            group = tree.root.add(f'File diagnostics ({len(diagnostics)})')
            if len(diagnostics) <= _DIAGNOSTICS_PAGE:
                self._add_diagnostics(group, range(len(diagnostics)))
            else:
                for first in range(0, len(diagnostics), _DIAGNOSTICS_PAGE):
                    page = range(first, min(first + _DIAGNOSTICS_PAGE, len(diagnostics)))
                    group.add(f'{page.start + 1}–{page.stop}', data=page)
        if feedback.index_notes:
            group = tree.root.add(f'Project index notes ({len(feedback.index_notes)})')
            for note in feedback.index_notes:
                group.add_leaf(Text(note))
        if not roots:
            tree.root.set_label("No lineage for the open file" if path else "Open a dbt resource")
            return
        tree.root.set_label(Text(path))
        left = _TREE_ROWS
        clipped = False
        status = []

        def add(parent, label, data=None, *, branch=False):
            nonlocal left, clipped
            if left <= 0:
                clipped = True
                return None
            left -= 1
            return parent.add(Text(label), data=data, expand=False) if branch else parent.add_leaf(Text(label), data=data)

        def target(key):
            resource = index.resources.get(key)
            return (resource.path, resource.row) if resource and resource.path else None

        def walk(parent, key, edges, chain, seen):
            nonlocal clipped
            for child in edges.get(key, ()):
                resource = index.resources[child]
                repeated = child in seen or child in chain
                note = ' · cycle' if child in chain else ' · shown above' if repeated else ''
                node = add(parent, resource.label + note, target(child), branch=not repeated and bool(edges.get(child)))
                if node is None:
                    return
                if not repeated:
                    seen.add(child)
                    if len(chain) < _TREE_DEPTH:
                        walk(node, child, edges, (*chain, child), seen)
                    elif edges.get(child):
                        clipped = True

        column_outputs = {}
        column_consumers = defaultdict(list)
        for key, result in feedback.related.items():
            for item in result.columns:
                column_outputs[key, item.column.name] = item.origins
                for origin in item.origins:
                    if origin.key:
                        column_consumers[origin.key, origin.column].append((key, item.column.name))

        def trace_columns(parent, key, column, inputs, chain, seen):
            nonlocal clipped
            if inputs:
                edges = ((edge.key, edge.relation, edge.column) for edge in column_outputs.get((key, column), ()))
            else:
                edges = ((child, index.resources[child].label, name)
                         for child, name in column_consumers.get((key, column), ()))
            for child, relation, name in edges:
                pair = (child, name)
                next_edges = column_outputs if inputs else column_consumers
                repeated = pair in seen or pair in chain
                note = ' · cycle' if pair in chain else ' · shown above' if repeated else ''
                node = add(parent, f"{relation}.{name}{note}", target(child),
                           branch=not repeated and bool(next_edges.get(pair)))
                if node is None:
                    return
                if not repeated:
                    seen.add(pair)
                    if len(chain) < _TREE_DEPTH:
                        trace_columns(node, child, name, inputs, (*chain, pair), seen)
                    elif next_edges.get(pair):
                        clipped = True

        for resource in roots:
            root = add(tree.root, resource.label, target(resource.key), branch=True)
            if root is None:
                break
            root.expand()
            for label, edges in (('Upstream', index.upstream), ('Downstream', index.downstream)):
                group = add(root, f"{label} ({len(edges.get(resource.key, ()))})", branch=bool(edges.get(resource.key)))
                if group is not None and edges.get(resource.key):
                    group.expand()
                    walk(group, resource.key, edges, (resource.key,), set())
            columns = feedback.related.get(resource.key, DbtAnalysis()).columns
            defs = tuple(item.column for item in columns) or resource.columns
            known = {col.name: col for col in defs}
            # Source files can have no declared types yet still have proven uses from SQL.
            for key, column in column_consumers:
                if key == resource.key:
                    known.setdefault(column, None)
            if not known:
                continue
            group = add(root, 'Columns', branch=True)
            if group is None:
                break
            for name, column in known.items():
                pair = (resource.key, name)
                dtype = column.type if column is not None and column.type else 'unknown type'
                inputs, users = column_outputs.get(pair), column_consumers.get(pair)
                output = add(group, f"{name} · {dtype}", branch=bool(inputs or users))
                if output is None:
                    break
                for label, exists, incoming in (('From', inputs, True), ('Used by', users, False)):
                    if exists and (branch := add(output, label, branch=True)) is not None:
                        branch.expand()
                        trace_columns(branch, *pair, incoming, (pair,), set())
        if clipped:
            status.append('Lineage display truncated')
        if feedback.trace_limited:
            status.append('Column trace limited to nearby SQL')
        self.query_one(Static).update(' · '.join(status) or 'Enter opens the selected resource or diagnostic')

    def _add_diagnostics(self, parent: TreeNode, page: range) -> None:
        for index in page:
            problem = self._diagnostics[index]
            parent.add_leaf(_diagnostic_label(self.ide, problem), data=problem)

    @on(Tree.NodeExpanded)
    def _expand_diagnostics(self, event: Tree.NodeExpanded) -> None:
        node = event.node
        if isinstance(node.data, range) and not node.children and self.ide.dbt_authoring.current:
            self._add_diagnostics(node, node.data)

    @on(Tree.NodeSelected)
    def _open(self, event: Tree.NodeSelected) -> None:
        if isinstance(event.node.data, Diagnostic):
            self.ide.dbt_authoring.jump_to_diagnostic(event.node.data)
            return
        if isinstance(event.node.data, tuple) and self.ide.dbt_authoring.current:
            path, row = event.node.data
            self.ide.open_dbt_file(self.ide._dbt_home, path, row=row)
