"""dbt project explorer and explicit workspace actions."""
from __future__ import annotations

import asyncio
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static, TabbedContent, TabPane, TextArea, Tree
from textual.widgets.option_list import Option

from d8r.dbt import DbtError
from d8r.dbt_column_rename import plan_column_rename
from d8r.dbt_docs import DbtDocs
from d8r.dbt_edits import DbtDraft, model_ref, singular_test
from d8r.dbt_files import DbtFile, DbtFiles, DbtModel, file_revision
from d8r.dbt_git import git_diff
from d8r.dbt_references import DbtReferences
from d8r.dbt_rename import DbtRenamePlan, apply_rename, plan_rename
from .dbt_authoring import DbtLineage, DbtProblems
from .dbt_completion import DbtTestCompletion, model_columns
from .editing import DraftTextArea
from .palette import CommandPalette, EditorPane
from .query_editor import QueryTextArea
from .session import Session

if TYPE_CHECKING:
    from .app import D8RApp


class DbtDialog(ModalScreen):
    BINDINGS = [Binding("escape", "cancel", "Close")]
    DEFAULT_CSS = """
    DbtDialog {
        align: center middle;
        background: $background 65%;
        .dbt-dialog {
            width: 88; max-width: 98%; height: auto; max-height: 95%;
            border: thick $primary; background: $surface; padding: 1 2;
        }
        Static { height: auto; margin-bottom: 1; }
        .dbt-buttons { height: auto; }
        Button { margin-right: 1; }
        TextArea { height: 8; margin-top: 1; }
        #dbt-diff-text { height: 1fr; min-height: 12; }
        .dbt-dialog.diff { height: 88%; }
        #dbt-message { color: $text-warning; }
    }
    """

    def action_cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#dbt-close")
    def _close(self) -> None:
        self.dismiss(None)

    def _message(self, text: str) -> None:
        self.query_one("#dbt-message", Static).update(text)


class DBTScreen(DbtDialog):
    """One source-scoped setting. No file writes or execution when opening a project."""

    def __init__(self, ide: D8RApp, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        self.source = ide.session.source_key()

    def compose(self) -> ComposeResult:
        with Vertical(classes="dbt-dialog"):
            yield Static("Workspace project home (directory containing dbt_project.yml)")
            yield Input(self.ide.session.dbt_project_dir, placeholder="Local project directory", id="dbt-project-dir")
            yield Static("", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Open & Save Project Home", id="dbt-open", variant="primary")
                yield Button("Close", id="dbt-close")

    @on(Button.Pressed, "#dbt-open")
    @on(Input.Submitted, "#dbt-project-dir")
    def _open_project(self) -> None:
        try:
            if not self.ide.session.dbt_enabled or self.ide.session.source_key() != self.source:
                raise DbtError("The source changed. Reopen project setup.")
            project = DbtFiles(self.query_one("#dbt-project-dir", Input).value.strip())
            if not self.ide.set_dbt_project_dir(str(project.root)):
                raise DbtError("Project home could not be saved. Check workspace storage permissions.")
        except (DbtError, OSError) as exc:
            self._message(str(exc))
        else:
            self.dismiss(str(project.root))


class DbtPortScreen(DbtDialog):
    def compose(self) -> ComposeResult:
        with Vertical(classes="dbt-dialog"):
            yield Static("Generate & Serve Documentation")
            yield Static("Generate from saved files without a warehouse connection. Serve on localhost only.")
            yield Input("8080", type="integer", placeholder="Port", id="dbt-docs-port")
            yield Static("", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Generate & Serve", id="dbt-docs-start", variant="primary")
                yield Button("Close", id="dbt-close")

    @on(Button.Pressed, "#dbt-docs-start")
    @on(Input.Submitted, "#dbt-docs-port")
    def _start(self) -> None:
        value = self.query_one("#dbt-docs-port", Input).value
        if not value.isascii() or not value.isdecimal() or len(value) > 5 or not 1 <= int(value) <= 65535:
            self._message("Choose a port between 1 and 65535.")
        else:
            self.dismiss(int(value))


class DbtModelScreen(DbtDialog):
    DEFAULT_CSS = """
    DbtModelScreen #dbt-models { height: auto; max-height: 18; }
    """

    def __init__(self, paths: tuple[str, ...]) -> None:
        super().__init__()
        self.paths = paths

    def compose(self) -> ComposeResult:
        with Vertical(classes="dbt-dialog"):
            yield Static("Select a dbt model")
            yield OptionList(*(Option(Text(path), id=path) for path in self.paths), id="dbt-models", markup=False)
            yield Button("Close", id="dbt-close")

    @on(OptionList.OptionSelected, "#dbt-models")
    def _model(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option_id)


class DbtTestScreen(DbtDialog):
    DEFAULT_CSS = """
    DbtTestScreen .dbt-dialog { height: 24; }
    DbtTestScreen #dbt-test-editor { height: 1fr; min-height: 6; layers: base overlay; }
    DbtTestScreen #dbt-test-sql { height: 1fr; margin: 0; }
    DbtTestScreen #dbt-test-palette {
        layer: overlay; position: absolute; display: none;
        max-height: 12; text-wrap: nowrap; text-overflow: ellipsis;
    }
    DbtTestScreen #dbt-test-palette.open { display: block; }
    DbtTestScreen .completion-details {
        layer: overlay; position: absolute; display: none;
        background: $surface; color: $foreground; border: round $primary 65%;
        padding: 0 1; max-height: 8; overflow-y: auto; text-wrap: wrap;
    }
    """

    def __init__(self, session: Session, project: DbtFiles, model: DbtModel) -> None:
        super().__init__()
        self.session, self.project, self.model = session, project, model
        self.source = session.source
        self.dialect = self.source.dialect
        source_key, home = session.source_key(), str(project.root)
        drafts = {}
        for page in session.workspace["pages"]:
            link = page.get("dbt_file")
            if (page["source"] == source_key and link and link["project_dir"] == home
                    and link["path"] in {model.path, model.properties}
                    and file_revision(page["document"]) != link["revision"]):
                drafts[link["path"]] = page["document"]
        columns = model_columns(project, model, self.dialect, drafts)
        self.completion = DbtTestCompletion(model, project.name, self.dialect, columns)

    def compose(self) -> ComposeResult:
        with Vertical(classes="dbt-dialog"):
            yield Static(f"Create a singular data test for {self.model.name}", markup=False)
            yield Input(placeholder="test_name", id="dbt-test-name")
            with EditorPane(id="dbt-test-editor"):
                editor = QueryTextArea(f"select *\nfrom {model_ref(self.model)}\nwhere ", id="dbt-test-sql",
                                       show_line_numbers=True, dialect=self.dialect)
                editor.dbt_template = True
                editor.move_cursor((2, len("where ")))
                yield editor
                yield CommandPalette(self.session, local_completion=self.completion,
                                     id="dbt-test-palette", markup=False)
            yield Static("Return only failing rows. Ctrl+Space shows local SQL suggestions.",
                         id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Create Test", id="dbt-test-create", variant="primary")
                yield Button("Close", id="dbt-close")

    def action_cancel(self) -> None:
        palette = self.query_one(CommandPalette)
        if palette.is_open:
            palette.close()
        else:
            super().action_cancel()

    @on(Button.Pressed, "#dbt-test-create")
    def _create(self) -> None:
        try:
            if not self.session.dbt_enabled or self.session.source is not self.source:
                raise DbtError("The source changed. Reopen the data test dialog.")
            path = self.project.test_file(self.query_one("#dbt-test-name", Input).value.strip())
            text = self.query_one("#dbt-test-sql", TextArea).text
            if text.strip().casefold().endswith("where"):
                raise DbtError("Add the failing-row predicate after WHERE.")
            body = singular_test(text, self.model, self.project.name, dialect=self.dialect)
        except (DbtError, OSError) as exc:
            self._message(str(exc))
        else:
            self.dismiss(DbtDraft(path, DbtFile("", ""), body))


class DbtDiffScreen(DbtDialog):
    def __init__(self, path: str, diff: str) -> None:
        super().__init__()
        self.path, self.diff = path, diff

    def compose(self) -> ComposeResult:
        with Vertical(classes="dbt-dialog diff"):
            yield Static(f"Git Diff · {self.path}", markup=False)
            yield DraftTextArea(self.diff, read_only=True, soft_wrap=False, id="dbt-diff-text")
            yield Button("Close", id="dbt-close")


class DbtReferencesScreen(DbtDialog):
    DEFAULT_CSS = """
    DbtReferencesScreen #dbt-usage-list { height: auto; max-height: 24; }
    """

    def __init__(self, results: DbtReferences) -> None:
        super().__init__()
        self.results = results

    def compose(self) -> ComposeResult:
        results = self.results
        suffix = ' · some uses could not be verified' if results.limited else ''
        with Vertical(classes='dbt-dialog'):
            yield Static(f'References · {results.label}', markup=False)
            if results.locations:
                yield OptionList(*(Option(Text(f'{item.path}:{item.row + 1}:{item.column + 1}'
                                               + (f' · {item.detail}' if item.detail else '')))
                                   for item in results.locations), id='dbt-usage-list', markup=False)
                yield Static(f'Enter opens a reference{suffix}', markup=False)
            else:
                yield Static(f'No statically resolved references{suffix}', markup=False)
            yield Button('Close', id='dbt-close')

    @on(OptionList.OptionSelected, '#dbt-usage-list')
    def _choose(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.results.locations[event.option_index])


class DbtRenameScreen(DbtDialog):
    """Review every rewrite (and any file move) together before writing."""

    DEFAULT_CSS = """
    DbtRenameScreen #dbt-rename-diff { height: 1fr; min-height: 8; margin-bottom: 1; }
    """

    def __init__(self, ide: D8RApp, home: str, path: str, *, column_offset: int | None = None) -> None:
        super().__init__()
        self.ide, self.home, self.path = ide, home, path
        self.context = (ide._workspace_key, ide._document_identity, ide.editor.text)
        self.column_offset = column_offset
        self.dialect = ide.editor.dialect
        self.dirty = ide._dbt_rename_dirty(home)
        self.plan: DbtRenamePlan | None = None
        self.working = False

    def compose(self) -> ComposeResult:
        symbol = 'column' if self.column_offset is not None else 'model'
        with Vertical(classes="dbt-dialog"):
            yield Static(f"Rename dbt {symbol} · {self.path}", markup=False)
            yield Input(placeholder=f"New {symbol} name", id="dbt-rename-name")
            review = DraftTextArea(read_only=True, soft_wrap=False, id="dbt-rename-diff")
            review.display = False
            yield review
            yield Static("Preview the definition and reference edits before applying.", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Preview Rename", id="dbt-rename-submit", variant="primary")
                yield Button("Cancel", id="dbt-rename-cancel")

    @on(Button.Pressed, "#dbt-rename-cancel")
    def action_cancel(self) -> None:
        if not self.working:
            self.dismiss(None)

    def _current(self) -> bool:
        ide = self.ide
        return (ide.session.dbt_enabled and ide._dbt_home == self.home
                and (ide._workspace_key, ide._document_identity, ide.editor.text) == self.context
                and ide._dbt_rename_dirty(self.home) == self.dirty)

    @on(Button.Pressed, "#dbt-rename-submit")
    @on(Input.Submitted, "#dbt-rename-name")
    def _submit(self) -> None:
        if self.working or not self._current():
            return
        name = self.query_one(Input).value.strip()
        self.working = True
        for button in self.query(Button):
            button.disabled = True
        self.run_worker(self._review_or_apply(name), group="dbt-rename", exit_on_error=False)

    async def _review_or_apply(self, name: str) -> None:
        def preview():
            project = DbtFiles(self.home)
            if self.column_offset is None:
                plan = plan_rename(project, self.path, name, dirty=self.dirty)
            else:
                plan = plan_column_rename(project, self.path, self.column_offset, name, dialect=self.dialect,
                                          dirty=self.dirty, revision=file_revision(self.context[2]))
            diff = []
            for edit in plan.edits:
                diff.append(f"\n{edit.path} → {edit.destination}\n" if edit.path != edit.destination else f"\n{edit.path}\n")
                diff.extend(unified_diff(edit.before.text.splitlines(keepends=True), edit.text.splitlines(keepends=True),
                                         fromfile=edit.path, tofile=edit.destination))
            return plan, "".join(diff)

        try:
            if self.plan is None:
                self._message("Checking project references…")
                plan, diff = await asyncio.to_thread(preview)
                if not self._current():
                    self._message("The workspace or its drafts changed. Reopen rename.")
                    return
                self.plan = plan
                self.query_one(Input).display = False
                self.query_one(Vertical).add_class("diff")
                review = self.query_one("#dbt-rename-diff", TextArea)
                review.load_text(diff)
                review.display = True
                review.focus()
                self.query_one("#dbt-rename-submit", Button).label = "Apply Rename"
                self._message(f"Review {len(plan.edits)} file(s). Apply writes to disk; nothing runs.")
            else:
                self._message("Rechecking the reviewed files, then applying…")
                plan = self.plan
                saved = await asyncio.to_thread(lambda: apply_rename(DbtFiles(self.home), plan))
                self.ide.finish_dbt_rename(self.home, plan, saved)
                self.dismiss(None)
        except (DbtError, OSError) as exc:
            self._message(str(exc))
        finally:
            self.working = False
            for button in self.query(Button):
                button.disabled = False


class DbtExplorer(Vertical):
    """Display project paths and explicit dbt actions in the shared explorer."""

    _TESTS_NODE = ("group", "tests")

    def __init__(self, ide: D8RApp, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        self.home: str | None = None
        self.workspace: str | None = None
        self.docs = DbtDocs()

    def compose(self) -> ComposeResult:
        with TabbedContent(initial="dbt-panel-files", id="dbt-panels"):
            with TabPane("Files", id="dbt-panel-files"):
                with Horizontal(classes="dbt-explorer-actions"):
                    yield Button("Project Home", id="dbt-home", compact=True)
                    yield Button("Refresh", id="dbt-refresh", compact=True)
                yield Tree("dbt", id="dbt-tree")
                yield Static("", id="dbt-explorer-status", markup=False)
            with TabPane("Lineage", id="dbt-panel-lineage"):
                yield DbtLineage(self.ide)
            with TabPane("Problems", id="dbt-panel-problems"):
                yield DbtProblems(self.ide)

    def refresh_project(self, *, force: bool = False) -> None:
        home = self.ide.session.dbt_project_dir if self.ide.session.dbt_enabled else ""
        workspace = self.ide.session.source_key()
        same = (home, workspace) == (self.home, self.workspace)
        if same and not force:
            return
        if not same and self.home:
            self.run_worker(self.docs.stop(), group="dbt-docs-stop", exit_on_error=False)
        tree = self.query_one("#dbt-tree", Tree)
        folders = {node.data for node in self._expanded(tree.root) if isinstance(node.data, str)} if same else set()
        tests_expanded = next((node.is_expanded or not node.children for node in tree.root.children
                               if same and node.data == self._TESTS_NODE), True)
        self.home, self.workspace = home, workspace
        tree.clear()
        tree.root.set_label(Text(Path(home).name if home else "No dbt project open"))
        actions = tree.root.add("Actions", data="actions", expand="actions" in folders)
        for key, label in (
            ("test", "Create Data Test"), ("reload", "Reload File from Disk"),
            ("rename", "Rename Open Model"),
            ("docs", "Generate & Serve Documentation"), ("stop-docs", "Stop Documentation Server"),
        ):
            actions.add_leaf(label, data=("action", key))
        tree.root.expand()
        self.query_one("#dbt-explorer-status", Static).update("Loading files…" if home else "Open a project home to browse dbt files.")
        if home:
            self.run_worker(self._populate(home, workspace, folders, tests_expanded), group="dbt-files",
                            exclusive=True, exit_on_error=False)

    @staticmethod
    def _expanded(node):
        for child in node.children:
            if child.is_expanded:
                yield child
                yield from DbtExplorer._expanded(child)

    async def _populate(self, home: str, workspace: str, expanded: set[str], tests_expanded: bool) -> None:
        def read_project():
            project = DbtFiles(home)
            roots = {}
            warning = project.config_error
            if not warning:
                for key, kind in (("test-paths", "tests"), ("model-paths", "models")):
                    try:
                        roots[kind] = tuple(path.relative_to(project.root) for path in project.directories(key, kind))
                    except DbtError as exc:
                        warning = warning or str(exc)
            return project.paths(), roots, warning

        status = self.query_one("#dbt-explorer-status", Static)
        try:
            paths, roots, warning = await asyncio.to_thread(read_project)
        except (DbtError, OSError) as exc:
            if (home, workspace) == (self.home, self.workspace):
                status.update(str(exc))
            return
        if (home, workspace) != (self.home, self.workspace):
            return
        tree = self.query_one("#dbt-tree", Tree)

        def add_files(root, paths: list[str], prefix: Path = Path()) -> None:
            folders = {prefix: root}
            for path in paths:
                relative = Path(path)
                parent = root
                key = prefix
                for folder in relative.relative_to(prefix).parts[:-1]:
                    key /= folder
                    if key not in folders:
                        folders[key] = parent.add(Text(folder), data=key.as_posix(), expand=key.as_posix() in expanded)
                    parent = folders[key]
                parent.add_leaf(Text(relative.name), data=("file", path))

        test_roots = roots.get("tests", ())
        files, tests = [], []
        for path in paths:
            relative = Path(path)
            if any(relative.is_relative_to(root) for root in test_roots) and (
                relative.suffix.lower() == ".sql" or relative.parts[0] == "tests"
            ):
                tests.append(path)
            else:
                files.append(path)
        add_files(tree.root, files)
        if test_roots:
            model_folders = {path.parts[0] for path in roots.get("models", ()) if path.parts}
            after = next((node for node in reversed(tree.root.children) if isinstance(node.data, str)
                          and node.data in model_folders), None)
            node = tree.root.add("tests", data=self._TESTS_NODE, expand=tests_expanded, after=after)
            # With one test root, show its contents directly. With multiple
            # roots, keep project-relative folders to distinguish same-name tests.
            add_files(node, tests, test_roots[0] if len(test_roots) == 1 else Path())
        status.update(warning or "Enter open · Ctrl+click / F12 follow dbt references")

    @on(Button.Pressed, "#dbt-home")
    def _home(self) -> None:
        if not self.ide.session.dbt_enabled:
            return
        self.app.push_screen(DBTScreen(self.ide), lambda path: self.ide.show_dbt_explorer() if path else None)

    @on(Button.Pressed, "#dbt-refresh")
    def _refresh(self) -> None:
        if not self.ide.session.dbt_enabled:
            return
        self.refresh_project(force=True)
        self.ide.dbt_authoring.invalidate()

    @on(TabbedContent.TabActivated, "#dbt-panels")
    def _panel_changed(self) -> None:
        self.ide.dbt_authoring.request()

    @on(Tree.NodeSelected, "#dbt-tree")
    def _select(self, event: Tree.NodeSelected) -> None:
        if not self.ide.session.dbt_enabled or not isinstance(event.node.data, tuple):
            return
        kind, value = event.node.data
        if kind == "file" and self.home == self.ide.session.dbt_project_dir:
            self.ide.open_dbt_file(self.home, value)
        elif kind == "action":
            self._select_action(value)

    def _select_action(self, value: str) -> None:
        if value == "stop-docs":
            self.run_worker(self._stop_docs(), group="dbt-docs-stop", exit_on_error=False)
        elif value == "docs":
            self._docs()
        elif value == "test":
            self.ide.action_create_dbt_test()
        elif value == "reload":
            self.ide.reload_dbt_file()
        elif value == "rename":
            self.ide.action_dbt_rename()

    def _docs(self) -> None:
        home = self.ide.session.dbt_project_dir
        workspace = self.ide.session.source_key()
        if not home:
            self.ide._set_status("Open a dbt project home first")
            return

        def start(port: int | None) -> None:
            if port is not None and (home, workspace) == (self.home, self.workspace) and self.ide.session.dbt_enabled:
                self.ide._flush_workspace()
                self.run_worker(self._serve(home, workspace, port), group="dbt-docs", exit_on_error=False)

        self.app.push_screen(DbtPortScreen(), start)

    async def _serve(self, home: str, workspace: str, port: int) -> None:
        if (home, workspace) != (self.home, self.workspace) or not self.ide.session.dbt_enabled:
            return
        self.ide._set_status("Generating documentation from saved project files…")
        try:
            url = await self.docs.start(DbtFiles(home), port)
        except (DbtError, OSError) as exc:
            if (home, workspace) == (self.home, self.workspace):
                self.ide._set_status(str(exc))
        else:
            if (home, workspace) == (self.home, self.workspace) and self.ide.session.dbt_enabled:
                self.ide._set_status(f"dbt docs: {url} · Stop Documentation Server in dbt Actions")
            else:
                await self.docs.stop()

    async def _stop_docs(self) -> None:
        await self.docs.stop()
        self.ide._set_status("dbt documentation stopped")

    async def show_diff(self, home: str, path: str, draft: str, page_id: str) -> None:
        try:
            diff = await asyncio.to_thread(lambda: git_diff(DbtFiles(home), path, draft))
        except (DbtError, OSError) as exc:
            self.ide._set_status(str(exc))
            return
        if self.ide.session.dbt_enabled and not self.ide._modal_open() and page_id == self.ide._document_identity:
            self.app.push_screen(DbtDiffScreen(path, diff))

    async def on_unmount(self) -> None:
        await self.docs.stop()
