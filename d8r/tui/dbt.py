"""dbt project explorer and explicit workspace actions."""
from __future__ import annotations

import asyncio
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from difflib import unified_diff
from pathlib import Path
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.suggester import SuggestFromList
from textual.widgets import Button, Input, OptionList, Select, Static, Switch, TabbedContent, TabPane, TextArea, Tree
from textual.widgets.option_list import Option

from d8r.dbt import DbtError
from d8r.dbt_column_rename import plan_column_rename
from d8r.dbt_docs import DbtDocs
from d8r.dbt_edits import DbtDraft, model_ref, singular_test
from d8r.dbt_files import DbtFile, DbtFiles, DbtModel, file_revision
from d8r.dbt_git import git_diff
from d8r.dbt_profiles import DbtTargetSettings
from d8r.dbt_references import DbtReferences
from d8r.dbt_rename import DbtRenamePlan, apply_rename, plan_rename
from d8r.dbt_run import (
    DbtModelCommand, DbtNodeResult, DbtRun, DbtRunResult, dbt_named_selectors, find_dbt_runtime,
)
from d8r.dbt_vars import read_project_vars, save_project_vars, validate_run_vars
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


@dataclass(frozen=True)
class DbtRunSelection:
    run: DbtRun | None
    result: DbtRunResult | None
    message: str
    jump: DbtNodeResult | None = None
    rerun: bool = False
    vars_override: str = field(default="", repr=False)


class DbtRunScreen(DbtDialog):
    """Run dbt on saved files, or inspect results from a previous run."""

    BINDINGS = [
        Binding("f8", "cancel", "Cancel dbt", show=False),
        Binding("ctrl+enter", "submit_run", "Run dbt", show=False),
        Binding("ctrl+r", "rerun", "Rerun selection", show=False),
    ]
    DEFAULT_CSS = """
    DbtRunScreen #dbt-run-nodes { max-height: 14; }
    DbtRunScreen #dbt-run-actions { margin-bottom: 1; }
    DbtRunScreen #dbt-run-actions Button { width: 1fr; min-width: 0; }
    DbtRunScreen .dbt-run-vars { height: auto; }
    DbtRunScreen #dbt-run-vars { height: 6; margin: 0; }
    """
    _PAGE_SIZE = 60
    _FAILURES = {"error", "fail", "failed", "runtime error", "runtime_error"}

    def __init__(self, prepare: Callable[[str], DbtRun] | None, *, sql_source: str,
                 auto_close: bool = False, completed: tuple[DbtRun, DbtRunResult] | None = None,
                 can_rerun: bool = True, compile_reason: str = "", run_label: str = "",
                 vars_override: str = "") -> None:
        super().__init__()
        if prepare is None and completed is None:
            raise ValueError("DbtRunScreen needs a command or a completed result.")
        self._prepare = prepare
        self._sql_source = sql_source
        self._auto_close = auto_close
        self._completed = completed
        self._can_rerun = can_rerun
        self._compile_reason = compile_reason
        self._run_label = run_label
        self._vars_override = vars_override
        self._approval = asyncio.Event() if compile_reason else None
        self._prepared: DbtRun | None = None
        self._execution: asyncio.Task[DbtRunSelection] | None = None
        self._selection: DbtRunSelection | None = None
        self._result_nodes: list[DbtNodeResult] = []
        self._first = 0

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="dbt-dialog"):
            yield Static("dbt command · saved project files")
            yield Static("", id="dbt-run-target", markup=False)
            yield Static(f"Native SQL connection (separate from dbt): {self._sql_source}", markup=False)
            if self._run_label:
                with Vertical(classes="dbt-run-vars", id="dbt-run-inputs"):
                    yield Static("Variable overrides (--vars) · YAML mapping, e.g. {lookback_days: 7}. "
                                 "Leave blank to use project defaults. Do not enter secrets.", markup=False)
                    yield DraftTextArea(self._vars_override, id="dbt-run-vars")
            yield Static("" if self._run_label else "Checking the project and workspace target…",
                         id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons", id="dbt-run-actions"):
                if self._run_label:
                    yield Button(self._run_label, id="dbt-run-start", variant="primary", compact=True)
                yield Button("Cancel", id="dbt-run-dismiss", variant="primary", compact=True)
                if self._approval is not None:
                    approve = Button("Compile & Switch", id="dbt-switch-compile", variant="primary", compact=True)
                    approve.display = False
                    yield approve
                rerun = Button("Rerun (Ctrl+R)", id="dbt-run-rerun", compact=True,
                               tooltip="Rerun the same selection")
                rerun.display = False
                yield rerun
            help_text = Static("", id="dbt-run-help", markup=False)
            help_text.display = False
            yield help_text
            nodes = OptionList(id="dbt-run-nodes", markup=False)
            nodes.display = False
            yield nodes

    def on_mount(self) -> None:
        if self._completed is not None:
            run, result = self._completed
            self._show_command(run)
            self._show_result(self._outcome(run, result))
        elif not self._run_label:
            self._start()

    def _start(self) -> None:
        if self._prepare is not None:
            self._message("Checking the project and workspace target…")
            self._execution = asyncio.create_task(self._run(self._prepare, self._vars_override))
            self._execution.add_done_callback(self._finished)

    @on(Button.Pressed, "#dbt-run-start")
    def action_submit_run(self) -> None:
        if not self._run_label or self._execution is not None or self._selection is not None:
            return
        editor = self.query_one("#dbt-run-vars", DraftTextArea)
        try:
            self._vars_override = validate_run_vars(editor.text)
        except DbtError as exc:
            self._message(str(exc))
            editor.focus()
            return
        self.query_one("#dbt-run-inputs").display = False
        self.query_one("#dbt-run-start", Button).display = False
        self.query_one("#dbt-run-dismiss", Button).focus()
        self._start()

    def _show_command(self, run: DbtRun) -> None:
        target = (f"{run.target.profile} / {run.target.name}" if run.target
                  else "Not resolved from project profiles")
        selection = f"\nSelection: {run.selector}" if run.selector else ""
        variables = "\nPer-run variable overrides applied" if run.vars_override else ""
        logs = "Temporary for this preview" if run.verb == "show" else str(run.log_dir / "dbt.log")
        self.query_one("#dbt-run-target", Static).update(
            f"Command: dbt {run.verb}{selection}\nRuntime: {run.runtime.label}\n"
            f"dbt profile / target: {target}{variables}\nProject: {run.root}\nLog: {logs}")

    def _progress(self, text: str) -> None:
        task = self._execution
        if text and task is not None and not task.cancelling() and self.is_mounted:
            # The runtime sends only classified progress, never subprocess output.
            self._message(text)

    async def _run(self, prepare: Callable[[str], DbtRun], vars_override: str) -> DbtRunSelection:
        try:
            run = await asyncio.to_thread(prepare, vars_override)
            self._prepared = run
            self._show_command(run)
            if self._approval is not None:
                self._message(f"{self._compile_reason}\n"
                              "Compile can connect to the dbt target and evaluate project macros. "
                              "It will not run or build the model.")
                approve = self.query_one("#dbt-switch-compile", Button)
                approve.display = True
                await self._approval.wait()
                approve.display = False
            self._message("Running dbt… (Esc or F8 cancels)")
            return self._outcome(run, await run.execute(self._progress))
        except DbtError as exc:
            return DbtRunSelection(self._prepared, None, f"dbt could not run · {exc}")
        except OSError:
            return DbtRunSelection(self._prepared, None, "Could not access the dbt project files. Check permissions.")

    @staticmethod
    def _outcome(run: DbtRun, result: DbtRunResult) -> DbtRunSelection:
        return DbtRunSelection(run, result, result.message or f"dbt {run.verb} · exit {result.exit_code}")

    def _finished(self, task: asyncio.Task[DbtRunSelection]) -> None:
        if self._execution is not task or not self.app.is_running or not self.is_mounted:
            return
        self._execution = None
        selection = (DbtRunSelection(self._prepared, None, "dbt cancelled · changes already committed may remain.")
                     if task.cancelled() else task.result())
        run, result = selection.run, selection.result
        if (self._auto_close and self.app.screen is self and run and result and result.exit_code == 0
                and ((run.verb == "show" and result.preview_available)
                     or (run.verb == "compile" and result.compiled_sql))):
            self.dismiss(selection)
        else:
            self._show_result(selection)

    def _show_result(self, selection: DbtRunSelection) -> None:
        self._selection = selection
        result = selection.result
        summary = selection.message
        if result is not None:
            counts = Counter(node.status for node in result.nodes)
            suffix = f" · {len(result.nodes)} node(s)"
            if counts:
                suffix += " (" + ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())) + ")"
            exit_summary = f"Exit {result.exit_code}{suffix}"
            summary = f"{exit_summary}\n{result.message}" if result.message else exit_summary
            self._result_nodes = sorted(result.nodes, key=self._node_order)
        self._message(summary)
        button = self.query_one("#dbt-run-dismiss", Button)
        button.label = "Close"
        button.disabled = False
        self.query_one("#dbt-run-rerun", Button).display = self._can_rerun
        if self._result_nodes:
            self._show_nodes(0)
            if self.app.screen is self:
                self.query_one("#dbt-run-nodes", OptionList).focus()
        elif self.app.screen is self:
            button.focus()

    @classmethod
    def _node_order(cls, node: DbtNodeResult) -> int:
        if node.status.casefold() in cls._FAILURES:
            return 0 if node.path else 1
        return 2 if node.path else 3

    def _show_nodes(self, first: int) -> None:
        self._first = first
        stop = min(first + self._PAGE_SIZE, len(self._result_nodes))
        options = [Option("‹ Previous page", id="previous")] if first else []
        for index in range(first, stop):
            node = self._result_nodes[index]
            prompt = Text.assemble((node.status, "bold"), " · ", node.unique_id)
            if node.path:
                line = f":{node.row + 1}" if node.row is not None else ""
                prompt.append(f"\n  ↳ {node.path}{line}")
            if node.reason:
                prompt.append(f"\n  {node.reason}")
            options.append(Option(prompt, id=str(index)))
        if stop < len(self._result_nodes):
            options.append(Option("Next page ›", id="next"))
        listing = self.query_one("#dbt-run-nodes", OptionList)
        listing.display = True
        listing.set_options(options)
        listing.highlighted = int(bool(first))
        help_text = self.query_one("#dbt-run-help", Static)
        help_text.display = True
        help_text.update(
            f"Nodes {first + 1}–{stop} of {len(self._result_nodes)} · Enter opens nodes with a local file (↳) or changes pages.")

    @on(OptionList.OptionSelected, "#dbt-run-nodes")
    def _open_node(self, event: OptionList.OptionSelected) -> None:
        if self._selection is None:
            return
        if event.option_id == "previous":
            self._show_nodes(self._first - self._PAGE_SIZE)
        elif event.option_id == "next":
            self._show_nodes(self._first + self._PAGE_SIZE)
        elif event.option_id is not None:
            node = self._result_nodes[int(event.option_id)]
            if node.path:
                self.dismiss(replace(self._selection, jump=node))
            else:
                self.query_one("#dbt-run-help", Static).update(
                    "This node has no editable file in the current project. Select another node or page.")

    @on(Button.Pressed, "#dbt-run-rerun")
    def action_rerun(self) -> None:
        if self._can_rerun and self._selection is not None:
            run = self._selection.run
            self.dismiss(replace(self._selection, rerun=True,
                                 vars_override=run.vars_override if run else self._vars_override))

    @on(Button.Pressed, "#dbt-switch-compile")
    def _approve_compile(self) -> None:
        if self._approval is not None:
            self._approval.set()

    @on(Button.Pressed, "#dbt-run-dismiss")
    def action_cancel(self) -> None:
        task = self._execution
        if task is None:
            self.dismiss(self._selection)
        elif self._approval is not None and not self._approval.is_set():
            self.dismiss(None)
        elif not task.done() and not task.cancelling():
            self._message("Cancelling dbt…")
            self.query_one("#dbt-run-dismiss", Button).disabled = True
            task.cancel()

    async def on_unmount(self) -> None:
        task, self._execution = self._execution, None
        if task is not None:
            if not task.done() and not task.cancelling():
                task.cancel()
            await asyncio.gather(task, return_exceptions=True)


class DBTScreen(DbtDialog):
    """Save the project, target and runtime settings for one source workspace."""

    DEFAULT_CSS = """
    DBTScreen .dbt-buttons { layout: vertical; }
    DBTScreen .dbt-buttons Button { width: 100%; min-width: 0; margin-right: 0; }
    """

    def __init__(self, ide: D8RApp, *, focus_field: str = "dbt-project-dir", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        self.source = ide.session.source_key()
        self.focus_field = focus_field

    def compose(self) -> ComposeResult:
        settings = self.ide.session.dbt_target_settings(self.ide.session.dbt_project_dir)
        with VerticalScroll(classes="dbt-dialog"):
            yield Static("Workspace project home (directory containing dbt_project.yml)")
            yield Input(self.ide.session.dbt_project_dir, placeholder="Local project directory", id="dbt-project-dir")
            yield Static("Path to dbt Profiles (directory or profiles.yml)")
            yield Input(settings.profiles_path, placeholder="Discover profiles.yml", id="dbt-profiles-path")
            yield Static("Workspace Default Target Override")
            yield Input(settings.override, placeholder="Use the profile default", id="dbt-target-override")
            yield Static("Check Setup to see the resolved target and available outputs.", id="dbt-profile-targets",
                         markup=False)
            yield Static("dbt runtime override (virtual environment directory or executable)")
            yield Input(self.ide.session.dbt_runtime_path, placeholder="Blank: automatically discover dbt Core",
                        id="dbt-runtime-path")
            yield Static("", id="dbt-runtime-status", markup=False)
            yield Static("", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Open & Save Project Home", id="dbt-open", variant="primary", compact=True)
                yield Button("Check Setup", id="dbt-list-targets", compact=True)
                yield Button("Close", id="dbt-close", compact=True)

    def on_mount(self) -> None:
        if self.ide.session.dbt_project_dir:
            self._check_setup()
        self.query_one(f"#{self.focus_field}", Input).focus()

    def _project(self) -> DbtFiles:
        if not self.ide.session.dbt_enabled or self.ide.session.source_key() != self.source:
            raise DbtError("The source changed. Reopen project setup.")
        return DbtFiles(self.query_one("#dbt-project-dir", Input).value.strip())

    def _selection(self) -> DbtTargetSettings:
        return DbtTargetSettings(self.query_one("#dbt-profiles-path", Input).value.strip(),
                                 self.query_one("#dbt-target-override", Input).value.strip())

    def _runtime_path(self) -> str:
        return self.query_one("#dbt-runtime-path", Input).value.strip()

    def _check_runtime(self, project: DbtFiles) -> None:
        status = self.query_one("#dbt-runtime-status", Static)
        override = self._runtime_path()
        try:
            runtime = find_dbt_runtime(project.root, override)
        except DbtError as exc:
            status.update(str(exc))
        except OSError:
            status.update("Cannot inspect the dbt runtime. Check file permissions.")
        else:
            status.update(f"{'Selected' if override else 'Auto-discovered'} runtime: {runtime.label}")

    @on(Button.Pressed, "#dbt-list-targets")
    def _check_setup(self) -> None:
        target_input = self.query_one("#dbt-target-override", Input)
        target_input.suggester = None
        status = self.query_one("#dbt-profile-targets", Static)
        try:
            project = self._project()
        except (DbtError, OSError) as exc:
            status.update(str(exc))
            self.query_one("#dbt-runtime-status", Static).update("")
            return
        self._check_runtime(project)
        settings = self._selection()
        try:
            profile = settings.profile(project.root, project.config)
        except (DbtError, OSError) as exc:
            status.update(str(exc))
            return
        if profile is None:
            status.update("No profile found. Choose a profiles path to validate a target, or link for offline authoring.")
            return
        names = list(profile.outputs)
        session = self.ide.session
        matched = profile.suggest(session.source.postgres or session.saved_source_profile(session.active_id) or {})
        if matched:
            names.remove(matched)
            names.insert(0, matched)
        target_input.suggester = SuggestFromList(names)
        outputs = ", ".join(names[:8]) + (" … type to complete" if len(names) > 8 else "")
        default = profile.default or "templated or invalid"
        try:
            target = profile.select(settings.override)
            resolved = f"dbt target: {target.profile} / {target.name}"
        except DbtError as exc:
            resolved = str(exc)
        status.update(f"{resolved}\nProfile default: {default} · Outputs: {outputs}"
                      + (f" · Suggested: {matched}" if matched else ""))

    @on(Button.Pressed, "#dbt-open")
    @on(Input.Submitted, "#dbt-project-dir")
    @on(Input.Submitted, "#dbt-profiles-path")
    @on(Input.Submitted, "#dbt-target-override")
    @on(Input.Submitted, "#dbt-runtime-path")
    def _open_project(self) -> None:
        try:
            project = self._project()
            settings = self._selection()
            target = settings.resolve(project.root, project.config)
            if settings.profiles_path and target:
                settings = DbtTargetSettings(str(target.profiles_dir), settings.override)
            runtime_path = self._runtime_path()
            if runtime_path:
                find_dbt_runtime(project.root, runtime_path)
            if not self.ide.set_dbt_project_dir(str(project.root), target_settings=settings, runtime_path=runtime_path):
                raise DbtError("Project home could not be saved. Check workspace storage permissions.")
        except (DbtError, OSError) as exc:
            self._message(str(exc))
        else:
            self.dismiss(str(project.root))


class DbtVarsScreen(DbtDialog):
    """Review and explicitly save the project vars mapping without running dbt."""

    BINDINGS = [Binding("ctrl+s", "save", "Save project variables", show=False)]
    DEFAULT_CSS = """
    DbtVarsScreen .dbt-dialog { height: 72%; }
    DbtVarsScreen #dbt-project-vars { height: 1fr; margin: 0 0 1 0; }
    """

    def __init__(self, home: str) -> None:
        super().__init__()
        self._home = home
        self._before: DbtFile | None = None
        self._original = ""

    def compose(self) -> ComposeResult:
        with Vertical(classes="dbt-dialog"):
            yield Static("dbt Project Variables · dbt_project.yml", markup=False)
            yield Static("Edit the YAML mapping under vars:. Package-specific variables are nested mappings. "
                         "Clear the editor to save an empty mapping.", markup=False)
            yield DraftTextArea(read_only=True, show_line_numbers=True, id="dbt-project-vars")
            yield Static("Loading project variables…", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Save (Ctrl+S)", id="dbt-vars-save", disabled=True, variant="primary", compact=True)
                yield Button("Cancel", id="dbt-close", compact=True)

    async def on_mount(self) -> None:
        try:
            self._before, self._original = await asyncio.to_thread(lambda: read_project_vars(DbtFiles(self._home)))
        except (DbtError, OSError) as exc:
            self._message(str(exc))
            return
        editor = self.query_one("#dbt-project-vars", DraftTextArea)
        editor.load_text(self._original)
        editor.read_only = False
        editor.focus()
        self.query_one("#dbt-vars-save", Button).disabled = False
        self._message("")

    @on(Button.Pressed, "#dbt-vars-save")
    async def action_save(self) -> None:
        editor = self.query_one("#dbt-project-vars", DraftTextArea)
        if (before := self._before) is None or editor.read_only:
            return
        text = editor.text
        if text == self._original:
            self.dismiss(None)
            return
        button = self.query_one("#dbt-vars-save", Button)
        editor.read_only = button.disabled = True
        try:
            saved = await asyncio.to_thread(lambda: save_project_vars(DbtFiles(self._home), before, text))
        except (DbtError, OSError) as exc:
            self._message(str(exc))
            editor.read_only = button.disabled = False
            editor.focus()
        else:
            self.dismiss(saved if saved != before else None)


@dataclass(frozen=True)
class DbtRunOptions:
    command: DbtModelCommand
    selector: str
    state_path: str
    defer: bool
    vars_override: str = field(default="", repr=False)


class DbtRunOptionsScreen(DbtDialog):
    """Choose an explicit dbt command and a selector defined in selectors.yml."""

    DEFAULT_CSS = """
    DbtRunOptionsScreen #dbt-run-options-state { margin-bottom: 1; }
    DbtRunOptionsScreen #dbt-run-options-vars { height: 5; margin: 0; }
    DbtRunOptionsScreen .dbt-buttons Button { width: 1fr; min-width: 0; }
    """
    _COMMANDS: tuple[DbtModelCommand, ...] = ("run", "build", "test")

    def __init__(self, home: str) -> None:
        super().__init__()
        self.home = home
        self._selectors: tuple[str, ...] = ()

    def compose(self) -> ComposeResult:
        with VerticalScroll(classes="dbt-dialog"):
            yield Static("Run a named dbt selection")
            yield Static("Command")
            yield Select(((command.title(), command) for command in self._COMMANDS), value="build",
                         allow_blank=False, id="dbt-run-options-command")
            yield Static("Named selector from selectors.yml")
            yield Select([], prompt="Choose a named selector", id="dbt-run-options-selector", disabled=True)
            yield Static("Local state directory (optional)")
            yield Input(placeholder="Absolute path or relative to the dbt project", id="dbt-run-options-state")
            yield Static("Defer to the state directory (requires manifest.json)")
            yield Switch(animate=False, id="dbt-run-options-defer")
            yield Static("Variable overrides (--vars) · optional YAML mapping. Do not enter secrets.", markup=False)
            yield DraftTextArea(id="dbt-run-options-vars")
            yield Static("", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Run Selection", variant="primary", id="dbt-run-options-run", disabled=True, compact=True)
                yield Button("Close", id="dbt-close", compact=True)

    def on_mount(self) -> None:
        try:
            self._selectors = dbt_named_selectors(self.home)
        except DbtError as exc:
            self._message(str(exc))
            return
        except OSError:
            self._message("Cannot read selectors.yml. Check the project and file permissions.")
            return
        if not self._selectors:
            self._message("No named selectors found. Add a selector to selectors.yml and reopen this dialog.")
            return
        selector = self.query_one("#dbt-run-options-selector", Select)
        selector.set_options((Text(name), name) for name in self._selectors)
        selector.disabled = False
        self.query_one("#dbt-run-options-run", Button).disabled = False
        selector.focus()

    @on(Button.Pressed, "#dbt-run-options-run")
    def _submit(self) -> None:
        command = self.query_one("#dbt-run-options-command", Select).value
        selector_input = self.query_one("#dbt-run-options-selector", Select)
        selector = selector_input.value
        if not isinstance(selector, str) or selector not in self._selectors:
            self._message("Choose a named selector from selectors.yml.")
            selector_input.focus()
            return
        if command not in self._COMMANDS:
            self._message("Choose Run, Build or Test.")
            return
        state_input = self.query_one("#dbt-run-options-state", Input)
        state_path = state_input.value.strip()
        defer = self.query_one("#dbt-run-options-defer", Switch).value
        if defer and not state_path:
            self._message("Defer requires a local state directory containing manifest.json.")
            state_input.focus()
            return
        vars_input = self.query_one("#dbt-run-options-vars", DraftTextArea)
        try:
            vars_override = validate_run_vars(vars_input.text)
        except DbtError as exc:
            self._message(str(exc))
            vars_input.focus()
            return
        self.dismiss(DbtRunOptions(command, selector, state_path, defer, vars_override))


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
            ("debug", "Debug dbt Connection"), ("selection", "Run Named dbt Selection"),
            ("results", "Show Last dbt Results"),
            ("vars", "Edit Project Variables"),
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
        elif value == "debug":
            self.ide.action_dbt_debug()
        elif value == "selection":
            self.ide.action_dbt_selection()
        elif value == "results":
            self.ide.action_dbt_results()
        elif value == "vars":
            self.ide.action_dbt_vars()

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
