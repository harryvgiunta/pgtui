"""A source-bound dbt setup interview with an explicit, immutable file preview."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label, Select, Static, TextArea

from d8r.ai.client import AIError
from d8r.ai.dbt_setup import WORKLOADS, SetupAdvice, recommend_setup
from d8r.dbt import DbtError
from d8r.dbt_setup import (
    CHECKS, MATERIALIZATIONS, TARGETS, DbtScaffold, SetupInput, plan_scaffold,
    setup_directory, setup_inputs, setup_name, write_scaffold,
)
from d8r.engine import DataSource
from .add_source import AddSourceModal

if TYPE_CHECKING:
    from .app import D8RApp


@dataclass(frozen=True)
class _Review:
    plan: DbtScaffold
    source: DataSource
    source_key: str
    previous_home: str


class DbtSetupScreen(ModalScreen[tuple[str, str] | None]):
    BINDINGS = [Binding("escape", "cancel", "Close setup")]
    DEFAULT_CSS = """
    DbtSetupScreen { align: center middle; background: $background 65%; }
    #dbt-setup { width: 100; max-width: 98%; height: 95%; border: round $primary;
                 background: $surface; padding: 0 1; }
    #dbt-setup-title { text-style: bold; margin-bottom: 1; }
    #dbt-setup-questions, #dbt-setup-review, #dbt-setup-preview { height: 1fr; }
    #dbt-setup-questions Vertical { height: auto; }
    #dbt-setup Static, #dbt-setup Label { height: auto; }
    #dbt-setup Label { margin-top: 1; }
    #dbt-setup Button { margin-right: 1; }
    #dbt-setup-source-help, #dbt-setup-ai-status, #dbt-setup-review-summary { margin-bottom: 1; }
    #dbt-setup-connect { margin-top: 1; }
    #dbt-setup-message { color: $text-warning; }
    #dbt-setup-create { margin-left: 3; }
    #dbt-setup-actions { height: auto; }
    """

    def __init__(self, ide: D8RApp) -> None:
        super().__init__()
        self.ide = ide
        self.session = ide.session
        self.stage = 1
        self.target = next((key for key, value in TARGETS.items() if value.source_kind == self.session.source.kind),
                           "postgres")
        self.review: _Review | None = None
        self.written = False
        self.advice: SetupAdvice | None = None
        self._advice_snapshot: tuple | None = None
        self._advice_worker = None
        self._advice_generation = 0

    def compose(self) -> ComposeResult:
        with Vertical(id="dbt-setup"):
            yield Static("New dbt project · AI setup interview", id="dbt-setup-title")
            with VerticalScroll(id="dbt-setup-questions"):
                with Vertical(id="dbt-setup-step-1"):
                    yield Static("1 / 4 · Where should we create your project?")
                    yield Label("Which target should dbt use?")
                    yield Select([(value.label, key) for key, value in TARGETS.items()], value=self.target,
                                 allow_blank=False, id="dbt-setup-target")
                    yield Label("Project directory (empty or new, with an existing parent)")
                    yield Input(placeholder="Local project directory", id="dbt-setup-directory")
                    yield Label("Project name (lowercase letters, digits and underscores)")
                    yield Input(placeholder="e.g. sales_analytics", id="dbt-setup-name", max_length=48)
                with Vertical(id="dbt-setup-step-2"):
                    yield Static("2 / 4 · Which source should the starter models read?")
                    yield Static("", id="dbt-setup-source-help", markup=False)
                    yield Static("Reuse a connected source or open the secure connection form. Enter passwords and "
                                 "tokens only in that form; its Remember option uses the OS keyring. Adding a "
                                 "source is separate from creating the project.")
                    yield Label("Link the project to this PGTUI source")
                    yield Select([], prompt="Connect a matching source", id="dbt-setup-source")
                    yield Button("Open secure source form", id="dbt-setup-connect", compact=True)
                    yield Label("Upstream table, view or local Parquet input")
                    yield Select([], prompt="Choose the input", id="dbt-setup-relation")
                    with Vertical(id="dbt-setup-custom"):
                        yield Label("Source schema (exact name, no SQL quotes)", id="dbt-setup-schema-label")
                        yield Input("public", id="dbt-setup-schema")
                        yield Label("Source table/view (exact name, no SQL quotes)")
                        yield Input(id="dbt-setup-table")
                    yield Label("What should the starter models prioritize?")
                    yield Select([(label, key) for key, label in WORKLOADS.items()], value="explore",
                                 allow_blank=False, id="dbt-setup-workload")
                    yield Checkbox("My input should always have rows", id="dbt-setup-expects-rows")
                    yield Static("Continue asks the configured AI provider using only the target type and the last "
                                 "two choices, not your names, paths, connection details or data.")
                with Vertical(id="dbt-setup-step-3"):
                    yield Static("3 / 4 · How should we model and test the input?")
                    yield Static("", id="dbt-setup-ai-status", markup=False)
                    yield Button("Use AI choices", id="dbt-setup-use-advice", compact=True, disabled=True)
                    yield Label("Staging model name (a second model summarizes its row count)")
                    yield Input("stg_records", id="dbt-setup-model", max_length=48)
                    yield Label("Materialize the starter models as")
                    yield Select([(label, key) for key, label in MATERIALIZATIONS.items()], value="view",
                                 allow_blank=False, id="dbt-setup-materialization")
                    yield Label("What should dbt test?")
                    yield Select([(label, key) for key, label in CHECKS.items()], value="summary",
                                 allow_blank=False, id="dbt-setup-check")
                    yield Label("Optional: which column must be unique and non-null?")
                    yield Select([], prompt="No key check", id="dbt-setup-key")
            with Vertical(id="dbt-setup-review"):
                yield Static("", id="dbt-setup-review-summary", markup=False)
                yield TextArea(read_only=True, id="dbt-setup-preview", tab_behavior="focus")
            yield Static("", id="dbt-setup-message", markup=False)
            with Horizontal(id="dbt-setup-actions"):
                yield Button("Cancel", id="dbt-setup-cancel")
                yield Button("Back", id="dbt-setup-back")
                yield Button("Continue", id="dbt-setup-next", variant="primary")
                yield Button("Create & Link", id="dbt-setup-create", variant="success")

    def on_mount(self) -> None:
        self._show_step(1)

    def _show_step(self, step: int) -> None:
        self.stage = step
        for index in (1, 2, 3):
            self.query_one(f"#dbt-setup-step-{index}").display = step == index
        questions = self.query_one("#dbt-setup-questions", VerticalScroll)
        questions.display = step != 4
        questions.scroll_home(animate=False)
        self.query_one("#dbt-setup-review").display = step == 4
        self.query_one("#dbt-setup-back", Button).disabled = step == 1 or self.written
        next_button = self.query_one("#dbt-setup-next", Button)
        next_button.display = step != 4
        next_button.label = "Preview scaffold" if step == 3 else "Continue"
        self.query_one("#dbt-setup-create", Button).display = step == 4
        self._message("")
        control = ("target", "source", "model", "preview")[step - 1]
        self.query_one(f"#dbt-setup-{control}").focus()

    def _message(self, text: str) -> None:
        widget = self.query_one("#dbt-setup-message", Static)
        widget.update(text)
        widget.display = bool(text)

    def _value(self, field: str) -> str:
        value = self.query_one(f"#dbt-setup-{field}", Select).value
        return value if isinstance(value, str) else ""

    def _input(self, field: str) -> str:
        return self.query_one(f"#dbt-setup-{field}", Input).value.strip()

    def _sources(self, preferred: str = "") -> None:
        picker = self.query_one("#dbt-setup-source", Select)
        preferred = preferred or self._value("source") or self.session.active_id
        options = {source.id: Content(source.display) for source in self.session.sources.values()
                   if source.kind == TARGETS[self.target].source_kind and self.session.source_connected(source.id)}
        picker.set_options((label, key) for key, label in options.items())
        picker.value = preferred if preferred in options else next(iter(options), Select.NULL)
        self._refresh_relations()

    def _selected_source(self) -> DataSource:
        source = self.session.sources.get(self._value("source"))
        if (source is None or source.kind != TARGETS[self.target].source_kind
                or not self.session.source_connected(source.id)):
            raise DbtError("Connect or select a source for this target first.")
        return source

    @on(Select.Changed, "#dbt-setup-source")
    def _source_changed(self, event: Select.Changed) -> None:
        if self.stage == 2 and event.value == event.select.value:
            self._refresh_relations()

    def _refresh_relations(self) -> None:
        picker = self.query_one("#dbt-setup-relation", Select)
        previous = picker.value
        try:
            source = self._selected_source()
        except DbtError:
            picker.set_options([])
        else:
            options = [(Content(name), name) for name in setup_inputs(source)]
            if self.target != "duckdb":
                options.append((Content("Specify a relation not in the catalog"), -1))
                self.query_one("#dbt-setup-schema", Input).value = source.default_schema or "public"
            picker.set_options(options)
            values = [value for _, value in options]
            picker.value = previous if previous in values else next(iter(values), Select.NULL)
        self._custom_relation()

    @on(Select.Changed, "#dbt-setup-relation")
    def _custom_relation(self) -> None:
        self.query_one("#dbt-setup-custom").display = self.query_one("#dbt-setup-relation", Select).value == -1
        postgres = self.target == "postgres"
        self.query_one("#dbt-setup-schema-label").display = postgres
        self.query_one("#dbt-setup-schema").display = postgres

    def _upstream(self) -> SetupInput:
        source = self._selected_source()
        chosen = self.query_one("#dbt-setup-relation", Select).value
        if chosen == -1 and self.target != "duckdb":
            schema = "main" if self.target == "d1" else self.query_one("#dbt-setup-schema", Input).value
            table = self.query_one("#dbt-setup-table", Input).value
            if schema.strip() and table.strip():
                return SetupInput(schema, table)
        elif entry := setup_inputs(source).get(chosen):
            return entry
        raise DbtError("Choose a relation or enter the exact upstream table name and schema.")

    @on(Button.Pressed, "#dbt-setup-connect")
    def _connect(self) -> None:
        if self.stage != 2 or self.ide.refuse_busy("Connect a dbt source"):
            return

        def added(source: DataSource | None) -> None:
            if source is not None:
                self.ide._source_added(source, activate=False)
                if self.is_mounted and self.stage == 2:
                    self._sources(source.id)
                    self._message("")

        backend = "parquet" if self.target == "duckdb" else self.target
        self.app.push_screen(AddSourceModal(self.session, backend=backend), added)

    def _cancel_advice(self) -> None:
        self._advice_generation += 1
        if self._advice_worker is not None:
            self._advice_worker.cancel()
            self._advice_worker = None

    def _advice_key(self) -> tuple:
        source_id = self._value("source")
        return (self.target, self._value("workload"), self.query_one("#dbt-setup-expects-rows", Checkbox).value,
                id(self.session.sources.get(source_id)), self.session.source_connected(source_id),
                self.session.ai_provider_revision, self.session.ai_config)

    def _request_advice(self) -> None:
        self._cancel_advice()
        self.advice = None
        self._advice_snapshot = None
        self.query_one("#dbt-setup-use-advice", Button).disabled = True
        status = self.query_one("#dbt-setup-ai-status", Static)
        config = self.session.ai_config
        try:
            config.validate()
        except ValueError:
            status.update("AI guidance is unavailable. Configure Settings → AI providers or use the choices below.")
            return
        status.update("Asking the assistant for model and test recommendations…")
        generation = self._advice_generation
        snapshot = self._advice_key()
        target, workload, expects_rows = snapshot[:3]

        async def request() -> None:
            try:
                advice = await recommend_setup(config, target, workload, expects_rows)
            except (AIError, ValueError, TimeoutError):
                advice = None
            if generation != self._advice_generation or not self.is_mounted or self.stage != 3:
                return
            self._advice_worker = None
            if snapshot != self._advice_key():
                advice = None
            self.advice = advice
            if advice is None:
                status.update("AI guidance unavailable for this step. You can choose the models and tests below.")
            else:
                self._advice_snapshot = snapshot
                status.update(f"AI recommendation: {MATERIALIZATIONS[advice.materialization]}. {CHECKS[advice.check]}.")
                self.query_one("#dbt-setup-use-advice", Button).disabled = False

        self._advice_worker = self.run_worker(request, name="dbt setup guidance", exit_on_error=False)

    @on(Button.Pressed, "#dbt-setup-use-advice")
    def _use_advice(self) -> None:
        if self.stage != 3 or self.advice is None:
            return
        if self._advice_snapshot != self._advice_key():
            self.query_one("#dbt-setup-use-advice", Button).disabled = True
            self.query_one("#dbt-setup-ai-status", Static).update("The provider or source changed. Choose settings below.")
            return
        self.query_one("#dbt-setup-materialization", Select).value = self.advice.materialization
        self.query_one("#dbt-setup-check", Select).value = self.advice.check

    @on(Button.Pressed, "#dbt-setup-back")
    def _back(self) -> None:
        if self.stage > 1 and not self.written:
            self._cancel_advice()
            self.review = None
            self._show_step(self.stage - 1)

    def _current_home(self, source_key: str) -> str:
        return self.session.workspace["source_workspaces"].get(source_key, {}).get("dbt_project_dir", "")

    def _preview(self) -> None:
        source = self._selected_source()
        key = self.session.source_key(source.id)
        plan = plan_scaffold(self._input("directory"), self._input("name"), self.target, self._upstream(),
                             model=self._input("model"), materialization=self._value("materialization"),
                             check=self._value("check"), key=self._value("key"))
        previous_home = self._current_home(key)
        self.review = _Review(plan, source, key, previous_home)
        note = f"4 / 4 · Review {len(plan.files)} new files in {plan.directory.path}\nLink source: {source.display}."
        if previous_home:
            note += f" This replaces the workspace link to {previous_home}, not its files."
        note += "\nScroll to review all contents, including the README for external verification. Nothing will run."
        if self._advice_worker is not None:
            self._cancel_advice()
            self.query_one("#dbt-setup-ai-status", Static).update("Continued without AI guidance for this preview.")
        self.query_one("#dbt-setup-review-summary", Static).update(note)
        self.query_one("#dbt-setup-preview", TextArea).load_text(plan.preview())
        self._show_step(4)

    def _apply(self) -> None:
        review = self.review
        if review is None or self.ide.refuse_busy("Create dbt project"):
            return
        home = str(review.plan.directory.path)
        valid_homes = {review.previous_home, home} if self.written else {review.previous_home}
        source_id = review.source.id
        if (self.session.sources.get(source_id) is not review.source or not self.session.source_connected(source_id)
                or self.session.source_key(source_id) != review.source_key
                or self._current_home(review.source_key) not in valid_homes):
            raise DbtError("The selected source or project link changed. Reopen setup to review the target.")
        if not self.written:
            write_scaffold(review.plan)
            self.written = True
        if not self.ide.link_new_dbt_project(review.source_key, home):
            self.query_one("#dbt-setup-back", Button).disabled = True
            self.query_one("#dbt-setup-create", Button).label = "Retry Link"
            self.query_one("#dbt-setup-cancel", Button).label = "Close"
            raise DbtError("Project files were created, but the workspace link was not saved. Retry Link or open "
                           "the project home manually after fixing workspace storage.")
        self.dismiss((source_id, home))

    @on(Button.Pressed, "#dbt-setup-create")
    def _approve(self) -> None:
        if self.stage == 4:
            try:
                self._apply()
            except (DbtError, OSError) as exc:
                self._message(str(exc))

    @on(Button.Pressed, "#dbt-setup-next")
    def _next(self) -> None:
        try:
            if self.stage == 1:
                setup_name(self._input("name"), "Project name")
                setup_directory(self._input("directory"))
                self.target = self._value("target")
                self.query_one("#dbt-setup-source-help", Static).update(TARGETS[self.target].profile_help)
                self._sources()
                self._show_step(2)
            elif self.stage == 2:
                upstream = self._upstream()
                picker = self.query_one("#dbt-setup-key", Select)
                previous = picker.value
                picker.set_options((Content(name), name) for name in upstream.columns)
                picker.value = previous if previous in upstream.columns else Select.NULL
                self._show_step(3)
                self._request_advice()
            elif self.stage == 3:
                self._preview()
        except (DbtError, OSError) as exc:
            self._message(str(exc))

    @on(Button.Pressed, "#dbt-setup-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_unmount(self) -> None:
        self._cancel_advice()
