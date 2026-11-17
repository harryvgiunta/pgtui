"""Manual local dbt import/export, with a guarded, undoable editor handoff."""
from __future__ import annotations

from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Select, Static, TextArea

from d8r.dbt import DbtError, DbtProject
from d8r.sql import SqlError

if TYPE_CHECKING:
    from .app import D8RApp


class DBTScreen(ModalScreen[None]):
    """Opening a form does not run dbt, SQL, network calls or read credentials."""

    BINDINGS = [Binding("escape", "cancel", "Close")]
    DEFAULT_CSS = """
    DBTScreen {
        align: center middle;
        background: $background 65%;
        #dbt-dialog {
            width: 108; max-width: 98%; height: 94%;
            border: thick $primary; background: $surface; padding: 1 2;
        }
        #dbt-title { text-style: bold; height: auto; }
        #dbt-form { height: 1fr; }
        #dbt-form Static { height: auto; margin-top: 1; }
        #dbt-sql { height: 16; min-height: 8; border: solid $primary; }
        #dbt-description { height: 4; border: solid $secondary; }
        .dbt-buttons { height: auto; margin-top: 1; }
        Button { margin-right: 1; }
        #dbt-message { height: auto; max-height: 6; margin-top: 1; }
    }
    """

    def __init__(self, ide: "D8RApp", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        self.target = ide._ai_target()
        self.editor = (ide.function_editor.query_one("#fn-body", TextArea)
                       if ide.function_mode else ide.editor)
        self.edit_serial = getattr(self.editor, "user_edit_serial", None)
        self.dialect = ide.session.sources[self.target.source_id].dialect
        self.project: DbtProject | None = None
        self.loaded_import = False
        self.imported_config: dict = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="dbt-dialog"):
            yield Static("dbt · local SQL/model exchange", id="dbt-title")
            with VerticalScroll(id="dbt-form"):
                yield Static("No dbt runtime, credentials, SQL execution or network. Advanced Jinja needs compiled artifacts. "
                             "Mappings must match exact manifest relation names or literal YAML sources.", markup=False)
                yield Static("Workspace project home (directory containing dbt_project.yml)")
                yield Input(self.ide.session.dbt_project_dir, placeholder="Local project directory", id="dbt-project-dir")
                yield Button("Open & save project home", id="dbt-open")
                yield Static("Import model (relative to project home)")
                yield Select([], prompt="Open a project first", id="dbt-model")
                yield Button("Load model SQL", id="dbt-load", disabled=True)
                yield Static("Model name")
                yield Input(placeholder="model_name", id="dbt-name")
                yield Static("Materialization")
                yield Select([(name, name) for name in ("view", "table", "incremental", "ephemeral")],
                             value="view", allow_blank=False, id="dbt-materialized")
                yield Static("Description")
                yield TextArea("", id="dbt-description")
                yield Static("Native SQL · review/edit before Apply or Export")
                yield TextArea(self.target.document, id="dbt-sql", show_line_numbers=True)
                yield Button("Use expanded editor SQL", id="dbt-editor")
                yield Checkbox("I consent to replacing existing model SQL and updating its YAML metadata", id="dbt-overwrite")
                yield Static("Export preserves unrelated YAML values, but YAML formatting/comments may change. "
                             "No SQL is executed. Imported SQL is applied only by the button below.", markup=False)
            yield Static("Open a project to list models. Your editor is unchanged.", id="dbt-message", markup=False)
            with Horizontal(classes="dbt-buttons"):
                yield Button("Apply imported SQL", id="dbt-apply", variant="primary", disabled=True)
                yield Button("Export model files", id="dbt-export")
                yield Button("Close", id="dbt-close")

    def _message(self, text: str) -> None:
        self.query_one("#dbt-message", Static).update(text)

    def _guard(self) -> None:
        current_editor = (self.ide.function_editor.query_one("#fn-body", TextArea)
                          if self.ide.function_mode else self.ide.editor)
        if (current_editor is not self.editor or self.ide._ai_target() != self.target
                or getattr(current_editor, "user_edit_serial", None) != self.edit_serial):
            raise DbtError("The source, page, function or editor changed. Close and reopen dbt exchange before applying or exporting.")

    def _project(self) -> DbtProject:
        directory = self.query_one("#dbt-project-dir", Input).value.strip()
        if self.project is None or str(self.project.root) != directory:
            raise DbtError("Open & save this project home first.")
        return DbtProject(directory, dialect=self.dialect)

    def _native(self, document: str) -> str:
        self._guard()
        with self.ide.session.target_source(self.target.source_id):
            outcome = self.ide.session.compile(document)
        if not outcome.ok:
            raise DbtError(outcome.error or "SQL expansion failed; fix the document before export.")
        return outcome.sql

    @on(Button.Pressed)
    def _pressed(self, event: Button.Pressed) -> None:
        event.stop()
        try:
            if event.button.id == "dbt-close":
                self.action_cancel()
            elif event.button.id == "dbt-open":
                self._guard()
                project = DbtProject(self.query_one("#dbt-project-dir", Input).value.strip(), dialect=self.dialect)
                if not self.ide.set_dbt_project_dir(str(project.root)):
                    raise DbtError("Project home could not be saved. Check workspace storage permissions.")
                self.project = project
                self.query_one("#dbt-project-dir", Input).value = str(project.root)
                self.query_one("#dbt-model", Select).set_options([(path, path) for path in project.models])
                self.query_one("#dbt-load", Button).disabled = not project.models
                self.loaded_import = False
                self.imported_config = {}
                self.query_one("#dbt-apply", Button).disabled = True
                self._message(f"Opened {project.name}: {len(project.models)} SQL models. Choose a model or export reviewed SQL.")
            elif event.button.id == "dbt-load":
                self._guard()
                path = self.query_one("#dbt-model", Select).value
                if not isinstance(path, str):
                    raise DbtError("Select a model to import.")
                imported = self._project().import_model(path)
                self.query_one("#dbt-sql", TextArea).load_text(imported.sql)
                self.query_one("#dbt-name", Input).value = imported.name
                materialized = self.query_one("#dbt-materialized", Select)
                values = ("view", "table", "incremental", "ephemeral")
                materialized.set_options([(value, value) for value in dict.fromkeys((*values, imported.materialized))])
                materialized.value = imported.materialized
                self.query_one("#dbt-description", TextArea).load_text(imported.description)
                self.loaded_import = True
                self.imported_config = imported.config
                self.query_one("#dbt-apply", Button).disabled = False
                self._message("Loaded actual compiled SQL; review before Apply." if imported.used_compiled else
                              "Resolved literal ref/source/config locally; review before Apply. Editor unchanged.")
            elif event.button.id == "dbt-editor":
                self.query_one("#dbt-sql", TextArea).load_text(self._native(self.target.document))
                self.loaded_import = False
                self.imported_config = {}
                self.query_one("#dbt-apply", Button).disabled = True
                self._message("Expanded saved functions locally. Review SQL, then Export; nothing executed.")
            elif event.button.id == "dbt-export":
                project = self._project()
                result = project.export_model(
                    self._native(self.query_one("#dbt-sql", TextArea).text),
                    self.query_one("#dbt-name", Input).value.strip(),
                    str(self.query_one("#dbt-materialized", Select).value),
                    self.query_one("#dbt-description", TextArea).text,
                    overwrite=self.query_one("#dbt-overwrite", Checkbox).value,
                    config=self.imported_config,
                )
                self.query_one("#dbt-overwrite", Checkbox).value = False
                self.project = DbtProject(project.root, dialect=self.dialect)
                self.query_one("#dbt-model", Select).set_options([(path, path) for path in self.project.models])
                self.query_one("#dbt-load", Button).disabled = not self.project.models
                self._message(f"Exported {result.sql_path.relative_to(project.root)} and {result.yaml_path.relative_to(project.root)}. Editor unchanged; nothing executed.")
            elif event.button.id == "dbt-apply":
                self._guard()
                if not self.loaded_import:
                    raise DbtError("Load and review a model before applying it.")
                if self.ide.refuse_busy("Apply dbt import"):
                    return
                self.ide.apply_dbt_import(self.query_one("#dbt-sql", TextArea).text)
                self.dismiss(None)
        except (DbtError, SqlError, OSError) as exc:
            self._message(str(exc))

    def action_cancel(self) -> None:
        self.dismiss(None)
