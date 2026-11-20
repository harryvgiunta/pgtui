"""Function explorer and form components for the shared workspace.

A table-valued function here is a named native SQL query with positional
arguments bound to unquoted @parameters. Calls expand locally; no server
function is installed. The explorer and form share the query workspace and
AI panel. Run preview saves and executes SELECT * FROM name(arguments).

Definitions are saved locally through `Session` and restored on the next launch.
Draft edits and AI Apply are autosaved separately; only Save or Run preview changes definitions.
"""

from __future__ import annotations

import json
from time import monotonic

from uuid import uuid4
from typing import TYPE_CHECKING

from textual import on
from textual.binding import Binding
from textual.message import Message
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Text
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, OptionList, Select, Static, TextArea
from textual.widgets.option_list import Option

from d8r.ai.context import AIProposal
from d8r.sql import FnDef, format_parameters, is_identifier

from .ai import AIPanel, AITarget

from .palette import CommandPalette, EditorPane, prompt
from .query_editor import QueryTextArea
from .query_folds import QueryFoldView

if TYPE_CHECKING:
    from .app import D8RApp

NEW = "＋ New function…"

HINT = "select a function · ctrl+r preview · F8 cancel · ctrl+f find · ctrl+d delete · ctrl+space complete · F4 AI complete · ctrl+q back"

PARAMETER_GUIDE = """1. Declare ordered parameters
min_id, source:relation, field:identifier
Bare names are SQL expressions.
No @ or defaults in this field.

2. Write one native SQL query
SELECT @field FROM @source
WHERE id > @min_id

Keep @parameters unquoted.
Identifier: one column/name.
Relation: table or schema.table.
For split names: @schema.@table,
declare both as identifier.
Double-quote names with spaces.
Expressions use SQL single-quoted text.

3. Save locally; preview explicitly
Save checks expansion, not server SQL.
Run preview saves, then runs the query.

4. Insert from the Functions picker
Search, Down to rows, Enter inserts.
Edit selected opens the saved draft.
Quality templates install locally only;
queries run only when you request it."""


class FnBodyPane(EditorPane):
    """Inherit the pane's priority Tab binding; navigate only without an acceptance."""

    def action_palette_tab(self) -> bool:
        if super().action_palette_tab():
            return True
        self.palette.close()
        self.screen.focus_next()
        return True


class FnExplorer(Vertical):
    """Searchable source-local call picker; editing and installation are explicit."""

    BINDINGS = [
        Binding("down", "enter_list", "Function list", show=False, priority=True),
        Binding("up", "focus_search", "Search functions", show=False, priority=True),
    ]

    DEFAULT_CSS = """
    FnExplorer #fn-search { height: 3; }
    FnExplorer .fn-library-actions { height: auto; }
    FnExplorer #fn-list { min-height: 4; height: 1fr; }
    """

    class InsertRequested(Message):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    class EditRequested(Message):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name

    class InstallQualityRequested(Message):
        pass

    def __init__(self, *, manage: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self._manage = manage
        self._functions: tuple[FnDef, ...] = ()

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search functions, parameters or descriptions", id="fn-search")
        if self._manage:
            with Horizontal(classes="fn-library-actions"):
                yield Button("Edit selected", id="fn-edit-selected", compact=True)
                yield Button("New", id="fn-new", compact=True)
            yield Button("Use current query page as function", id="query-to-fn", compact=True,
                         tooltip="Copy the current query into a new function draft; nothing is saved or run")
            yield Button("Install quality templates", id="fn-install-quality", compact=True,
                         tooltip="Store native SQL definitions locally for this source; no SQL is run; collisions cancel the entire install")
        yield OptionList(id="fn-list")
        if self._manage:
            with VerticalScroll(id="fn-param-guide"):
                yield Static("How parameters work", classes="guide-title")
                yield Static(PARAMETER_GUIDE, markup=False)
        else:
            yield Static("No matching saved functions. Add or install templates in Functions for this source.",
                         id="fn-picker-empty", markup=False)

    def focus_search(self) -> None:
        self.query_one("#fn-search", Input).focus()

    def show_functions(self, functions, selected: str = "") -> None:
        """Accept this workspace editor's library, never fetch schema or rows."""
        self._functions = tuple(functions.values() if hasattr(functions, "values") else functions)
        self._draw(selected)

    def _draw(self, selected: str = "") -> None:
        search = self.query_one("#fn-search", Input).value.casefold().split()
        rows = []
        for fn in self._functions:
            signature = format_parameters(fn) or "no arguments"
            detail = f"{signature} · {fn.doc}" if fn.doc else signature
            haystack = f"{fn.name} {detail}".casefold()
            if all(term in haystack for term in search):
                rows.append((fn.name, detail))
        if self._manage and not search:
            rows.append((NEW, "define a new function"))
        listing = self.query_one("#fn-list", OptionList)
        listing.clear_options()
        listing.add_options(Option(prompt(name, detail), id=name) for name, detail in rows)
        listing.highlighted = next((i for i, (name, _) in enumerate(rows) if name == selected),
                                   0 if rows else None)
        if self._manage:
            self.query_one("#fn-edit-selected", Button).disabled = not rows
        else:
            self.query_one("#fn-picker-empty", Static).display = not rows

    @on(Input.Changed, "#fn-search")
    def _filter(self, event: Input.Changed) -> None:
        event.stop()
        self._draw()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action in {"enter_list", "focus_search"}:
            if not self.is_mounted:
                return False
            if action == "enter_list":
                return self.query_one("#fn-search", Input).has_focus
            listing = self.query_one("#fn-list", OptionList)
            return listing.has_focus and listing.highlighted in (None, 0)
        return super().check_action(action, parameters)

    def action_enter_list(self) -> None:
        listing = self.query_one("#fn-list", OptionList)
        listing.focus()
        if listing.option_count and listing.highlighted is None:
            listing.highlighted = 0

    def action_focus_search(self) -> None:
        self.focus_search()

    @on(OptionList.OptionSelected, "#fn-list")
    def _choose(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        name = str(event.option_id or "")
        if name == NEW:
            self.post_message(self.EditRequested(""))
        elif name:
            self.post_message(self.InsertRequested(name))

    @on(Button.Pressed, "#fn-edit-selected")
    def _edit_selected(self, event: Button.Pressed) -> None:
        event.stop()
        listing = self.query_one("#fn-list", OptionList)
        if listing.highlighted is not None:
            name = str(listing.get_option_at_index(listing.highlighted).id or "")
            self.post_message(self.EditRequested("" if name == NEW else name))

    @on(Button.Pressed, "#fn-new")
    def _new(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.EditRequested(""))

    @on(Button.Pressed, "#fn-install-quality")
    def _install(self, event: Button.Pressed) -> None:
        event.stop()
        self.post_message(self.InstallQualityRequested())


class FunctionPicker(ModalScreen[FnDef | None]):
    """Reuse the source-local library for a draft-only function reference."""

    BINDINGS = [Binding("escape", "dismiss", "Back to query", show=False, priority=True)]

    DEFAULT_CSS = """
    FunctionPicker { align: center middle; }
    FunctionPicker > FnExplorer {
        width: 80;
        max-width: 95%;
        height: 65%;
        padding: 1 2;
        background: $surface;
        border: round $primary;
    }
    """

    def __init__(self, functions: tuple[FnDef, ...]) -> None:
        super().__init__()
        self._functions = {fn.name: fn for fn in functions}

    def compose(self) -> ComposeResult:
        explorer = FnExplorer(manage=False)
        explorer.border_title = "Insert a saved function into chat"
        yield explorer

    def on_mount(self) -> None:
        explorer = self.query_one(FnExplorer)
        explorer.show_functions(self._functions)
        explorer.focus_search()

    @on(FnExplorer.InsertRequested)
    def _choose(self, event: FnExplorer.InsertRequested) -> None:
        event.stop()
        self.dismiss(self._functions.get(event.name))


class FnEditor(Vertical):
    """A persistent function draft in the workspace's middle editor slot."""

    # In the SQL body, completion has priority over the form's Back action.
    BINDINGS = [Binding("escape", "back", "Back to query", show=False)]

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "back" and self.query_one(QueryFoldView).has_focus_within:
            return False  # Esc closes the fold view before leaving the function.
        return super().check_action(action, parameters)


    def __init__(self, ide: "D8RApp", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        draft = ide.session.workspace.get("function_draft")
        self.selected = draft["selected"] if draft else ""
        self.source_id = ide.session.restore_source(draft["source"], activate=False) if draft else ide.session.active_id
        self._draft_identity = draft["identity"] if draft else str(uuid4())
        self._initial_fields = draft if draft else {
            "name": "", "description": "", "parameters": "", "body": "",
            "arguments": "", "cursor": [0, 0],
        }
        self._draft_ready = False
        self._draft_timer = None
        self._draft_changed_at = 0.0
        self._draft_body: TextArea | None = None
        self._draft_inputs: dict[str, Input] = {}
        self._draft_snapshot: dict = {}
        self._leaving = False

    @property
    def session(self):
        return self.ide.session

    def compose(self) -> ComposeResult:
        from .settings import binding_tooltip

        with Horizontal(id="fn-actions", classes="pane-controls"):
            yield Static("Function", classes="pane-title")
            yield Button("Ask AI", id="fn-ai", compact=True)
            yield Button("Query", id="fn-back", compact=True,
                         tooltip=binding_tooltip(self.ide.BINDINGS, "back_or_quit", context="Preserve function draft"))
            yield Button(Text("[x]"), id="fn-close", classes="pane-close", compact=True,
                         tooltip=binding_tooltip(self.ide.BINDINGS, "back_or_quit", context="Preserve function draft"))
        with VerticalScroll(id="fn-editor"):
            yield Select(
                self.session.source_options(),
                value=self.source_id, allow_blank=False, id="fn-source", compact=True,
            )
            yield Button("Reconnect target", id="fn-reconnect", disabled=self.session.source_connected(self.source_id), compact=True)
            with Horizontal(classes="fn-field"):
                yield Static("Name")
                yield Input(self._initial_fields["name"], placeholder="monthly", id="fn-name")
            with Horizontal(classes="fn-field"):
                yield Static("Description")
                yield Input(self._initial_fields["description"], placeholder="one line on what it returns", id="fn-doc")
            with Horizontal(classes="fn-field"):
                yield Static("Parameters")
                yield Input(self._initial_fields["parameters"], placeholder="min_amount, source:relation, field:identifier", id="fn-params")
            with Horizontal(classes="fn-field"):
                yield Static("Preview arguments")
                yield Input(self._initial_fields["arguments"], placeholder="data, id, 1 (SQL values or declared names, in order)", id="fn-args")
            with FnBodyPane(id="fn-body-pane"):
                yield QueryTextArea(self._initial_fields["body"], id="fn-body", show_line_numbers=True,
                                    dialect=self.session.sources[self.source_id].dialect)
                yield QueryFoldView()
                yield CommandPalette(
                    self.session, source_id=self.source_id, workspace_actions=False,
                    parameters=self._parameter_names, id="fn-palette", markup=False,
                )
        with Horizontal(id="fn-buttons"):
            yield Button("Save", id="fn-save", variant="primary", compact=True)
            yield Button("Run preview", id="fn-preview", compact=True,
                         tooltip=binding_tooltip(self.ide.BINDINGS, "preview_function") + " · "
                         + binding_tooltip(self.ide.BINDINGS, "run_or_chat") + " / "
                         + binding_tooltip(self.ide.BINDINGS, "run") + " · Function editor focused; saves then runs")
            yield Button("Delete", id="fn-delete", variant="error", compact=True,
                         tooltip=binding_tooltip(self.ide.BINDINGS, "delete_function", context="Function editor focused"))
        yield Static("", id="fn-status", markup=False)

    def on_mount(self) -> None:
        self.preview_grid.cursor_type = "cell"
        self.query_one("#fn-reconnect").display = not self.session.source_connected(self.source_id)
        self._draft_body = self.query_one("#fn-body", TextArea)
        self._draft_body.border_title = "SQL body · @param binds its declared kind"
        self._draft_inputs = {
            key: self.query_one(selector, Input) for key, selector in (
                ("name", "#fn-name"), ("description", "#fn-doc"),
                ("parameters", "#fn-params"), ("arguments", "#fn-args"),
            )
        }
        self._draft_body.move_cursor(tuple(self._initial_fields["cursor"]))
        keys = list(self.session.functions_for(self.source_id))
        self._draw_list(keys.index(self.selected) if self.selected in keys else len(keys))
        self.query_one("#fn-delete", Button).disabled = self.selected not in self.session.functions_for(self.source_id)
        self._status("restored draft" if (self.session.workspace.get("function_draft") or {}).get("identity") == self._draft_identity
                     else f"{self.selected} · saved locally" if self.selected else "new function")
        self._draft_ready = True
        self._target_changed()
        self._queue_draft_save()

    def start_draft(self, *, focus: str = "", new_name: str = "", new_body: str = "") -> None:
        """Explicit New/query conversion starts a fresh draft on the workspace source."""
        self.flush_draft()
        self.source_id = self.session.active_id
        with self.prevent(Select.Changed):
            source = self.query_one("#fn-source", Select)
            source.set_options(self.session.source_options())
            source.value = self.source_id
        self._select(focus)
        if not focus:
            with self.prevent(Input.Changed, TextArea.Changed):
                self.query_one("#fn-name", Input).value = new_name
                self.query_one("#fn-body", TextArea).load_text(new_body)
        self._select_target(self.source_id)
        self._queue_draft_save()

    def restore_workspace(self, draft: dict | None) -> None:
        """Load this source's draft after the outgoing form has been flushed."""
        if self._draft_timer is not None:
            self._draft_timer.stop()
            self._draft_timer = None
        self.source_id = self.session.active_id
        self.query_one("#fn-body", QueryTextArea).dialect = self.session.sources[self.source_id].dialect
        self.selected = draft["selected"] if draft else ""
        self._draft_identity = draft["identity"] if draft else str(uuid4())
        self._draft_snapshot = {}
        with self.prevent(Select.Changed, Input.Changed, TextArea.Changed, TextArea.SelectionChanged):
            select = self.query_one("#fn-source", Select)
            select.set_options(self.session.source_options())
            select.value = self.source_id
            for key, widget in self._draft_inputs.items():
                widget.value = draft[key] if draft else ""
            self._draft_body.load_text(draft["body"] if draft else "")
            self._draft_body.move_cursor(tuple(draft["cursor"]) if draft else (0, 0))
        palette = self.query_one(CommandPalette)
        palette.source_id = self.source_id
        palette.close()
        self.query_one("#fn-reconnect", Button).disabled = self.session.source_connected()
        self.query_one("#fn-reconnect").display = not self.session.source_connected()
        self.query_one("#fn-delete", Button).disabled = self.selected not in self.session.functions_for(self.source_id)
        self.preview_grid.clear(columns=True)
        self._draw_list()
        self._status("restored draft" if draft else "new function")

    def on_unmount(self) -> None:
        self._leaving = True
        self.flush_draft()

    def _capture_draft(self) -> None:
        if self._draft_body is not None:
            self._draft_snapshot = {
                "identity": self._draft_identity,
                "source": self.session.source_key(self.source_id),
                "selected": self.selected,
                **{key: widget.value for key, widget in self._draft_inputs.items()},
                "body": self._draft_body.text,
                "cursor": list(self._draft_body.cursor_location),
            }

    def _queue_draft_save(self) -> None:
        if not self._draft_ready or self._leaving or not self.ide._function_started:
            return
        # Keep one timer alive during a key repeat; capture text only at flush.
        self._draft_changed_at = monotonic()
        if self._draft_timer is None:
            self._draft_timer = self.set_timer(0.2, self._save_draft_when_idle)

    def _save_draft_when_idle(self) -> None:
        self._draft_timer = None
        remaining = 0.2 - (monotonic() - self._draft_changed_at)
        if remaining > 0:
            self._draft_timer = self.set_timer(max(0.01, remaining), self._save_draft_when_idle)
        else:
            self.flush_draft()

    def flush_draft(self) -> None:
        if not self.ide._function_started:
            return
        if self._draft_timer is not None:
            self._draft_timer.stop()
            self._draft_timer = None
        self._capture_draft()
        if self._draft_snapshot and not self.ide.save_workspace(function_draft=self._draft_snapshot):
            if not self._leaving:
                self._status(self.session.workspace_error, error=True)

    def _target_changed(self) -> None:
        if self.ide.function_mode:
            self.ai_panel.target_changed()

    @on(TextArea.SelectionChanged, "#fn-body")
    def _body_caret_changed(self) -> None:
        self._queue_draft_save()

    @on(Select.Changed, "#fn-source")
    def _source_changed(self, event: Select.Changed) -> None:
        if (not self._draft_ready or not isinstance(event.value, str) or event.value != event.select.value
                or event.value == self.source_id):
            return
        if not self.ide.select_source(event.value, open_function=True):
            with self.prevent(Select.Changed):
                event.select.value = self.source_id

    def _select_target(self, source_id: str) -> None:
        self.ide._cancel_connection()
        self.source_id = source_id
        self.query_one("#fn-body", QueryTextArea).dialect = self.session.sources[self.source_id].dialect
        self.query_one("#fn-reconnect", Button).disabled = self.session.source_connected(source_id)
        self.query_one("#fn-reconnect").display = not self.session.source_connected(source_id)
        self._target_changed()
        palette = self.query_one(CommandPalette)
        palette.source_id = self.source_id
        palette.close()
        palette.sync()
        self.preview_grid.clear(columns=True)
        self._status(f"target · {self.session.sources[self.source_id].display}")
        self._queue_draft_save()

    def retarget(self, source, previous_key: str) -> None:
        """Adopt a reconnected source's registry id when the draft's target died.

        A disconnected target is registered under its identity key; the live
        source that replaces it gets its own id, and the old key is dropped.
        Nothing else re-reads `source_id`, so a Save or preview after
        reconnecting would name a key no longer in the registry.
        """
        if self.session.source_key(source.id) != previous_key:
            return
        if self.source_id != source.id and self.source_id in self.session.sources:
            return
        select = self.query_one("#fn-source", Select)
        with self.prevent(Select.Changed):
            select.set_options(self.session.source_options())
            select.value = source.id
        self._select_target(source.id)

    def _source_connected(self, source) -> None:
        if source is None:
            return
        self.ide._source_added(source)
        self.ide._open_fn()

    @on(Button.Pressed, "#fn-reconnect")
    def _reconnect_target(self) -> None:
        self.ide.reconnect_source(self.source_id, self._source_connected)

    def _parameter_names(self) -> list[str]:
        """Read this form's unsaved signature; invalid/duplicate names aren't offers."""
        names = self.query_one("#fn-params", Input).value.split(",")
        return list(dict.fromkeys(name for part in names if is_identifier(name := part.partition(":")[0].strip())))

    @on(Input.Changed, "#fn-params")
    def _parameters_changed(self) -> None:
        self.query_one(CommandPalette).close()
        self.query_one(CommandPalette).sync()


    @property
    def ai_panel(self) -> AIPanel:
        return self.ide.ai_panel

    def _ai_target(self) -> AITarget:
        name, params, body, doc = self._fields()
        return AITarget(
            (self.session.source_key(self.source_id), self._draft_identity), self.source_id, body,
            tuple(part.strip() for part in params.split(",") if part.strip()),
            json.dumps({"name": name, "description": doc, "parameters": params,
                        "arguments": self.query_one("#fn-args", Input).value}, ensure_ascii=False),
            function_name=self.selected,
        )

    def _apply_ai_draft(self, proposal: AIProposal) -> None:
        fn = proposal.function
        assert fn is not None
        self.query_one("#fn-name", Input).value = fn.name.strip()
        self.query_one("#fn-doc", Input).value = fn.doc.strip()
        self.query_one("#fn-params", Input).value = format_parameters(fn)
        self.query_one("#fn-args", Input).value = proposal.arguments
        self.query_one("#fn-body", TextArea).load_text(proposal.body)
        self.query_one(CommandPalette).close()
        self.preview_grid.clear(columns=True)
        self._status("AI draft applied · review, then Save or Run preview")


    @on(Input.Changed, "#fn-name")
    @on(Input.Changed, "#fn-doc")
    @on(Input.Changed, "#fn-params")
    @on(Input.Changed, "#fn-args")
    @on(TextArea.Changed, "#fn-body")
    def _ai_draft_changed(self) -> None:
        if self._draft_ready:
            self._target_changed()
            self._queue_draft_save()

    @on(Button.Pressed, "#fn-ai")
    def _make_with_ai(self, event: Button.Pressed) -> None:
        event.stop()
        self.query_one(CommandPalette).close()
        self.ide.action_ai()

    @on(Button.Pressed, "#fn-back")
    @on(Button.Pressed, "#fn-close")
    def _back_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_back()

    @property
    def preview_grid(self) -> DataTable:
        return self.ide.query_one("#fn-grid", DataTable)

    # -- drawing -------------------------------------------------------------

    def _draw_list(self, highlight: int | None = None) -> None:
        """Refresh the picker using the visible workspace editor's source."""
        source_id = self.source_id if self.ide.function_mode else self.session.active_id
        functions = self.session.functions_for(source_id)
        selected = self.selected
        if highlight is not None:
            keys = list(functions)
            selected = keys[highlight] if 0 <= highlight < len(keys) else NEW
        self.ide.query_one(FnExplorer).show_functions(functions, selected)

    def _select(self, name: str) -> None:
        """Highlight `name` in the list and load it (or a blank form) into the fields."""
        keys = [*self.session.functions_for(self.source_id)]
        index = keys.index(name) if name in keys else len(keys)
        self.selected = keys[index] if index < len(keys) else ""
        self._draw_list(highlight=index)
        self._load(self.selected)

    def _load(self, name: str) -> None:
        """Put one function's text in the fields; a new function shows blanks."""
        self._draft_identity = str(uuid4())
        self.query_one(CommandPalette).close()
        fn = self.session.functions_for(self.source_id).get(name)
        self.query_one("#fn-delete", Button).disabled = fn is None
        with self.prevent(Input.Changed, TextArea.Changed, TextArea.SelectionChanged):
            self.query_one("#fn-name", Input).value = fn.name if fn else ""
            self.query_one("#fn-doc", Input).value = fn.doc if fn else ""
            self.query_one("#fn-params", Input).value = format_parameters(fn) if fn else ""
            self.query_one("#fn-args", Input).value = ""
            self.query_one("#fn-body", TextArea).load_text(fn.body if fn else "")
        self.preview_grid.clear(columns=True)
        self._target_changed()
        self._queue_draft_save()
        self._status(f"{fn.name} · saved locally" if fn else "new function")

    def _status(self, text: str, error: bool = False) -> None:
        widget = self.query_one("#fn-status", Static)
        widget.update(text)
        widget.set_class(error, "error")

    # -- acting --------------------------------------------------------------

    def select_function(self, name: str) -> None:
        self.ide._open_fn(focus="" if name == NEW else name, restore_draft=False)

    def _fields(self) -> tuple[str, str, str, str]:
        return (
            self.query_one("#fn-name", Input).value,
            self.query_one("#fn-params", Input).value,
            self.query_one("#fn-body", TextArea).text,
            self.query_one("#fn-doc", Input).value,
        )

    def _save(self) -> str | None:
        """Save the fields; return the error text, or None on success."""
        if self.ide.refuse_busy("Save function"):
            return self.session.busy
        name, params_text, body, doc = self._fields()
        try:
            with self.session.target_source(self.source_id):
                fn = self.session.save_fn(name, params_text, body, doc)
        except ValueError as exc:
            self._status(str(exc), error=True)
            return str(exc)
        self.selected = fn.name
        self._draw_list(list(self.session.functions_for(self.source_id)).index(fn.name))
        self.query_one("#fn-delete", Button).disabled = False
        self._target_changed()
        self._queue_draft_save()
        self._status(f'function "{fn.name}" saved locally · {len(self.session.functions_for(self.source_id))} in library')
        return None

    @on(Button.Pressed, "#fn-save")
    def _save_pressed(self, event: Button.Pressed) -> None:
        self._save()

    @on(Button.Pressed, "#fn-preview")
    def _preview_pressed(self, event: Button.Pressed) -> None:
        self.action_preview()

    @on(Button.Pressed, "#fn-delete")
    def _delete_pressed(self, event: Button.Pressed) -> None:
        self.action_delete()

    def action_preview(self) -> None:
        """Save, then run a call with the arguments field; rows render in the grid."""
        if self.ai_panel.has_focus_within:
            return
        if self._save() is not None:
            return
        name = self.query_one("#fn-name", Input).value.strip()
        args_text = self.query_one("#fn-args", Input).value
        identity = self._draft_identity

        def publish(source, outcome) -> None:
            if self.is_mounted and self.source_id == source.id and self._draft_identity == identity:
                self._show_preview(outcome)
                self.ide._set_status(outcome.status if outcome.error else "Ready")

        self.ide.submit_sql(self.session.sources[self.source_id], "Function preview",
                            lambda source, stop: self.session.fn_preview(name, args_text, source_id=source.id),
                            publish)

    def _show_preview(self, outcome) -> None:
        grid = self.preview_grid
        grid.clear(columns=True)
        if outcome.error:
            self._status(f"preview failed · {outcome.error}", error=True)
            return
        grid.add_columns(
            *[
                Text.assemble(column, (" · " + dtype, "dim")) if dtype else Text(column)
                for column, dtype in zip(outcome.columns, outcome.dtypes)
            ]
        )
        if outcome.rows:
            grid.add_rows([[_cell(value) for value in row] for row in outcome.rows])
            grid.move_cursor(row=0, column=0)
        self._status(f"preview · {outcome.status}")

    def action_delete(self) -> None:
        if self.ai_panel.has_focus_within:
            return
        if self.ide.refuse_busy("Delete function"):
            return
        name = self.query_one("#fn-name", Input).value.strip()
        if name not in self.session.functions_for(self.source_id):
            self._status("nothing to delete — this function is not saved", error=True)
            return
        try:
            with self.session.target_source(self.source_id):
                self.session.delete_fn(name)
        except ValueError as exc:
            self._status(str(exc), error=True)
            return
        self._select("")
        self._status(f'function "{name}" deleted from local memory')

    def action_back(self) -> None:
        """Return to the query without unmounting editors or cancelling chats."""
        self.ide.return_to_query()


def _cell(value: object) -> str:
    """A cell as the preview grid shows it: NULL for nothing, compact otherwise."""
    return "NULL" if value is None else str(value)
