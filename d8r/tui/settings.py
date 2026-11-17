"""Settings workspace: persistent sidebar navigation and central category details.

Rows read live state whenever a category is drawn. Changes apply to the IDE
immediately; connection and provider forms retain their explicit Add/Save steps.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.content import Content
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, OptionList, Select, Static, TextArea
from textual.widgets.option_list import Option

from d8r.ai import client as ai_client
from d8r.ai.client import AIConfig
from d8r.ai.config import import_yolo_config

from d8r.credentials import CredentialError
from d8r.themes import THEMES, theme_label

from .add_source import AddSourceModal
from .dbt import DBTScreen
from .palette import prompt

if TYPE_CHECKING:
    from .app import D8RApp

# Sidebar order is also the keyboard navigation order.
MENUS: dict[str, str] = {
    "general": "General",
    "menus": "Show/Hide Menus",
    "theme": "Theme",
    "values": "Value cache",
    "sources": "Data source",
    "dbt": "DBT",
    "ai": "AI providers",
    "keys": "Keybindings",
}

@dataclass(frozen=True)
class Row:
    """One menu row: what it reads as, and what selecting it does."""

    label: str
    detail: str = ""
    # pane | intellisense | entry-screen | default-rows | value-cache | value-limit | clear-values |
    # source | ai | ai-auto-accept | ai-sample-data | ai-schema-refreshes | add-source.
    action: str = ""
    value: str = ""


def binding_tooltip(bindings: list[Binding], action: str, *, context: str = "") -> str:
    """Describe an actual binding, with its focus scope when it is local."""
    matches = [binding for binding in bindings if binding.action == action]
    keys = " / ".join(
        binding.key_display or "+".join(part.title() for part in binding.key.split(",")[0].split("+"))
        for binding in matches
    )
    description = matches[0].description if matches else ""
    return f"{description} · {keys}" + (f" · {context}" if context else "") if matches else ""


def binding_rows() -> list[Row]:
    """Every key the IDE answers to, read from the classes that declare them.

    Nothing is retyped here: each row is the `Binding` the widget or the app
    actually carries, so the reference cannot drift from the keymap.
    """
    from textual.app import App as TextualApp

    from .app import D8RApp
    from .results import ResultsTable
    from .palette import EditorPane
    from .ai import AIDiagnosticsScreen, AIPanel, DeleteOptionList
    from .results import ExportModal

    groups = (
        ("app", D8RApp.BINDINGS),
        ("document", EditorPane.BINDINGS),
        ("results", ResultsTable.BINDINGS),
        ("settings", SettingsScreen.BINDINGS),
        ("add source", AddSourceModal.BINDINGS),
        ("AI", AIPanel.BINDINGS),
        ("AI logs", AIDiagnosticsScreen.BINDINGS),
        ("page / conversation lists", DeleteOptionList.BINDINGS),
        ("export", ExportModal.BINDINGS),
        ("dbt exchange", DBTScreen.BINDINGS),
        ("default rows", DefaultRowsScreen.BINDINGS),
        ("value cache limit", ValueCacheLimitScreen.BINDINGS),
        ("AI provider", AIProviderScreen.BINDINGS),
        # Only what the framework still answers to: `IdeScreen` drops Textual's
        # tab/shift+tab focus keys, so listing them would be a lie.
        ("textual", TextualApp.BINDINGS),
    )
    rows: list[Row] = []
    for group, bindings in groups:
        for binding in bindings:
            key = str(binding.key)
            # Multi-key aliases (`ctrl+c,super+c`) are for terminals, not people.
            if not binding.description or "," in key:
                continue
            rows.append(Row(key, f"{binding.description} · {group}"))
    return rows


class SettingsScreen(ModalScreen[None]):
    """The `\\settings` menu; every change lands in the IDE behind it at once."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("left", "sidebar", "Categories", show=False),
        Binding("right", "details", "Details", show=False),
    ]

    def __init__(self, ide: "D8RApp", *, menu: str = "general", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        self.menu = menu
        self.rows: list[Row] = []

    @property
    def session(self):
        """The session whose state the menu reads and writes."""
        return self.ide.session

    def compose(self) -> ComposeResult:
        with Vertical(id="settings"):
            yield Static("Settings", id="settings-title")
            with Horizontal(id="settings-workspace"):
                with Vertical(id="settings-sidebar-pane"):
                    yield Static("CATEGORIES", id="settings-nav-title")
                    yield OptionList(
                        *(Option(label, id=name) for name, label in MENUS.items()),
                        id="settings-sidebar",
                    )
                with Vertical(id="settings-content"):
                    yield Static("", id="settings-heading")
                    yield Static("", id="settings-description", markup=False)
                    yield OptionList(id="settings-menu")
            yield Static("", id="settings-hint")

    def on_mount(self) -> None:
        sidebar = self.query_one("#settings-sidebar", OptionList)
        sidebar.highlighted = sidebar.get_option_index(self.menu)
        if self.menu == "theme":
            self.action_details()
        else:
            sidebar.focus()
        self._draw()

    # -- drawing ------------------------------------------------------------

    def _draw(self, highlight: int = 0) -> None:
        """Draw the current menu from live state, with one row highlighted.

        Named `_draw`, not `_render`: `Widget._render` is Textual's own hook for
        the visual a widget paints, and a screen has to leave it alone.
        """
        self.rows = self._row_source()
        menu = self.query_one("#settings-menu", OptionList)
        menu.clear_options()
        menu.add_options(
            [Option(self._row_prompt(row), id=f"row-{index}") for index, row in enumerate(self.rows)]
        )
        if self.rows:
            menu.highlighted = min(max(highlight, 0), len(self.rows) - 1)
        self.query_one("#settings-heading", Static).update(MENUS[self.menu])
        descriptions = {
            "general": "Control editor completion, the startup entry screen and default query rows. Row changes require Save in the editor.",
            "menus": "Show or hide workspace panes without changing your document.",
            "theme": "Choose a palette with Enter or a click. Colors change immediately and are saved for next time. "
                     "Nord is the default palette.",
            "values": "The schema explorer caches explicitly requested column values in memory. "
                      "Completion uses cached metadata only and never queries while typing. "
                      "Values never go to disk and are cleared at exit. Changing these settings clears cached values.",
            "sources": "Connect PostgreSQL, Cloudflare D1 or a local Parquet file. SQL dialect and engine version follow the selected source.",
            "dbt": "Exchange native SQL and local dbt models. Project home belongs to this source workspace. "
                   "No dbt installation, execution, network or credential profiles; imports require manual Apply.",
            "ai": "Provider forms require Save & use to activate changes. Permission toggles save immediately. Schema refreshes control AI metadata lookups, not IDE indexing. "
                  "Test AI Queries permits bounded read-only execution and PostgreSQL estimated plans with ANALYZE disabled; reads can still be expensive. "
                  "Plans may expose schema details and literals even when Allow Sample Data is off. Query-check failures and bounded row counts are also shared; result samples require Allow Sample Data. "
                  "Sample records per read governs query checks, not plans. Request timeout also caps checks and plans at 30 seconds. "
                  "Automatically applied edits update validated, unchanged drafts only; they never run queries or save functions. "
                  "AI completion while typing sends the current SQL window and permitted cached schema after an idle pause; "
                  "it never samples rows or auto-applies; SQL literals may contain sensitive data. F4 or Ctrl+Shift+Space requests it manually. "
                  "Disabling reads does not remove data already shared in chat.",
            "keys": "Keyboard shortcuts from the running app. This reference is read-only.",
        }
        self.query_one("#settings-description", Static).update(
            f"{descriptions[self.menu]}\nSettings: {self.session.settings_path}\n"
            "Credentials use the system keyring when remembered. Workspaces and chats remain plaintext."
            + (f"\n{self.session.credential_warning}" if self.session.credential_warning else "")
        )
        action = {
            "general": "enter / click toggle or edit",
            "menus": "enter / click toggle",
            "theme": "enter / click apply and save theme",
            "values": "enter / click toggle, edit or clear",
            "sources": "enter / click select or add",
            "dbt": "enter / click open project exchange",
            "ai": "enter / click toggle or edit",
            "keys": "read-only reference",
        }[self.menu]
        self.query_one("#settings-hint", Static).update(
            f"↑↓ navigate · {action} · tab switch panel · esc back / close"
        )

    def _row_prompt(self, row: Row) -> Content:
        if row.action != "theme":
            return prompt(row.label, row.detail)
        theme = THEMES[row.value]
        swatches = Content.assemble(*(
            ("  ", f"on {color}" if color else "") for color in
            (theme.background, theme.surface, theme.primary, theme.secondary, theme.accent)
        ))
        return Content.assemble(swatches, "  ", prompt(row.label, row.detail))

    def _row_source(self) -> list[Row]:
        if self.menu == "theme":
            return [
                Row(theme_label(name),
                    ("active · " if name == self.ide.theme else "") + ("dark" if theme.dark else "light"),
                    "theme", name)
                for name, theme in THEMES.items()
            ]
        if self.menu == "menus":
            return [
                Row(
                    title,
                    "visible" if self.ide.pane_visible(name) else "hidden",
                    "pane",
                    name,
                )
                for name, title in self.ide.PANE_TITLES.items()
            ]
        if self.menu == "values":
            return [
                Row("Cache column values", "on" if self.session.value_cache_enabled else "off", "value-cache"),
                Row("Values per column", str(self.session.value_cache_limit), "value-limit"),
                Row("Clear cache", f"{self.session.cached_value_columns} cached columns", "clear-values"),
            ]
        if self.menu == "keys":
            return binding_rows()
        if self.menu == "sources":
            return [Row("Add data source", "PostgreSQL, Cloudflare D1 or Local Parquet", "add-source")] + [
                Row(
                    label,
                    "active" if source_id == self.session.active_id else
                    f"{len(self.session.sources[source_id].datasets)} tables" if source_id in self.session.sources else
                    "select to reconnect",
                    "source",
                    source_id,
                )
                for label, source_id in self.session.source_options()
            ]
        if self.menu == "dbt":
            return [Row("Project home / import / export", self.session.dbt_project_dir or "not configured", "dbt")]
        if self.menu == "ai":
            config = self.session.ai_config
            return [
                Row("Providers", next((profile["name"] for profile in self.session.ai_providers
                                       if profile["id"] == self.session.active_ai_provider), "not configured"),
                    "ai", "ai-provider"),
                Row("Endpoint", config.base_url or "not configured", "ai", "ai-base-url"),
                Row("Model", config.model or "not configured", "ai", "ai-model"),
                Row("API key", "configured" if config.api_key else "not set", "ai", "ai-api-key"),
                Row("Request headers", "configured" if config.headers else "not set", "ai", "ai-headers"),
                Row("AI turns/tool rounds", str(config.max_tool_rounds), "ai", "ai-tool-rounds"),
                Row("Tool calls per round", str(config.max_tool_calls), "ai", "ai-tool-calls"),
                Row("Sample records per read", f"{config.sample_rows} · samples and query checks", "ai", "ai-sample-rows"),
                Row("Maximum attempts", str(config.max_attempts), "ai", "ai-attempts"),
                Row("Request timeout", f"{config.timeout:g} seconds · requests, checks and plans", "ai", "ai-timeout"),
                Row("Allow Sample Data", "on" if self.session.ai_allow_sample_data else "off", "ai-sample-data"),
                Row("Allow Schema Refreshes", "on" if self.session.ai_allow_schema_refreshes else "off", "ai-schema-refreshes"),
                Row("Test AI Queries", "on · read-only checks + PostgreSQL plans" if self.session.ai_test_queries else
                    "off · no query checks or plans", "ai-test-queries"),
                Row("Automatically Apply AI Edits", "on · validated drafts only" if self.session.ai_auto_accept else
                    "off · review and Apply manually", "ai-auto-accept"),
                Row("AI completion while typing", "on · sends SQL/cached schema after pauses" if self.session.ai_completion else
                    "off · F4 / Ctrl+Shift+Space requests manually", "ai-completion"),
            ]
        return [
            Row("Intellisense", "on" if self.session.intellisense else "off", "intellisense"),
            Row("Default rows returned", str(self.session.default_rows) if self.session.default_rows else
                "0 · safe 10,000-row preview cap", "default-rows"),
            Row("Entry screen", "on · shown at next startup" if self.session.entry_screen else
                "off · open workspace directly", "entry-screen"),
        ]

    # -- acting -------------------------------------------------------------

    @on(OptionList.OptionHighlighted, "#settings-sidebar")
    def _category_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        name = event.option.id
        if name in MENUS and name != self.menu:
            self.menu = name
            self._draw()

    @on(OptionList.OptionSelected, "#settings-sidebar")
    def _category_selected(self, event: OptionList.OptionSelected) -> None:
        self.action_details()

    def action_sidebar(self) -> None:
        self.query_one("#settings-sidebar", OptionList).focus()

    def action_details(self) -> None:
        self.query_one("#settings-menu", OptionList).focus()

    @on(OptionList.OptionSelected, "#settings-menu")
    def _selected(self, event: OptionList.OptionSelected) -> None:
        """Carry out the row, then redraw keeping its place in the list."""
        if not 0 <= event.option_index < len(self.rows):
            return
        row = self.rows[event.option_index]
        if not row.action:
            return
        if row.action == "pane":
            self.ide.toggle_pane(row.value)
        elif row.action == "theme":
            self.ide.update_settings(theme=row.value)
        elif row.action == "intellisense":
            self.ide.update_settings(intellisense=not self.session.intellisense)
        elif row.action == "entry-screen":
            self.ide.update_settings(entry_screen=not self.session.entry_screen)
        elif row.action == "ai-auto-accept":
            self.ide.update_settings(ai_auto_accept=not self.session.ai_auto_accept)
        elif row.action == "ai-completion":
            self.ide.update_settings(ai_completion=not self.session.ai_completion)
        elif row.action == "ai-sample-data":
            self.ide.update_settings(ai_allow_sample_data=not self.session.ai_allow_sample_data)
        elif row.action == "ai-schema-refreshes":
            self.ide.update_settings(ai_allow_schema_refreshes=not self.session.ai_allow_schema_refreshes)
        elif row.action == "ai-test-queries":
            self.ide.update_settings(ai_test_queries=not self.session.ai_test_queries)
        elif row.action == "value-cache":
            self.ide.update_settings(value_cache_enabled=not self.session.value_cache_enabled)
        elif row.action == "clear-values":
            self.ide.clear_value_cache()
        elif row.action == "value-limit":
            def limit_saved(_) -> None:
                self._draw(event.option_index)
                self.action_details()

            self.app.push_screen(ValueCacheLimitScreen(self.ide), limit_saved)
            return
        elif row.action == "default-rows":
            def rows_saved(_) -> None:
                self._draw(event.option_index)
                self.action_details()

            self.app.push_screen(DefaultRowsScreen(self.ide), rows_saved)
            return
        elif row.action == "dbt":
            self.app.push_screen(DBTScreen(self.ide), lambda _: self._draw(event.option_index))
            return
        elif row.action == "ai":
            self.app.push_screen(
                AIProviderScreen(self.ide, focus_field=row.value),
                lambda _: self._draw(event.option_index),
            )
            return
        elif row.action == "add-source":
            if self.ide.refuse_busy("Connect data source"):
                return
            def source_added(source) -> None:
                self.ide._source_added(source)
                self._draw(event.option_index)

            self.app.push_screen(AddSourceModal(self.session), source_added)
            return
        elif row.action == "source":
            self.ide.select_source(row.value)
        self._draw(highlight=event.option_index)

    def action_back(self) -> None:
        """Return to categories without losing the page; Escape there closes."""
        if self.query_one("#settings-sidebar", OptionList).has_focus:
            self.dismiss(None)
        else:
            self.action_sidebar()


class DefaultRowsScreen(ModalScreen[None]):
    """Default query limit, persisted only on explicit Save."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    DefaultRowsScreen { align: center middle; background: $background; }
    #rows-settings { width: 76; max-width: 100%; height: auto; max-height: 100%; padding: 1 2; border: round $accent; }
    #rows-settings Static { height: auto; margin-top: 1; }
    #rows-settings-buttons { height: auto; margin-top: 1; }
    #rows-settings-buttons Button { margin-right: 1; min-width: 10; }
    #rows-settings-error { color: $error; }
    """

    def __init__(self, ide: "D8RApp", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="rows-settings"):
            yield Static("Default rows returned")
            yield Static(
                "Caps rows fetched for the results grid; SQL is never rewritten. "
                "An authored LIMIT remains part of your query. Writes are not capped. "
                "SQL preview only expands local functions and does not check server grammar. "
                "AI sample reads keep their own limits.", markup=False,
            )
            yield Static("Whole number from 0 to 1,000,000; 0 uses the safe 10,000-row preview cap.")
            yield Input(str(self.ide.session.default_rows), type="integer", id="default-rows")
            yield Static("", id="rows-settings-error", markup=False)
            with Horizontal(id="rows-settings-buttons"):
                yield Button("Save", id="rows-settings-save", variant="primary")
                yield Button("Cancel", id="rows-settings-cancel", tooltip=binding_tooltip(self.BINDINGS, "cancel"))

    def on_mount(self) -> None:
        self.query_one("#default-rows", Input).focus()

    @on(Button.Pressed, "#rows-settings-save")
    def _save_rows(self) -> None:
        field = self.query_one("#default-rows", Input)
        try:
            value = int(field.value)
            if not 0 <= value <= 1_000_000:
                raise ValueError
        except ValueError:
            self.query_one("#rows-settings-error", Static).update(
                "Enter a whole number from 0 to 1,000,000 (0 means no default limit)."
            )
            field.focus()
            return
        try:
            self.ide.session.update_settings(default_rows=value)
        except ValueError as exc:
            self.query_one("#rows-settings-error", Static).update(str(exc))
            return
        self.dismiss(None)

    @on(Button.Pressed, "#rows-settings-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class ValueCacheLimitScreen(ModalScreen[None]):
    """Bound per-column distinct reads, persisted only on explicit Save."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    ValueCacheLimitScreen { align: center middle; background: $background; }
    #value-cache-settings { width: 76; max-width: 100%; height: auto; max-height: 100%; padding: 1 2; border: round $accent; }
    #value-cache-settings Static { height: auto; margin-top: 1; }
    #value-cache-settings-buttons { height: auto; margin-top: 1; }
    #value-cache-settings-buttons Button { margin-right: 1; min-width: 10; }
    #value-cache-settings-error { color: $error; }
    """

    def __init__(self, ide: "D8RApp", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide

    def compose(self) -> ComposeResult:
        with VerticalScroll(id="value-cache-settings"):
            yield Static("Values per column")
            yield Static(
                "Limits distinct values read for each requested column in autocomplete and the schema explorer. "
                "Changing this limit clears cached values; new values are read only when needed, never across every column "
                "at startup. Values stay in memory, never on disk, and are cleared at exit.", markup=False,
            )
            yield Static("Whole number from 1 to 10,000.")
            yield Input(str(self.ide.session.value_cache_limit), type="integer", id="value-cache-limit")
            yield Static("", id="value-cache-settings-error", markup=False)
            with Horizontal(id="value-cache-settings-buttons"):
                yield Button("Save", id="value-cache-settings-save", variant="primary")
                yield Button("Cancel", id="value-cache-settings-cancel", tooltip=binding_tooltip(self.BINDINGS, "cancel"))

    def on_mount(self) -> None:
        self.query_one("#value-cache-limit", Input).focus()

    @on(Button.Pressed, "#value-cache-settings-save")
    def _save_limit(self) -> None:
        field = self.query_one("#value-cache-limit", Input)
        try:
            value = int(field.value)
            if not 1 <= value <= 10_000:
                raise ValueError
        except ValueError:
            self.query_one("#value-cache-settings-error", Static).update(
                "Enter a whole number from 1 to 10,000."
            )
            field.focus()
            return
        try:
            self.ide.session.update_settings(value_cache_limit=value)
        except ValueError as exc:
            self.query_one("#value-cache-settings-error", Static).update(str(exc))
            return
        self.ide.value_cache_changed()
        self.dismiss(None)

    @on(Button.Pressed, "#value-cache-settings-cancel")
    def action_cancel(self) -> None:
        self.dismiss(None)


class AIProviderScreen(ModalScreen[None]):
    """Named provider drafts; discovery is explicit and never commits settings."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    AIProviderScreen { align: center middle; background: $background; }
    #ai-settings { width: 90; max-width: 100%; height: auto; max-height: 100%; padding: 1 2; border: round $accent; }
    #ai-settings Static { height: auto; margin-top: 1; }
    #ai-settings-buttons, #ai-profile-buttons, #ai-model-buttons { height: auto; margin-top: 1; }
    #ai-settings Button { margin-right: 1; min-width: 10; }
    #ai-headers { height: 5; }
    #ai-settings-error { color: $error; }
    """

    def __init__(self, ide: "D8RApp", *, focus_field: str = "ai-base-url", **kwargs) -> None:
        super().__init__(**kwargs)
        self.ide = ide
        self.focus_field = focus_field
        self._profile_id = ide.session.active_ai_provider
        self._discovery_worker = None
        self._discovery_generation = 0
        self._provider_form_closed = False
        self._delete_pending = False

    def _provider_options(self) -> list[tuple[Content, str]]:
        return [(Content(profile["name"] + (" · active" if profile["id"] == self.ide.session.active_ai_provider else "")),
                 profile["id"]) for profile in self.ide.session.ai_providers]

    def compose(self) -> ComposeResult:
        config = self.ide.session.ai_config
        name = next((profile["name"] for profile in self.ide.session.ai_providers
                     if profile["id"] == self._profile_id), "Default")
        with VerticalScroll(id="ai-settings"):
            yield Static("AI providers · OpenAI chat completions")
            yield Static(
                "Choose a saved provider or create one. Save & use activates this draft; Cancel discards it. "
                "Only Populate models contacts the endpoint from this form, sending no SQL. "
                "Remember stores API keys and header text in your system keyring; otherwise both stay session-only. "
                "Header values are visible while editing. Use HTTPS for remote providers.", markup=False,
            )
            yield Static("Saved providers")
            yield Select(self._provider_options(), value=self._profile_id or Select.NULL,
                         prompt="New provider", id="ai-provider")
            with Horizontal(id="ai-profile-buttons"):
                yield Button("New provider", id="ai-provider-new")
                yield Button("Delete provider", id="ai-provider-delete", disabled=not self._profile_id)
            yield Static("Provider name")
            yield Input(name, id="ai-provider-name")
            yield Static("Base URL (including /v1), or full /chat/completions URL")
            yield Input(config.base_url, placeholder="https://provider.example/v1", id="ai-base-url")
            yield Static("API key (token only; leave blank for header-only or unauthenticated providers)")
            yield Input(config.api_key, password=True, id="ai-api-key")
            yield Static("Request headers · one Name: value per line")
            yield TextArea(config.headers, soft_wrap=True, tab_behavior="focus", id="ai-headers")
            yield Checkbox("Remember API key and headers in system keyring", value=self.ide.session.ai_remember_key,
                           id="ai-remember-credentials")
            yield Static(self.ide.session.credential_warning, markup=False)
            with Horizontal(id="ai-model-buttons"):
                yield Button("Populate models", id="ai-models-populate")
                yield Button("Cancel lookup", id="ai-models-cancel", disabled=True)
            yield Static("Uses GET /models with the URL, key and headers above. No automatic discovery.",
                         id="ai-models-status", markup=False)
            yield Select([], prompt="Populate models to choose an ID", id="ai-model-list")
            yield Static("Model ID · choose above or enter manually")
            yield Input(config.model, placeholder="Exact provider model ID", id="ai-model")
            yield Static("AI turns/tool rounds (1–50)")
            yield Input(str(config.max_tool_rounds), type="integer", id="ai-tool-rounds")
            yield Static("Rounds count model context batches, not chat messages. After the round budget, "
                         "one final answer is requested with tools disabled.", markup=False)
            yield Static("Tool calls per round (1–16)")
            yield Input(str(config.max_tool_calls), type="integer", id="ai-tool-calls")
            yield Static("Schema, sample, history, validation reads and query checks count as tool calls; "
                         "several calls may share a round.", markup=False)
            yield Static("Sample records per read / query check (1–100)")
            yield Input(str(config.sample_rows), type="integer", id="ai-sample-rows")
            yield Static("Bounds sample reads and query-check result rows; smaller explicit query limits are honored, "
                         "including zero. Aggregates can still scan many source rows. Query checks share failures and "
                         "bounded row counts; result samples are sent only with Allow Sample Data on.", markup=False)
            yield Static("Maximum attempts (1–5; transient failures only)")
            yield Input(str(config.max_attempts), type="integer", id="ai-attempts")
            yield Static("1 initial attempt + remaining attempts as shared retries per message, "
                         "independent of the round budget. Model discovery makes one attempt.", markup=False)
            yield Static("Request timeout in seconds (without stream progress)")
            yield Input(str(config.timeout), type="number", id="ai-timeout")
            yield Static("Active streams may take longer; keepalive comments do not count as progress. "
                         "Model discovery and each context-tool read or query check have a total deadline of at most 30 seconds.", markup=False)
            yield Static("", id="ai-settings-error", markup=False)
            with Horizontal(id="ai-settings-buttons"):
                yield Button("Save & use", id="ai-settings-save", variant="primary")
                yield Button("Import yolo key", id="ai-settings-import")
                yield Button("Cancel", id="ai-settings-cancel", tooltip=binding_tooltip(self.BINDINGS, "cancel"))

    def on_mount(self) -> None:
        self.query_one(f"#{self.focus_field}").focus()

    def _read_config(self) -> AIConfig:
        try:
            return AIConfig(
                base_url=self.query_one("#ai-base-url", Input).value.strip(),
                model=self.query_one("#ai-model", Input).value.strip(),
                api_key=self.query_one("#ai-api-key", Input).value.strip(),
                headers=self.query_one("#ai-headers", TextArea).text,
                max_attempts=int(self.query_one("#ai-attempts", Input).value),
                timeout=float(self.query_one("#ai-timeout", Input).value),
                max_tool_rounds=int(self.query_one("#ai-tool-rounds", Input).value),
                max_tool_calls=int(self.query_one("#ai-tool-calls", Input).value),
                sample_rows=int(self.query_one("#ai-sample-rows", Input).value),
            )
        except ValueError:
            raise ValueError("Enter whole numbers for attempts, tool rounds, tool calls and sample records, "
                             "and a number for the timeout.") from None

    def _connection_snapshot(self) -> tuple:
        return (self._profile_id, *(self.query_one("#" + field, Input).value for field in
                                  ("ai-base-url", "ai-api-key", "ai-timeout")),
                self.query_one("#ai-headers", TextArea).text)

    def _invalidate_discovery(self, *, clear_models: bool = True) -> None:
        self._discovery_generation += 1
        if self._discovery_worker is not None:
            self._discovery_worker.cancel()
            self._discovery_worker = None
        if self.is_mounted and not self._provider_form_closed:
            self.query_one("#ai-models-populate", Button).disabled = False
            self.query_one("#ai-models-cancel", Button).disabled = True
            if clear_models:
                with self.prevent(Select.Changed):
                    self.query_one("#ai-model-list", Select).set_options([])
                self.query_one("#ai-models-status", Static).update("Populate models to list this provider's models; manual IDs also work.")

    @on(Input.Changed, "#ai-base-url, #ai-api-key, #ai-timeout")
    @on(TextArea.Changed, "#ai-headers")
    def _connection_changed(self) -> None:
        self._invalidate_discovery()
        self._delete_pending = False
        self.query_one("#ai-provider-delete", Button).label = "Delete provider"
        self.query_one("#ai-settings-error", Static).update("")

    def _load_profile(self, provider_id: str) -> None:
        self._invalidate_discovery()
        self._profile_id = provider_id
        profile = next((item for item in self.ide.session.ai_providers if item["id"] == provider_id), None)
        config = self.ide.session.ai_provider_config(provider_id) if profile else AIConfig()
        with self.prevent(Input.Changed, TextArea.Changed, Select.Changed):
            self.query_one("#ai-provider", Select).value = provider_id or Select.NULL
            self.query_one("#ai-provider-name", Input).value = profile["name"] if profile else "New provider"
            for field, value in (("ai-base-url", config.base_url), ("ai-api-key", config.api_key),
                                 ("ai-model", config.model), ("ai-tool-rounds", config.max_tool_rounds),
                                 ("ai-tool-calls", config.max_tool_calls), ("ai-sample-rows", config.sample_rows),
                                 ("ai-attempts", config.max_attempts), ("ai-timeout", config.timeout)):
                self.query_one("#" + field, Input).value = str(value)
            self.query_one("#ai-headers", TextArea).load_text(config.headers)
            self.query_one("#ai-remember-credentials", Checkbox).value = profile["remember"] if profile else True
        self._delete_pending = False
        self.query_one("#ai-provider-delete", Button).label = "Delete provider"
        self.query_one("#ai-provider-delete", Button).disabled = not provider_id
        self.query_one("#ai-settings-error", Static).update("")

    @on(Select.Changed, "#ai-provider")
    def _provider_selected(self, event: Select.Changed) -> None:
        provider_id = event.value if isinstance(event.value, str) else ""
        if provider_id != self._profile_id:
            self._load_profile(provider_id)

    @on(Button.Pressed, "#ai-provider-new")
    def _new_provider(self) -> None:
        self._load_profile("")
        self.query_one("#ai-provider-name", Input).focus()

    @on(Button.Pressed, "#ai-provider-delete")
    def _delete_provider(self) -> None:
        if not self._profile_id:
            return
        if not self._delete_pending:
            self._delete_pending = True
            self.query_one("#ai-provider-delete", Button).label = "Confirm delete"
            self.query_one("#ai-settings-error", Static).update("Delete this saved provider and its remembered credentials? Click Confirm delete.")
            return
        try:
            self.ide.session.delete_ai_provider(self._profile_id)
            self.ide.ai_chats.providers_changed()
        except ValueError as exc:
            self.query_one("#ai-settings-error", Static).update(str(exc))
            return
        with self.prevent(Select.Changed):
            self.query_one("#ai-provider", Select).set_options(self._provider_options())
        self._load_profile(self.ide.session.active_ai_provider)

    @on(Button.Pressed, "#ai-models-populate")
    def _populate_models(self) -> None:
        self._invalidate_discovery()
        try:
            config = self._read_config()
            config.validate_connection()
            if self._profile_id and any(not getattr(config, field) for field in
                                       self.ide.session.ai_provider_missing_secrets(self._profile_id)):
                raise CredentialError("Saved AI credentials are unavailable. Unlock the vault and reopen PGTUI, or re-enter the key and headers before discovery.")
        except ValueError as exc:
            self.query_one("#ai-settings-error", Static).update(str(exc))
            return
        self.query_one("#ai-settings-error", Static).update("")
        self.query_one("#ai-models-status", Static).update("Loading models…")
        self.query_one("#ai-models-populate", Button).disabled = True
        self.query_one("#ai-models-cancel", Button).disabled = False
        self._discovery_worker = self.run_worker(
            self._discover(config, self._discovery_generation, self._connection_snapshot()), exit_on_error=False,
        )

    def _discovery_current(self, generation: int, snapshot: tuple) -> bool:
        return (not self._provider_form_closed and self.is_mounted and generation == self._discovery_generation
                and snapshot == self._connection_snapshot())

    async def _discover(self, config: AIConfig, generation: int, snapshot: tuple) -> None:
        try:
            models = await ai_client.discover_models(config)
            if not self._discovery_current(generation, snapshot):
                return
            with self.prevent(Select.Changed):
                choices = self.query_one("#ai-model-list", Select)
                choices.set_options([(Content(model), model) for model in models])
                current = self.query_one("#ai-model", Input).value
                choices.value = current if current in models else Select.NULL
            self.query_one("#ai-models-status", Static).update(f"{len(models)} models available. Choose one below or enter an ID manually.")
            choices.focus()
        except asyncio.CancelledError:
            raise
        except ai_client.AIError as exc:
            if self._discovery_current(generation, snapshot):
                self.query_one("#ai-models-status", Static).update("Discovery failed; manual model entry remains available.")
                self.query_one("#ai-settings-error", Static).update(str(exc))
        except Exception:
            if self._discovery_current(generation, snapshot):
                self.query_one("#ai-settings-error", Static).update("Unable to load models. Check the provider settings or enter a model ID manually.")
        finally:
            if self._discovery_current(generation, snapshot):
                self._discovery_worker = None
                self.query_one("#ai-models-populate", Button).disabled = False
                self.query_one("#ai-models-cancel", Button).disabled = True

    @on(Button.Pressed, "#ai-models-cancel")
    def _cancel_discovery(self) -> None:
        self._invalidate_discovery()
        self.query_one("#ai-models-status", Static).update("Model lookup cancelled. Manual model entry remains available.")

    @on(Select.Changed, "#ai-model-list")
    def _model_selected(self, event: Select.Changed) -> None:
        if isinstance(event.value, str):
            self.query_one("#ai-model", Input).value = event.value

    @on(Button.Pressed, "#ai-settings-save")
    def _save_ai(self) -> None:
        try:
            config = self._read_config()
            remember = self.query_one("#ai-remember-credentials", Checkbox).value
            self.ide.session.save_ai_provider(
                self.query_one("#ai-provider-name", Input).value.strip(), config,
                provider_id=self._profile_id or None, remember=remember,
            )
        except CredentialError as exc:
            self.query_one("#ai-settings-error", Static).update(
                f"{exc} Unlock your system keyring and retry, or uncheck Remember to use session-only credentials."
            )
            return
        except ValueError as exc:
            self.query_one("#ai-settings-error", Static).update(str(exc))
            return
        self.ide.ai_chats.providers_changed()
        self.ide.notify("Provider saved and selected; credentials use the system keyring." if remember else
                        "Provider saved and selected; credentials are session-only.", title="AI provider")
        self.action_cancel()

    @on(Button.Pressed, "#ai-settings-import")
    def _import_ai(self) -> None:
        try:
            config = import_yolo_config()
        except ValueError as exc:
            self.query_one("#ai-settings-error", Static).update(str(exc))
            return
        self._load_profile("")
        self.query_one("#ai-provider-name", Input).value = "YOLO"
        self.query_one("#ai-base-url", Input).value = config.base_url
        self.query_one("#ai-model", Input).value = config.model
        self.query_one("#ai-api-key", Input).value = config.api_key
        self.query_one("#ai-settings-error", Static).update("Imported into a new draft only; Save & use commits it.")

    def on_unmount(self) -> None:
        self._provider_form_closed = True
        self._invalidate_discovery(clear_models=False)

    @on(Button.Pressed, "#ai-settings-cancel")
    def action_cancel(self) -> None:
        self._provider_form_closed = True
        self._invalidate_discovery(clear_models=False)
        self.dismiss(None)
