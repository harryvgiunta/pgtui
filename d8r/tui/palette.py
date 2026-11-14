"""Cached native SQL suggestions and cancellable, explicitly accepted AI edits."""
from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Literal

from rich.cells import cell_len
from rich.text import Text

from textual import events, on
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.message import Message
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

from d8r.completion.engine import complete
from d8r.completion.models import Signature
from .session import Session

ACTIONS: tuple[tuple[str, str, str], ...] = (
    ("Run", "ctrl+enter · execute the document", "run"),
    ("SQL preview", "ctrl+k · expand functions without executing", "compile"),
    ("Data source…", "ctrl+o · PostgreSQL, Cloudflare D1 or local Parquet", "add-source"),
    ("Functions…", "open the SQL function library", "fn"),
    ("Query to function", "create a function draft from this document", "query-to-fn"),
    ("Export results", "save selected or buffered rows as CSV", "export-results"),
    ("AI", "chat and review an AI-proposed replacement", "ai"),
    ("History", "show/hide the history pane", "history"),
    ("Pages", "show/hide the query pages pane", "toggle_pages"),
    ("Queries", "show/hide the query pages pane", "toggle_pages"),
    ("Results", "show/hide the results pane", "toggle-results"),
    ("SQL", "show/hide the SQL pane", "toggle-sql"),
    ("Schema", "show/hide the schema pane", "toggle-schema"),
    ("Settings", "every setting, full screen", "settings"),
)


@dataclass(frozen=True)
class Entry:
    label: str
    insert: str = ""
    detail: str = ""
    action: str = ""
    cursor_back: int = 0
    replace_document: bool = False
    kind: str = "keyword"
    documentation: str = ""
    start: int | None = None
    end: int | None = None
    table: str | None = None

    @property
    def identity(self) -> tuple:
        return self.label, self.insert, self.action, self.kind, self.replace_document, self.table


@dataclass(frozen=True)
class View:
    """All spans are absolute document offsets, including per-entry overrides."""
    start: int
    token: str
    entries: list[Entry] = field(default_factory=list)
    phase: Literal["command", "argument", "ai"] = "argument"
    end: int | None = None
    signature: Signature | None = None
    hint: str = ""

    @property
    def labels(self) -> list[str]:
        return [entry.label for entry in self.entries]

    def highlighted_is_typed(self, index: int) -> bool:
        entry = self.entries[index]
        return (self.phase != "ai" and not entry.action and not entry.replace_document
                and entry.insert.strip().lstrip("\\") == self.token.strip())


def prompt(label: str, detail: str = "") -> Content:
    return Content.assemble(label, ("  " + detail, "dim")) if detail else Content(label)


def _offset(document: str, location: tuple[int, int]) -> int:
    row, column = location
    return sum(len(line) + 1 for line in document.split("\n")[:row]) + column


def _location(document: str, offset: int) -> tuple[int, int]:
    offset = max(0, min(offset, len(document)))
    return document.count("\n", 0, offset), offset - document.rfind("\n", 0, offset) - 1


def view_for(session: Session, doc: str, line: str, column: int, *,
             parameters: Iterable[str] = (), row: int = 0,
             query_history: bool = True, manual: bool = False) -> View | None:
    before = line[:column]
    offset = _offset(doc, (row, column))
    action = re.fullmatch(r"\s*\\([^\\]*)", before)
    if action:
        token = action.group(1)
        entries = [Entry(label, detail=detail, action=name, kind="action")
                   for label, detail, name in ACTIONS
                   if token.casefold() in label.casefold() or token.casefold() in name.casefold()]
        return View(offset - column + before.index("\\"), token, entries, "command", offset) if entries else None
    source = session.source
    result = complete(doc, offset, session.schema, dialect=source.dialect,
                      parameters=tuple(parameters), usage=session.completion_usage(),
                      values=lambda table, column: session.cached_values(source, table, column),
                      manual=manual)
    context = result.context
    entries = [Entry(item.label, item.insert, item.detail, cursor_back=item.cursor_back,
                     kind=item.kind, documentation=item.documentation,
                     start=item.start, end=item.end, table=item.table) for item in result.items]
    if query_history and not context.blocked and (context.token or not doc.strip()):
        seen: set[str] = set()
        for item in session.history:
            if (item.doc not in seen
                    and item.doc.casefold().startswith(doc.casefold()) and item.doc != doc):
                seen.add(item.doc)
                entries.append(Entry(item.doc.replace("\n", " ")[:96], item.doc,
                                     "query history", replace_document=True, kind="history"))
                if len(seen) == 10:
                    break
    if not entries and result.signature is None and not result.hint:
        return None
    return View(context.start, context.token, entries, end=context.end,
                signature=result.signature, hint=result.hint)


class CompletionDetails(VerticalScroll):
    """A passive, scrollable preview which never steals the editor's focus."""

    can_focus = False

    def __init__(self, **kwargs) -> None:
        self._body = Static(Content(""), markup=False)
        super().__init__(self._body, **kwargs)

    @property
    def content(self) -> Content:
        return self._body.content

    def update(self, content: Content) -> None:
        self._body.update(content)

    def on_mouse_down(self, event: events.MouseDown) -> None:
        event.stop()


class CommandPalette(OptionList):
    """An unfocusable list with a bounded, plain-text documentation/preview surface."""
    can_focus = False

    class ActionPerformed(Message):
        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    class HistoryPreviewChanged(Message):
        def __init__(self, palette: CommandPalette, document: str | None) -> None:
            super().__init__()
            self.palette = palette
            self.document = document

    def __init__(self, session: Session, *, source_id: str | None = None,
                 workspace_actions: bool = True,
                 parameters: Callable[[], Iterable[str]] | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.session = session
        self.source_id = source_id
        self.workspace_actions = workspace_actions
        self.parameters = parameters
        self.editor = None
        self.details: CompletionDetails | None = None
        self._view: View | None = None
        self._preferred_width = 40
        self._dismissed = False
        self._history_dismissed = False
        self._preview_document: str | None = None
        self._sync_pending = False
        self._sync_scheduled = False
        self._placement_scheduled = False
        self._detail_layout: tuple[str, int] | None = None
        self._detail_lines = 0
        self._generation = 0
        self._ai_task: asyncio.Task | None = None
        self._ai_snapshot: tuple | None = None
        self._ai_status = ""
        self._last_document = ""
        self._edit_serial = 0
        self._accepting = False

    @property
    def is_open(self) -> bool:
        return self.has_class("open")

    @property
    def view(self) -> View | None:
        return self._view

    def attach(self, editor) -> None:
        self.editor = editor
        self._last_document = editor.text
        self._edit_serial = getattr(editor, "user_edit_serial", 0)
        with self.session.target_source(self.source_id):
            editor.dialect = self.session.source.dialect
        self.set_interval(0.1, self._guard_ai)

    def on_unmount(self) -> None:
        self.cancel_ai()

    def on_mouse_down(self, event: events.MouseDown) -> None:
        if self.editor is not None:
            self.editor.focus()
        event.stop()

    def _parameters(self) -> tuple[str, ...]:
        return tuple(self.parameters()) if self.parameters is not None else ()

    def _snapshot(self) -> tuple:
        editor = self.editor
        with self.session.target_source(self.source_id):
            source = self.session.source
            schema = self.session.schema
            schema_identity = id(schema)
            return (editor.text, editor.cursor_location, source, id(source), schema, schema_identity,
                    tuple(schema.fns), self._parameters(), self.session.ai_config,
                    getattr(self.app, "_document_identity", None),
                    getattr(getattr(self.app, "function_editor", None), "_draft_identity", None),
                    getattr(self.app, "function_mode", False), self.session.intellisense,
                    self.session.ai_completion, self.session.ai_allow_schema_refreshes,
                    getattr(editor, "user_edit_serial", 0), getattr(editor, "document_load_serial", 0),
                    getattr(editor, "selection_revision", 0))

    def _eligible(self, *, ai: bool = False) -> bool:
        editor = self.editor
        return bool(editor is not None and editor.is_mounted and editor.is_on_screen
                    and self.app.focused is editor and not editor.read_only
                    and editor.selection.start == editor.selection.end
                    and self.session.intellisense and (not ai or not self.session.busy)
                    and not self._dismissed and not getattr(self.app, "_closing", False))

    def cancel_ai(self) -> None:
        self._generation += 1
        task, self._ai_task = self._ai_task, None
        if task is not None and not task.done():
            task.cancel()
        self._ai_snapshot = None
        self._ai_status = ""
        if self._view is not None and self._view.phase == "ai":
            self._hide()

    def _guard_ai(self) -> None:
        if self._ai_snapshot is not None and (
                not self._eligible(ai=True) or self._snapshot() != self._ai_snapshot):
            self.close()

    def document_changed(self) -> None:
        self._guard_ai()
        editor = self.editor
        if editor is None or editor.text == self._last_document:
            return
        self._last_document = editor.text
        serial = getattr(editor, "user_edit_serial", 0)
        typed = (serial != self._edit_serial
                 and serial > getattr(editor, "programmatic_edit_serial", 0))
        self._edit_serial = serial
        row, column = editor.cursor_location
        if typed and column and editor.document[row][column - 1] in {"\\", "@"}:
            self._dismissed = False
            self._history_dismissed = False
        self.cancel_ai()
        self.queue_sync()
        if typed and not self._accepting and self.session.ai_completion and self._eligible():
            self.request_ai(automatic=True)

    def caret_changed(self) -> None:
        self._guard_ai()
        if self.is_open or self._sync_pending:
            self.queue_sync()

    def queue_sync(self) -> None:
        self._sync_pending = True
        if not self._sync_scheduled:
            self._sync_scheduled = True
            # Coalesce key-repeat and selection messages without delaying Tab.
            self.set_timer(0.025, self._flush_queued_sync)

    def _flush_queued_sync(self) -> None:
        self._sync_scheduled = False
        self.flush_sync()

    def flush_sync(self) -> None:
        if self._sync_pending:
            self.sync()

    def sync(self, *, respect_dismissal: bool = False, manual: bool = False) -> None:
        self._sync_pending = False
        self._guard_ai()
        editor = self.editor
        if editor is None:
            return
        if not self._eligible():
            self.close()
            return
        if self._view is not None and self._view.phase == "ai":
            return
        row, column = editor.cursor_location
        line = editor.document[row]
        if not editor.text.strip():
            self._history_dismissed = False
        with self.session.target_source(self.source_id):
            editor.dialect = self.session.source.dialect
            view = view_for(self.session, editor.text, line, column, parameters=self._parameters(),
                            row=row, query_history=self.workspace_actions and not self._history_dismissed,
                            manual=manual)
        if view is not None and not self.workspace_actions:
            view = replace(view, entries=[entry for entry in view.entries if not entry.action])
        if view is None:
            self._hide()
            self._render_details()
        else:
            self._show(view)

    def accept_table(self, table: str) -> bool:
        """Accept a registered explorer table by identity, never display spelling."""
        editor = self.editor
        if (editor is None or editor.read_only or self._dismissed or not self.session.intellisense
                or editor.selection.start != editor.selection.end):
            return False
        row, column = editor.cursor_location
        with self.session.target_source(self.source_id):
            view = view_for(self.session, editor.text, editor.document[row], column,
                            parameters=self._parameters(), row=row, query_history=False)
        choices = [] if view is None else [index for index, entry in enumerate(view.entries)
                                          if entry.table == table and entry.kind == "table"]
        if not choices:
            return False
        self.cancel_ai()
        self.screen.set_focus(editor)
        editor.scroll_cursor_visible()
        self._show(view)
        return self._accept(choices[0])

    def manual_local(self) -> None:
        self.cancel_ai()
        self._dismissed = False
        self._history_dismissed = False
        self.sync(manual=True)

    def request_ai(self, *, automatic: bool = False) -> None:
        if not automatic:
            self._dismissed = False
        self.cancel_ai()
        if not self._eligible(ai=True) or (automatic and not self.session.ai_completion):
            return
        document = self.editor.text
        if len(document.strip()) < 3:
            return
        row, column = self.editor.cursor_location
        if self.editor.document[row][:column].lstrip().startswith("\\"):
            return
        if automatic and not any(character.isspace() for character in document.strip()):
            return
        self._ai_snapshot = self._snapshot()
        generation = self._generation
        self._ai_task = asyncio.create_task(self._generate_ai(generation, automatic))

    async def _generate_ai(self, generation: int, automatic: bool) -> None:
        from d8r.ai import sql_completion
        snapshot = self._ai_snapshot
        try:
            if automatic:
                await asyncio.sleep(0.75)
            if not self._ai_current(generation, snapshot):
                return
            with self.session.target_source(self.source_id):
                source = self.session.source
                document = self.editor.text
                cursor = _offset(document, self.editor.cursor_location)
                from d8r.completion.context import analyze
                context = analyze(document, cursor, self.session.schema, dialect=source.dialect,
                                  parameters=self._parameters())
                if context.blocked:
                    return
                request = sql_completion.CompletionRequest(
                    document=document, cursor=cursor, start=context.start, end=context.end,
                    dialect=source.dialect, server_version=source.server_version,
                    schema=self.session.schema, parameters=self._parameters(),
                    allow_schema=self.session.ai_allow_schema_refreshes)
                config = self.session.ai_config
            self._ai_status = "AI · requesting suggestions… · Esc cancels"
            self._render_details()
            suggestions = await sql_completion.complete_sql(config, request)
            if not self._ai_current(generation, snapshot):
                return
            self._ai_status = "AI · review only · Tab/Enter accepts · Esc dismisses" if suggestions else "AI · no suggestions"
            if suggestions:
                entries = [Entry(f"AI suggestion {index + 1}", item.insert, item.detail,
                                 cursor_back=item.cursor_back, kind="AI",
                                 documentation=item.insert)
                           for index, item in enumerate(suggestions)]
                self._show(View(request.start, document[request.start:cursor], entries,
                                "ai", request.end))
            self._render_details()
        except asyncio.CancelledError:
            pass
        except Exception:
            if self._ai_current(generation, snapshot):
                # Provider exception text may contain URLs, credentials or response bodies.
                self._ai_status = "AI completion unavailable · check provider settings and retry"
                self._render_details()
        finally:
            if generation == self._generation:
                self._ai_task = None

    def _ai_current(self, generation: int, snapshot: tuple | None) -> bool:
        return (generation == self._generation and snapshot is not None
                and self._eligible(ai=True) and self._snapshot() == snapshot)

    def _show(self, view: View) -> None:
        identity = None
        if (self._view is not None
                and (self._view.start, self._view.token, self._view.phase) == (view.start, view.token, view.phase)
                and self.highlighted is not None and self.highlighted < len(self._view.entries)):
            identity = self._view.entries[self.highlighted].identity
        previous = self._view
        self._view = view
        if previous is None or previous.entries != view.entries:
            self.clear_options()
            prompts = [prompt(entry.label.replace("\n", " "), f"{entry.kind} · {entry.detail}")
                       for entry in view.entries]
            self._preferred_width = min(86, max(40, max((item.cell_length for item in prompts), default=40) + 5))
            self.add_options(Option(item, id=f"opt-{index}") for index, item in enumerate(prompts))
        self.highlighted = next((i for i, entry in enumerate(view.entries)
                                 if entry.identity == identity), 0) if view.entries else None
        if view.entries:
            self.add_class("open")
        else:
            self.remove_class("open")
        self.reposition()
        self.queue_reposition()

    def queue_reposition(self) -> None:
        if not self._placement_scheduled:
            self._placement_scheduled = True
            self.call_after_refresh(self._flush_reposition)

    def _flush_reposition(self) -> None:
        self._placement_scheduled = False
        self.reposition()

    def _detail_content(self) -> Content:
        parts: list = []
        view = self._view
        if self._ai_status:
            parts.append((self._ai_status + "\n", "bold"))
        if view is not None:
            if view.signature is not None:
                signature = view.signature
                parts.append((signature.label + "(", "bold"))
                for index, parameter in enumerate(signature.parameters):
                    if index:
                        parts.append(signature.separator)
                    parts.append((parameter, "bold reverse" if index == signature.active_parameter else "dim"))
                parts.append(")" + (" → " + signature.return_type if signature.return_type else "") + "\n")
                if signature.documentation:
                    parts.append(signature.documentation[:500] + "\n")
            if self.highlighted is not None and self.highlighted < len(view.entries):
                entry = view.entries[self.highlighted]
                parts.append(f"{entry.kind} · {entry.detail}\n")
                if entry.documentation:
                    parts.append(entry.documentation[:2048] + "\n")
            if view.hint:
                parts.append(view.hint)
        return Content.assemble(*parts)

    def _render_details(self) -> None:
        if self.details is not None:
            content = self._detail_content()
            if content != self.details.content:
                self.details.update(content)
            self.styles.visibility = "visible" if self._place(content) else "hidden"

    def _place(self, detail: Content) -> bool:
        pane, editor = self.parent, self.editor
        if editor is None or pane is None:
            return False
        visible = editor.content_region.intersection(pane.content_region).intersection(self.screen.region)
        caret = editor.cursor_screen_offset
        if caret not in visible or visible.width < 8:
            if self.details is not None:
                self.details.display = False
            return False
        below, above = visible.bottom - caret.y - 1, caret.y - visible.y
        below_caret = below >= min(self.option_count, 8) + 5 or below >= above
        available = below if below_caret else above
        detail_width = min(86, max((cell_len(line) for line in detail.plain.splitlines()), default=0) + 6)
        width = min(max(self._preferred_width, detail_width), visible.width)
        layout = (detail.plain, width)
        if layout != self._detail_layout:
            self._detail_layout = layout
            self._detail_lines = len(Text(detail.plain).wrap(self.app.console, max(1, width - 6)))
        detail_lines = self._detail_lines
        detail_height = min(8, max(3, detail_lines + 2), max(0, available - 3)) if detail.plain.strip() else 0
        height = min(min(self.option_count, 10) + 2, available - detail_height) if self.option_count and self.is_open else 0
        if height < 3:
            height = 0
            detail_height = min(8, available) if detail.plain.strip() else 0
        left = max(visible.x, min(caret.x, visible.right - width))
        top = caret.y + 1 if below_caret else caret.y - height - detail_height
        self.styles.offset = (left - pane.content_region.x, top - pane.content_region.y)
        self.styles.width = width
        self.styles.height = max(1, height)
        if self.details is not None:
            self.details.display = detail_height >= 2
            self.details.styles.offset = (left - pane.content_region.x, top + height - pane.content_region.y)
            self.details.styles.width = width
            self.details.styles.height = max(1, detail_height)
            self.details.border_subtitle = "Alt+Up/Down scroll" if detail_lines > max(0, detail_height - 2) else ""
        return height >= 3

    def reposition(self) -> None:
        self._render_details()
        self._publish_history_preview()

    def _hide(self) -> None:
        self._sync_pending = False
        self._view = None
        self.remove_class("open")
        if self.details is not None:
            self.details.display = False
        self._publish_history_preview()

    def close(self) -> None:
        self.cancel_ai()
        self._hide()

    @on(OptionList.OptionHighlighted)
    def _highlighted(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_list is self:
            self._render_details()
            self._publish_history_preview()

    def _publish_history_preview(self) -> None:
        document = None
        if (self.is_open and self.styles.visibility == "visible" and self._view is not None
                and self.highlighted is not None and self.highlighted < len(self._view.entries)):
            entry = self._view.entries[self.highlighted]
            if entry.replace_document:
                document = entry.insert
        if document == self._preview_document:
            return
        self._preview_document = document
        self.post_message(self.HistoryPreviewChanged(self, document).set_sender(self))

    def escape(self) -> None:
        if self._ai_snapshot is not None:
            self.close()
            self._dismissed = True
        elif self._view is not None and any(entry.replace_document for entry in self._view.entries):
            self._history_dismissed = True
            self.sync()
        elif self.is_open or (self.details is not None and self.details.display):
            self.close()
            self._dismissed = True
        else:
            self._dismissed = False
            self.sync(manual=True)

    def move(self, delta: int) -> None:
        if self.option_count:
            self.highlighted = (self.highlighted + delta) % self.option_count if self.highlighted is not None else 0
            self.scroll_to_highlight()

    def accept_highlighted(self, *, newline: bool = False) -> bool:
        self._guard_ai()
        if self._sync_pending or (self._view is not None and self._view.phase != "ai"):
            self.sync(respect_dismissal=True)
        return self._accept(self.highlighted, newline=newline) if self.highlighted is not None else False

    def accept(self, index: int, *, newline: bool = False) -> bool:
        self._guard_ai()
        if self._sync_pending or (self._view is not None and self._view.phase != "ai"):
            self.sync(respect_dismissal=True)
        return self._accept(index, newline=newline)

    def _accept(self, index: int, *, newline: bool = False) -> bool:
        view, editor = self._view, self.editor
        if view is None or not self._eligible() or not 0 <= index < len(view.entries):
            return False
        if view.phase == "ai" and (self.styles.visibility != "visible" or self.details is None or not self.details.display):
            return False
        entry = view.entries[index]
        document = editor.text
        start = view.start if entry.start is None else entry.start
        end = view.end if entry.end is None else entry.end
        end = _offset(document, editor.cursor_location) if end is None else end
        if entry.replace_document:
            start, end = 0, len(document)
        insert = "" if entry.action else entry.insert
        if not 0 <= start <= end <= len(document):
            return False
        if not entry.action and document[start:end] == insert:
            landing = start + len(insert) - entry.cursor_back
            if entry.kind == "function" and landing != _offset(document, editor.cursor_location):
                self.cancel_ai()
                editor.cursor_location = _location(document, landing)
                self.sync()
                return True
            return False
        self.cancel_ai()
        self._accepting = True
        try:
            # Separate the replacement from adjacent typing; one undo restores the original.
            editor.history.checkpoint()
            editor.replace(insert, _location(document, start), _location(document, end),
                           maintain_selection_offset=False)
            editor.cursor_location = _location(editor.text, start + len(insert) - entry.cursor_back)
            editor.history.checkpoint()
            self._last_document = editor.text
            self._edit_serial = getattr(editor, "user_edit_serial", 0)
        finally:
            self._accepting = False
        if entry.action:
            self.close()
            self.post_message(self.ActionPerformed(entry.action))
        elif view.phase == "ai":
            self.close()
        else:
            self.sync()
        return True


class EditorPane(Vertical):
    """Scope completion keys to editors, leaving results column selection intact."""
    BINDINGS = [
        Binding("up", "palette_up", "Cursor up", show=False, priority=True),
        Binding("down", "palette_down", "Cursor down", show=False, priority=True),
        Binding("enter", "palette_enter", "Newline", show=False, priority=True),
        Binding("escape", "palette_escape", "Close the palette", show=False, priority=True),
        Binding("ctrl+a", "select_all", "Select all", show=False, priority=True),
        Binding("tab", "palette_tab", "Accept suggestion", show=False, priority=True),
        Binding("ctrl+space", "complete_local", "SQL completion", show=False, priority=True),
        Binding("ctrl+shift+space,f4", "complete_ai", "AI completion", show=False, priority=True),
        Binding("alt+up", "completion_details_up", "Scroll completion details up", show=False, priority=True),
        Binding("alt+down", "completion_details_down", "Scroll completion details down", show=False, priority=True),
    ]

    @property
    def editor(self) -> TextArea:
        return self.query_one(TextArea)

    @property
    def palette(self) -> CommandPalette:
        return self.query_one(CommandPalette)

    async def on_mount(self) -> None:
        self.palette.attach(self.editor)
        details = CompletionDetails(classes="completion-details")
        details.display = False
        await self.mount(details)
        self.palette.details = details
        self.watch(self.editor, "scroll_x", self._queue_palette_placement, init=False)
        self.watch(self.editor, "scroll_y", self._queue_palette_placement, init=False)

    def _queue_palette_placement(self) -> None:
        self.palette.queue_reposition()

    def on_resize(self) -> None:
        self.call_after_refresh(self._resize_palette)

    def _resize_palette(self) -> None:
        if self.app.focused is self.editor:
            self.editor.scroll_cursor_visible()
        self.palette.reposition()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if event.widget is self.editor:
            self.palette.close()

    @on(TextArea.Changed)
    def _editor_changed(self) -> None:
        self.palette.document_changed()

    @on(TextArea.SelectionChanged)
    def _editor_caret_moved(self) -> None:
        self.palette.caret_changed()

    @on(OptionList.OptionSelected)
    def _palette_clicked(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        # AI publication never edits the document; only keyboard acceptance does.
        if self.palette.view is None or self.palette.view.phase != "ai":
            self.palette.accept(event.option_index)
        self.editor.focus()

    def action_palette_up(self) -> None:
        self.palette.flush_sync()
        if self.palette.is_open:
            self.palette.move(-1)
        else:
            self.editor.action_cursor_up()

    def action_palette_down(self) -> None:
        self.palette.flush_sync()
        if self.palette.is_open:
            self.palette.move(1)
        else:
            self.editor.action_cursor_down()

    def action_palette_enter(self) -> None:
        if self.palette.accept_highlighted(newline=True):
            return
        self.palette.close()
        if not self.editor.read_only:
            self.editor.user_edit_serial += 1
            start, end = self.editor.selection
            self.editor.replace("\n", start, end, maintain_selection_offset=False)

    def action_palette_escape(self) -> None:
        self.palette.escape()

    def action_select_all(self) -> None:
        self.editor.select_all()

    def action_palette_tab(self) -> bool:
        return self.palette.accept_highlighted()

    def action_complete_local(self) -> None:
        self.palette.manual_local()

    def action_complete_ai(self) -> None:
        self.palette.request_ai()

    def action_completion_details_up(self) -> None:
        if self.palette.details is not None and self.palette.details.display:
            self.palette.details.scroll_relative(y=-3, animate=False)

    def action_completion_details_down(self) -> None:
        if self.palette.details is not None and self.palette.details.display:
            self.palette.details.scroll_relative(y=3, animate=False)
