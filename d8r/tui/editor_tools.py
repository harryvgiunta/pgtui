"""Local, draft-only TextArea tools. No Session, database or persistence dependencies."""
from __future__ import annotations

from collections.abc import Callable

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea
from textual.widgets.text_area import Location, Selection

from .sql_layout import LayoutEdit, layout_edits


def _mapped(index: int, edits: list[LayoutEdit]) -> int:
    delta = 0
    for edit in edits:
        if index < edit.start:
            break
        if index < edit.end:
            return edit.start + delta + min(index - edit.start, len(edit.whitespace))
        delta += len(edit.whitespace) - (edit.end - edit.start)
    return index + delta


def format_editor(editor: TextArea, dialect: str) -> bool:
    """Format the selection (or document) in one undo step, retaining caret/selection."""
    if editor.read_only:
        return False
    doc = editor.document
    anchor, caret = (doc.get_index_from_location(loc) for loc in editor.selection)
    text = editor.text
    start, end = sorted((anchor, caret)) if anchor != caret else (0, len(text))
    edits = list(layout_edits(text, dialect, start, end))
    if not edits:
        return False
    pos = edits[0].start
    chunks: list[str] = []
    for edit in edits:
        chunks.extend((text[pos:edit.start], edit.whitespace))
        pos = edit.end
    first = doc.get_location_from_index(edits[0].start)
    last = doc.get_location_from_index(pos)
    editor.history.checkpoint()
    editor.replace("".join(chunks), first, last, maintain_selection_offset=False)
    editor.selection = Selection(*(doc.get_location_from_index(_mapped(i, edits)) for i in (anchor, caret)))
    editor.history.checkpoint()
    return True


Reveal = Callable[[TextArea, Location, Location], None]


class FindBar(Horizontal):
    """Mount once outside the editor pane. The host supplies the active SQL editor.

    Call open(editor) for Ctrl+F, open(editor, from_toolbar=True) for a button,
    and close(restore_focus=False) on document/page/source switches. Optional
    reveal(editor, start, end) must synchronously unfold the hit before navigation.
    """

    BINDINGS = [
        Binding("shift+enter,shift+f3", "previous", "Previous match", show=False),
        Binding("f3", "next", "Next match", show=False),
        Binding("escape", "dismiss", "Close find", show=False, priority=True),
    ]
    DEFAULT_CSS = """
    FindBar { display: none; height: 1; }
    FindBar > Input { width: 1fr; height: 1; border: none; padding: 0 1; }
    FindBar > Button { width: auto; min-width: 4; padding: 0 1; }
    FindBar > Static { width: auto; padding: 0 1; }
    """

    _editor: TextArea | None = None
    _reveal: Reveal | None = None
    _anchor = 0

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Find literal text", id="find-query", compact=True)
        yield Static("", id="find-status")
        yield Button("Prev", id="find-prev", compact=True)
        yield Button("Next", id="find-next", compact=True)
        yield Button("Close", id="find-close", compact=True)

    def _available(self, editor: TextArea) -> bool:
        return (self.is_mounted and editor.is_mounted and self.screen is self.app.screen
                and editor.screen is self.screen and not isinstance(self.screen, ModalScreen)
                and editor.focusable and all(node.display for node in editor.ancestors_with_self))

    def open(self, editor: TextArea, *, reveal: Reveal | None = None, from_toolbar: bool = False) -> bool:
        if not self._available(editor):
            return False
        if not (from_toolbar or self.screen.focused is editor or self.has_focus_within):
            return False  # A priority binding must not pull focus out of another input.
        field = self.query_one(Input)
        if editor is not self._editor or not self.has_focus_within:
            self._anchor = editor.document.get_index_from_location(min(editor.selection))
            selected = editor.selected_text
            field.value = selected if len(selected) <= 512 and not any(c in selected for c in "\r\n") else ""
        self._editor, self._reveal = editor, reveal
        self.display = True
        self.screen.set_focus(field)
        field.action_select_all()
        self._search(initial=True)
        return True

    def close(self, restore_focus: bool = True) -> None:
        editor = self._editor
        self._editor = self._reveal = None
        self.display = False
        if self.is_mounted:
            self.query_one(Input).value = ""
        if restore_focus and editor is not None and self._available(editor):
            self.screen.set_focus(editor)

    def _search(self, *, previous: bool = False, initial: bool = False) -> None:
        editor = self._editor
        if editor is None or not self._available(editor) or not self.has_focus_within:
            return
        needle = self.query_one(Input).value
        status = self.query_one("#find-status", Static)
        if not needle:
            status.update("")
            return
        doc = editor.document
        text = editor.text
        if initial:
            origin = self._anchor
        else:
            location = min(editor.selection) if previous else max(editor.selection)
            origin = doc.get_index_from_location(location)
        hit = text.rfind(needle, 0, origin) if previous else text.find(needle, origin)
        wrapped = hit == -1
        if wrapped:
            hit = text.rfind(needle) if previous else text.find(needle)
        if hit == -1:
            status.update("No match")
            return
        start = doc.get_location_from_index(hit)
        end = doc.get_location_from_index(hit + len(needle))
        if self._reveal is not None:
            self._reveal(editor, start, end)
        editor.selection = Selection(start, end)
        editor.scroll_cursor_visible(center=True, animate=False)
        status.update("Wrapped" if wrapped else "Match")

    @on(Input.Changed)
    def _changed(self, event: Input.Changed) -> None:
        event.stop()
        if event.value == event.input.value:
            self._search(initial=True)

    @on(Input.Submitted)
    def _submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.action_next()

    @on(Button.Pressed)
    def _button(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "find-close":
            self.close()
        else:
            self._search(previous=event.button.id == "find-prev")
            self.screen.set_focus(self.query_one(Input))

    def action_next(self) -> None:
        self._search()

    def action_previous(self) -> None:
        self._search(previous=True)

    def action_dismiss(self) -> None:
        self.close()
