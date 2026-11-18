"""Keyboard conventions shared by the workspace's multiline draft editors."""
from textual.binding import Binding
from textual.widgets import TextArea


class DraftTextArea(TextArea):
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all text", show=False),
        # Some macOS terminals send Esc-b / Esc-f for Option+Left / Right.
        Binding("alt+left,alt+b", "cursor_word_left", "Previous word", show=False, priority=True),
        Binding("alt+right,alt+f", "cursor_word_right", "Next word", show=False, priority=True),
        Binding("alt+up", "cursor_page_up", "Page up", show=False),
        Binding("alt+down", "cursor_page_down", "Page down", show=False),
    ]
