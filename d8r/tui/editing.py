"""Input conventions shared by the workspace's multiline draft editors."""
from textual import events
from textual.binding import Binding
from textual.widgets import TextArea
from textual.widgets.text_area import Selection


class DraftTextArea(TextArea):
    BINDINGS = [
        Binding("ctrl+a", "select_all", "Select all text", show=False),
        # Some macOS terminals send Esc-b / Esc-f for Option+Left / Right.
        Binding("alt+left,alt+b", "cursor_word_left", "Previous word", show=False, priority=True),
        Binding("alt+right,alt+f", "cursor_word_right", "Next word", show=False, priority=True),
        Binding("alt+up", "cursor_page_up", "Page up", show=False),
        Binding("alt+down", "cursor_page_down", "Page down", show=False),
    ]

    def on_click(self, event: events.Click) -> None:
        if event.button != 1 or event.chain not in (2, 3) or event.shift or event.ctrl or event.meta:
            return
        offset = event.get_content_offset(self)
        if offset is None or offset.x < self.gutter_width or offset.y + self.scroll_offset.y >= self.wrapped_document.height:
            return
        event.prevent_default()
        event.stop()
        if not self.selection.is_empty:
            return  # A release after a drag is also a Click; keep the dragged range.

        # TextArea's mouse-down already mapped the clicked cell through tabs,
        # wrapping, scrolling and folds to this document location.
        row, column = self.cursor_location
        if event.chain == 3:
            self.select_line(row)
            return
        line = self.document[row]
        if column == len(line) or line[column].isspace():
            return
        start, end = column, column + 1
        while start and not line[start - 1].isspace():
            start -= 1
        while end < len(line) and not line[end].isspace():
            end += 1
        self.selection = Selection((row, start), (row, end))
        self.record_cursor_width()
