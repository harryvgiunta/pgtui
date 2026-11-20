"""Read-only SQL fold projection. Mount one QueryFoldView beside its QueryTextArea.

All offsets point into the editor's current document. No folded text enters a
TextArea, the undo history, or an app-level Changed/SelectionChanged event.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, OptionList

from d8r.sql import SqlError, cte_declarations, lex_sql, split_statements

if TYPE_CHECKING:
    from .query_editor import QueryTextArea

_NEWLINE = re.compile(r"\r\n|\n|\r")


@dataclass(frozen=True)
class Section:
    start: int
    end: int
    title: str


def sections(sql: str, dialect: str) -> list[Section]:
    """Exact nested source ranges. Reuse native routine/trigger statement boundaries."""
    tokens = tuple(t for t in lex_sql(sql, dialect=dialect) if t.kind not in {"whitespace", "comment"})
    result = []
    cursor = 0
    for n, statement in enumerate(split_statements(sql, dialect=dialect), 1):
        # split_statements strips surrounding whitespace and the terminating ';'.
        # Ignore empty/comment-only statements using token offsets. Searching
        # for statement text could match identical SQL inside an earlier comment.
        first = bisect_left(tokens, cursor, key=lambda t: t.start)
        while tokens[first].text == ";":
            cursor = tokens[first].end
            first += 1
        while sql[cursor].isspace():
            cursor += 1
        start = cursor
        end = start + len(statement)
        after = end
        while after < len(sql) and sql[after].isspace():
            after += 1
        cursor = after + 1 if sql.startswith(";", after) else end
        # A summary must not echo a whole one-line query when collapsed.
        verb = tokens[first].text.upper() if tokens[first].kind == "word" else "SQL"
        result.append(Section(start, cursor, f"Statement {n}: {verb}"))

    ctes = {opening: name for name, opening in cte_declarations(tokens)}
    stack: list[int] = []
    for i, token in enumerate(tokens):
        if token.text == "(":
            stack.append(i)
        elif token.text == ")" and stack:
            opening = stack.pop()
            start = tokens[opening].start
            if opening in ctes:
                name = tokens[ctes[opening]]
                title, start = f"CTE {' '.join(name.text.splitlines())}", name.start
            elif (tokens[opening + 1].kind == "word" and tokens[opening + 1].text.upper()
                  in {"SELECT", "WITH", "TABLE", "VALUES", "FROM"}):
                title = "Subquery"
            else:
                continue
            if _NEWLINE.search(sql, start, token.end):
                result.append(Section(start, token.end, title))
    return sorted(result, key=lambda r: (r.start, -r.end))


@dataclass(frozen=True)
class _Row:
    offset: int
    section: Section | None
    header: bool
    label: Text


def _rows(sql: str, regions: list[Section], collapsed: set[Section], starts: list[int]) -> list[_Row]:
    rows: list[_Row] = []
    parents: list[Section] = []
    cursor = 0

    def emit_to(end: int) -> None:
        nonlocal cursor
        while cursor < end:
            row = bisect_right(starts, cursor) - 1
            stop = min(end, starts[row + 1]) if row + 1 < len(starts) else end
            value = sql[cursor:stop].rstrip("\r\n")
            # Skip whitespace-only gaps between sections, not blank rows in SQL.
            if value.strip() or (parents and cursor == starts[row]):
                label = Text(f"{row + 1:>4}  ", style="dim")
                label.append(value, style="none")
                rows.append(_Row(cursor, parents[-1] if parents else None, False, label))
            cursor = stop

    for region in regions:
        if region.start < cursor:  # covered by a collapsed ancestor
            continue
        while parents and parents[-1].end <= region.start:
            emit_to(parents[-1].end)
            parents.pop()
        emit_to(region.start)
        first = bisect_right(starts, region.start)
        last = bisect_right(starts, region.end - 1)
        mark = "▸" if region in collapsed else "▾"
        label = Text(f"{first:>4}  {'  ' * min(len(parents), 8)}{mark} {region.title}  [L{first}–{last}]", style="bold")
        rows.append(_Row(region.start, region, True, label))
        if region in collapsed:
            cursor = region.end
        else:
            parents.append(region)
    while parents:
        emit_to(parents[-1].end)
        parents.pop()
    emit_to(len(sql))
    return rows


class QueryFoldView(Vertical):
    """Browse/toggle individual regions. Esc preserves selection; Edit here moves it."""

    DEFAULT_CSS = """
    QueryFoldView { display: none; height: 1fr; }
    QueryFoldView > OptionList { height: 1fr; }
    QueryFoldView > Horizontal { height: 1; }
    QueryFoldView Button { min-width: 0; margin-right: 1; }
    """
    BINDINGS = [
        Binding("space", "toggle_section", "Toggle section", show=False),
        Binding("e", "edit_here", "Edit here", show=False),
        Binding("escape", "return_to_editor", "Back to editor", show=False),
    ]

    def compose(self) -> ComposeResult:
        options = OptionList(compact=True)
        options.tooltip = "Up/Down: navigate · Enter or click: toggle section / edit line · Esc: back"
        yield options
        with Horizontal():
            yield Button("Toggle (Space)", compact=True, classes="fold-toggle")
            yield Button("Edit here (E)", compact=True, classes="fold-edit")
            yield Button("Back (Esc)", compact=True, classes="fold-back")

    def open_for(self, editor: QueryTextArea) -> bool:
        sql = editor.text
        try:
            regions = sections(sql, editor.dialect)
        except SqlError:
            self.notify("Finish the open SQL quote or comment before folding.")
            return False
        if not regions:
            self.notify("No SQL sections to fold.")
            return False
        self._sql = sql
        self._regions = regions
        self._starts = [0, *(m.end() for m in _NEWLINE.finditer(sql))]
        # Show sections inside a statement at entry; collapse all other bodies.
        self._collapsed = {r for i, r in enumerate(regions) if not (
            r.title.startswith("Statement ") and i + 1 < len(regions) and regions[i + 1].start < r.end)}
        self._editor = editor
        self.display = True
        editor.display = False
        self._render_rows(editor.document.get_index_from_location(editor.cursor_location))
        self.query_one(OptionList).focus()
        return True

    def _render_rows(self, offset: int, section: Section | None = None) -> None:
        self._rows = _rows(self._sql, self._regions, self._collapsed, self._starts)
        options = self.query_one(OptionList)
        options.set_options(row.label for row in self._rows)
        if section is None:
            options.highlighted = max(0, bisect_right(self._rows, offset, key=lambda row: row.offset) - 1)
        else:
            options.highlighted = next(i for i, row in enumerate(self._rows) if row.header and row.section == section)
        options.scroll_to_highlight()

    def close(self, *, focus: bool) -> None:
        self.display = False
        self._editor.display = True
        if focus:
            self._editor.focus()
        self._sql = ""
        self._regions.clear()
        self._collapsed.clear()
        self._rows.clear()
        self._starts.clear()
        self.query_one(OptionList).clear_options()

    @on(Button.Pressed, ".fold-back")
    def action_return_to_editor(self) -> None:
        self._editor.reveal_for_edit()

    @on(Button.Pressed, ".fold-edit")
    def action_edit_here(self) -> None:
        i = self.query_one(OptionList).highlighted
        if self.display and i is not None:
            location = self._editor.document.get_location_from_index(self._rows[i].offset)
            self._editor.reveal_for_edit()
            self._editor.move_cursor(location)

    @on(Button.Pressed, ".fold-toggle")
    def action_toggle_section(self) -> None:
        options = self.query_one(OptionList)
        i = options.highlighted
        if self.display and i is not None and (section := self._rows[i].section):
            self._collapsed.symmetric_difference_update({section})
            self._render_rows(section.start, section)
        options.focus()

    @on(OptionList.OptionSelected)
    def _activate_row(self, event: OptionList.OptionSelected) -> None:
        event.stop()  # not a completion-palette click
        if self.display and event.option is event.option_list.highlighted_option:
            # A queued activation from a replaced page/projection has no target.
            if self._rows[event.option_index].header:
                self.action_toggle_section()
            else:
                self.action_edit_here()
