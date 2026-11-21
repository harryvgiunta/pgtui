"""Canonical SQL editor with lexical colors and inline section folding.

Presentation never queries a source, validates SQL, or rewrites the document.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import re

from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.binding import Binding
from textual.document._document_navigator import DocumentNavigator
from textual.geometry import Offset, Size
from textual.message import Message
from textual.reactive import reactive
from textual.strip import Strip
from textual.widgets.text_area import Location, Selection

from d8r.sql import SqlError, Token, _relation_end, _relation_sites, cte_declarations, lex_sql

from d8r.completion.catalog import KEYWORDS
from .editing import DraftTextArea
from .query_folds import FoldedDocument, LineFold, hidden_lines, line_folds

_TOKENS = re.compile(
    r"(?P<comment>--[^\n]*|/\*[\s\S]*?(?:\*/|\Z))"
    r"|(?P<dollar>(?P<tag>\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$)[\s\S]*?(?:(?P=tag)|\Z))"
    r"|(?P<string>'(?:[^']|'')*(?:'|\Z))"
    r'|(?P<identifier>"(?:[^"]|"")*(?:"|\Z))'
    r"|(?P<parameter>(?<![A-Za-z0-9_])@[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<number>(?<![A-Za-z0-9_])(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9$]*)"
    r"|(?P<operator>::|!=|>=|<=|<>|!~|[=<>~+*/%|-])"
    r"|(?P<punctuation>[(),.;])"
)
_SQLITE_TOKENS = re.compile(
    r"(?P<comment>--[^\n]*|/\*[\s\S]*?(?:\*/|\Z))"
    r"|(?P<string>'(?:[^']|'')*(?:'|\Z))"
    r'|(?P<identifier>"(?:[^"]|"")*(?:"|\Z)|`(?:[^`]|``)*(?:`|\Z)|\[[^\]]*(?:\]|\Z))'
    r"|(?P<parameter>(?<![A-Za-z0-9_])@[A-Za-z_][A-Za-z_0-9]*)"
    r"|(?P<number>(?<![A-Za-z0-9_])(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)"
    r"|(?P<word>[A-Za-z_][A-Za-z_0-9$]*)"
    r"|(?P<operator>!=|>=|<=|<>|[=<>~+*/%|-])"
    r"|(?P<punctuation>[(),.;])"
)
_KEYWORDS = frozenset(word.lower() for word in KEYWORDS)
_CALL_TAIL = re.compile(r"\s*\(")


def _table_offsets(tokens: tuple[Token, ...], dialect: str) -> set[int]:
    """Color written relation heads, not matching spellings elsewhere in SQL."""
    sig = tuple(t for t in tokens if t.kind not in {"whitespace", "comment"})
    offsets = {sig[i].start for i, _ in cte_declarations(sig)}

    def word(i: int) -> str:
        return sig[i].text.upper() if 0 <= i < len(sig) and sig[i].kind == "word" else ""

    def mark(i: int, *, allow_columns: bool = False) -> None:
        if i >= len(sig) or sig[i].kind not in {"word", "quoted_identifier"}:
            return
        end = _relation_end(sig, i)
        if not allow_columns and end < len(sig) and sig[end].text == "(":
            return  # FROM function(...) is a call, unlike INSERT INTO t(col).
        offsets.update(sig[j].start for j in range(i, end, 2))

    # Reuse the expander's scoped FROM/JOIN/comma logic (which distinguishes
    # expression FROM, e.g. extract(... FROM col), from a query relation).
    # Supply query context for UPDATE ... FROM and DuckDB FROM-first SELECTs.
    scan: list[Token] = []
    original: list[int] = []
    commands = [""]
    for i, t in enumerate(sig):
        verb = word(i)
        head = not i or sig[i - 1].text in {";", "(", ")"}
        if t.text == "(":
            commands.append("")
        elif t.text == ")" and len(commands) > 1:
            commands.pop()
        elif t.text == ";":
            commands = [""]
        elif head and verb in {"SELECT", "DELETE", "MERGE", "UPDATE", "INSERT", "CREATE", "ALTER"}:
            commands[-1] = verb
        duck_from = dialect == "duckdb" and verb == "FROM" and (not i or sig[i - 1].text == ";")
        if (head and verb in {"UPDATE", "MERGE"}) or duck_from:
            scan.append(Token("word", "SELECT", t.start, t.start))
            original.append(i)
        # DELETE/MERGE USING introduces a source, unlike JOIN USING(columns).
        if (verb == "USING" and commands[-1] in {"MERGE", "DELETE"}
                and i + 1 < len(sig) and sig[i + 1].text != "("):
            scan.append(Token("word", "FROM", t.start, t.end))
        else:
            scan.append(t)
        original.append(i)

        if ((verb == "INTO" and word(i - 1) in {"INSERT", "REPLACE", "MERGE"})
            or (verb == "INTO" and word(i - 3) == "INSERT" and word(i - 2) == "OR")
            or (head and verb in {"UPDATE", "TABLE", "TRUNCATE"})
            or (verb == "REFERENCES" and ("CREATE" in commands or "ALTER" in commands))):
            j = i + 1
            if verb == "UPDATE" and word(j) == "OR" and dialect == "sqlite":
                j += 2
            if verb == "TRUNCATE" and word(j) == "TABLE":
                j += 1
            mark(j + (word(j) == "ONLY"), allow_columns=True)
        elif verb in {"TABLE", "VIEW"}:
            j = i - 1
            while word(j) in {"TEMP", "TEMPORARY", "UNLOGGED", "MATERIALIZED", "VIRTUAL"}:
                j -= 1
            if word(j) == "REPLACE" and word(j - 1) == "OR":
                j -= 2
            if word(j) in {"CREATE", "ALTER", "DROP"}:
                j = i + 1
                if word(j) == "IF":
                    j += 3 if word(j + 1) == "NOT" else 2
                mark(j + (word(j) == "ONLY"), allow_columns=True)
    for i in _relation_sites(tuple(scan)):
        if word(original[i]) not in {"VALUES", "ROWS"}:
            mark(original[i])
    return offsets


class QueryTextArea(DraftTextArea):
    """SQL editor; caret, selection, wrapping and undo remain TextArea's."""

    BINDINGS = [Binding("shift+f7", "toggle_fold", "Toggle section at caret", show=False)]

    class FoldsChanged(Message):
        """Visual positions changed without an edit to SQL."""

    class DefinitionRequested(Message):
        def __init__(self, offset: int) -> None:
            super().__init__()
            self.offset = offset

    file_format = reactive("sql")
    dbt_navigation = False
    dbt_template = False
    dialect = reactive("postgres")

    def __init__(self, *args, dialect: str = "postgres", **kwargs) -> None:
        self.user_edit_serial = 0
        self.programmatic_edit_serial = 0
        self.document_load_serial = 0
        self.selection_revision = 0
        self._syntax_revision = 0
        self._fold_regions: dict[int, LineFold] | None = None
        self._collapsed: set[int] = set()
        super().__init__(*args, **kwargs)
        self.dialect = dialect

    def load_text(self, text: str) -> None:
        self.programmatic_edit_serial = self.user_edit_serial
        self.document_load_serial += 1
        super().load_text(text)

    def _set_document(self, text: str, language: str | None) -> None:
        self._collapsed.clear()
        self._fold_regions = None
        super()._set_document(text, language)

    def _sections(self) -> dict[int, LineFold]:
        if self._fold_regions is None:
            self._fold_regions = line_folds(self.text, self.dialect) if self.file_format == "sql" else {}
        return self._fold_regions

    def _keep_endpoints(self, rows: set[int], selection: Selection) -> set[int]:
        markers = self._sections()
        return {row for row in rows if not any(row < point[0] < markers[row].stop for point in selection)}

    def action_toggle_folds(self) -> None:
        """Expand all if any section is folded; otherwise collapse all."""
        self._toggle_to(set() if self._collapsed else set(self._sections()))

    def action_toggle_fold(self) -> None:
        row, _ = self.cursor_location
        if row in self._sections():
            self._toggle_to(self._collapsed ^ {row})

    def _toggle_to(self, rows: set[int]) -> None:
        if self.selection.is_empty:
            row, _ = self.cursor_location
            for start, stop in hidden_lines(self._sections(), rows):
                if start <= row < stop:
                    self.move_cursor((start - 1, len(self.document[start - 1])))
                    break
        elif rows:
            # A fold must never change an authored selection, including Run's input.
            rows = self._keep_endpoints(rows, self.selection)
        self._set_collapsed(rows)

    def _set_collapsed(self, rows: set[int]) -> None:
        if rows == self._collapsed:
            return
        wrapped = self.wrapped_document
        top = wrapped.offset_to_location(Offset(0, self.scroll_offset.y))
        self._collapsed = rows
        self._project_folds()
        if self.is_mounted:
            y = wrapped.location_to_offset(top).y
            self.scroll_to(y=y, animate=False, immediate=True, force=True)
            self.post_message(self.FoldsChanged())

    def _project_folds(self) -> None:
        wrapped = self.wrapped_document
        wrapped.project(hidden_lines(self._sections(), self._collapsed) if self._collapsed else ())
        self._line_cache.clear()
        self._refresh_size()
        self._recompute_cursor_offset()
        if self.is_mounted and self.has_focus:
            self.app.cursor_position = self.cursor_screen_offset
        self.refresh()

    def _rewrap_and_refresh_virtual_size(self) -> None:
        if isinstance(self.wrapped_document, FoldedDocument):
            self.wrapped_document.wrap(self.wrap_width, self.indent_width)
        else:
            self.wrapped_document = FoldedDocument(self.document, self.wrap_width, self.indent_width)
            self.navigator = DocumentNavigator(self.wrapped_document)
        self._project_folds()

    def _refresh_size(self) -> None:
        if not self.soft_wrap and isinstance(self.wrapped_document, FoldedDocument) and self.wrapped_document.folded:
            width, _ = self.document.get_size(self.indent_width)
            self.virtual_size = Size(width + self.gutter_width + 1, self.wrapped_document.height)
            self._refresh_scrollbars()
        else:
            super()._refresh_size()

    def watch_dialect(self) -> None:
        self._set_collapsed(set())
        self._fold_regions = None
        self._line_cache.clear()
        self.refresh()

    def watch_file_format(self) -> None:
        self._set_collapsed(set())
        self._fold_regions = None
        self._line_cache.clear()
        self.tab_behavior = "indent" if self.file_format == "yaml" else "focus"
        self.indent_width = 2 if self.file_format == "yaml" else 4
        self.refresh()

    def _watch_selection(self, previous_selection, selection) -> None:
        if previous_selection != selection:
            self.selection_revision += 1
            wrapped = self.wrapped_document
            if isinstance(wrapped, FoldedDocument) and wrapped.folded:
                self._set_collapsed(self._keep_endpoints(self._collapsed, selection))
                if not self.soft_wrap:
                    self._line_cache.clear()  # Plain TextArea caches by raw, not folded, row.
        super()._watch_selection(previous_selection, selection)

    def _vertical_target(self, target: Location, delta: int) -> Location:
        wrapped = self.wrapped_document
        if not wrapped.folded or all(not row < target[0] < self._sections()[row].stop for row in self._collapsed):
            return target
        x, y = wrapped.location_to_offset(self.cursor_location)
        if delta > 0 and y + delta >= wrapped.height:
            row = self.cursor_location[0]
            return (row, len(self.document[row]))
        return wrapped.offset_to_location(Offset(max(x, self.navigator.last_x_offset), y + delta))

    def get_cursor_up_location(self) -> Location:
        return self._vertical_target(super().get_cursor_up_location(), -1)

    def get_cursor_down_location(self) -> Location:
        return self._vertical_target(super().get_cursor_down_location(), 1)

    def _render_line(self, y: int) -> Strip:
        strip = super()._render_line(y)
        if not self.show_line_numbers:
            return strip
        line = self.wrapped_document.visual_line(y + self.scroll_offset.y)
        if line is None or line[1] or line[0] not in self._sections():
            return strip
        column = self.gutter_width - 2  # Reuse one gutter padding cell.
        style = next(iter(strip.crop(column, column + 1))).style
        mark = "▸" if line[0] in self._collapsed else "▾"
        return Strip.join((strip.crop(0, column), Strip([Segment(mark, style)], 1), strip.crop(column + 1)))

    def _on_mouse_down(self, event: events.MouseDown) -> None:
        if self.dbt_navigation and self.file_format == "sql" and event.button == 1 and event.ctrl:
            if event.x - self.gutter.left >= self.gutter_width:
                location = self.get_target_document_location(event)
                self.post_message(self.DefinitionRequested(self.document.get_index_from_location(location)))
                event.prevent_default()
                event.stop()
                return
        if not self.show_line_numbers or event.button != 1 or event.x - self.gutter.left != self.gutter_width - 2:
            return  # Textual's normal mouse-selection handler still runs.
        y = event.y - self.gutter.top + self.scroll_offset.y
        line = self.wrapped_document.visual_line(y)
        if line is not None and line[1] == 0 and line[0] in self._sections():
            event.prevent_default()
            event.stop()
            self._toggle_to(self._collapsed ^ {line[0]})

    def on_key(self, event: events.Key) -> None:
        # Restoration, caret movement and provider publication never count as typing.
        if event.is_printable or event.key in {"enter", "backspace", "delete", "ctrl+v", "ctrl+x"}:
            self.user_edit_serial += 1

    def on_paste(self, event: events.Paste) -> None:
        self.user_edit_serial += 1

    def action_delete_left(self) -> None:
        self.user_edit_serial += 1
        super().action_delete_left()

    def action_delete_right(self) -> None:
        self.user_edit_serial += 1
        super().action_delete_right()

    def action_cut(self) -> None:
        self.user_edit_serial += 1
        super().action_cut()

    def action_paste(self) -> None:
        self.user_edit_serial += 1
        super().action_paste()

    def action_undo(self) -> None:
        self.user_edit_serial += 1
        super().action_undo()

    def action_redo(self) -> None:
        self.user_edit_serial += 1
        super().action_redo()

    COMPONENT_CLASSES = {
        "query--keyword", "query--function", "query--string", "query--comment",
        "query--number", "query--parameter", "query--operator", "query--punctuation",
        "query--table",
    }
    DEFAULT_CSS = """
    QueryTextArea {
        & .query--keyword {
            color: $text-primary;
            text-style: bold;
        }
        & .query--function { color: $text-accent; text-style: bold; }
        & .query--table { color: $secondary-lighten-1; text-style: bold; }
        & .query--string { color: $text-success; }
        & .query--comment { color: $text-muted; text-style: italic; }
        & .query--number { color: $text-warning; }
        & .query--parameter { color: $text-secondary; text-style: italic; }
        & .query--operator { color: $text-accent; }
        & .query--punctuation { color: $text-muted; }
    }
    """

    def get_component_rich_style(
        self, *names: str, partial: bool = False, default: Style | None = None,
    ) -> Style:
        # TextArea applies the cursor-line style after get_line. Inheriting the
        # editor's foreground there would paint over every syntax token.
        return super().get_component_rich_style(
            *names, partial=partial or names == ("text-area--cursor-line",), default=default,
        )

    def _build_highlight_map(self) -> None:
        # TextArea invokes this after loads, edits, undo and redo. A revision
        # avoids joining the entire document again for every visible line.
        self._syntax_revision += 1
        old, self._fold_regions = self._fold_regions, None
        if self._collapsed and old is not None:
            # Rematch unchanged, unambiguous headers after source rows move.
            before = Counter(fold.heading for fold in old.values())
            wanted = {old[row].heading for row in self._collapsed if before[old[row].heading] == 1}
            current = self._sections()
            after = Counter(fold.heading for fold in current.values())
            self._collapsed = self._keep_endpoints(
                {row for row, fold in current.items() if fold.heading in wanted and after[fold.heading] == 1},
                self.selection,
            )
            self._project_folds()
        super()._build_highlight_map()

    def get_line(self, line_index: int) -> Text:
        line = super().get_line(line_index)
        if self.file_format != "sql":
            return line
        syntax_key = (self._syntax_revision, self.dialect)
        if getattr(self, "_syntax_key", None) != syntax_key:
            text = self.text
            self._syntax_key = syntax_key
            self._syntax_lines: dict[int, list[tuple[int, int, str]]] = {}
            starts = [0]
            starts.extend(m.end() for m in re.finditer(r"\r\n|[\r\n]", text))
            tables: set[int] = set()
            try:
                lexed = lex_sql(text, dialect=self.dialect)
                tables = _table_offsets(lexed, self.dialect)
                tokens = ((token.kind, token.start, token.end) for token in lexed)
            except SqlError:
                # Unfinished input still has useful presentation colors. This
                # tolerant scanner neither executes nor validates SQL grammar.
                scanner = _SQLITE_TOKENS if self.dialect == "sqlite" else _TOKENS
                tokens = ((token.lastgroup, *token.span()) for token in scanner.finditer(text))
            for kind, start, end in tokens:
                if start in tables:
                    kind = "table"
                elif kind == "word":
                    if (start and text[start - 1] == ".") or text[end:end + 1] == ".":
                        continue
                    if text[start:end].lower() in _KEYWORDS:
                        kind = "keyword"
                    elif _CALL_TAIL.match(text, end):
                        kind = "function"
                    else:
                        continue
                elif kind in {"identifier", "quoted_identifier", "whitespace"}:
                    continue
                elif kind in {"dollar", "dollar_string"}:
                    kind = "string"
                elif kind == "symbol":
                    kind = "punctuation" if text[start:end] in "(),.;" else "operator"
                first = bisect_right(starts, start) - 1
                last = bisect_right(starts, end - 1) - 1
                for row in range(first, last + 1):
                    self._syntax_lines.setdefault(row, []).append((max(0, start - starts[row]), end - starts[row], kind))
        for start, end, kind in self._syntax_lines.get(line_index, ()):
            line.stylize(self.get_component_rich_style(f"query--{kind}", partial=True), start, min(end, len(line.plain)))
        return line
