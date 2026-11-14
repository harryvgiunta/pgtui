"""Presentation-only native SQL colors using TextArea's public line hook.

Color visible lines without parsing, querying a source, or rewriting the document.
Incomplete quotes and calls are ordinary editor input, not validation errors.
"""

from __future__ import annotations

import re

from rich.style import Style
from rich.text import Text
from textual import events
from textual.reactive import reactive
from textual.widgets import TextArea

from d8r.sql import SqlError, lex_sql

from d8r.completion.catalog import KEYWORDS

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


class QueryTextArea(TextArea):
    """SQL editor; caret, selection, wrapping and undo remain TextArea's."""

    dialect = reactive("postgres")

    def __init__(self, *args, dialect: str = "postgres", **kwargs) -> None:
        self.user_edit_serial = 0
        self.programmatic_edit_serial = 0
        self.document_load_serial = 0
        self.selection_revision = 0
        self._syntax_revision = 0
        super().__init__(*args, **kwargs)
        self.dialect = dialect

    def load_text(self, text: str) -> None:
        self.programmatic_edit_serial = self.user_edit_serial
        self.document_load_serial += 1
        super().load_text(text)

    def _watch_selection(self, previous_selection, selection) -> None:
        if previous_selection != selection:
            self.selection_revision += 1
        super()._watch_selection(previous_selection, selection)

    def on_key(self, event: events.Key) -> None:
        # Restoration, caret movement and provider publication never count as typing.
        if event.is_printable or event.key in {"backspace", "delete", "ctrl+v", "ctrl+x"}:
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
    }
    DEFAULT_CSS = """
    QueryTextArea {
        & .query--keyword {
            color: $text-primary;
            text-style: bold;
        }
        & .query--function { color: $text-accent; text-style: bold; }
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
        super()._build_highlight_map()

    def get_line(self, line_index: int) -> Text:
        line = super().get_line(line_index)
        syntax_key = (self._syntax_revision, self.dialect)
        if getattr(self, "_syntax_key", None) != syntax_key:
            text = self.text
            self._syntax_key = syntax_key
            self._syntax_lines: dict[int, list[tuple[int, int, str]]] = {}
            starts = [0]
            starts.extend(index + 1 for index, char in enumerate(text) if char == "\n")
            from bisect import bisect_right
            try:
                tokens = ((token.kind, token.start, token.end) for token in lex_sql(text, dialect=self.dialect))
            except SqlError:
                # Unfinished input still has useful presentation colors. This
                # tolerant scanner neither executes nor validates SQL grammar.
                scanner = _SQLITE_TOKENS if self.dialect == "sqlite" else _TOKENS
                tokens = ((token.lastgroup, *token.span()) for token in scanner.finditer(text))
            for kind, start, end in tokens:
                if kind == "word":
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
