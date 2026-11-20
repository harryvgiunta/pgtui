"""Canonical SQL editor with lexical colors and an optional read-only fold view.

Presentation never queries a source, validates SQL, or rewrites the document.
"""

from __future__ import annotations

from bisect import bisect_right
import re

from rich.style import Style
from rich.text import Text
from textual import events
from textual.reactive import reactive

from d8r.sql import SqlError, Token, _relation_end, _relation_sites, cte_declarations, lex_sql

from d8r.completion.catalog import KEYWORDS
from .editing import DraftTextArea
from .query_folds import QueryFoldView

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

    dialect = reactive("postgres")

    def __init__(self, *args, dialect: str = "postgres", **kwargs) -> None:
        self.user_edit_serial = 0
        self.programmatic_edit_serial = 0
        self.document_load_serial = 0
        self.selection_revision = 0
        self._syntax_revision = 0
        self._fold_view: QueryFoldView | None = None
        super().__init__(*args, **kwargs)
        self.dialect = dialect

    def load_text(self, text: str) -> None:
        self.programmatic_edit_serial = self.user_edit_serial
        self.document_load_serial += 1
        super().load_text(text)

    def action_toggle_folds(self) -> None:
        """Toolbar/shortcut entry point. Mount QueryFoldView in the same pane."""
        if self._fold_view is not None:
            self.reveal_for_edit()
        elif self.parent is not None:
            view = self.parent.query_one(QueryFoldView)
            if view.open_for(self):
                self._fold_view = view

    def reveal_for_edit(self, *, focus: bool = True) -> None:
        """Show the unmodified editor before find/format/jump. Keep its selection."""
        view, self._fold_view = self._fold_view, None
        if view is not None:
            view.close(focus=focus)

    def _reset_folds(self) -> None:
        if self._fold_view is not None:
            self.reveal_for_edit(focus=self._fold_view.has_focus_within)

    def watch_dialect(self) -> None:
        self._reset_folds()

    def _watch_selection(self, previous_selection, selection) -> None:
        if previous_selection != selection:
            self.selection_revision += 1
            self._reset_folds()
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
        self._reset_folds()  # includes edits, format, undo, and even identical page loads
        super()._build_highlight_map()

    def get_line(self, line_index: int) -> Text:
        line = super().get_line(line_index)
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
