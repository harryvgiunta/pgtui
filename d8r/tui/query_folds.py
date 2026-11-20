"""SQL section boundaries and their projection into a TextArea's visual rows."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
import re

from textual.document._wrapped_document import WrappedDocument
from textual.geometry import Offset
from textual.widgets.text_area import Location

from d8r.sql import SqlError, cte_declarations, lex_sql, split_statements


_NEWLINE = re.compile(r"\r\n|\n|\r")


@dataclass(frozen=True)
class Section:
    start: int
    end: int
    kind: str


def sections(sql: str, dialect: str) -> list[Section]:
    """Exact nested source ranges. Reuse native routine/trigger statement boundaries."""
    tokens = tuple(t for t in lex_sql(sql, dialect=dialect) if t.kind not in {"whitespace", "comment"})
    result = []
    cursor = 0
    for statement in split_statements(sql, dialect=dialect):
        # split_statements strips surrounding whitespace and the terminating ';'.
        # Ignore empty/comment-only statements using token offsets. Searching
        # for statement text could match identical SQL inside an earlier comment.
        first = bisect_left(tokens, cursor, key=lambda t: t.start)
        while tokens[first].text == ";":
            cursor = tokens[first].end
            first += 1
        while sql[cursor].isspace():
            cursor += 1
        end = cursor + len(statement)
        after = end
        while after < len(sql) and sql[after].isspace():
            after += 1
        cursor = after + 1 if sql.startswith(";", after) else end
        result.append(Section(tokens[first].start, cursor, "statement"))

    ctes = {opening: name for name, opening in cte_declarations(tokens)}
    stack: list[int] = []
    for i, token in enumerate(tokens):
        if token.text == "(":
            stack.append(i)
        elif token.text == ")" and stack:
            opening = stack.pop()
            start = tokens[opening].start
            if opening in ctes:
                kind, start = "cte", tokens[ctes[opening]].start
            elif (tokens[opening + 1].kind == "word" and tokens[opening + 1].text.upper()
                  in {"SELECT", "WITH", "TABLE", "VALUES", "FROM"}):
                kind = "subquery"
            else:
                continue
            if _NEWLINE.search(sql, start, token.end):
                result.append(Section(start, token.end, kind))
    return sorted(result, key=lambda r: (r.start, -r.end))


@dataclass(frozen=True)
class LineFold:
    stop: int  # First visible row after a fold, or the end of the document.
    heading: str


def line_folds(sql: str, dialect: str) -> dict[int, LineFold]:
    """Fold whole body lines only; a shared closing line still belongs in the editor."""
    try:
        regions = sections(sql, dialect)
    except SqlError:
        return {}  # Unfinished quotes or comments: don't guess what to hide.
    starts = [0, *(m.end() for m in _NEWLINE.finditer(sql))]
    ends = starts[1:] + [len(sql)]
    markers: dict[int, LineFold] = {}
    statement_headers: set[int] = set()
    for section in regions:
        header = bisect_right(starts, section.start) - 1
        last = bisect_right(starts, section.end - 1) - 1
        # A close followed by a new CTE, SELECT, alias, or comment is not ours
        # to hide. Standalone ')' and trailing separators may go with the body.
        stop = last if sql[section.end:ends[last]].strip(" \t\r\n,;") else last + 1
        if stop <= header + 1:
            continue
        # One arrow per physical row. In 'WITH a AS (', prefer the local CTE
        # over folding the whole statement and hiding its sibling CTEs.
        if header not in markers or header in statement_headers and section.kind != "statement":
            markers[header] = LineFold(stop, sql[starts[header]:ends[header]].rstrip("\r\n"))
            if section.kind == "statement":
                statement_headers.add(header)
            else:
                statement_headers.discard(header)
    return markers


def hidden_lines(markers: dict[int, LineFold], collapsed: set[int]) -> list[tuple[int, int]]:
    """Merge hidden half-open line ranges without duplicating nested bodies."""
    ranges: list[tuple[int, int]] = []
    for row in sorted(collapsed):
        start, stop = row + 1, markers[row].stop
        if ranges and start <= ranges[-1][1]:
            ranges[-1] = (ranges[-1][0], max(stop, ranges[-1][1]))
        else:
            ranges.append((start, stop))
    return ranges


class FoldedDocument(WrappedDocument):
    """Adapt Textual 8's wrapped-row map; the document and column maps stay canonical.

    Textual renders through _offset_to_line_info. Keep its original map for
    wrap_range's incremental edits, and project only that map, not SQL text.
    """

    _unfolded_rows: list[tuple[int, int]] | None = None
    _cuts: list[int]
    _cut_ends: list[int]
    _removed: list[int]

    @property
    def folded(self) -> bool:
        return self._unfolded_rows is not None

    @property
    def height(self) -> int:
        return len(self._offset_to_line_info)

    def visual_line(self, y: int) -> tuple[int, int] | None:
        return self._offset_to_line_info[y] if 0 <= y < self.height else None

    def project(self, ranges: Iterable[tuple[int, int]]) -> None:
        self._unproject()
        source = self._offset_to_line_info
        start = 0
        visible: list[tuple[int, int]] = []
        self._cuts, self._cut_ends, self._removed = [], [], [0]
        for top, stop in ranges:
            first = self._line_index_to_offsets[top][0]
            end = self._line_index_to_offsets[stop][0] if stop < self.document.line_count else len(source)
            visible.extend(source[start:first])
            self._cuts.append(first)
            self._cut_ends.append(end)
            self._removed.append(self._removed[-1] + end - first)
            start = end
        if start:
            visible.extend(source[start:])
            self._unfolded_rows = source
            self._offset_to_line_info = visible

    def _unproject(self) -> None:
        if self._unfolded_rows is not None:
            self._offset_to_line_info = self._unfolded_rows
            self._unfolded_rows = None

    def wrap(self, width: int, tab_width: int | None = None) -> None:
        self._unfolded_rows = None
        super().wrap(width, tab_width)

    def wrap_range(self, start: Location, old_end: Location, new_end: Location) -> None:
        self._unproject()
        super().wrap_range(start, old_end, new_end)

    def location_to_offset(self, location: Location) -> Offset:
        offset = super().location_to_offset(location)
        if self.folded and (i := bisect_right(self._cuts, offset.y) - 1) >= 0:
            if offset.y < self._cut_ends[i]:
                # E.g. the old top of the viewport was just folded away.
                return Offset(0, self._cuts[i] - self._removed[i] - 1)
            return Offset(offset.x, offset.y - self._removed[i + 1])
        return offset

    def offset_to_location(self, offset: Offset) -> Location:
        if self.folded and not self._width:
            # Textual's no-wrap fast path treats y as a source row rather than
            # looking it up. Correct y before delegating its column math.
            row, _ = self._offset_to_line_info[max(0, min(offset.y, self.height - 1))]
            offset = Offset(offset.x, row)
        return super().offset_to_location(offset)
