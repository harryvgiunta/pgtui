"""Local layout for query clauses. The database, not the formatter, owns SQL grammar."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
import re

from d8r.sql import SqlError, Token, cte_declarations, lex_sql, split_statements


@dataclass
class _Scope:
    close: str
    indent: int
    query: bool
    clause: str = ""
    cases: int = 0
    between: bool = False
    projection: list[int] | None = None
    distinct_on: bool = False

    def finish_projection(self, end: int, breaks: dict[int, int]) -> None:
        if self.projection is None:
            return
        # DuckDB allows a trailing comma: the clause or closing delimiter that
        # follows it is not another projection.
        if self.projection[-1] == end:
            self.projection.pop()
        if len(self.projection) >= _VERTICAL_PROJECTIONS:
            breaks.update((i, self.indent + 1) for i in self.projection)
        self.projection = None
        self.distinct_on = False


_JOIN_PREFIX = frozenset("NATURAL INNER LEFT RIGHT FULL OUTER CROSS ASOF SEMI ANTI POSITIONAL".split())
_SET = frozenset({"UNION", "INTERSECT", "EXCEPT"})
_CLAUSES = frozenset({"WHERE", "HAVING", "WINDOW", "QUALIFY", "LIMIT", "OFFSET"})
_PAIRS = {"(": ")", "[": "]", "{": "}"}
_VERTICAL_PROJECTIONS = 5  # Keep small lists compact, independent of existing line breaks.


def _breaks(
    tokens: list[Token], dialect: str, line_starts: set[int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Clause and projection breaks keyed by token index (ignoring expression nesting)."""
    # Qualified names and prefixed parameters can contain clause keywords.
    words = [
        t.text.upper() if t.kind == "word" and not (
            i and (tokens[i - 1].text == "." or
                   (tokens[i - 1].text in {":", "$"} and tokens[i - 1].end == t.start))
        ) else ""
        for i, t in enumerate(tokens)
    ]
    heads = {"SELECT", "WITH"} | ({"FROM"} if dialect == "duckdb" else set())
    head = 0
    while dialect != "sqlite" and head < len(tokens) and tokens[head].text == "(":
        head += 1
    if head == len(words) or words[head] not in heads:
        return {}, {}
    clauses = _CLAUSES if dialect == "duckdb" else _CLAUSES - {"QUALIFY"}
    if dialect == "sqlite":
        clauses = clauses - {"OFFSET", "WINDOW"}
    ctes = dict(cte_declarations(tokens))
    bodies = set(ctes.values())
    scopes = [_Scope("", 0, True)]
    lines: dict[int, int] = {}
    continuations: dict[int, int] = {}
    projections: dict[int, int] = {}
    for i, (word, token) in enumerate(zip(words, tokens)):
        scope = scopes[-1]
        text = token.text
        prev = words[i - 1] if i else ""
        following = words[i + 1] if i + 1 < len(words) else ""
        # A projection or relation may already start on its own line. Indent
        # that line under its clause, without changing gaps between literals.
        if i in line_starts and scope.query and scope.clause and not scope.cases and (
            prev in {"SELECT", "DISTINCT", "FROM", "JOIN", "WHERE", "HAVING", "BY", "ON", "AND", "OR"}
            or tokens[i - 1].text == "," or text == ","
        ):
            continuations[i] = scope.indent + (2 if scope.clause == "ON" else 1)
        if text in _PAIRS:
            query = text == "(" and (following in heads or i in bodies)
            scopes.append(_Scope(_PAIRS[text], scope.indent + int(query), query))
            if query:
                # Don't turn pathological nesting into quadratically more whitespace.
                if scopes[-1].indent > 64:
                    return {}, {}
                lines[i + 1] = scopes[-1].indent
        elif text in (")", "]", "}"):
            if text != scope.close:
                return {}, {}
            scopes.pop()
            parent = scopes[-1]
            if parent.distinct_on:
                # DISTINCT ON (...) is a SELECT modifier, not its first expression.
                parent.projection = [i + 1]
                parent.distinct_on = False
            if scope.query:
                scope.finish_projection(i, projections)
                lines[i] = parent.indent
        elif not scope.query:
            continue
        elif word == "CASE":
            scope.cases += 1
        elif scope.cases:
            if word == "END":
                scope.cases -= 1
        elif i in ctes and i and tokens[i - 1].text == ",":
            lines[i] = scope.indent
        elif word == "SELECT":
            scope.finish_projection(i, projections)
            if i:
                lines[i] = scope.indent
            scope.clause = word
            first = i + 1 + int(following in {"ALL", "DISTINCT"})
            scope.projection = [first]
            scope.distinct_on = (following == "DISTINCT" and first + 1 < len(tokens)
                                 and words[first] == "ON" and tokens[first + 1].text == "(")
        elif word == "FROM" and not (
            words[i - 2:i] == ["IS", "DISTINCT"] or words[i - 3:i] == ["IS", "NOT", "DISTINCT"]
        ):
            scope.finish_projection(i, projections)
            if i:
                lines[i] = scope.indent
            scope.clause = word
        elif word == "JOIN" and scope.clause in {"FROM", "JOIN", "ON"}:
            first = i
            while first and words[first - 1] in _JOIN_PREFIX:
                first -= 1
            lines[first] = scope.indent
            scope.clause = word
        elif word == "ON" and scope.clause == "JOIN":
            lines[i] = scope.indent + 1
            scope.clause = word
        elif (word in _SET or word in clauses or
              (word in {"ORDER", "GROUP"} and following == "BY") or
              (word == "OFFSET" and scope.clause == "LIMIT") or
              (word == "WINDOW" and i + 3 < len(tokens) and words[i + 2] == "AS"
               and tokens[i + 3].text == "(")):
            scope.finish_projection(i, projections)
            lines[i] = scope.indent
            scope.clause = word
        elif scope.projection is not None and (
            (dialect == "postgres" and word == "INTO") or
            (dialect != "sqlite" and word == "FETCH" and following in {"FIRST", "NEXT"}) or
            (dialect == "postgres" and word == "FOR" and following in {"UPDATE", "SHARE", "NO", "KEY"})
        ):
            scope.finish_projection(i, projections)
        elif text == "," and scope.projection is not None:
            scope.projection.append(i + 1)
        elif word == "BETWEEN":
            scope.between = True
            continue
        elif word == "AND":
            if not scope.between and scope.clause in {"WHERE", "HAVING", "ON", "QUALIFY"}:
                lines[i] = scope.indent + 1
            scope.between = False
        else:
            continue
        if i in lines and word != "AND":
            scope.between = False
    if len(scopes) != 1:
        return {}, {}
    scopes[0].finish_projection(len(tokens), projections)
    return continuations | lines, projections


@dataclass(frozen=True)
class LayoutEdit:
    start: int
    end: int
    whitespace: str


def _statements(sql: str, dialect: str) -> Iterator[list[Token]]:
    # Use native compound-statement boundaries, not a naive semicolon split. Locate
    # them on real token boundaries: the next query might also occur inside a comment.
    try:
        statements = split_statements(sql, dialect=dialect)
        raw = lex_sql(sql, dialect=dialect)
    except SqlError:
        return  # Incomplete quote, comment or dollar body: don't guess its boundary.
    # Without server settings an ordinary PostgreSQL string containing a backslash
    # before a quote has two possible boundaries (standard_conforming_strings).
    if dialect == "postgres" and any(t.kind == "string" and t.text.startswith("'") and "\\'" in t.text
                                     for t in raw):
        return
    pos = 0
    for statement in statements:
        while pos < len(raw):
            while pos < len(raw) and raw[pos].kind == "whitespace":
                pos += 1
            significant = pos
            while significant < len(raw) and raw[significant].kind in {"whitespace", "comment"}:
                significant += 1
            if significant == len(raw):
                return
            if raw[significant].text != ";":
                break
            pos = significant + 1  # An empty or comment-only statement.
        first = pos
        boundary = raw[pos].start + len(statement)
        while pos < len(raw) and raw[pos].start < boundary:
            pos += 1
        yield list(raw[first:pos])


def layout_edits(sql: str, dialect: str, start: int, end: int) -> Iterator[LayoutEdit]:
    """Plan ordered edits inside [start, end), lexing the *whole* document for context."""
    if dialect not in {"postgres", "duckdb", "sqlite"} or start == end:
        return
    newline_match = re.search(r"\r\n|\r|\n", sql)
    newline = newline_match[0] if newline_match else "\n"
    for raw in _statements(sql, dialect):
        if raw[0].start >= end:
            break
        significant = [i for i, t in enumerate(raw) if t.kind not in {"whitespace", "comment"}]
        tokens = [raw[i] for i in significant]
        if not tokens:
            continue
        # COPY payloads and shell commands can contain arbitrary text that looks like SQL.
        if tokens[0].text.upper() == "COPY" or any(t.text == "\\" for t in tokens):
            break
        if raw[-1].end <= start:
            continue
        # SQLite's Tcl-style named parameter suffix is opaque, not a SQL subquery.
        if dialect == "sqlite" and re.search(r"[$:@][\w:$]+\(", sql[raw[0].start:raw[-1].end]):
            continue
        line_starts = {i for i, j in enumerate(significant) if j and raw[j - 1].kind == "whitespace"
                       and ("\n" in raw[j - 1].text or "\r" in raw[j - 1].text)}
        breaks, projections = _breaks(tokens, dialect, line_starts)
        breaks.update(projections)
        for i, raw_index in enumerate(significant):
            if i not in breaks or not raw_index:
                continue
            at = raw[raw_index].start
            previous = raw[raw_index - 1]
            before = previous.start if previous.kind == "whitespace" else at
            if not (start <= before and at < end):
                continue
            gap = sql[before:at]
            # Insert absent whitespace only at known boundaries. A projection
            # can also touch its SELECT keyword or an intervening block comment.
            if not gap and previous.text not in {"(", ")", ","} and tokens[i].text != ")":
                # A tight @name can lex differently once separated from a word.
                if i not in projections or tokens[i].text == "@":
                    continue
            if gap.strip(" \t\r\n"):
                continue  # Python's definition of whitespace is wider than native SQL's.
            # Keep every existing line ending (including the LF after a line comment)
            # and blank line. Never collapse whitespace between adjacent string literals.
            after_break = max(gap.rfind("\n"), gap.rfind("\r")) + 1
            replacement = (gap[:after_break] if after_break else newline) + "    " * breaks[i]
            if replacement != gap:
                yield LayoutEdit(before, at, replacement)


def format_sql(sql: str, dialect: str = "postgres") -> str:
    """Whitespace-only query layout; unsupported or uncertain constructs pass through."""
    chunks: list[str] = []
    last = 0
    for edit in layout_edits(sql, dialect, 0, len(sql)):
        chunks.extend((sql[last:edit.start], edit.whitespace))
        last = edit.end
    chunks.append(sql[last:])
    return "".join(chunks)
