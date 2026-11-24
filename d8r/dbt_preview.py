"""Select a bounded dbt preview query without re-rendering or translating SQL."""
from __future__ import annotations

import re
from dataclasses import dataclass

from d8r.dbt import DbtError
from d8r.dbt_index import _tags
from d8r.sql import SqlError, Token, cte_declarations, lex_sql, relation_parts, split_statements

_READ = frozenset({"SELECT", "WITH", "VALUES", "TABLE"})
_WRITE = frozenset({"INSERT", "UPDATE", "DELETE", "MERGE", "INTO"})


@dataclass(frozen=True)
class _Cte:
    name: Token
    close: Token
    depth: int


def _tokens(sql: str, dialect: str) -> tuple[Token, ...]:
    return tuple(token for token in lex_sql(sql, dialect=dialect) if token.kind not in {"comment", "whitespace"})


def _ctes(tokens: tuple[Token, ...]) -> list[_Cte]:
    declarations = {opening: name for name, opening in cte_declarations(tokens)}
    stack: list[int] = []
    ctes: list[_Cte] = []
    for i, token in enumerate(tokens):
        if token.kind != "symbol":
            continue
        if token.text == "(":
            stack.append(i)
        elif token.text == ")" and stack:
            name = declarations.get(stack.pop())
            if name is not None:
                ctes.append(_Cte(tokens[name], token, len(stack)))
    return ctes


def dbt_preview_query(compiled: str, dialect: str, *, source: str, caret: int | None) -> tuple[str, str]:
    """Return one query and its label; map a literal top-level CTE by name, not by line number."""
    try:
        statements = split_statements(compiled, dialect=dialect)
        if len(statements) != 1:
            raise DbtError("dbt preview requires one compiled query.")
        query = statements[0]
        tokens = _tokens(query, dialect)
        words = {token.text.upper() for token in tokens if token.kind == "word"}
        if not tokens or tokens[0].text.upper() not in _READ or words & _WRITE:
            raise DbtError("dbt preview requires a query without SQL writes. Use Run or Build for materialization.")
        if caret is None:
            return query, "Model"

        chunks: list[str] = []
        start = 0
        for tag in _tags(source, strict=True):
            chunks.extend((source[start:tag.start], re.sub(r"[^\r\n]", " ", source[tag.start:tag.end])))
            start = tag.end
        chunks.append(source[start:])
        authored = _ctes(_tokens("".join(chunks), dialect))
        containing = [cte for cte in authored if cte.name.start <= caret < cte.close.end]
        if not containing or any(cte.depth for cte in containing):
            raise DbtError("Place the caret inside a top-level CTE in the dbt source.")
        name = relation_parts(containing[0].name.text, dialect=dialect)
        top = [cte for cte in _ctes(tokens) if cte.depth == 0]
        matches = [cte for cte in top if relation_parts(cte.name.text, dialect=dialect) == name]
        if not name or len(matches) != 1 or tokens[0].text.upper() != "WITH":
            raise DbtError("This source CTE cannot be matched unambiguously in compiled SQL.")
        end = top[-1].close.end
        tail = next((tok for tok in tokens if tok.start >= end), None)
        if tail is None or tail.kind != "word" or tail.text.upper() not in _READ:
            raise DbtError("The compiled WITH clause has an unsupported CTE boundary.")
        label = matches[0].name.text
        # Retain the entire WITH clause (recursive and injected dependencies included).
        return f"{query[:end]}\nSELECT * FROM {label}", f"CTE {label}"
    except SqlError:
        raise DbtError("Cannot find safe SQL preview boundaries in this model.") from None
