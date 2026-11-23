"""Conservative local SELECT lint. Positions are in the shadow SQL, not rendered dbt SQL."""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

from sqlglot import exp
from sqlglot.optimizer.scope import walk_in_scope
from sqlglot.tokens import Token, TokenType

from d8r.sql import RoutineDef, relation_parts


@dataclass(frozen=True)
class WarningSpan:
    start: int
    end: int
    message: str


_FAMILIES = {
    'number': {'smallint', 'int', 'integer', 'bigint', 'int2', 'int4', 'int8', 'numeric', 'decimal',
               'real', 'float', 'float4', 'float8', 'double', 'double precision', 'hugeint', 'tinyint',
               'utinyint', 'usmallint', 'uinteger', 'ubigint', 'uhugeint'},
    'text': {'text', 'varchar', 'character varying', 'char', 'character', 'bpchar', 'string'},
    'boolean': {'bool', 'boolean'},
    'date': {'date'},
    'time': {'time', 'time without time zone', 'timetz', 'time with time zone'},
    'timestamp': {'timestamp', 'timestamptz', 'timestamp with time zone', 'timestamp without time zone'},
    'interval': {'interval'},
}
_TYPE = {name: family for family, names in _FAMILIES.items() for name in names}
_ARITHMETIC = (exp.Add, exp.Sub, exp.Mul, exp.Div, exp.Mod)
_NUMERIC_PREFIX = re.compile(r'^\s*[+-]?(?:[0-9]|\.[0-9])', re.ASCII)
_PG_BOOL_LITERALS = {'1', '0', 't', 'tr', 'tru', 'true', 'f', 'fa', 'fal', 'fals', 'false',
                     'y', 'ye', 'yes', 'n', 'no', 'on', 'of', 'off'}
_DUCK_BOOL_LITERALS = {'1', '0', 't', 'true', 'f', 'false', 'y', 'yes', 'n', 'no'}

# Fallback signatures for engines or offline workspaces without this routine
# in the cached catalog. Completion snippets are not exhaustive signatures.
_COMMON_ARITY = {'abs': (1, 1), 'lower': (1, 1), 'upper': (1, 1), 'nullif': (2, 2),
                 'replace': (3, 3), 'round': (1, 2)}
_BUILTIN_ARITY = {
    'postgres': _COMMON_ARITY | {'count': (1, 1), 'now': (0, 0), 'split_part': (3, 3)},
    'duckdb': _COMMON_ARITY | {'count': (0, 1), 'now': (0, 0), 'random': (0, 0), 'string_agg': (1, 2)},
    'sqlite': _COMMON_ARITY | {'count': (0, 1), 'random': (0, 0), 'coalesce': (2, None),
                               'group_concat': (1, 2), 'ifnull': (2, 2), 'instr': (2, 2),
                               'json': (1, 1), 'typeof': (1, 1)},
}
_CLOSING = {TokenType.R_PAREN: TokenType.L_PAREN, TokenType.R_BRACKET: TokenType.L_BRACKET,
            TokenType.R_BRACE: TokenType.L_BRACE}
_OPENING = set(_CLOSING.values())


def _family(typ: str, dialect: str) -> str:
    typ = re.sub(r'\(\s*\d+(?:\s*,\s*\d+)?\s*\)', '', typ.strip().lower()).strip()
    if dialect != 'sqlite':
        return _TYPE.get(typ, '')
    # Respect SQLite's affinity precedence. Only compare declarations that
    # imply a storage family; BLOB/ANY and semantic types like DATE or JSON
    # do not promise a single class in SQLite.
    if 'int' in typ:
        return 'number'
    if any(part in typ for part in ('char', 'clob', 'text')):
        return 'text'
    if 'blob' not in typ and (any(part in typ for part in ('real', 'floa', 'doub')) or
                             typ in {'numeric', 'decimal', 'bool', 'boolean'}):
        return 'number'
    return ''


def types_incompatible(actual: str, expected: str, *, dialect: str) -> bool:
    a, b = _family(actual, dialect), _family(expected, dialect)
    return bool(a and b and a != b)


def _authored_calls(statements: Sequence[exp.Expr | None], tokens: Sequence[Token], dialect: str
                    ) -> Iterator[tuple[tuple[str, ...], int, Token, exp.Func]]:
    """Count commas in actual calls, not the parser's normalized arguments."""
    sites = {start: node for query in statements if isinstance(query, exp.Query)
             for node in query.walk() if isinstance(node, exp.Func) and not isinstance(node, exp.Unnest)
             and (start := node.meta_get('start')) is not None}
    if not sites:
        return

    def name(token: Token) -> str:
        return token.text if dialect == 'postgres' and token.token_type is TokenType.IDENTIFIER else token.text.lower()

    stack: list[tuple[int, int, bool]] = []
    for i, token in enumerate(tokens):
        kind = token.token_type
        if kind in _OPENING:
            stack.append((i, 0, False))
        elif kind is TokenType.ORDER_BY and stack:
            opening, commas, _ = stack[-1]
            stack[-1] = opening, commas, True
        elif kind is TokenType.COMMA and stack and not stack[-1][2]:
            opening, commas, _ = stack[-1]
            stack[-1] = opening, commas + 1, False
        elif kind is TokenType.R_PAREN and stack:
            opening, commas, _ = stack.pop()
            if not opening or tokens[opening].token_type is not TokenType.L_PAREN:
                continue
            head = tokens[opening - 1]
            if (node := sites.get(head.start)) is None:
                continue
            qualifiers = [name(head)]
            j = opening - 2
            while j >= 1 and tokens[j].token_type is TokenType.DOT:
                qualifiers.append(name(tokens[j - 1]))
                j -= 2
            yield tuple(reversed(qualifiers)), (0 if i == opening + 1 else commas + 1), head, node
        elif kind in _CLOSING and stack:
            stack.pop()


def call_warnings(statements: Sequence[exp.Expr | None], tokens: Sequence[Token], *, dialect: str,
                  routines: Sequence[RoutineDef]) -> Iterator[WarningSpan]:
    calls = list(_authored_calls(statements, tokens, dialect))
    if not calls:
        return
    names = {parts[-1] for parts, _, _, _ in calls}
    folded = {name.lower() for name in names}
    short_names = not any('.' in name for name in names)
    signatures: dict[str, list[tuple[tuple[str, ...], RoutineDef]]] = defaultdict(list)
    for routine in routines:
        if routine.kind == 'procedure':
            continue
        if short_names and routine.name.rpartition('.')[-1].strip('"').replace('""', '"').lower() not in folded:
            continue
        parts = relation_parts(routine.name, dialect=dialect)
        if parts:
            parts = parts if dialect == 'postgres' else tuple(part.lower() for part in parts)
            if parts[-1] in names:
                signatures[parts[-1]].append((parts, routine))

    arities = _BUILTIN_ARITY.get(dialect, {})
    for parts, count, head, node in calls:
        name = parts[-1]
        overloads = [routine for identity, routine in signatures.get(name, ()) if identity[-len(parts):] == parts]
        if overloads:
            if any((routine.min_args is None or count >= routine.min_args) and
                   (routine.variadic or count <= len(routine.parameters)) for routine in overloads):
                continue
            noun = 'argument' if count == 1 else 'arguments'
            message = f'No cached overload of {".".join(parts)!r} accepts {count} {noun}.'
        elif len(parts) == 1 and name in arities:
            least, most = arities[name]
            if count >= least and (most is None or count <= most):
                continue
            expected = f'at least {least}' if most is None else str(least) if least == most else f'{least}–{most}'
            noun = 'argument' if expected == '1' else 'arguments'
            message = f'Built-in {name}() expects {expected} {noun}; got {count}.'
        else:
            continue
        if dialect == 'duckdb' and any(isinstance(arg, (exp.Columns, exp.Star)) for arg in walk_in_scope(node)):
            continue  # A star expansion can supply a different argument count.
        yield WarningSpan(head.start, head.end + 1, message)


def _value(node: exp.Expr) -> exp.Expr:
    if isinstance(node, exp.Alias):
        node = node.this
    return node.unnest() if isinstance(node, exp.Paren) else node


def _warning(node: exp.Expr, message: str) -> WarningSpan | None:
    node = _value(node)
    while isinstance(node, (exp.Cast, exp.Neg)):
        node = _value(node.this)
    if isinstance(node, exp.Column):
        node = node.this
    start, end = node.meta_get('start'), node.meta_get('end')
    if isinstance(start, int) and isinstance(end, int):
        return WarningSpan(start, end + 1, message)
    return None


def _aggregate(node: exp.Expr) -> bool:
    if not isinstance(node, exp.AggFunc):
        return False
    parent = node.parent
    if isinstance(parent, (exp.Filter, exp.WithinGroup)):
        parent = parent.parent
    return not isinstance(parent, exp.Window)


def select_warnings(select: exp.Select, *, dialect: str, column_type: Callable[[exp.Column], str],
                    correlated: bool = False) -> Iterator[WarningSpan]:
    expressions = [_value(item) for item in select.expressions]
    having = select.args.get('having')
    aggregate_inputs = expressions + ([having.this] if having is not None else [])
    group = select.args.get('group')
    order = select.args.get('order') if dialect == 'sqlite' else None
    evidence = aggregate_inputs + ([order] if order is not None else [])
    aggregates = [node for expr in evidence for node in walk_in_scope(expr) if _aggregate(node)]
    sqlite_extremum = dialect == 'sqlite' and sum(isinstance(node, (exp.Min, exp.Max)) for node in aggregates) == 1
    # With a single built-in extremum, SQLite deliberately picks bare values
    # from an extremal row (ties notwithstanding), not from just any group row.
    if (not sqlite_extremum and (group is not None or aggregates) and
            (group is None or all(key == 'expressions' or not val for key, val in group.args.items()))):
        grouped = set()
        aliases = {item.alias: _value(item) for item in select.expressions if isinstance(item, exp.Alias)}
        for term in group.expressions if group is not None else ():
            term = _value(term)
            if isinstance(term, exp.Literal) and term.is_int and len(term.this) < 6:
                position = int(term.this) - 1
                if 0 <= position < len(expressions):
                    grouped.add(expressions[position])
            else:
                grouped.add(term)
                if isinstance(term, exp.Column) and not term.table and term.name in aliases:
                    grouped.add(aliases[term.name])
        grouped_columns = [term for term in grouped if isinstance(term, exp.Column)]
        grouped_names = {term.name for term in grouped_columns if not term.table}
        grouped_by_name = {term.name for term in grouped_columns}
        for expr in aggregate_inputs:
            pending = [expr]
            while pending:
                node = pending.pop()
                if isinstance(node, exp.Query) or node in grouped or _aggregate(node):
                    continue
                if isinstance(node, (exp.Filter, exp.WithinGroup)) and _aggregate(node.this):
                    continue
                if isinstance(node, exp.Column) and not node.is_star:
                    # A qualified/unqualified spelling pair may be the same input.
                    if node.name in grouped_names or not node.table and node.name in grouped_by_name:
                        continue
                    if (dialect != 'postgres' and having is not None and expr is having.this and
                            not node.table and node.name in aliases):
                        continue
                    if correlated and not column_type(node):
                        continue
                    if dialect == 'sqlite':
                        message = (f'{node.name!r} is not grouped or aggregated; SQLite can choose its value from '
                                   'any row in the group. Verify the value is constant per group.')
                    else:
                        message = f'{node.name!r} is not explicitly grouped or aggregated.'
                        if any(not term.table or not node.table or term.table == node.table for term in grouped_columns):
                            message += ' Verify that a grouped key determines its value.'
                    if (warning := _warning(node, message)) is not None:
                        yield warning
                else:
                    pending.extend(node.iter_expressions())

    def kind(node: exp.Expr) -> str:
        node = _value(node)
        if isinstance(node, exp.Column):
            return _family(column_type(node), dialect)
        if isinstance(node, exp.Cast) and isinstance(node.to, exp.DataType):
            return _family(node.to.sql(dialect=dialect), dialect)
        if isinstance(node, exp.Boolean):
            return 'boolean'
        if isinstance(node, exp.Literal) and not node.is_string:
            return 'number'
        return ''

    for node in walk_in_scope(select):
        arithmetic = isinstance(node, _ARITHMETIC)
        if not arithmetic and not isinstance(node, (exp.And, exp.Or, exp.Not)):
            continue
        for arg in node.iter_expressions():
            operand = _value(arg)
            message = ''
            if isinstance(operand, exp.Literal) and operand.is_string:
                literal = operand.this
                if dialect == 'sqlite':
                    if not _NUMERIC_PREFIX.match(literal):
                        conversion = 'zero in arithmetic' if arithmetic else 'false in a logical expression'
                        message = f'SQLite treats this nonnumeric text literal as {conversion}.'
                elif arithmetic:
                    other = node.expression if arg is node.this else node.this
                    if (kind(other) == 'number' and not _NUMERIC_PREFIX.match(literal) and
                            literal.strip().lower().lstrip('+-') not in {'nan', 'inf', 'infinity'}):
                        message = 'This string literal is not numeric; verify the arithmetic operand type.'
                elif (dialect == 'postgres' and literal.strip().lower() not in _PG_BOOL_LITERALS or
                      dialect == 'duckdb' and literal.lower() not in _DUCK_BOOL_LITERALS):
                    message = 'This string literal is not a recognized boolean value.'
            elif dialect != 'sqlite':
                family = kind(operand)
                if arithmetic and family in {'boolean', 'text'}:
                    message = 'This arithmetic operand has a known nonnumeric type.'
                elif not arithmetic and (family in {'date', 'time', 'timestamp', 'interval'} or
                                         dialect == 'postgres' and family in {'number', 'text'}):
                    message = 'This logical operand has a known non-boolean type.'
            if message and (warning := _warning(operand, message)) is not None:
                yield warning
