"""Immutable completion-only context and edit proposals; never executable query ASTs."""
from __future__ import annotations

from dataclasses import dataclass

from d8r.sql import ForeignKey


@dataclass(frozen=True)
class Field:
    name: str
    type: str = ""
    table: str | None = None
    column: str | None = None
    doc: str = ""
    values: tuple[object, ...] | None = None


@dataclass(frozen=True)
class Relation:
    name: str
    reference: str
    fields: tuple[Field, ...] = ()
    table: str | None = None
    doc: str = ""
    foreign_keys: tuple[ForeignKey, ...] = ()
    outer: bool = False


@dataclass(frozen=True)
class Call:
    name: str
    argument: int = 0
    start: int = 0


@dataclass(frozen=True)
class Context:
    start: int
    end: int
    token: str = ""
    qualifier: str = ""
    quote: str = ""
    clause: str = "statement"
    blocked: bool = False
    relations: tuple[Relation, ...] = ()
    ctes: tuple[Relation, ...] = ()
    outputs: tuple[Field, ...] = ()
    join_target: Relation | None = None
    join_left: tuple[Relation, ...] = ()
    call: Call | None = None
    value_field: Field | None = None
    value_prefix: str = ""
    literal: bool = False
    used_columns: tuple[str, ...] = ()
    statement_start: int = 0
    statement_end: int = 0
    # Visible aliases include CTE aliases and explicit query aliases; avoid collisions.
    identifiers: tuple[str, ...] = ()


@dataclass(frozen=True)
class Suggestion:
    label: str
    insert: str
    detail: str = ""
    kind: str = "keyword"
    documentation: str = ""
    cursor_back: int = 0
    # Optional absolute span override (e.g. qualified star expansion).
    start: int | None = None
    end: int | None = None
    table: str | None = None


@dataclass(frozen=True)
class Signature:
    label: str
    parameters: tuple[str, ...]
    active_parameter: int
    return_type: str = ""
    documentation: str = ""
    separator: str = ", "


@dataclass(frozen=True)
class Completion:
    context: Context
    items: tuple[Suggestion, ...] = ()
    signature: Signature | None = None
    hint: str = ""
