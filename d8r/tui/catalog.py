"""PostgreSQL catalog branches and a read-only local definition viewer."""
from __future__ import annotations

from dataclasses import dataclass

from rich.text import Text
from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Static, TextArea

from d8r.engine import DataSource
from d8r.sql import CatalogObject, RoutineDef, quote_identifier, relation_parts


@dataclass(frozen=True)
class DefinitionLink:
    kind: str
    name: str
    title: str
    oid: int | None = None


def definition_leaf(node, label: Text, link: DefinitionLink) -> None:
    node.add_leaf(label, data={"kind": "definition", "name": link.name, "link": link})


def add_catalog_groups(root, source: DataSource) -> None:
    groups: dict[str, list[CatalogObject | RoutineDef]] = {name: [] for name in (
        "Sequences", "Extensions", "Collations", "Types", "Routines",
    )}
    for obj in source.objects:
        group = {"sequence": "Sequences", "extension": "Extensions", "collation": "Collations",
                 "composite": "Types", "domain": "Types"}[obj.kind]
        groups[group].append(obj)
    # DuckDB routine hints do not have a PostgreSQL definition target.
    if source.dialect == "postgres":
        groups["Routines"].extend(source.routines)
    for name, entries in groups.items():
        if entries:
            root.add(Text(f"{name} ({len(entries)})"), data={"kind": "catalog_group", "name": name,
                                                             "entries": entries, "by_schema": name != "Extensions"})


def expand_catalog_group(node) -> None:
    data = node.data
    if data.get("loaded"):
        return
    data["loaded"] = True
    if data["by_schema"]:
        groups = {}
        for obj in data["entries"]:
            parts = relation_parts(obj.name) or (obj.name,)
            schema = quote_identifier(parts[-2]) if len(parts) > 1 else "Unqualified"
            groups.setdefault(schema, []).append(obj)
        for schema, entries in groups.items():
            node.add(Text(f"{schema} ({len(entries)})"), data={"kind": "catalog_group", "name": schema,
                                                             "entries": entries, "by_schema": False})
        return
    for obj in data["entries"]:
        if isinstance(obj, RoutineDef):
            title = f"{obj.name}({', '.join(obj.parameters)})"
            detail = obj.kind + (f" → {obj.return_type}" if obj.return_type else "")
            label = Text.assemble(title, ("  " + detail, "dim"))
            if obj.kind == "aggregate":
                node.add_leaf(label, data={"kind": "note", "name": obj.name})
            else:
                definition_leaf(node, label, DefinitionLink("routine", obj.name, title, obj.oid))
        else:
            detail = " · ".join(filter(None, (obj.kind, obj.detail)))
            label = Text.assemble(obj.name, ("  " + detail, "dim"))
            link = DefinitionLink(obj.kind, obj.name, obj.name)
            if obj.columns:
                child = node.add(label, data={"kind": "schema_group", "name": obj.name})
                for column in obj.columns:
                    child.add_leaf(Text.assemble(column.name, ("  " + column.type, "dim")),
                                   data={"kind": "note", "name": column.name})
                definition_leaf(child, Text("Definition"), link)
            else:
                definition_leaf(node, label, link)


class CatalogDefinitionScreen(ModalScreen):
    """Definition text is deliberately not a query draft, result, or history item."""

    BINDINGS = [Binding("escape", "close", "Close definition")]
    DEFAULT_CSS = """
    CatalogDefinitionScreen { align: center middle; background: $background 65%; }
    CatalogDefinitionScreen > Vertical { width: 88%; height: 84%; padding: 1 2; border: round $accent; background: $surface; }
    CatalogDefinitionScreen TextArea { height: 1fr; }
    CatalogDefinitionScreen Horizontal { height: auto; align-horizontal: right; }
    """

    def __init__(self, link: DefinitionLink, definition: str) -> None:
        super().__init__()
        self.link = link
        self.definition = definition

    def compose(self) -> ComposeResult:
        kind = self.link.kind
        title = "View query" if kind in {"view", "materialized view"} else f"{kind.capitalize()} definition"
        with Vertical():
            yield Static(Text(f"{title} · {self.link.title}"), classes="pane-title")
            yield TextArea(self.definition, id="catalog-definition-text", read_only=True, show_line_numbers=True)
            with Horizontal():
                yield Button("Copy", id="catalog-definition-copy")
                yield Button("Close", id="catalog-definition-close")

    @on(Button.Pressed, "#catalog-definition-copy")
    def _copy(self) -> None:
        self.app.copy_to_clipboard(self.definition)

    @on(Button.Pressed, "#catalog-definition-close")
    def action_close(self) -> None:
        self.dismiss()
