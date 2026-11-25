"""A navigable tree of literal YAML field names and values."""
from __future__ import annotations

from typing import Literal

from rich.style import Style
from rich.text import Text
from textual import events, on
from textual.message import Message
from textual.widgets import Tree
from textual.widgets.tree import TreeNode

from d8r.yaml_form import YamlFormDocument, YamlPath


_Part = Literal["key", "value"]
_CONTAINERS = {"mapping", "list"}
_KEY = Style(meta={"yaml_part": "key"})
_VALUE = Style(dim=True, meta={"yaml_part": "value"})


def _preview(value: str, limit: int = 100) -> str:
    """Only the display is abbreviated; activation carries the actual path."""
    return "".join(char if char.isprintable() else ascii(char)[1:-1] for char in value[:limit]) + (
        "…" if len(value) > limit else ""
    )


def _item_field(doc: YamlFormDocument, path: YamlPath) -> YamlPath | None:
    """Put a list item's identifying field on its row instead of inventing an index label."""
    if not isinstance(path[-1], int) or doc.kind(path) != "mapping":
        return None
    children = doc.children(path)
    name = (*path, "name")
    if name in children and doc.kind(name) not in _CONTAINERS:
        return name
    return next((child for child in children if doc.kind(child) not in _CONTAINERS), None)


def _label(doc: YamlFormDocument, path: YamlPath, *, field: YamlPath | None = None, filename: str = "") -> Text:
    item = bool(path) and isinstance(path[-1], int)
    label = Text("- " if item else filename)
    target = field or path
    named = bool(target) and isinstance(target[-1], str)
    if named:
        label.append(_preview(target[-1]) or '""', _KEY)
        label.append(":")
    kind = doc.kind(target)
    if kind not in _CONTAINERS:
        value = _preview(doc.value(target)) or '""'
    elif not doc.children(target):
        value = "{}" if kind == "mapping" else "[]"
    elif item:
        value = _preview(doc.first_line(target, 101))
    else:
        return label  # The group's contents are already on the rows beneath it.
    if named or not item:
        label.append(" ")
    label.append(value, _VALUE)
    return label


class YamlFieldTree(Tree[YamlPath]):
    """Show the whole document, independently of the editor that consumes activations."""

    class Activated(Message):
        def __init__(self, tree: YamlFieldTree, document: YamlFormDocument, path: YamlPath, part: _Part) -> None:
            super().__init__()
            self.tree = tree
            self.document = document
            self.path = path
            self.part = part

        @property
        def control(self) -> YamlFieldTree:
            return self.tree

    def __init__(self, label: str | Text, **kwargs) -> None:
        self._revision = 0
        self._hit_style = Style(meta={"yaml_revision": self._revision})
        super().__init__(label, **kwargs)
        self.guide_depth = 2
        self.auto_expand = False
        self.root.allow_expand = False
        self._document: YamlFormDocument | None = None
        self._by_path: dict[YamlPath, TreeNode[YamlPath]] = {}
        self._item_fields: dict[YamlPath, YamlPath] = {}

    def render_label(self, node: TreeNode[YamlPath], base_style: Style, style: Style) -> Text:
        label = super().render_label(node, base_style, style)
        # Textual recycles node IDs on reset. Include the revision on the arrow too.
        label.stylize(self._hit_style)
        return label

    @property
    def current_path(self) -> YamlPath | None:
        node = self.cursor_node
        if node is None or node.data is None or self._by_path.get(node.data) is not node:
            return None
        return node.data

    @property
    def current_field(self) -> YamlPath | None:
        path = self.current_path
        return self._item_fields.get(path, path) if path is not None else None

    def show_document(self, doc: YamlFormDocument | None, filename: str, selection: YamlPath = (), *,
                      keep_folds: bool = False) -> None:
        """Rebuild from a snapshot, revealing the selection even beneath a saved fold."""
        folds = {}
        if keep_folds and self._document is not None:
            folds = {path: self._document.kind(path) for path, node in self._by_path.items()
                     if node.allow_expand and not node.is_expanded}

        def expanded(path: YamlPath, kind: str, field: YamlPath | None = None) -> bool:
            reveal_child = len(selection) > len(path) and selection[:len(path)] == path and selection != field
            return folds.get(path) != kind or reveal_child

        self.unselect()
        self.hover_line = -1
        self.root.remove_children()
        self._by_path.clear()
        self._item_fields.clear()
        self._revision += 1
        self._hit_style = Style(meta={"yaml_revision": self._revision})
        self._document = doc
        if doc is None:
            self.reset("Document")
            self.root.allow_expand = False
            self.root.collapse()
        else:
            kind = doc.kind(())
            self.reset(_label(doc, (), filename=filename), ())
            self.root.allow_expand = kind in _CONTAINERS
            self._by_path[()] = self.root
            if self.root.allow_expand and expanded((), kind):
                self.root.expand()
            else:
                self.root.collapse()

            def add_children(parent: TreeNode[YamlPath], path: YamlPath) -> None:
                for child in doc.children(path):
                    if child == self._item_fields.get(path):
                        continue  # This field occupies the item heading; show it just once.
                    kind = doc.kind(child)
                    group = kind in _CONTAINERS
                    field = _item_field(doc, child) if group else None
                    if field is not None:
                        self._item_fields[child] = field
                    node = parent.add(_label(doc, child, field=field), child, allow_expand=group,
                                      expand=group and expanded(child, kind, field))
                    self._by_path[child] = node
                    if group:
                        add_children(node, child)

            add_children(self.root, ())

        while selection and selection not in self._by_path:
            selection = selection[:-1]
        # Resolve native visible line numbers before moving the cursor to a new node.
        self.get_node_at_line(0)
        self.move_cursor(self._by_path.get(selection, self.root))
        if not selection:
            self.scroll_home(animate=False, immediate=True)

    def on_click(self, event: events.Click) -> None:
        meta = event.style.meta
        line = meta.get("line")
        if line is None:
            return
        node = self.get_node_at_line(line)
        if (event.button != 1 or node is None or meta.get("yaml_revision", self._revision) != self._revision
                or meta.get("node", node.id) != node.id):
            event.prevent_default()
            return
        if meta.get("toggle"):
            self.cursor_line = line  # Select the group, leaving disclosure to Tree.
            return
        part = meta.get("yaml_part")
        if part in ("key", "value"):
            event.prevent_default()  # Don't also dispatch the generic row activation.
            self.cursor_line = line
            self._activate(node, part)

    @on(Tree.NodeSelected)
    def _selected(self, event: Tree.NodeSelected[YamlPath]) -> None:
        self._activate(event.node)

    def _activate(self, node: TreeNode[YamlPath], part: _Part | None = None) -> None:
        doc, path = self._document, node.data
        if doc is None or path is None or self._by_path.get(path) is not node:
            return
        path = self._item_fields.get(path, path)
        if doc.kind(path) in _CONTAINERS:
            named = bool(path) and isinstance(path[-1], str)
            if doc.children(path):
                if not named:
                    node.toggle()
                    return
                part = "key"  # A populated group can't change type.
            elif part is None and named:
                part = "key"
        self.post_message(self.Activated(self, doc, path, part or "value"))
