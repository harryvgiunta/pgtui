"""Inline field editing over the workspace's current YAML draft."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static, Tree

from d8r.yaml_form import YamlFormDocument, YamlFormError, YamlPath
from .yaml_fields import YamlFields
from .yaml_tree import YamlFieldTree


def _resource(collection: YamlPath) -> str | None:
    match collection:
        case ("sources",):
            return "Source"
        case ("sources", int(), "tables"):
            return "Table"
        case ("sources", int(), "tables", int(), "columns"):
            return "Column"
    return None


@dataclass(frozen=True)
class _FieldEdit:
    document: YamlFormDocument
    path: YamlPath
    operation: str
    promote: YamlPath | None = None

    @property
    def resource(self) -> str | None:
        return _resource(self.path) if self.operation == "append_named" else None

    @property
    def adds_key(self) -> bool:
        return self.operation == "add" and (self.promote is not None or self.document.kind(self.path) == "mapping")

    @property
    def subject(self) -> str:
        return self.resource or ("Field" if self.adds_key else "Item")


class YamlForm(Vertical):
    """Browse, edit, and confirm removals without leaving the workspace screen."""

    BINDINGS = [Binding("a", "add_field", "Add field or item", show=False),
                Binding("c", "add_child", "Add named resource", show=False),
                Binding("delete", "remove_field", "Remove field or item", show=False),
                Binding("escape", "keep_field", "Keep field", show=False)]
    DEFAULT_CSS = """
    YamlForm {
        display: none; height: 1fr;
        #yaml-tree { height: 1fr; }
        .yaml-tools { height: 1; }
        #yaml-tools.narrow { layout: vertical; height: auto; }
        .yaml-tools Button { width: auto; min-width: 5; margin-right: 1; }
        #yaml-removal { display: none; }
        #yaml-remove-label { width: 1fr; text-overflow: ellipsis; text-wrap: nowrap; }
    }
    """

    def __init__(self, apply_edit: Callable[[str, str], bool], **kwargs) -> None:
        super().__init__(**kwargs)
        self._apply_edit = apply_edit
        self._document: YamlFormDocument | None = None
        self._filename = ""
        self._pending: _FieldEdit | None = None

    @property
    def text(self) -> str:
        return self._document.text if self._document is not None else ""

    @property
    def dirty(self) -> bool:
        return self._pending is not None and self._pending.operation != "remove" and self.query_one(YamlFields).dirty

    def compose(self) -> ComposeResult:
        yield YamlFieldTree("Document", id="yaml-tree")
        yield YamlFields(id="yaml-fields")
        with Horizontal(id="yaml-tools", classes="yaml-tools"):
            yield Button("Add Field (A)", id="yaml-add", compact=True)
            yield Button(id="yaml-add-child", compact=True)
            yield Button("Remove (Del)", id="yaml-remove", compact=True)
        with Horizontal(id="yaml-removal", classes="yaml-tools"):
            yield Static(id="yaml-remove-label", markup=False)
            yield Button("Remove", id="yaml-confirm", variant="error", compact=True)
            yield Button("Keep", id="yaml-keep", compact=True)

    def on_mount(self) -> None:
        self.query_one(YamlFieldTree).show_document(self._document, self._filename)
        self._sync_tools()

    def open_document(self, text: str, filename: str) -> None:
        doc = YamlFormDocument(text)
        self._filename = filename
        self._show_document(doc)

    def clear_document(self) -> None:
        self._filename = ""
        self._show_document(None)

    def _show_document(self, doc: YamlFormDocument | None) -> None:
        self._document = doc
        if self.is_mounted:
            self._close_fields()
            self.query_one(YamlFieldTree).show_document(doc, self._filename)
            self._sync_tools()

    def _close_fields(self) -> None:
        self._pending = None
        self.query_one(YamlFields).close()
        self.query_one("#yaml-removal").display = False
        self.query_one("#yaml-tools").display = True

    def on_resize(self, event: events.Resize) -> None:
        # Stack short, single-line buttons rather than clipping the contextual action.
        self.query_one("#yaml-tools").set_class(event.size.width < 52, "narrow")

    def _add_targets(self, path: YamlPath | None) -> tuple[_FieldEdit | None, _FieldEdit | None]:
        """Resolve both toolbar actions from the same row and document snapshot."""
        doc = self._document
        if doc is None or path is None:
            return None, None
        kind = doc.kind(path)
        ordinary = _FieldEdit(doc, path, "add") if kind in {"mapping", "list"} else None
        stem = Path(self._filename).stem.casefold()
        if stem in {"dbt_project", "profiles", "packages", "dependencies", "package-lock", "selectors"}:
            return ordinary, None
        if not path and kind == "mapping":
            keys = doc.children(())
            if ("sources",) in keys or (not keys and stem == "sources"):
                return ordinary, _FieldEdit(doc, ("sources",), "append_named")
        if _resource(path):
            return _FieldEdit(doc, path, "append_named"), None

        # A scalar property remains a useful landing spot after adding it: offer
        # the same additions as its owning resource, but don't change Delete.
        owner = path[:-1] if path and isinstance(path[-1], str) and kind not in {"mapping", "list"} else path
        if not owner or not isinstance(owner[-1], int):
            return ordinary, None
        resource = _resource(owner[:-1])
        if resource is None or doc.kind(owner) not in {"mapping", "text"}:
            return ordinary, None
        promote = owner if doc.kind(owner) == "text" else None
        add_field = _FieldEdit(doc, owner, "add", promote)
        if resource == "Column":  # C adds a sibling; it needn't touch the existing column.
            return add_field, _FieldEdit(doc, owner[:-1], "append_named")
        child = (*owner, "tables" if resource == "Source" else "columns")
        return add_field, _FieldEdit(doc, child, "append_named", promote)

    @on(Tree.NodeHighlighted, "#yaml-tree")
    def _sync_tools(self) -> None:
        path = self.query_one(YamlFieldTree).current_path
        primary, contextual = self._add_targets(path)
        add = self.query_one("#yaml-add", Button)
        add.disabled = primary is None
        add.label = f"Add {primary.subject} (A)" if primary else "Add (A)"
        child = self.query_one("#yaml-add-child", Button)
        child.display = contextual is not None
        if contextual is not None:
            child.label = f"Add {contextual.subject} (C)"
        self.query_one("#yaml-remove", Button).disabled = not path

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "keep_field":
            return self._pending is not None and self._pending.operation == "remove"
        if action in {"add_field", "add_child", "remove_field"}:
            return self.query_one(YamlFieldTree).has_focus or self.query_one("#yaml-tools").has_focus_within
        return super().check_action(action, parameters)

    def _settle_for(self, path: YamlPath | None) -> YamlPath | None:
        """A click made during a rename still addresses the same branch after it lands."""
        if path is None:
            return None
        pending = self._pending
        if pending and pending.operation == "edit" and pending.path and path[:len(pending.path)] == pending.path:
            key, _, _ = self.query_one(YamlFields).values()
            if key is not None:
                path = (*pending.path[:-1], key, *path[len(pending.path):])
        return path if self.finish_edit(next_selection=path) else None

    @on(YamlFieldTree.Activated)
    def _activate(self, event: YamlFieldTree.Activated) -> None:
        if self._document is not event.document:
            return
        if self._pending and self._pending.operation == "edit" and self._pending.path == event.path:
            self.query_one(YamlFields).focus_part(event.part)
        else:
            path = self._settle_for(event.path)
            if path is not None and self._document is not None:
                self._start(_FieldEdit(self._document, path, "edit"), event.part)

    @on(Button.Pressed, "#yaml-add")
    def action_add_field(self) -> None:
        self._start_add(child=False)

    @on(Button.Pressed, "#yaml-add-child")
    def action_add_child(self) -> None:
        self._start_add(child=True)

    def _start_add(self, *, child: bool) -> None:
        path = self._settle_for(self.query_one(YamlFieldTree).current_path)
        primary, contextual = self._add_targets(path)
        target = contextual if child else primary
        if target is not None:
            self._start(target, "key")

    def _start(self, edit: _FieldEdit, part: str) -> None:
        doc, path = edit.document, edit.path
        resource = edit.resource
        adding = edit.operation != "edit"
        if adding:
            key = "" if edit.adds_key else None
            part = "key" if key is not None else "value"
        else:
            key = path[-1] if path and isinstance(path[-1], str) else None
        self._pending = edit
        self.query_one(YamlFields).begin(key=key, kind="text" if adding else doc.kind(path),
                                         value="" if adding else doc.value(path), part=part,
                                         has_children=not adding and bool(doc.children(path)),
                                         adding=adding, resource=resource)

    @on(Button.Pressed, "#yaml-remove")
    def action_remove_field(self) -> None:
        tree = self.query_one(YamlFieldTree)
        target = tree.current_path
        if self._pending and self._pending.operation == "edit" and self._pending.path == tree.current_field:
            target = self._pending.path  # Remove the inline field, not its containing list item.
        path = self._settle_for(target)
        if self._document is None or not path:
            return
        self._pending = _FieldEdit(self._document, path, "remove")
        subject = (f"field {path[-1]!r}" if isinstance(path[-1], str)
                   else f"item {tree.cursor_node.label.plain.removeprefix('- ')}")
        self.query_one("#yaml-remove-label", Static).update(f"Remove {subject} and its contents?")
        self.query_one("#yaml-tools").display = False
        self.query_one("#yaml-removal").display = True
        self.query_one("#yaml-keep").focus()

    @on(YamlFields.Cancelled)
    @on(Button.Pressed, "#yaml-keep")
    def action_keep_field(self) -> None:
        self._close_fields()
        self.query_one(YamlFieldTree).focus()

    @on(YamlFields.Submitted)
    @on(Button.Pressed, "#yaml-confirm")
    def _submit(self) -> None:
        if self.finish_edit(explicit=True):
            self.query_one(YamlFieldTree).focus()

    def on_descendant_blur(self, event: events.DescendantBlur) -> None:
        if self._pending is not None:
            self.call_after_refresh(self._finish_if_left, self._pending)

    def _finish_if_left(self, pending: _FieldEdit) -> None:
        if self._pending is pending and not self.has_focus_within:
            self.finish_edit()

    def finish_edit(self, *, explicit: bool = False, next_selection: YamlPath | None = None) -> bool:
        """Commit valid live fields on navigation. Only a confirmation can delete."""
        pending = self._pending
        if pending is None:
            return True
        fields = self.query_one(YamlFields)
        op, path, basis = pending.operation, pending.path, pending.document
        if not explicit and (op == "remove" or not fields.dirty):
            self._close_fields()
            return True
        try:
            if self._document is not basis:
                raise YamlFormError("The draft changed. Cancel and select the field again.")
            if op == "remove":
                after, selected = basis.change(op, path), path[:-1]
            else:
                key, kind, value = fields.values()
                # Promotion and append run on private snapshots. Only their combined
                # result goes through the workspace identity guard and undo stack.
                working = basis.as_named_mapping(pending.promote) if pending.promote is not None else basis
                if op == "append_named":
                    after, selected = working.append_named_item(path, value)
                else:
                    after = working.change(op, path, key=key, kind=kind, value=value)
                    if op == "add":
                        selected = (*path, key if key is not None else len(working.children(path)))
                    else:
                        selected = (*path[:-1], key) if key is not None else path
            candidate = YamlFormDocument(after) if after != basis.text else basis
            if not self._apply_edit(basis.text, after):
                raise YamlFormError("The workspace draft changed. Cancel and select the field again.")
        except YamlFormError as exc:
            if op == "remove":
                self.query_one("#yaml-remove-label", Static).update(str(exc))
            else:
                fields.show_error(str(exc))
            return False
        self._close_fields()
        if self._document is basis and candidate is not basis:
            self._document = candidate
            target = selected if next_selection is None else next_selection
            self.query_one(YamlFieldTree).show_document(candidate, self._filename, target, keep_folds=True)
            self._sync_tools()
        return True
