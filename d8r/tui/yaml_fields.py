"""Inline controls for one field of the YAML draft. The owner applies edits."""
from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import Button, Input, Select, Static, TextArea

from .editing import DraftTextArea


_KINDS = ("text", "integer", "number", "boolean", "null", "mapping", "list")
_NO_VALUE = {"null", "mapping", "list"}


class YamlFields(Vertical):
    """A live, nonmodal editor. Only begin/close change the editing session."""

    class Submitted(Message):
        """The owner should validate and apply the current values."""

    class Cancelled(Message):
        """The owner should discard the current edit."""

    BINDINGS = [
        Binding("ctrl+s", "apply", "Apply", show=False, priority=True),
        Binding("enter", "submit_text", "Apply", show=False, priority=True),
        Binding("shift+enter", "line_break", "Insert newline", show=False, priority=True),
        Binding("escape", "cancel", "Cancel", show=False, priority=True),
        Binding("tab", "screen.focus_next", "Next field", show=False),
        Binding("shift+tab", "screen.focus_previous", "Previous field", show=False),
    ]
    DEFAULT_CSS = """
    YamlFields {
        display: none; height: auto;
        .field-inputs { height: auto; }
        #yaml-key { width: 2fr; margin-right: 1; }
        #yaml-kind { width: 11; margin-right: 1; }
        #yaml-value { width: 3fr; height: 1; }
        .field-actions { height: 1; }
        Button { width: auto; min-width: 6; margin-right: 1; }
        #yaml-error { display: none; height: auto; max-height: 3; overflow-y: auto; color: $text-warning; }
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._initial: tuple[str | None, str, str] | None = None

    def compose(self) -> ComposeResult:
        with Horizontal(classes="field-inputs"):
            self._name_input = Input(placeholder="Field name", id="yaml-key", compact=True)
            self._kind_input = Select([(kind.title(), kind) for kind in _KINDS], value="text",
                                      allow_blank=False, id="yaml-kind", compact=True)
            self._scalar = DraftTextArea(placeholder="Value", tab_behavior="focus", compact=True,
                                         tooltip="Enter applies; Shift+Enter inserts a line break", id="yaml-value")
            yield self._name_input
            yield self._kind_input
            yield self._scalar
        with Horizontal(classes="field-actions"):
            yield Button("Apply", id="yaml-apply", variant="primary", compact=True)
            yield Button("Cancel", id="yaml-cancel", compact=True)
        yield Static(id="yaml-error", markup=False)

    def begin(self, *, key: str | None, kind: str, value: str, part: str,
              has_children: bool, adding: bool, resource: str | None = None) -> None:
        """Show a field and route the very next keystroke to the requested control."""
        self._initial = (key, kind, "" if kind in _NO_VALUE else value)
        self._name_input.value = key if key is not None else ""
        self._name_input.display = key is not None
        self._kind_input.expanded = False
        self._kind_input.value = kind
        self._kind_input.display = resource is None
        self._kind_input.disabled = has_children or resource is not None
        self._scalar.placeholder = f"{resource} name" if resource else "Value"
        self._scalar.load_text(self._initial[2])
        self._scalar.display = kind not in _NO_VALUE
        self._resize_scalar()
        self.query_one("#yaml-apply", Button).label = f"Add {resource}" if resource else "Add" if adding else "Apply"
        self.show_error("")
        self.display = True
        self.focus_part(part)

    def focus_part(self, part: str) -> None:
        """Refocus a displayed field without reloading or committing its draft."""
        if self._initial is None:
            return

        if part == "value" and self._scalar.display:
            target = self._scalar
            self._scalar.select_all()
        elif part == "kind" and not self._kind_input.disabled:
            target = self._kind_input
        elif self._name_input.display:
            target = self._name_input
            self._name_input.select_all()
        elif not self._kind_input.disabled:
            target = self._kind_input
        else:
            target = self.query_one("#yaml-cancel", Button)
        self.screen.set_focus(target)

    def values(self) -> tuple[str | None, str, str]:
        if self._initial is None:
            return None, "text", ""
        key = None if self._initial[0] is None else self._name_input.value
        kind = self._kind_input.value
        assert isinstance(kind, str)  # The Select has no blank option.
        return key, kind, "" if kind in _NO_VALUE else self._scalar.text

    @property
    def dirty(self) -> bool:
        return self._initial is not None and self.values() != self._initial

    def close(self) -> None:
        self.display = False
        if self._initial is not None:
            self._initial = None
            self._kind_input.expanded = False
            self._kind_input.value = "text"
            self._kind_input.disabled = False
            self._name_input.value = ""
            self._scalar.load_text("")
            self.show_error("")

    def show_error(self, message: str) -> None:
        error = self.query_one("#yaml-error", Static)
        error.update(message)
        error.display = bool(message)

    @on(Select.Changed, "#yaml-kind")
    def _kind_changed(self, event: Select.Changed) -> None:
        event.stop()
        # Another begin() may already have superseded the queued change.
        self._scalar.display = self._kind_input.value not in _NO_VALUE

    @on(TextArea.Changed, "#yaml-value")
    def _resize_scalar(self) -> None:
        self._scalar.styles.height = min(4, self._scalar.document.line_count)

    def check_consume_key(self, key: str, character: str | None = None) -> bool:
        # Keep ancestor tree shortcuts out of the whole row, including its buttons.
        return key in {"a", "c", "delete", "enter", "shift+enter", "ctrl+s", "escape"}

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "submit_text":
            # Preserve native Enter on the type menu and both buttons.
            return self.screen.focused in (self._name_input, self._scalar)
        if action == "line_break":
            return self.screen.focused is self._scalar
        if action == "cancel" and self._kind_input.expanded:
            return False  # Escape dismisses the type menu first.
        return super().check_action(action, parameters)

    def action_submit_text(self) -> None:
        self.action_apply()

    def action_line_break(self) -> None:
        self._scalar.replace("\n", *self._scalar.selection, maintain_selection_offset=False)

    @on(Button.Pressed, "#yaml-apply")
    def _apply_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_apply()

    def action_apply(self) -> None:
        if self._initial is not None:
            self.post_message(self.Submitted())

    @on(Button.Pressed, "#yaml-cancel")
    def _cancel_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        self.action_cancel()

    def action_cancel(self) -> None:
        if self._initial is not None:
            self.post_message(self.Cancelled())
