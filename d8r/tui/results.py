"""Buffered result selection and explicit CSV export; NULL is an empty CSV/TSV cell."""

from __future__ import annotations

import asyncio
import csv
import io
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

from rich.style import Style
from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Text
from textual.coordinate import Coordinate
from textual.geometry import Size
from textual.screen import ModalScreen
from textual.widgets import Button, DataTable, Input, Select, Static, Tree

from .session import ResultPage, RunOutcome, Session

MIN_COLUMN_WIDTH = 4


_PLAN_QUALS = ("Filter", "Index Cond", "Recheck Cond", "Hash Cond", "Merge Cond", "Join Filter",
               "One-Time Filter", "TID Cond", "TID Range Cond", "Run Condition")
_PLAN_WHITESPACE = str.maketrans({"\n": r"\n", "\r": r"\r", "\t": r"\t"})


def _plan_label(node: dict) -> Text:
    label = Text(node["Node Type"], style="bold")
    if join := node.get("Join Type"):
        label.append(f" · {join} join")
    if index := node.get("Index Name"):
        label.append(f" using {index}")
    relation = node.get("Relation Name") or node.get("CTE Name") or node.get("Function Name")
    if relation:
        label.append(f" on {relation}")
        if (alias := node.get("Alias")) and alias != relation:
            label.append(f" as {alias}")
    if subplan := node.get("Subplan Name"):
        label.append(f" · {subplan}")
    label.plain = label.plain.translate(_PLAN_WHITESPACE)
    label.append(f'  rows={node["Plan Rows"]:,}  cost={node["Startup Cost"]:,.2f}..{node["Total Cost"]:,.2f}', "dim")
    return label


class PlanTree(Tree[None]):
    """An in-memory view of the engine's bounded EXPLAIN (FORMAT JSON) result."""

    DEFAULT_CSS = "PlanTree { height: 1fr; }"
    BINDINGS = [
        Binding("pageup", "page_up", "Plan page up", show=False),
        Binding("pagedown", "page_down", "Plan page down", show=False),
        Binding("home", "scroll_home", "First plan line", show=False),
        Binding("end", "scroll_end", "Last plan line", show=False),
        Binding("left", "scroll_left", "Scroll plan left", show=False),
        Binding("right", "scroll_right", "Scroll plan right", show=False),
    ]

    def show_plan(self, result: dict) -> None:
        root = self.root
        root.remove_children()
        if result["truncated"]:
            root.set_label(Text(f'The plan exceeded {result["plan_limit"]:,} characters and was withheld in full.'))
            root.allow_expand = False
        else:
            root.set_label(Text("Estimated plan · ANALYZE off · costs in planner units", style="bold"))
            root.allow_expand = True
            root.expand()
            pending = [(root, item["Plan"]) for item in reversed(result["plan"])]
            while pending:
                parent, plan = pending.pop()
                children = plan.get("Plans", ())
                quals = [(name, plan[name]) for name in _PLAN_QUALS if name in plan]
                node = parent.add(_plan_label(plan), expand=True, allow_expand=bool(children or quals))
                for name, expression in quals:
                    node.add_leaf(Text.assemble((f"{name}: ", "dim"), expression.translate(_PLAN_WHITESPACE)))
                pending.extend((node, child) for child in reversed(children))
        self.cursor_line = 0
        self.scroll_home(animate=False, immediate=True)


@dataclass(frozen=True)
class ResultSnapshot:
    columns: tuple[str, ...]
    rows: tuple[tuple[object, ...], ...]
    total: int
    selected: tuple[int, ...]
    page: ResultPage | None = None


class ResultsTable(DataTable):
    """Buffered cell ranges, whole-row selection, and column resizing."""

    COMPONENT_CLASSES = DataTable.COMPONENT_CLASSES | {"results--selected-row"}
    DEFAULT_CSS = """
    ResultsTable {
        border: solid $border;
        border-subtitle-align: left;
        & > .results--selected-row {
            background: $primary-darken-2;
            color: $text;
            text-style: bold;
        }
    }
    """
    BINDINGS = [
        Binding("alt+left", "narrow_column", "Narrow column", show=False),
        Binding("alt+right", "widen_column", "Widen column", show=False),
        Binding("ctrl+left", "app.result_page(-1)", "Previous page", show=False),
        Binding("ctrl+right", "app.result_page(1)", "Next page", show=False),
        Binding("shift+left", "extend_selection(0,-1)", "Extend selection left", show=False),
        Binding("shift+right", "extend_selection(0,1)", "Extend selection right", show=False),
        Binding("shift+up", "extend_selection(-1,0)", "Extend selection up", show=False),
        Binding("shift+down", "extend_selection(1,0)", "Extend selection down", show=False),
        Binding("shift+space", "select_row", "Select row", show=False),
        Binding("ctrl+space", "select_column", "Select column", show=False),
        Binding("space", "toggle_row", "Toggle row", show=False),
        Binding("ctrl+a", "select_all_rows", "Select buffered rows", show=False),
        Binding("escape", "clear_selection", "Clear selection", show=False),
        Binding("ctrl+c,super+c", "copy_selection", "Copy selection", show=False),
    ]

    def __init__(self, *args, session: Session | None = None, **kwargs) -> None:
        self._session = session
        self.raw_columns: tuple[str, ...] = ()
        self.raw_rows: tuple[tuple[object, ...], ...] = ()
        self.result_total = 0
        self._page: ResultPage | None = None
        self.selected_rows: set[int] = set()
        self._cell_bounds: tuple[int, int, int, int] | None = None
        self._range_anchor: Coordinate | None = None
        self._selection_mode: Literal["cell", "row", "column"] = "cell"
        self._extending = False
        self._dragging = False
        super().__init__(*args, **kwargs)
        self.border_title = "Drag / Shift+arrows select · Ctrl+C copy · Alt+←/→ resize"
        self._selection_caption()

    @property
    def page_offset(self) -> int:
        return self._page.offset if self._page else 0

    def show_result(self, outcome: RunOutcome) -> None:
        """Replace the grid and its raw buffer, resetting selection and cursor."""
        self.clear(columns=True)
        self.raw_columns = tuple(outcome.columns) if not outcome.error else ()
        self.raw_rows = tuple(tuple(row) for row in outcome.rows) if self.raw_columns else ()
        self.result_total = outcome.total if self.raw_columns else 0
        self._page = outcome.page
        self._end_drag()
        self.action_clear_selection()
        if not self.raw_columns:
            return
        self.add_columns(*[
            Text.assemble(name, (" · " + outcome.dtypes[index], "dim"))
            if index < len(outcome.dtypes) and outcome.dtypes[index] else Text(name)
            for index, name in enumerate(self.raw_columns)
        ])
        for index, row in enumerate(self.raw_rows):
            self.add_row(*[
                Text("NULL", style="dim") if value is None else Text(str(value)) for value in row
            ], label=str(self.page_offset + index + 1))
        self.move_cursor(row=0, column=0)

    def _selection_caption(self) -> None:
        if self._cell_bounds is not None:
            top, left, bottom, right = self._cell_bounds
            selected = f"{(bottom - top + 1) * (right - left + 1):,} cells"
        else:
            selected = f"{len(self.selected_rows):,} rows"
        extent = "more rows available" if self.result_total > len(self.raw_rows) else "all returned rows"
        if self.page_offset and self.result_total <= len(self.raw_rows):
            extent = "last page"
        self.border_subtitle = f"{selected} selected · {len(self.raw_rows):,} buffered · {extent}"

    def _selection_changed(self) -> None:
        # Textual 8's row and line caches do not include owner selection state.
        self._clear_caches()
        self._selection_caption()
        self.refresh()

    def _get_row_style(self, row_index: int, base_style: Style) -> Style:
        # DataTable uses this hook for every cell and the trailing row fill.
        style = super()._get_row_style(row_index, base_style)
        if row_index in self.selected_rows:
            style += self.get_component_rich_style("results--selected-row")
        return style

    def _render_cell(self, row_index: int, column_index: int, base_style: Style,
                     width: int, cursor: bool = False, hover: bool = False):
        selected = row_index in self.selected_rows
        if self._cell_bounds is not None:
            top, left, bottom, right = self._cell_bounds
            selected = (left <= column_index <= right and
                        (top <= row_index <= bottom or
                         row_index == -1 and self._selection_mode == "column"))
        if selected:
            base_style += self.get_component_rich_style("results--selected-row")
            hover = False
        return super()._render_cell(row_index, column_index, base_style, width, cursor, hover)

    def watch_cursor_coordinate(self, old_coordinate: Coordinate, new_coordinate: Coordinate) -> None:
        super().watch_cursor_coordinate(old_coordinate, new_coordinate)
        if not self._extending:
            self._range_anchor = None
            self._selection_mode = "cell"
            if self._cell_bounds is not None:
                self._cell_bounds = None
                self._selection_changed()

    def _select_to(self, target: Coordinate) -> None:
        if not self.raw_rows:
            return
        anchor = self._range_anchor or self.cursor_coordinate
        target = Coordinate(max(0, min(len(self.raw_rows) - 1, target.row)),
                            max(0, min(len(self.raw_columns) - 1, target.column)))
        self._extending = True
        try:
            self.move_cursor(row=target.row, column=target.column, animate=False)
        finally:
            self._extending = False
        self._range_anchor = anchor
        top, bottom = sorted((anchor.row, target.row))
        left, right = sorted((anchor.column, target.column))
        self.selected_rows = set()
        self._cell_bounds = None
        if self._selection_mode == "row":
            self.selected_rows = set(range(top, bottom + 1))
        else:
            if self._selection_mode == "column":
                top, bottom = 0, len(self.raw_rows) - 1
            self._cell_bounds = top, left, bottom, right
        self._selection_changed()

    def action_extend_selection(self, rows: int, columns: int) -> None:
        self._set_hover_cursor(False)
        self._select_to(Coordinate(self.cursor_row + rows, self.cursor_column + columns))

    def action_select_row(self) -> None:
        self._range_anchor = self.cursor_coordinate
        self._selection_mode = "row"
        self._select_to(self.cursor_coordinate)

    def action_select_column(self) -> None:
        self._range_anchor = self.cursor_coordinate
        self._selection_mode = "column"
        self._select_to(self.cursor_coordinate)

    def action_toggle_row(self) -> None:
        if not self.raw_rows:
            return
        self._range_anchor = self.cursor_coordinate
        self._selection_mode = "row"
        self._cell_bounds = None
        self.selected_rows ^= {self.cursor_row}
        self._selection_changed()

    def action_select_all_rows(self) -> None:
        self._range_anchor = None
        self._cell_bounds = None
        self._selection_mode = "row"
        self.selected_rows = set(range(len(self.raw_rows)))
        self._selection_changed()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "clear_selection":
            # Without a selection, let Escape dismiss the enclosing results window.
            return self._cell_bounds is not None or bool(self.selected_rows)
        return super().check_action(action, parameters)

    def action_clear_selection(self) -> None:
        self._range_anchor = None
        self._cell_bounds = None
        self._selection_mode = "cell"
        self.selected_rows = set()
        self._selection_changed()

    def _mouse_cell(self, event: events.MouseEvent) -> Coordinate | None:
        meta = event.style.meta
        row, column = meta.get("row"), meta.get("column")
        if not isinstance(row, int) or not isinstance(column, int) or meta.get("out_of_bounds"):
            return None
        return Coordinate(row, column)

    def _on_mouse_down(self, event: events.MouseDown) -> None:
        event.prevent_default()
        if event.button != 1 or not self.raw_rows:
            return
        cell = self._mouse_cell(event)
        if cell is None:
            return
        event.stop()
        self.focus()
        if cell.row == -1 and cell.column == -1:
            self.action_select_all_rows()
            return
        mode = "column" if cell.row == -1 else "row" if cell.column == -1 else "cell"
        target = Coordinate(max(0, cell.row), max(0, cell.column))
        if not event.shift or self._selection_mode != mode:
            self._range_anchor = target
        self._selection_mode = mode
        if mode == "row" and (event.ctrl or event.meta):
            self.move_cursor(row=target.row, column=target.column)
            self.action_toggle_row()
            return
        self._select_to(target)
        self._dragging = True
        self.capture_mouse()

    def _on_mouse_move(self, event: events.MouseMove) -> None:
        if not self._dragging:
            return
        event.prevent_default()
        event.stop()
        cell = self._mouse_cell(event)
        if cell is not None:
            self._select_to(cell)

    def _end_drag(self) -> None:
        if self._dragging:
            self._dragging = False
            self.release_mouse()

    def _on_mouse_up(self, event: events.MouseUp) -> None:
        event.prevent_default()
        if self._dragging:
            event.stop()
            self._end_drag()

    async def _on_click(self, event: events.Click) -> None:
        # Selection happens on press/move. A release-generated click must not
        # collapse a dragged range or move the cursor a second time.
        event.prevent_default()
        event.stop()

    def _selected_indices(self) -> tuple[list[int] | range, range]:
        if self._cell_bounds is not None:
            top, left, bottom, right = self._cell_bounds
            return range(top, bottom + 1), range(left, right + 1)
        if self.selected_rows:
            return sorted(self.selected_rows), range(len(self.raw_columns))
        return [self.cursor_row], range(self.cursor_column, self.cursor_column + 1)

    def action_copy_selection(self) -> None:
        if not self.raw_columns or not self.raw_rows:
            self.app.notify("No buffered result rows to copy.", severity="warning")
            return
        rows, columns = self._selected_indices()
        output = io.StringIO(newline="")
        writer = csv.writer(output, delimiter="\t")
        writer.writerow(self.raw_columns[column] for column in columns)
        writer.writerows([self.raw_rows[row][column] for column in columns] for row in rows)
        self.app.copy_to_clipboard(output.getvalue())
        self.app.notify(f"Copied {len(rows):,} row(s) × {len(columns):,} field(s) with headers as TSV. NULL is an empty cell.")

    def action_export(self) -> None:
        if not self.raw_columns:
            self.app.notify("No result columns to export. Run a query first.", severity="warning")
            return
        # Isolate even nested engine values from future result mutations.
        snapshot = ResultSnapshot(
            self.raw_columns, deepcopy(self.raw_rows), self.result_total,
            tuple(self._selected_indices()[0]) if self.selected_rows or self._cell_bounds else (),
            self._page,
        )
        self.app.push_screen(ExportModal(snapshot, session=self._session))

    def action_narrow_column(self) -> None:
        self._resize_column(-2)

    def action_widen_column(self) -> None:
        self._resize_column(2)

    def _resize_column(self, delta: int) -> None:
        columns = self.ordered_columns
        if not columns:
            return
        index = min(self.cursor_column, len(columns) - 1)
        column = columns[index]
        before = column.get_render_width(self)
        if column.auto_width:
            column.width = before - 2 * self.cell_padding
            column.auto_width = False
        column.width = max(MIN_COLUMN_WIDTH, column.width + delta)
        after = column.get_render_width(self)
        width, height = self.virtual_size
        self.virtual_size = Size(width + after - before, height)
        self._clear_caches()
        self.refresh_column(index)


class ExportModal(ModalScreen):
    """Export a captured buffer or explicitly re-read its full query result."""

    BINDINGS = [Binding("escape", "cancel", "Cancel")]
    DEFAULT_CSS = """
    ExportModal {
        align: center middle;
        background: $background 65%;
        #export-dialog {
            width: 78;
            max-width: 95%;
            height: 26;
            max-height: 95%;
            border: thick $primary;
            background: $surface;
            padding: 1 2;
        }
        #export-title { text-style: bold; margin-bottom: 1; }
        #export-form { height: 1fr; }
        #export-summary, #export-message { height: auto; margin: 1 0; }
        #export-buttons { height: auto; }
        Button { margin-right: 1; }
        #export-overwrite { display: none; }
    }
    """

    def __init__(self, snapshot: ResultSnapshot, *, session: Session | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.snapshot = snapshot
        self._session = session
        self._can_export_full = (session is not None and session.has_result_source(snapshot.page)
                                 and not session.tx_state().open)
        self._overwrite_path: Path | None = None
        self._busy = False

    def compose(self) -> ComposeResult:
        from .settings import binding_tooltip

        with Vertical(id="export-dialog"):
            yield Static("Export results to CSV", id="export-title")
            with VerticalScroll(id="export-form"):
                yield Static("Destination path (UTF-8 CSV; NULL becomes an empty cell)", markup=False)
                yield Input(placeholder="Enter a destination .csv path", id="export-path")
                options = [("All buffered rows", "all")]
                if self.snapshot.selected:
                    options.append(("Selected rows", "selected"))
                if self._can_export_full:
                    options.append(("Full query result (rerun)", "full"))
                yield Select(options, value="selected" if self.snapshot.selected else "all",
                             allow_blank=False, id="export-scope")
                yield Static("", id="export-summary", markup=False)
                yield Static("", id="export-message", markup=False)
            with Horizontal(id="export-buttons"):
                yield Button("Export CSV", variant="primary", id="export-save")
                yield Button("Overwrite file", variant="error", id="export-overwrite")
                yield Button("Cancel", id="export-cancel", tooltip=binding_tooltip(self.BINDINGS, "cancel"))

    def on_mount(self) -> None:
        self._update_summary()
        self.query_one("#export-path", Input).focus()

    def _indices(self) -> tuple[int, ...] | range | None:
        scope = self.query_one("#export-scope", Select).value
        if scope == "full":
            return None
        if scope == "selected":
            return self.snapshot.selected
        return range(len(self.snapshot.rows))

    def _update_summary(self) -> None:
        snapshot = self.snapshot
        scope = self.query_one("#export-scope", Select).value
        indices = self._indices()
        summary = self.query_one("#export-summary", Static)
        if indices is None:
            text = ("Export every row returned by this tab’s read-only query, plus headers.\n"
                    "The query runs again from the beginning. Its LIMIT and OFFSET still apply; data may have changed.")
            if snapshot.page.source.kind == "d1-live":
                text += "\nD1 reads separate pages. Use a unique ORDER BY; changes between pages can shift rows."
            summary.update(text)
            return
        extent = "More rows are available; their total was not counted." if snapshot.total > len(snapshot.rows) else "All returned rows are buffered."
        if snapshot.page and snapshot.page.offset:
            extent = f"Current page only, at offset {snapshot.page.offset:,}."
        elif snapshot.total > len(snapshot.rows) and not self._can_export_full:
            extent = "Full export requires a pageable read on its original connection."
        summary.update(
            f"Exporting {len(indices):,} {scope} row(s), plus headers.\n"
            f"{len(snapshot.rows):,} buffered rows. {extent}\n"
            "Only this captured buffer is exported; the query is not rerun."
        )

    @on(Select.Changed, "#export-scope")
    def _scope_changed(self) -> None:
        self._reset_confirmation()
        self._update_summary()

    @on(Input.Changed, "#export-path")
    def _path_changed(self) -> None:
        self._reset_confirmation()

    def _reset_confirmation(self) -> None:
        self._overwrite_path = None
        self.query_one("#export-overwrite", Button).display = False
        self.query_one("#export-message", Static).update("")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for selector in ("#export-path", "#export-scope", "#export-save",
                         "#export-overwrite", "#export-cancel"):
            self.query_one(selector).disabled = busy

    @on(Button.Pressed, "#export-save")
    def _export_pressed(self) -> None:
        self._begin_export(overwrite=False)

    @on(Button.Pressed, "#export-overwrite")
    def _overwrite_pressed(self) -> None:
        self._begin_export(overwrite=True)

    def _begin_export(self, *, overwrite: bool) -> None:
        if self._busy:
            return
        destination = self.query_one("#export-path", Input).value.strip()
        if not destination:
            self.query_one("#export-message", Static).update("Enter an explicit destination path.")
            return
        try:
            path = Path(destination).expanduser().absolute()
        except (OSError, ValueError, RuntimeError) as exc:
            self._show_error(exc)
            return
        if overwrite and path != self._overwrite_path:
            return
        indices = self._indices()
        self._set_busy(True)
        self.query_one("#export-message", Static).update("Writing CSV…")
        self.run_worker(self._export(path, indices, overwrite), name="Export CSV", exit_on_error=False)

    async def _export(self, path: Path, indices: tuple[int, ...] | range | None, overwrite: bool) -> None:
        try:
            if indices is None:
                if self._session.busy:
                    raise ValueError(f"{self._session.busy}; wait for it to finish before exporting the full result.")
                self._session.busy = "Full CSV export running"
            task = asyncio.create_task(asyncio.to_thread(self._write_csv, path, indices, overwrite))
            if indices is None:
                # Closing the app cancels the UI worker, not the database thread.
                task.add_done_callback(self._release_connection)
                count = await asyncio.shield(task)
            else:
                count = await task
        except FileExistsError:
            self._overwrite_path = path
            self.query_one("#export-overwrite", Button).display = True
            self.query_one("#export-message", Static).update(
                f"File already exists: {path}\nNothing was written. Choose Overwrite file to replace it."
            )
        except (OSError, ValueError, csv.Error) as exc:
            self._show_error(exc)
        else:
            self.app.notify(f"Exported {count:,} row(s) to {path}")
            self.dismiss(path)
        finally:
            if self.is_mounted:
                self._set_busy(False)

    def _release_connection(self, task: asyncio.Task) -> None:
        self._session.busy = ""
        if not task.cancelled():
            task.exception()  # Consume failures even if the dialog was unmounted.

    def _write_csv(self, path: Path, indices: tuple[int, ...] | range | None, overwrite: bool) -> int:
        # Never truncate an existing file before the last row has been fetched.
        # For a new file, exclusive creation must succeed before any query runs.
        output = (NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8", newline="", delete=False)
                  if overwrite else path.open("x", encoding="utf-8", newline=""))
        written = Path(output.name)
        try:
            with output:
                if indices is None:
                    count = self._session.export_result_csv(self.snapshot.page, output)
                else:
                    writer = csv.writer(output)
                    writer.writerow(self.snapshot.columns)
                    writer.writerows(self.snapshot.rows[index] for index in indices)
                    count = len(indices)
            if overwrite:
                written.replace(path)
            return count
        except BaseException:
            written.unlink(missing_ok=True)
            raise

    def _show_error(self, exc: Exception) -> None:
        message = f"CSV export failed: {exc}"
        self.query_one("#export-message", Static).update(message)
        self.app.notify(message, severity="error")

    @on(Button.Pressed, "#export-cancel")
    def action_cancel(self) -> None:
        # A committed write cannot be cancelled halfway by closing its screen.
        if not self._busy:
            self.dismiss(None)
