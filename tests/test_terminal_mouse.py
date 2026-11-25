"""Explorer hit testing through the same terminal driver used on macOS."""
from __future__ import annotations

import asyncio
import os
import signal
import sys
from contextlib import ExitStack
from io import StringIO

import pytest
from textual import events
from textual.widgets import TabbedContent, Tree

from d8r.tui.app import D8RApp
from d8r.tui.session import Session


@pytest.mark.skipif(os.name != "posix", reason="Exercises Textual's macOS/Linux terminal driver")
def test_ghostty_cell_clicks_navigate_and_close_explorer(parquet_source, monkeypatch):
    monkeypatch.setenv("TERM_PROGRAM", "ghostty")
    monkeypatch.setenv("COLUMNS", "110")
    monkeypatch.setenv("LINES", "35")
    read_fd, write_fd = os.pipe()
    with ExitStack() as stack:
        stdin = stack.enter_context(os.fdopen(read_fd))
        terminal = stack.enter_context(os.fdopen(write_fd, "wb", buffering=0))

        class TerminalOutput(StringIO):
            def write(self, data):
                # This terminal advertises in-band resizing but delivers cell mouse
                # coordinates. Enabling 2048 produces an immediate pixel-size report.
                if "\x1b[?2048$p" in data:
                    terminal.write(b"\x1b[?2048;2$y")
                if "\x1b[?2048h" in data:
                    terminal.write(b"\x1b[48;35;110;1120;1760t")
                return len(data)

        monkeypatch.setattr(sys, "__stdin__", stdin)
        monkeypatch.setattr(sys, "__stderr__", TerminalOutput())
        for sig in (signal.SIGTSTP, signal.SIGCONT, signal.SIGWINCH):
            stack.callback(signal.signal, sig, signal.getsignal(sig))

        async def exercise():
            app = D8RApp(Session({parquet_source.id: parquet_source}))
            released = asyncio.Event()

            def observe(message):
                if isinstance(message, events.MouseUp):
                    released.set()

            async with app.run_test(headless=False, message_hook=observe) as pilot:
                async def click(point):
                    x, y = map(int, point)
                    released.clear()
                    terminal.write(f"\x1b[<0;{x+1};{y+1}M\x1b[<0;{x+1};{y+1}m".encode())
                    await asyncio.wait_for(released.wait(), timeout=3)
                    await pilot.pause()

                tabs = app.query_one("#explorer-tabs", TabbedContent)
                await click(tabs.get_tab("tab-pages").region.center)
                assert tabs.active == "tab-pages"
                await click(tabs.get_tab("tab-schema").region.center)
                tree = app.query_one("#schema-tree", Tree)
                table = tree.root.children[0]
                await click((tree.content_region.x + 8, tree.content_region.y + table.line))
                assert tree.has_focus and table.is_expanded
                await pilot.press("down")
                assert tree.cursor_node is table.children[0]

                await click(app.query_one("#explorer-close").region.center)
                assert not app.query_one("#explorer-slot").display and app.editor.has_focus
                await pilot.press(*r"\schema", "enter")
                assert app.query_one("#explorer-slot").display

        asyncio.run(exercise())
