"""Use terminal cells for hit testing in Ghostty."""

from textual.drivers.linux_driver import LinuxDriver


class GhosttyDriver(LinuxDriver):
    def _query_in_band_window_resize(self) -> None:
        # Textual 8 couples in-band resize (2048) to pixel mouse reports (1016).
        # A cell report decoded as pixels lands near the top left, not on the
        # clicked widget. Keep SIGWINCH resizing and request SGR cells instead.
        self.write("\x1b[?2048l")
        if self._mouse:
            self.write("\x1b[?1016l\x1b[?1006h")
