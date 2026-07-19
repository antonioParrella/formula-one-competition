"""Tiny dependency-free progress indicators for the slow pipeline stages.

The optimiser deliberately avoids extra dependencies (the odds clients use
stdlib ``urllib`` for the same reason), so this is a ~60-line stand-in for
tqdm covering the two spots that otherwise run silently: the scipy fit and
the optimiser's restart loop.

Both indicators draw to **stderr and only when it is a TTY** — piped or
redirected runs (CI, ``> log.txt``) get clean, bar-free output. Lines are
cleared by overwriting with spaces rather than ANSI escapes, so they behave
on any terminal. Nothing here touches results; it is display only.
"""

import sys
import time

_BAR_W = 22


def _tty() -> bool:
    try:
        return sys.stderr.isatty()
    except Exception:
        return False


class Progress:
    """Determinate bar over a known number of steps (e.g. restarts)."""

    def __init__(self, total: int, label: str) -> None:
        self.total = max(int(total), 1)
        self.label = label
        self.n = 0
        self.t0 = time.perf_counter()
        self.on = _tty()
        self._last = 0
        self._draw()

    def _write(self, line: str) -> None:
        pad = max(self._last - len(line), 0)
        sys.stderr.write("\r" + line + " " * pad)
        sys.stderr.flush()
        self._last = len(line)

    def _draw(self, extra: str = "") -> None:
        if not self.on:
            return
        frac = self.n / self.total
        fill = int(_BAR_W * frac)
        bar = "█" * fill + "░" * (_BAR_W - fill)
        el = time.perf_counter() - self.t0
        self._write(f"  {self.label} [{bar}] {self.n}/{self.total} {el:4.0f}s  {extra}")

    def update(self, n: int = 1, extra: str = "") -> None:
        self.n += n
        self._draw(extra)

    def log(self, msg: str) -> None:
        """Print a line above the bar without clobbering it."""
        if self.on:
            sys.stderr.write("\r" + " " * self._last + "\r")
            sys.stderr.flush()
            self._last = 0
        print(msg)
        self._draw()

    def close(self) -> None:
        if self.on:
            sys.stderr.write("\r" + " " * self._last + "\r")
            sys.stderr.flush()
            self._last = 0


class Spinner:
    """Indeterminate spinner for an unbounded/early-stopping loop (the fit).

    The scipy solver converges at an unknown iteration, so a fake bar would
    mislead — this just shows liveness, iteration count and elapsed time,
    ticked once per solver iteration via a ``callback``.
    """

    FRAMES = "|/-\\"

    def __init__(self, label: str) -> None:
        self.label = label
        self.i = 0
        self.t0 = time.perf_counter()
        self.on = _tty()
        self._last = 0

    def tick(self, extra: str = "") -> None:
        if not self.on:
            return
        frame = self.FRAMES[self.i % len(self.FRAMES)]
        self.i += 1
        el = time.perf_counter() - self.t0
        line = f"  {self.label} {frame} iter {self.i:>3}  {el:4.0f}s  {extra}"
        pad = max(self._last - len(line), 0)
        sys.stderr.write("\r" + line + " " * pad)
        sys.stderr.flush()
        self._last = len(line)

    def close(self) -> None:
        if self.on:
            sys.stderr.write("\r" + " " * self._last + "\r")
            sys.stderr.flush()
            self._last = 0
