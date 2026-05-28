"""Off-the-Tk-thread figure rendering (#40).

Heavy matplotlib renders (3D patterns, MIMO sweeps) block the Tk event loop when
drawn on the GUI thread via pyplot/TkAgg, freezing the UI. This module renders to
a PNG using a *standalone* ``Figure`` + ``FigureCanvasAgg`` — it never touches the
global pyplot state or the TkAgg backend, so it is safe to run on a worker thread.
The GUI can then load the finished PNG back on the main thread.

``render_to_png`` is the synchronous, dependency-free core (unit-testable in CI);
``render_async`` runs it on a daemon thread and posts the result/exception back
through callbacks.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from .plotting import PLOT_DPI, PLOT_FORMAT


def render_to_png(
    draw_fn: Callable[[Figure], None],
    out_path: str,
    *,
    dpi: int = PLOT_DPI,
    fmt: str = PLOT_FORMAT,
    figsize=(8, 6),
) -> str:
    """Render synchronously to ``out_path`` using a standalone Agg canvas.

    ``draw_fn`` receives a fresh :class:`matplotlib.figure.Figure` and should
    populate it (add axes, plot, etc.). Thread-safe: uses no pyplot global state.
    Returns ``out_path``.
    """
    fig = Figure(figsize=figsize)
    FigureCanvasAgg(fig)  # attaches a canvas to fig
    try:
        draw_fn(fig)
        fig.savefig(out_path, format=fmt, dpi=dpi)
    finally:
        fig.clear()
    return out_path


def render_async(
    draw_fn: Callable[[Figure], None],
    out_path: str,
    *,
    on_done: Optional[Callable[[str], None]] = None,
    on_error: Optional[Callable[[BaseException], None]] = None,
    dpi: int = PLOT_DPI,
    fmt: str = PLOT_FORMAT,
    figsize=(8, 6),
) -> threading.Thread:
    """Render on a daemon thread; invoke ``on_done(path)`` or ``on_error(exc)``.

    Returns the started :class:`threading.Thread`. In a Tk app, the callbacks
    should marshal back to the main thread (e.g. ``root.after(0, ...)``) before
    touching widgets — matplotlib's Agg render is done by the time they fire, so
    only the lightweight image-load needs the GUI thread.
    """

    def _worker():
        try:
            render_to_png(draw_fn, out_path, dpi=dpi, fmt=fmt, figsize=figsize)
        except BaseException as exc:  # never let the worker thread die silently
            if on_error is not None:
                on_error(exc)
            return
        if on_done is not None:
            on_done(out_path)

    thread = threading.Thread(target=_worker, name="rflect-render", daemon=True)
    thread.start()
    return thread
