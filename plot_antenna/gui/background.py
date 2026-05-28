"""Reusable background-task runner for the Tk GUI (#24).

Several GUI actions ran long, non-GUI work (DOCX report assembly, batch
processing) — some already on a worker thread with ad-hoc ``root.after(0, ...)``
marshalling, but the report path ran *on the Tk thread* and froze the UI. This
helper centralises the pattern: run ``work_fn`` on a daemon thread and marshal
the completion/error callback back to the Tk main thread via ``root.after``.

Only thread-safe work belongs here (python-docx, file IO, numpy). matplotlib's
TkAgg backend is NOT thread-safe — pre-render figures via
``plot_antenna.async_render`` (standalone Agg) instead of drawing inside ``work_fn``.

The threading/marshalling logic is unit-tested with a fake root whose ``after``
runs the callback synchronously, so it is verified without a display.
"""

from __future__ import annotations

import threading
from typing import Any, Callable, Optional


def run_background(
    root: Any,
    work_fn: Callable[[], Any],
    *,
    on_done: Optional[Callable[[Any], None]] = None,
    on_error: Optional[Callable[[BaseException], None]] = None,
) -> threading.Thread:
    """Run ``work_fn()`` on a daemon thread; marshal the result back to the Tk
    thread.

    On success, ``root.after(0, lambda: on_done(result))`` is scheduled; on
    failure, ``root.after(0, lambda: on_error(exc))``. Returns the started
    thread (useful for tests / joining). The worker never raises into the
    interpreter — exceptions are routed to ``on_error``.
    """

    def _worker():
        try:
            result = work_fn()
        except BaseException as exc:  # noqa: BLE001 — routed to on_error, never swallowed
            if on_error is not None:
                root.after(0, lambda e=exc: on_error(e))
            return
        if on_done is not None:
            root.after(0, lambda r=result: on_done(r))

    thread = threading.Thread(target=_worker, name="rflect-bg", daemon=True)
    thread.start()
    return thread
