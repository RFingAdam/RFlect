"""Tests for the GUI background-task runner (#24).

Uses a fake root whose ``after`` runs the scheduled callback synchronously, so
the threading + marshalling contract is verified headlessly (no Tk display).
"""

from __future__ import annotations

import threading

from plot_antenna.gui.background import make_progress_marshaller, run_background


class _FakeRoot:
    """Records and immediately runs callbacks scheduled via ``after``."""

    def __init__(self):
        self.scheduled = []

    def after(self, delay, fn):
        self.scheduled.append(delay)
        fn()


def test_on_done_receives_result_marshalled_through_root():
    root = _FakeRoot()
    got = []
    t = run_background(root, lambda: 21 * 2, on_done=got.append)
    t.join(timeout=10)
    assert got == [42]
    assert root.scheduled == [0]  # marshalled with delay 0


def test_on_error_receives_exception():
    root = _FakeRoot()
    errors = []

    def _boom():
        raise ValueError("nope")

    t = run_background(root, _boom, on_error=errors.append)
    t.join(timeout=10)
    assert len(errors) == 1
    assert isinstance(errors[0], ValueError)


def test_work_runs_off_the_calling_thread():
    root = _FakeRoot()
    thread_ids = {}
    main_id = threading.get_ident()

    def _work():
        thread_ids["worker"] = threading.get_ident()
        return None

    t = run_background(root, _work, on_done=lambda r: None)
    t.join(timeout=10)
    assert thread_ids["worker"] != main_id


def test_no_callbacks_is_safe():
    root = _FakeRoot()
    t = run_background(root, lambda: 1)
    t.join(timeout=10)
    assert not t.is_alive()


class _FakeBar:
    def __init__(self):
        self.kw = {}

    def config(self, **kw):
        self.kw.update(kw)


def test_progress_marshaller_sets_maximum_once_and_updates_value():
    root = _FakeRoot()
    bar = _FakeBar()
    cb = make_progress_marshaller(root, bar)

    cb(1, 4)
    assert bar.kw["maximum"] == 4
    assert bar.kw["value"] == 1

    cb(3, 4)
    assert bar.kw["maximum"] == 4  # not re-set
    assert bar.kw["value"] == 3

    cb(4, 4)
    assert bar.kw["value"] == 4
    assert root.scheduled == [0, 0, 0]  # every update marshalled via after(0)


def test_progress_marshaller_handles_zero_total_gracefully():
    root = _FakeRoot()
    bar = _FakeBar()
    cb = make_progress_marshaller(root, bar)
    cb(0, 0)  # nothing known yet
    assert "maximum" not in bar.kw
    assert bar.kw["value"] == 0
