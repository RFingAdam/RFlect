"""Tests for off-thread figure rendering (#40)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")

from plot_antenna.async_render import render_async, render_to_png


def _draw(fig):
    ax = fig.add_subplot(111)
    ax.plot([0, 1, 2], [0, 1, 4])
    ax.set_title("test")


class TestRenderToPng:
    def test_writes_nonempty_png(self, tmp_path):
        out = tmp_path / "sync.png"
        result = render_to_png(_draw, str(out))
        assert result == str(out)
        assert os.path.getsize(out) > 0

    def test_does_not_disturb_pyplot_state(self, tmp_path):
        # Rendering via the standalone canvas must not create pyplot figures.
        import matplotlib.pyplot as plt

        plt.close("all")
        render_to_png(_draw, str(tmp_path / "x.png"))
        assert plt.get_fignums() == []


class TestRenderAsync:
    def test_on_done_fires_and_file_exists(self, tmp_path):
        out = tmp_path / "async.png"
        done = []
        t = render_async(_draw, str(out), on_done=lambda p: done.append(p))
        t.join(timeout=30)
        assert not t.is_alive()
        assert done == [str(out)]
        assert os.path.getsize(out) > 0

    def test_on_error_fires_on_bad_draw(self, tmp_path):
        errors = []

        def _boom(fig):
            raise RuntimeError("draw failed")

        t = render_async(_boom, str(tmp_path / "err.png"), on_error=lambda e: errors.append(e))
        t.join(timeout=30)
        assert not t.is_alive()
        assert len(errors) == 1
        assert isinstance(errors[0], RuntimeError)
