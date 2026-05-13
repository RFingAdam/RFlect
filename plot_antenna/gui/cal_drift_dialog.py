"""Tk dialog for the calibration drift history.

Three-column layout:
  • Left:   runs tree (group by antenna → band → date). Right-click to mark as
            baseline/current, edit notes, or delete. Buttons below: Import
            Historical, Change History Folder, Refresh.
  • Middle: notebook with Summary stats / Consistency / Missing-freq audit /
            Raw delta table.
  • Right:  matplotlib two-panel ΔdB plot + export buttons.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, Optional

import pandas as pd

from .. import cal_drift
from ..config import (
    ACCENT_BLUE_COLOR,
    DARK_BG_COLOR,
    LIGHT_TEXT_COLOR,
    SURFACE_COLOR,
)


class CalDriftDialog:
    """A modal-ish Toplevel window showing the calibration drift history."""

    def __init__(self, parent: tk.Misc, logger: Optional[Callable[[str], None]] = None):
        self._log = logger or (lambda msg: None)
        self.top = tk.Toplevel(parent)
        self.top.title("Calibration Drift History")
        self.top.geometry("1400x800")
        self.top.configure(bg=DARK_BG_COLOR)

        self.baseline_run_id: Optional[str] = None
        self.current_run_id: Optional[str] = None
        self._runs_df: pd.DataFrame = pd.DataFrame()
        self._canvas = None  # FigureCanvasTkAgg
        self._current_figure = None
        self._current_result: Optional[cal_drift.DriftResult] = None

        self._build_layout()
        self._refresh_runs()

    # ──────────────────────────────────────────────────────────────────
    # Layout
    # ──────────────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        paned = ttk.PanedWindow(self.top, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self._build_left_panel(paned)
        self._build_middle_panel(paned)
        self._build_right_panel(paned)

        # Footer status bar
        self.status_var = tk.StringVar(value="")
        status = ttk.Label(self.top, textvariable=self.status_var, anchor="w")
        status.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=(0, 4))

    # ─── Left ─────────────────────────────────────────────────────────
    def _build_left_panel(self, parent: ttk.PanedWindow) -> None:
        frame = ttk.Frame(parent)
        parent.add(frame, weight=1)

        header = ttk.Label(frame, text="Recorded runs", font=("TkDefaultFont", 11, "bold"))
        header.pack(anchor="w", padx=4, pady=(2, 4))

        cols = ("date", "rows", "role", "setup_group", "notes")
        self.tree = ttk.Treeview(frame, columns=cols, show="tree headings", selectmode="browse")
        self.tree.heading("#0", text="Antenna / Band / Run")
        self.tree.heading("date", text="Date")
        self.tree.heading("rows", text="Rows")
        self.tree.heading("role", text="Role")
        self.tree.heading("setup_group", text="Setup group")
        self.tree.heading("notes", text="Notes")
        self.tree.column("#0", width=210)
        self.tree.column("date", width=90, anchor="center")
        self.tree.column("rows", width=55, anchor="e")
        self.tree.column("role", width=70, anchor="center")
        self.tree.column("setup_group", width=140)
        self.tree.column("notes", width=140)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0), pady=2)
        vsb.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 4))
        self.tree.bind("<Button-3>", self._show_tree_context_menu)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 6))
        ttk.Button(btn_frame, text="Import Historical…", command=self._import_historical).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Change History Folder…", command=self._change_history_dir).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Refresh", command=self._refresh_runs).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Compare", command=self._compare_selected).pack(
            side=tk.RIGHT, padx=2
        )

    # ─── Middle ───────────────────────────────────────────────────────
    def _build_middle_panel(self, parent: ttk.PanedWindow) -> None:
        frame = ttk.Frame(parent)
        parent.add(frame, weight=1)

        self.sel_var = tk.StringVar(value="Baseline: —   Current: —")
        ttk.Label(frame, textvariable=self.sel_var).pack(anchor="w", padx=4, pady=4)

        self.notebook = ttk.Notebook(frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        self.stats_tree = self._make_tree(
            self.notebook, ("metric", "h", "v"), ("Metric", "H-Pol", "V-Pol")
        )
        self.consistency_tree = self._make_tree(
            self.notebook,
            ("field", "state", "baseline", "current"),
            ("Field", "State", "Baseline", "Current"),
        )
        self.audit_tree = self._make_tree(
            self.notebook, ("kind", "freqs"), ("Change", "Frequencies (MHz)")
        )
        self.delta_tree = self._make_tree(
            self.notebook, ("freq", "dh", "dv"), ("Freq (MHz)", "Δ H (dB)", "Δ V (dB)")
        )

        self.notebook.add(self.stats_tree.master, text="Summary")
        self.notebook.add(self.consistency_tree.master, text="Consistency")
        self.notebook.add(self.audit_tree.master, text="Missing-freq audit")
        self.notebook.add(self.delta_tree.master, text="Raw deltas")

    def _make_tree(self, parent, cols, headings) -> ttk.Treeview:
        container = ttk.Frame(parent)
        tree = ttk.Treeview(container, columns=cols, show="headings")
        for c, h in zip(cols, headings):
            tree.heading(c, text=h)
            tree.column(c, width=120, anchor="w")
        vsb = ttk.Scrollbar(container, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.LEFT, fill=tk.Y)
        return tree

    # ─── Right ────────────────────────────────────────────────────────
    def _build_right_panel(self, parent: ttk.PanedWindow) -> None:
        self.right_frame = ttk.Frame(parent)
        parent.add(self.right_frame, weight=2)

        header = ttk.Label(self.right_frame, text="Drift plot", font=("TkDefaultFont", 11, "bold"))
        header.pack(anchor="w", padx=4, pady=(2, 4))

        self.plot_container = ttk.Frame(self.right_frame)
        self.plot_container.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        btn_frame = ttk.Frame(self.right_frame)
        btn_frame.pack(fill=tk.X, padx=4, pady=(4, 6))
        ttk.Button(btn_frame, text="Export PNG…", command=lambda: self._export("png")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(btn_frame, text="Export PDF…", command=lambda: self._export("pdf")).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            btn_frame, text="Export Markdown…", command=lambda: self._export("markdown")
        ).pack(side=tk.LEFT, padx=2)

    # ──────────────────────────────────────────────────────────────────
    # Data loading
    # ──────────────────────────────────────────────────────────────────

    def _refresh_runs(self) -> None:
        self._runs_df = cal_drift.list_runs()
        for iid in self.tree.get_children():
            self.tree.delete(iid)

        hdir = cal_drift.history_dir()
        self.status_var.set(f"History: {hdir}   •   {len(self._runs_df)} run(s) recorded")

        if self._runs_df.empty:
            return

        # Group: antenna → band → date rows
        for antenna in sorted(self._runs_df["antenna"].unique()):
            ant_node = self.tree.insert("", "end", text=antenna, open=True)
            ant_df = self._runs_df[self._runs_df["antenna"] == antenna]
            for band in sorted(ant_df["band_label"].unique()):
                band_node = self.tree.insert(ant_node, "end", text=band, open=True)
                band_df = ant_df[ant_df["band_label"] == band].sort_values("date")
                for _, row in band_df.iterrows():
                    run_id = str(row["run_id"])
                    role = ""
                    if run_id == self.baseline_run_id:
                        role = "baseline"
                    elif run_id == self.current_run_id:
                        role = "current"
                    self.tree.insert(
                        band_node,
                        "end",
                        iid=run_id,
                        text=run_id,
                        values=(
                            row["date"],
                            row["rows_written"],
                            role,
                            str(row.get("setup_group", ""))[:60],
                            str(row.get("operator_notes", ""))[:60],
                        ),
                    )

    # ──────────────────────────────────────────────────────────────────
    # Actions
    # ──────────────────────────────────────────────────────────────────

    def _show_tree_context_menu(self, event) -> None:
        iid = self.tree.identify_row(event.y)
        if not iid or iid not in self._runs_df["run_id"].values:
            return
        self.tree.selection_set(iid)
        menu = tk.Menu(self.top, tearoff=0)
        menu.add_command(label="Set as Baseline", command=lambda: self._set_role(iid, "baseline"))
        menu.add_command(label="Set as Current", command=lambda: self._set_role(iid, "current"))
        menu.add_separator()
        menu.add_command(label="Edit setup group…", command=lambda: self._edit_setup_group(iid))
        menu.add_command(label="Edit notes…", command=lambda: self._edit_notes(iid))
        menu.add_command(label="Delete run", command=lambda: self._delete_run(iid))
        menu.tk_popup(event.x_root, event.y_root)

    def _set_role(self, run_id: str, role: str) -> None:
        if role == "baseline":
            self.baseline_run_id = run_id
        else:
            self.current_run_id = run_id
        b_short = (self.baseline_run_id or "—")[:12]
        c_short = (self.current_run_id or "—")[:12]
        self.sel_var.set(f"Baseline: {b_short}   Current: {c_short}")
        self._refresh_runs()
        # Keep selections on screen
        for iid in (self.baseline_run_id, self.current_run_id):
            if iid and self.tree.exists(iid):
                self.tree.see(iid)

    def _edit_notes(self, run_id: str) -> None:
        meta = cal_drift.get_run(run_id)
        if meta is None:
            return
        new_notes = simpledialog.askstring(
            "Edit notes",
            f"Notes for run {run_id}:",
            initialvalue=meta.operator_notes,
            parent=self.top,
        )
        if new_notes is None:
            return
        cal_drift.update_notes(run_id, new_notes)
        self._refresh_runs()

    def _edit_setup_group(self, run_id: str) -> None:
        meta = cal_drift.get_run(run_id)
        if meta is None:
            return
        new_group = simpledialog.askstring(
            "Edit setup group",
            f"Setup group for run {run_id}\n"
            "(free text; used to flag cross-epoch comparisons — leave blank for default):",
            initialvalue=meta.setup_group,
            parent=self.top,
        )
        if new_group is None:
            return
        cal_drift.set_setup_group(run_id, new_group)
        self._refresh_runs()

    def _delete_run(self, run_id: str) -> None:
        if not messagebox.askyesno("Delete run", f"Delete run {run_id}?", parent=self.top):
            return
        cal_drift.delete_run(run_id)
        if self.baseline_run_id == run_id:
            self.baseline_run_id = None
        if self.current_run_id == run_id:
            self.current_run_id = None
        self._refresh_runs()

    def _import_historical(self) -> None:
        initial = os.path.expanduser("~/Downloads/Calibration Data")
        if not os.path.exists(initial):
            initial = os.path.expanduser("~")
        directory = filedialog.askdirectory(
            title="Select folder of archived TRP Cal files",
            initialdir=initial,
            parent=self.top,
        )
        if not directory:
            return
        self.status_var.set(f"Importing from {directory}…")
        self.top.update_idletasks()
        result = cal_drift.import_historical_dir(directory)
        self._log(
            f"Cal drift import: ingested={result['ingested']}, "
            f"skipped_duplicate={result['skipped_duplicate']}, failed={result['failed']}"
        )
        messagebox.showinfo(
            "Import complete",
            f"Ingested: {result['ingested']}\n"
            f"Skipped (duplicate): {result['skipped_duplicate']}\n"
            f"Failed: {result['failed']}",
            parent=self.top,
        )
        self._refresh_runs()

    def _change_history_dir(self) -> None:
        current = cal_drift.history_dir()
        directory = filedialog.askdirectory(
            title="Select cal-drift history folder",
            initialdir=str(current),
            parent=self.top,
        )
        if not directory:
            return
        cal_drift.set_history_dir(directory)
        self._refresh_runs()

    def _compare_selected(self) -> None:
        if not self.baseline_run_id or not self.current_run_id:
            messagebox.showinfo(
                "Select runs",
                "Right-click two runs to mark them as Baseline and Current, " "then click Compare.",
                parent=self.top,
            )
            return
        try:
            result = cal_drift.compute_drift(self.baseline_run_id, self.current_run_id)
        except Exception as exc:
            messagebox.showerror("Compare failed", str(exc), parent=self.top)
            return
        self._current_result = result
        self._populate_metrics(result)
        self._render_plot(result)

    # ──────────────────────────────────────────────────────────────────
    # Metric panels
    # ──────────────────────────────────────────────────────────────────

    def _populate_metrics(self, result: cal_drift.DriftResult) -> None:
        # Summary stats
        for iid in self.stats_tree.get_children():
            self.stats_tree.delete(iid)
        s_h = result.stats["H"]
        s_v = result.stats["V"]
        metrics = [
            ("n", s_h["n"], s_v["n"]),
            ("mean (dB)", f"{s_h['mean']:.3f}", f"{s_v['mean']:.3f}"),
            ("std (dB)", f"{s_h['std']:.3f}", f"{s_v['std']:.3f}"),
            ("max |Δ| (dB)", f"{s_h['max_abs']:.3f}", f"{s_v['max_abs']:.3f}"),
            ("% > 0.5 dB", f"{s_h['pct_gt_0_5']:.1f}", f"{s_v['pct_gt_0_5']:.1f}"),
            ("% > 1.0 dB", f"{s_h['pct_gt_1']:.1f}", f"{s_v['pct_gt_1']:.1f}"),
        ]
        for row in metrics:
            self.stats_tree.insert("", "end", values=row)

        # Consistency
        for iid in self.consistency_tree.get_children():
            self.consistency_tree.delete(iid)
        self.consistency_tree.tag_configure("match", foreground="#2e7d32")
        self.consistency_tree.tag_configure("mismatch", foreground="#c62828")
        self.consistency_tree.tag_configure("unknown", foreground="#616161")
        for field_name, info in result.consistency.items():
            self.consistency_tree.insert(
                "",
                "end",
                values=(field_name, info["state"], info["baseline"], info["current"]),
                tags=(info["state"],),
            )

        # Missing-freq audit
        for iid in self.audit_tree.get_children():
            self.audit_tree.delete(iid)
        appeared = result.missing_audit["appeared_in_current"]
        disappeared = result.missing_audit["disappeared_in_current"]
        self.audit_tree.insert(
            "",
            "end",
            values=(
                f"Appeared in current ({len(appeared)})",
                ", ".join(f"{f:.1f}" for f in appeared[:50]) or "(none)",
            ),
        )
        self.audit_tree.insert(
            "",
            "end",
            values=(
                f"Disappeared from current ({len(disappeared)})",
                ", ".join(f"{f:.1f}" for f in disappeared[:50]) or "(none)",
            ),
        )

        # Raw deltas (cap at 1000 rows for UI perf)
        for iid in self.delta_tree.get_children():
            self.delta_tree.delete(iid)
        for _, row in result.deltas.head(1000).iterrows():
            dh = f"{row['d_h']:.3f}" if pd.notna(row["d_h"]) else ""
            dv = f"{row['d_v']:.3f}" if pd.notna(row["d_v"]) else ""
            self.delta_tree.insert("", "end", values=(f"{row['freq_mhz']:.1f}", dh, dv))

    # ──────────────────────────────────────────────────────────────────
    # Plot
    # ──────────────────────────────────────────────────────────────────

    def _render_plot(self, result: cal_drift.DriftResult) -> None:
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

        # Clear prior canvas
        for child in self.plot_container.winfo_children():
            child.destroy()

        fig = cal_drift.render_delta_plot(result)
        self._current_figure = fig
        canvas = FigureCanvasTkAgg(fig, master=self.plot_container)
        canvas.draw()
        toolbar = NavigationToolbar2Tk(canvas, self.plot_container)
        toolbar.update()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._canvas = canvas

    # ──────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────

    def _export(self, fmt: str) -> None:
        if self._current_result is None:
            messagebox.showinfo("Export", "Run a comparison first.", parent=self.top)
            return
        ext_map = {"png": ".png", "pdf": ".pdf", "markdown": ".md"}
        ext = ext_map[fmt]
        default_name = (
            f"drift_{self._current_result.baseline.antenna}_"
            f"{self._current_result.baseline.band_label}_"
            f"{self._current_result.baseline.date}_to_"
            f"{self._current_result.current.date}{ext}"
        )
        path = filedialog.asksaveasfilename(
            title=f"Export {fmt.upper()}",
            defaultextension=ext,
            initialfile=default_name,
            parent=self.top,
        )
        if not path:
            return
        try:
            if fmt == "png":
                cal_drift.render_delta_plot(self._current_result, out_path=path)
            elif fmt == "pdf":
                cal_drift.export_pdf(self._current_result, path)
            else:
                cal_drift.export_markdown(self._current_result, path)
        except Exception as exc:
            messagebox.showerror("Export failed", str(exc), parent=self.top)
            return
        self._log(f"Cal drift export → {path}")
