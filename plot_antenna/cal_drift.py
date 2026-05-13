"""Calibration drift tracking for RFlect.

Persists every Active Chamber Calibration run + lets operators compare any two
runs to catch real chamber/equipment drift and methodology inconsistencies.

History layout (at <history-dir>, default ~/.config/RFlect/cal_drift):
    runs.csv                   one row per CalRunMeta (append-only)
    points/{run_id}.csv        freq_mhz,hpol_dbm,vpol_dbm (immutable per run)

Resolution of <history-dir> (first hit wins):
    1. os.environ["RFLECT_CAL_DRIFT_DIR"]
    2. user_settings.json key "cal_drift_dir"
    3. get_user_data_dir() / "cal_drift"
"""

from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import logging
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_DEFAULT_HISTORY_SUBDIR = "cal_drift"
_ENV_VAR = "RFLECT_CAL_DRIFT_DIR"
_OUTLIER_DB = 0.5
_HARD_OUTLIER_DB = 1.0


# ──────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CalRunMeta:
    """One calibration-run record: identity + method signature for a TRP cal file."""

    run_id: str
    date: str  # "YYYY-MM-DD"
    time: str  # "HH:MM:SS"
    antenna: str
    band_label: str
    start_mhz: float
    stop_mhz: float
    output_path: str
    output_sha256: str
    power_file: str = ""
    power_sha256: str = ""
    gain_std_file: str = ""
    gain_std_sha256: str = ""
    hpol_ref_file: str = ""
    hpol_ref_sha256: str = ""
    vpol_ref_file: str = ""
    vpol_ref_sha256: str = ""
    source_level_dBm: Optional[float] = None
    signal_source: Optional[str] = None
    receiver: Optional[str] = None
    freq_step_mhz: Optional[float] = None
    rows_written: int = 0
    rows_missing: int = 0
    operator_notes: str = ""
    ingested_at: str = ""
    setup_group: str = ""  # user-assigned methodology epoch; empty == default


@dataclass
class DriftResult:
    """Outcome of comparing a baseline run to a current run."""

    deltas: pd.DataFrame  # freq_mhz, base_h, cur_h, d_h, base_v, cur_v, d_v, outlier_h, outlier_v
    stats: dict
    consistency: dict  # {field: (match|mismatch|unknown, base_value, cur_value)}
    missing_audit: dict  # {appeared_in_current: [...], disappeared_in_current: [...]}
    baseline: CalRunMeta
    current: CalRunMeta


# ──────────────────────────────────────────────────────────────────────────
# Parsing
# ──────────────────────────────────────────────────────────────────────────

_CAL_HEADER_RE = re.compile(r"^Freq\s+H-Pol\s+V-Pol", re.IGNORECASE)
_DATE_RE = re.compile(r"^Date:\s*(\S+)", re.IGNORECASE)
_TIME_RE = re.compile(r"^Time:\s*(\S+)", re.IGNORECASE)


def parse_trp_cal_file(path: str | os.PathLike) -> pd.DataFrame:
    """Parse a generated TRP Cal file → DataFrame(freq_mhz, hpol_dbm, vpol_dbm).

    Raises ValueError if the file does not contain the expected header.
    """
    p = Path(path)
    rows: List[Tuple[float, float, float]] = []
    in_data = False
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not in_data:
                if _CAL_HEADER_RE.match(stripped):
                    in_data = True
                continue
            if not stripped or not stripped[0].isdigit():
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                freq = float(parts[0])
                h = float(parts[1])
                v = float(parts[2])
            except ValueError:
                continue
            rows.append((freq, h, v))
    if not rows:
        raise ValueError(f"No cal data rows found in {p}")
    df = pd.DataFrame(rows, columns=["freq_mhz", "hpol_dbm", "vpol_dbm"])
    df = df.drop_duplicates(subset="freq_mhz").sort_values("freq_mhz").reset_index(drop=True)
    return df


def parse_cal_file_header(path: str | os.PathLike) -> Dict[str, str]:
    """Extract Date and Time fields from a TRP Cal file header.

    Returns {"date": "MM-DD-YY" or "", "time": "HH:MM:SS" or ""}.
    """
    out = {"date": "", "time": ""}
    p = Path(path)
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            if i > 20:
                break
            m = _DATE_RE.match(line.strip())
            if m:
                out["date"] = m.group(1)
                continue
            m = _TIME_RE.match(line.strip())
            if m:
                out["time"] = m.group(1)
    return out


def parse_ref_file_header(path: str | os.PathLike) -> Dict[str, Optional[str | float]]:
    """Extract method-consistency fields from an AP_HPol/VPol reference file header.

    Returns {signal_source, receiver, output_level_dBm, frequency_count,
    start_mhz, stop_mhz}. Any field that is absent from the header is None.
    """
    result: Dict[str, Optional[str | float]] = {
        "signal_source": None,
        "receiver": None,
        "output_level_dBm": None,
        "frequency_count": None,
        "start_mhz": None,
        "stop_mhz": None,
    }
    p = Path(path)
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i > 80:
                    break
                s = line.strip()
                if s.startswith("Signal Source:"):
                    result["signal_source"] = s.split(":", 1)[1].strip()
                elif "Receiver:" in s and result["receiver"] is None:
                    # "HPOL Receiver:" or "VPOL Receiver:" — first wins
                    result["receiver"] = s.split(":", 1)[1].strip()
                elif s.startswith("Output Level:"):
                    val = s.split(":", 1)[1].strip()
                    m = re.search(r"-?\d+(?:\.\d+)?", val)
                    if m:
                        result["output_level_dBm"] = float(m.group(0))
                elif s.startswith("Frequency Count:"):
                    try:
                        result["frequency_count"] = float(s.split(":", 1)[1].strip())
                    except ValueError:
                        pass
                elif s.startswith("Start Frequency:"):
                    m = re.search(r"-?\d+(?:\.\d+)?", s)
                    if m:
                        result["start_mhz"] = float(m.group(0))
                elif s.startswith("Stop Frequency:"):
                    m = re.search(r"-?\d+(?:\.\d+)?", s)
                    if m:
                        result["stop_mhz"] = float(m.group(0))
    except OSError:
        log.warning("Could not read ref-file header: %s", p)
    return result


def parse_cal_summary_file(path: str | os.PathLike) -> Dict[str, object]:
    """Parse a TRP Cal Summary file → {power_file, gain_std_file, hpol_file,
    vpol_file, output_file, freq_start_mhz, freq_stop_mhz, missing_freqs:[]}.

    Missing fields default to "" or []. Existing callers may pass non-existent
    paths; the function simply returns an empty-ish dict in that case.
    """
    out: Dict[str, object] = {
        "power_file": "",
        "gain_std_file": "",
        "hpol_file": "",
        "vpol_file": "",
        "output_file": "",
        "freq_start_mhz": None,
        "freq_stop_mhz": None,
        "missing_freqs": [],
    }
    p = Path(path)
    if not p.exists():
        return out
    in_missing = False
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("Missing Data Frequencies:"):
                in_missing = True
                continue
            if in_missing:
                m = re.match(r"(-?\d+(?:\.\d+)?)\s*MHz", s)
                if m:
                    out["missing_freqs"].append(float(m.group(1)))  # type: ignore[union-attr]
                continue
            if s.startswith("Power Measurement File:"):
                out["power_file"] = s.split(":", 1)[1].strip()
            elif s.startswith("Gain Standard File:"):
                out["gain_std_file"] = s.split(":", 1)[1].strip()
            elif s.startswith("HPol File:"):
                out["hpol_file"] = s.split(":", 1)[1].strip()
            elif s.startswith("VPol File:"):
                out["vpol_file"] = s.split(":", 1)[1].strip()
            elif s.startswith("Output File:"):
                out["output_file"] = s.split(":", 1)[1].strip()
            elif s.startswith("Frequency Range:"):
                nums = re.findall(r"-?\d+(?:\.\d+)?", s)
                if len(nums) >= 2:
                    out["freq_start_mhz"] = float(nums[0])
                    out["freq_stop_mhz"] = float(nums[1])
    return out


def _parse_cal_filename(path: str | os.PathLike) -> Dict[str, str]:
    """Best-effort extraction of antenna / band_label from a TRP Cal filename.

    Handles both current and legacy naming conventions:
        'TRP Cal BLPA 690-2700 1Amp 0 dBm 690.0-2700.0 04-14-26.txt'
        'TRP Cal BLPA 690-2700MHz 2023-06-21.txt'
    → {antenna: 'BLPA', band_label: '690-2700'}
    """
    name = Path(path).name
    m = re.match(r"TRP Cal\s+(?P<ant>[A-Za-z]+)\s+(?P<band>\d+-\d+)", name)
    if m:
        return {"antenna": m.group("ant").upper(), "band_label": m.group("band")}
    return {"antenna": "UNKNOWN", "band_label": ""}


def _normalize_date(date_str: str) -> str:
    """Normalize MM-DD-YY or M/D/YYYY date → YYYY-MM-DD. Empty on failure."""
    if not date_str:
        return ""
    for fmt in ("%m-%d-%y", "%m-%d-%Y", "%m/%d/%y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            return _dt.datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return ""


# ──────────────────────────────────────────────────────────────────────────
# SHA-256
# ──────────────────────────────────────────────────────────────────────────


def sha256_file(path: str | os.PathLike) -> str:
    """Streaming SHA-256 of a file. Returns '' on read failure."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ──────────────────────────────────────────────────────────────────────────
# History directory resolution + settings
# ──────────────────────────────────────────────────────────────────────────


def _user_data_dir() -> Path:
    """Return platform-appropriate RFlect user data dir (mirrors gui.utils.get_user_data_dir)."""
    import sys

    if sys.platform == "win32":
        base = os.getenv("LOCALAPPDATA", os.path.expanduser("~"))
        return Path(base) / "RFlect"
    if sys.platform == "darwin":
        return Path(os.path.expanduser("~/Library/Application Support/RFlect"))
    return Path(os.path.expanduser("~/.config/RFlect"))


def _user_settings_path() -> Path:
    return _user_data_dir() / "user_settings.json"


def _read_user_settings() -> dict:
    p = _user_settings_path()
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_user_settings(settings: dict) -> None:
    p = _user_settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def history_dir() -> Path:
    """Resolve the active cal-drift history directory (env > settings > default)."""
    env = os.environ.get(_ENV_VAR)
    if env:
        p = Path(os.path.expanduser(env))
    else:
        settings = _read_user_settings()
        configured = settings.get("cal_drift_dir")
        p = (
            Path(os.path.expanduser(configured))
            if configured
            else (_user_data_dir() / _DEFAULT_HISTORY_SUBDIR)
        )
    p.mkdir(parents=True, exist_ok=True)
    (p / "points").mkdir(parents=True, exist_ok=True)
    return p


def set_history_dir(path: str | os.PathLike) -> Path:
    """Persist a new cal-drift history directory to user_settings.json."""
    resolved = Path(os.path.expanduser(str(path))).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / "points").mkdir(parents=True, exist_ok=True)
    settings = _read_user_settings()
    settings["cal_drift_dir"] = str(resolved)
    _write_user_settings(settings)
    return resolved


# ──────────────────────────────────────────────────────────────────────────
# History I/O
# ──────────────────────────────────────────────────────────────────────────


_RUNS_COLUMNS = [f.name for f in fields(CalRunMeta)]


def _runs_csv_path(hdir: Optional[Path] = None) -> Path:
    return (hdir or history_dir()) / "runs.csv"


def _points_csv_path(run_id: str, hdir: Optional[Path] = None) -> Path:
    return (hdir or history_dir()) / "points" / f"{run_id}.csv"


def list_runs(antenna: Optional[str] = None, band: Optional[str] = None) -> pd.DataFrame:
    """Read the runs table, optionally filtered by antenna / band_label."""
    path = _runs_csv_path()
    if not path.exists():
        return pd.DataFrame(columns=_RUNS_COLUMNS)
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        log.warning("runs.csv is corrupt or unreadable; returning empty table")
        return pd.DataFrame(columns=_RUNS_COLUMNS)
    for col in _RUNS_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    df = df[_RUNS_COLUMNS]
    if antenna:
        df = df[df["antenna"].str.upper() == antenna.upper()]
    if band:
        df = df[df["band_label"] == band]
    return df.sort_values(["antenna", "band_label", "date"]).reset_index(drop=True)


def load_points(run_id: str) -> pd.DataFrame:
    """Load the per-frequency points for one run. Empty DF if not found."""
    path = _points_csv_path(run_id)
    if not path.exists():
        return pd.DataFrame(columns=["freq_mhz", "hpol_dbm", "vpol_dbm"])
    return pd.read_csv(path)


def get_run(run_id: str) -> Optional[CalRunMeta]:
    """Fetch a single run's metadata by run_id."""
    df = list_runs()
    hit = df[df["run_id"] == run_id]
    if hit.empty:
        return None
    row = hit.iloc[0].to_dict()
    return _row_to_meta(row)


def _row_to_meta(row: dict) -> CalRunMeta:
    kwargs = {}
    for f in fields(CalRunMeta):
        raw = row.get(f.name, "")
        if pd.isna(raw):
            raw = ""
        if f.type is int or f.name in {"rows_written", "rows_missing"}:
            try:
                kwargs[f.name] = int(float(raw)) if raw != "" else 0
            except (TypeError, ValueError):
                kwargs[f.name] = 0
        elif f.name in {"start_mhz", "stop_mhz"}:
            try:
                kwargs[f.name] = float(raw) if raw != "" else 0.0
            except (TypeError, ValueError):
                kwargs[f.name] = 0.0
        elif f.name in {"source_level_dBm", "freq_step_mhz"}:
            try:
                kwargs[f.name] = float(raw) if raw != "" else None
            except (TypeError, ValueError):
                kwargs[f.name] = None
        else:
            kwargs[f.name] = str(raw) if raw != "" else ""
    return CalRunMeta(**kwargs)


def _append_run_row(meta: CalRunMeta, hdir: Path) -> None:
    path = _runs_csv_path(hdir)
    new_file = not path.exists()
    row = {f.name: getattr(meta, f.name) for f in fields(CalRunMeta)}
    # Replace None with empty string so CSV round-trip is predictable.
    row = {k: ("" if v is None else v) for k, v in row.items()}
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_RUNS_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _write_points_csv(run_id: str, points: pd.DataFrame, hdir: Path) -> None:
    path = _points_csv_path(run_id, hdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(path, index=False)


def _run_id_from_sha(output_path: str, output_sha: str) -> str:
    h = hashlib.sha256((output_path + output_sha).encode("utf-8")).hexdigest()
    return h[:16]


def record_run(
    cal_result: dict,
    power_file: str = "",
    gain_std_file: str = "",
    hpol_ref_file: str = "",
    vpol_ref_file: str = "",
    operator_notes: str = "",
) -> Optional[CalRunMeta]:
    """Record one calibration run to the drift history. Idempotent on output SHA.

    `cal_result` is the dict returned by `generate_active_cal_file`:
    {output_path, summary_path, rows_written, rows_missing}.

    Returns the recorded CalRunMeta, or None if it was a duplicate.
    """
    output_path = cal_result.get("output_path", "")
    summary_path = cal_result.get("summary_path", "")
    if not output_path or not os.path.exists(output_path):
        log.warning("record_run: output file missing (%s)", output_path)
        return None

    output_sha = sha256_file(output_path)
    run_id = _run_id_from_sha(output_path, output_sha)
    hdir = history_dir()

    # Idempotency guard
    existing = list_runs()
    if not existing.empty and (existing["output_sha256"] == output_sha).any():
        log.info("record_run: skipping duplicate (sha=%s)", output_sha[:8])
        return None

    # Parse cal file + ref header + summary
    points = parse_trp_cal_file(output_path)
    header = parse_cal_file_header(output_path)
    filename_info = _parse_cal_filename(output_path)
    summary = parse_cal_summary_file(summary_path) if summary_path else {}
    ref_meta = parse_ref_file_header(hpol_ref_file) if hpol_ref_file else {}

    # Prefer summary-file paths when the caller did not provide them
    power_file = power_file or str(summary.get("power_file", "") or "")
    gain_std_file = gain_std_file or str(summary.get("gain_std_file", "") or "")
    hpol_ref_file = hpol_ref_file or str(summary.get("hpol_file", "") or "")
    vpol_ref_file = vpol_ref_file or str(summary.get("vpol_file", "") or "")

    freq_step = None
    if len(points) > 1:
        diffs = np.diff(points["freq_mhz"].to_numpy())
        freq_step = float(np.median(diffs))

    meta = CalRunMeta(
        run_id=run_id,
        date=_normalize_date(header.get("date", "")),
        time=header.get("time", ""),
        antenna=filename_info["antenna"],
        band_label=filename_info["band_label"],
        start_mhz=float(points["freq_mhz"].min()),
        stop_mhz=float(points["freq_mhz"].max()),
        output_path=output_path,
        output_sha256=output_sha,
        power_file=power_file,
        power_sha256=sha256_file(power_file) if power_file else "",
        gain_std_file=gain_std_file,
        gain_std_sha256=sha256_file(gain_std_file) if gain_std_file else "",
        hpol_ref_file=hpol_ref_file,
        hpol_ref_sha256=sha256_file(hpol_ref_file) if hpol_ref_file else "",
        vpol_ref_file=vpol_ref_file,
        vpol_ref_sha256=sha256_file(vpol_ref_file) if vpol_ref_file else "",
        source_level_dBm=ref_meta.get("output_level_dBm"),  # type: ignore[arg-type]
        signal_source=ref_meta.get("signal_source"),  # type: ignore[arg-type]
        receiver=ref_meta.get("receiver"),  # type: ignore[arg-type]
        freq_step_mhz=freq_step,
        rows_written=int(cal_result.get("rows_written") or len(points)),
        rows_missing=int(cal_result.get("rows_missing") or 0),
        operator_notes=operator_notes,
        ingested_at=_dt.datetime.now().isoformat(timespec="seconds"),
    )

    _write_points_csv(run_id, points, hdir)
    _append_run_row(meta, hdir)
    log.info("record_run: recorded %s (%s %s %s)", run_id, meta.antenna, meta.band_label, meta.date)
    return meta


def update_notes(run_id: str, notes: str) -> bool:
    """Overwrite operator_notes for one run. Returns True if updated."""
    return _update_string_field(run_id, "operator_notes", notes)


def set_setup_group(run_id: str, group: str) -> bool:
    """Assign a free-text setup_group tag to one run.

    The setup_group names a methodology epoch (e.g. "pre-2024-cable-change",
    "2026-v2-mount"). When two runs with different setup_group values are
    compared, compute_drift flags the setup_group field as mismatched on the
    consistency tab — a loud visual signal that the comparison may not be
    apples-to-apples.
    """
    return _update_string_field(run_id, "setup_group", group)


def _update_string_field(run_id: str, field: str, value: str) -> bool:
    path = _runs_csv_path()
    if not path.exists():
        return False
    df = pd.read_csv(path)
    mask = df["run_id"] == run_id
    if not mask.any():
        return False
    if field not in df.columns:
        df[field] = ""
    df[field] = df[field].astype("object")
    df.loc[mask, field] = value
    df.to_csv(path, index=False)
    return True


def delete_run(run_id: str) -> bool:
    """Remove one run (both row + points file). Returns True on success."""
    path = _runs_csv_path()
    if not path.exists():
        return False
    df = pd.read_csv(path)
    mask = df["run_id"] == run_id
    if not mask.any():
        return False
    df = df[~mask].reset_index(drop=True)
    df.to_csv(path, index=False)
    points_path = _points_csv_path(run_id)
    if points_path.exists():
        points_path.unlink()
    return True


# ──────────────────────────────────────────────────────────────────────────
# Historical import
# ──────────────────────────────────────────────────────────────────────────


def import_historical_dir(
    root: str | os.PathLike,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> dict:
    """Walk a directory, ingest every TRP Cal file found.

    Pairs each 'TRP Cal X.txt' with its sibling 'TRP Cal Summary X.txt' when
    present, falls back to filename-pattern scan for ref/gain/power inputs.

    Returns {ingested, skipped_duplicate, failed, run_ids: [...]}.
    """
    root_p = Path(os.path.expanduser(str(root)))
    result = {"ingested": 0, "skipped_duplicate": 0, "failed": 0, "run_ids": []}
    if not root_p.exists():
        return result

    cal_files = [
        p for p in root_p.rglob("TRP Cal *.txt") if not p.name.startswith("TRP Cal Summary")
    ]
    total = len(cal_files)
    for i, cal in enumerate(cal_files, start=1):
        if progress:
            progress(i, total, cal.name)
        summary = _find_matching_summary(cal)
        inputs = _resolve_inputs(cal, summary)
        try:
            meta = record_run(
                cal_result={
                    "output_path": str(cal),
                    "summary_path": str(summary) if summary else "",
                    "rows_written": 0,  # re-derived from points during record_run
                    "rows_missing": 0,
                },
                power_file=inputs.get("power_file", ""),
                gain_std_file=inputs.get("gain_std_file", ""),
                hpol_ref_file=inputs.get("hpol_file", ""),
                vpol_ref_file=inputs.get("vpol_file", ""),
            )
            if meta is None:
                result["skipped_duplicate"] += 1
            else:
                result["ingested"] += 1
                result["run_ids"].append(meta.run_id)
        except Exception as exc:  # pragma: no cover - defensive
            log.exception("import_historical_dir: failed on %s: %s", cal, exc)
            result["failed"] += 1
    return result


def _find_matching_summary(cal_file: Path) -> Optional[Path]:
    """Find the 'TRP Cal Summary *.txt' paired with a TRP Cal file."""
    want_suffix = cal_file.name[len("TRP Cal ") :]
    candidate = cal_file.parent / f"TRP Cal Summary {want_suffix}"
    if candidate.exists():
        return candidate
    # Fall back to loose sibling match
    for p in cal_file.parent.glob("TRP Cal Summary *.txt"):
        if cal_file.stem.split()[2:4] == p.stem.split()[3:5]:
            return p
    return None


def _resolve_inputs(cal_file: Path, summary: Optional[Path]) -> Dict[str, str]:
    """Determine power/gain-std/HPol/VPol input paths for a cal file.

    Prefers the summary file's recorded paths (remapped to local filesystem
    when possible); falls back to filename-pattern scan in the same directory.

    Filename-pattern scan matches ref/gain files to the antenna/band encoded
    in the cal filename (e.g. a 'TRP Cal BLPA 690-2700 ...' cal gets matched
    to the 'BLPA 690-2700 AP_*Pol.txt' ref files and the 'Howland BLPA*' gain
    standard — not the HORN gain file that may sit in the same directory).
    """
    resolved: Dict[str, str] = {
        "power_file": "",
        "gain_std_file": "",
        "hpol_file": "",
        "vpol_file": "",
    }
    parent = cal_file.parent
    filename_info = _parse_cal_filename(cal_file)
    ant = filename_info["antenna"].upper()
    band = filename_info["band_label"]  # e.g. "690-2700"

    if summary is not None:
        s = parse_cal_summary_file(summary)
        for src_key, dst_key in (
            ("power_file", "power_file"),
            ("gain_std_file", "gain_std_file"),
            ("hpol_file", "hpol_file"),
            ("vpol_file", "vpol_file"),
        ):
            ref = str(s.get(src_key, "") or "")
            if ref:
                # Summary paths are from a different machine (e.g. Windows);
                # re-map by basename in the same directory as the cal file.
                local = parent / Path(ref.replace("\\", "/")).name
                resolved[dst_key] = str(local) if local.exists() else ""

    # Fallback scan — match by antenna/band to avoid grabbing the wrong pair
    # when BLPA and HORN cal files coexist in the same directory.
    def _band_match(name: str) -> bool:
        return bool(band) and band in name

    def _ant_match(name: str) -> bool:
        return bool(ant) and ant in name.upper()

    if not resolved["hpol_file"] or not resolved["vpol_file"]:
        for p in parent.iterdir():
            n = p.name
            low = n.lower()
            if not low.endswith(".txt"):
                continue
            matches_ant_band = _ant_match(n) and _band_match(n)
            if low.endswith("_hpol.txt") and matches_ant_band and not resolved["hpol_file"]:
                resolved["hpol_file"] = str(p)
            elif low.endswith("_vpol.txt"):
                if n.startswith("P_") and _band_match(n) and not resolved["power_file"]:
                    resolved["power_file"] = str(p)
                elif matches_ant_band and not resolved["vpol_file"]:
                    resolved["vpol_file"] = str(p)
    if not resolved["gain_std_file"]:
        # Prefer a gain standard whose filename mentions this cal's antenna
        # ("Howland BLPA-19 ..." for BLPA cals, "Howland HORN ..." for HORN).
        best = None
        for p in parent.glob("*Gain*.txt"):
            if _ant_match(p.name):
                best = p
                break
            if best is None:  # keep first as last-resort fallback
                best = p
        if best is not None:
            resolved["gain_std_file"] = str(best)
    return resolved


# ──────────────────────────────────────────────────────────────────────────
# Drift compute
# ──────────────────────────────────────────────────────────────────────────


_CONSISTENCY_FIELDS = [
    "setup_group",
    "antenna",
    "band_label",
    "gain_std_file",
    "gain_std_sha256",
    "source_level_dBm",
    "signal_source",
    "receiver",
    "freq_step_mhz",
]


def _consistency(baseline: CalRunMeta, current: CalRunMeta) -> dict:
    out: dict = {}
    for name in _CONSISTENCY_FIELDS:
        a = getattr(baseline, name)
        b = getattr(current, name)
        if a in (None, "", 0) and b in (None, "", 0):
            state = "unknown"
        elif a in (None, "") or b in (None, ""):
            state = "unknown"
        else:
            state = "match" if a == b else "mismatch"
        # Special-case filename vs sha: if filenames differ but sha matches, call it match (renamed file)
        if name == "gain_std_file" and state == "mismatch":
            if baseline.gain_std_sha256 and baseline.gain_std_sha256 == current.gain_std_sha256:
                state = "match"
        out[name] = {"state": state, "baseline": a, "current": b}
    return out


def _stats_for_pol(d: np.ndarray) -> dict:
    if d.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "max_abs": 0.0, "pct_gt_0_5": 0.0, "pct_gt_1": 0.0}
    return {
        "n": int(d.size),
        "mean": float(np.mean(d)),
        "std": float(np.std(d)),
        "max_abs": float(np.max(np.abs(d))),
        "pct_gt_0_5": float(np.mean(np.abs(d) > _OUTLIER_DB) * 100.0),
        "pct_gt_1": float(np.mean(np.abs(d) > _HARD_OUTLIER_DB) * 100.0),
    }


def compute_drift(baseline_run_id: str, current_run_id: str) -> DriftResult:
    """Outer-join baseline + current on frequency; return deltas + stats + consistency."""
    base_meta = get_run(baseline_run_id)
    cur_meta = get_run(current_run_id)
    if base_meta is None or cur_meta is None:
        raise ValueError(f"Unknown run_id(s): {baseline_run_id}, {current_run_id}")

    base = load_points(baseline_run_id).rename(columns={"hpol_dbm": "base_h", "vpol_dbm": "base_v"})
    cur = load_points(current_run_id).rename(columns={"hpol_dbm": "cur_h", "vpol_dbm": "cur_v"})

    merged = (
        pd.merge(base, cur, on="freq_mhz", how="outer")
        .sort_values("freq_mhz")
        .reset_index(drop=True)
    )
    merged["d_h"] = merged["cur_h"] - merged["base_h"]
    merged["d_v"] = merged["cur_v"] - merged["base_v"]
    merged["outlier_h"] = merged["d_h"].abs() > _OUTLIER_DB
    merged["outlier_v"] = merged["d_v"].abs() > _OUTLIER_DB

    both = merged.dropna(subset=["base_h", "cur_h", "base_v", "cur_v"])
    stats = {
        "H": _stats_for_pol(both["d_h"].to_numpy()),
        "V": _stats_for_pol(both["d_v"].to_numpy()),
        "outlier_threshold_dB": _OUTLIER_DB,
        "hard_outlier_threshold_dB": _HARD_OUTLIER_DB,
    }

    base_freqs = set(base["freq_mhz"].tolist())
    cur_freqs = set(cur["freq_mhz"].tolist())
    missing_audit = {
        "appeared_in_current": sorted(cur_freqs - base_freqs),
        "disappeared_in_current": sorted(base_freqs - cur_freqs),
    }

    return DriftResult(
        deltas=merged,
        stats=stats,
        consistency=_consistency(base_meta, cur_meta),
        missing_audit=missing_audit,
        baseline=base_meta,
        current=cur_meta,
    )


# ──────────────────────────────────────────────────────────────────────────
# Plotting + reporting
# ──────────────────────────────────────────────────────────────────────────


def render_delta_plot(result: DriftResult, out_path: Optional[str | os.PathLike] = None):
    """Render a two-panel freq-vs-ΔdB plot. Returns the matplotlib Figure.

    If out_path is given, also saves as PNG (or PDF if the suffix says so).
    """
    import matplotlib

    if out_path is not None:
        matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    fig, (ax_h, ax_v) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    df = result.deltas

    for ax, pol, delta_col in ((ax_h, "H-Pol", "d_h"), (ax_v, "V-Pol", "d_v")):
        sub = df.dropna(subset=[delta_col])
        freq = sub["freq_mhz"].to_numpy()
        d = sub[delta_col].to_numpy()
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        ax.axhspan(-_OUTLIER_DB, _OUTLIER_DB, color="#c8e6c9", alpha=0.5, zorder=0)
        ax.axhspan(-_HARD_OUTLIER_DB, -_OUTLIER_DB, color="#ffe0b2", alpha=0.5, zorder=0)
        ax.axhspan(_OUTLIER_DB, _HARD_OUTLIER_DB, color="#ffe0b2", alpha=0.5, zorder=0)
        ax.plot(freq, d, color="#1976d2", linewidth=1.0)
        mask = np.abs(d) > _OUTLIER_DB
        if mask.any():
            ax.scatter(
                freq[mask],
                d[mask],
                color="#d32f2f",
                s=18,
                zorder=3,
                label=f"|Δ| > {_OUTLIER_DB} dB",
            )
            ax.legend(loc="upper right", fontsize=8)
        ax.set_ylabel(f"{pol} Δ (dB)")
        ax.grid(True, alpha=0.3)

    ax_v.set_xlabel("Frequency (MHz)")
    fig.suptitle(
        f"Calibration drift: {result.baseline.antenna} {result.baseline.band_label}  "
        f"{result.baseline.date} → {result.current.date}"
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150)
    return fig


def export_markdown(result: DriftResult, out_path: str | os.PathLike) -> None:
    """Write a Markdown summary of a DriftResult to out_path."""
    b = result.baseline
    c = result.current
    lines = [
        f"# Calibration Drift: {b.antenna} {b.band_label}",
        "",
        f"- **Baseline:** {b.date} ({b.run_id}) — {b.rows_written} rows",
        f"- **Current:** {c.date} ({c.run_id}) — {c.rows_written} rows",
        "",
        "## Summary stats",
        "",
        "| Pol | n | mean (dB) | std (dB) | max \\|Δ\\| | % > 0.5 dB | % > 1.0 dB |",
        "|-----|----|-----------|----------|-----------|------------|------------|",
    ]
    for pol in ("H", "V"):
        s = result.stats[pol]
        lines.append(
            f"| {pol} | {s['n']} | {s['mean']:.3f} | {s['std']:.3f} | "
            f"{s['max_abs']:.3f} | {s['pct_gt_0_5']:.1f} | {s['pct_gt_1']:.1f} |"
        )

    lines += [
        "",
        "## Method consistency",
        "",
        "| Field | State | Baseline | Current |",
        "|-------|-------|----------|---------|",
    ]
    for k, v in result.consistency.items():
        lines.append(f"| {k} | {v['state']} | {v['baseline']} | {v['current']} |")

    lines += ["", "## Frequency audit", ""]
    audit = result.missing_audit
    lines.append(f"- Appeared in current (not in baseline): {len(audit['appeared_in_current'])}")
    lines.append(
        f"- Disappeared in current (missing vs baseline): {len(audit['disappeared_in_current'])}"
    )

    # Worst-offender list
    df = result.deltas.dropna(subset=["d_h", "d_v"]).copy()
    if not df.empty:
        df["max_abs_d"] = np.maximum(df["d_h"].abs(), df["d_v"].abs())
        worst = df.sort_values("max_abs_d", ascending=False).head(10)
        lines += [
            "",
            "## Worst 10 points by |Δ|",
            "",
            "| Freq (MHz) | Δ H (dB) | Δ V (dB) |",
            "|------------|----------|----------|",
        ]
        for _, row in worst.iterrows():
            lines.append(f"| {row['freq_mhz']:.1f} | {row['d_h']:.3f} | {row['d_v']:.3f} |")

    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_pdf(result: DriftResult, out_path: str | os.PathLike) -> None:
    """Write a one-page PDF report via matplotlib (no reportlab dep)."""
    from matplotlib.backends.backend_pdf import PdfPages

    fig = render_delta_plot(result)
    with PdfPages(out_path) as pdf:
        pdf.savefig(fig)
        # Second page: stats + consistency as a text-only page.
        import matplotlib.pyplot as plt

        fig2 = plt.figure(figsize=(10, 7))
        ax = fig2.add_subplot(111)
        ax.axis("off")
        b = result.baseline
        c = result.current
        text = [
            f"Calibration Drift Report — {b.antenna} {b.band_label}",
            f"Baseline: {b.date} ({b.run_id})",
            f"Current:  {c.date} ({c.run_id})",
            "",
            "Summary statistics",
            "------------------",
        ]
        for pol in ("H", "V"):
            s = result.stats[pol]
            text.append(
                f"  {pol}: n={s['n']}, mean={s['mean']:.3f} dB, std={s['std']:.3f} dB, "
                f"max|Δ|={s['max_abs']:.3f} dB, >0.5dB={s['pct_gt_0_5']:.1f}%, "
                f">1dB={s['pct_gt_1']:.1f}%"
            )
        text.append("")
        text.append("Method consistency")
        text.append("------------------")
        for k, v in result.consistency.items():
            text.append(f"  {k}: {v['state']}  (base={v['baseline']!r}, cur={v['current']!r})")
        ax.text(
            0.01,
            0.99,
            "\n".join(text),
            family="monospace",
            fontsize=9,
            va="top",
            ha="left",
            transform=ax.transAxes,
        )
        pdf.savefig(fig2)


def result_to_dict(result: DriftResult, max_delta_rows: int = 500) -> dict:
    """Serialize a DriftResult to a JSON-safe dict (for MCP transport)."""
    deltas = result.deltas
    truncated = False
    if len(deltas) > max_delta_rows:
        deltas = deltas.head(max_delta_rows)
        truncated = True
    return {
        "baseline": asdict(result.baseline),
        "current": asdict(result.current),
        "stats": result.stats,
        "consistency": result.consistency,
        "missing_audit": result.missing_audit,
        "deltas": deltas.to_dict(orient="records"),
        "deltas_truncated": truncated,
    }
