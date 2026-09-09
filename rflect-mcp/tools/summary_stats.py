"""Deterministic active-measurement summaries without filename inference."""

import math
from collections import defaultdict


def active_summary(measurements, frequencies=None, dut_ids=None):
    groups = defaultdict(list)
    dut_labels = {}
    missing = 0
    for index, measurement in enumerate(measurements):
        if len(measurement.frequencies) != 1:
            missing += 1
            continue
        frequency = measurement.frequencies[0]
        if not isinstance(frequency, (int, float)) or not math.isfinite(frequency):
            missing += 1
            continue
        if frequencies is not None and frequency not in frequencies:
            continue
        value = measurement.data.get("TRP_dBm")
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            missing += 1
            continue
        dut = dut_ids[index] if dut_ids is not None else measurement.data.get("dut_id")
        if dut is None:
            label = "Unassigned DUT"
        else:
            key = str(dut)
            label = dut_labels.setdefault(key, f"DUT {len(dut_labels) + 1}")
        groups[(label, frequency)].append(float(value))
    paragraphs = []
    for (label, frequency), values in sorted(groups.items()):
        low, high = min(values), max(values)
        paragraphs.append(
            f"{label} at {frequency:g} MHz: {len(values)} measurement(s), "
            f"TRP {low:.2f} to {high:.2f} dBm, spread {high - low:.2f} dB."
        )
    if missing:
        paragraphs.append(
            f"{missing} measurement(s) excluded from TRP statistics because frequency or TRP data is missing or invalid."
        )
    if groups:
        paragraphs.append(
            "These ranges describe the loaded samples. They do not establish measurement "
            "uncertainty, statistical significance, antenna efficiency, or compliance. "
            "Compare against the applicable test limits and calibrated uncertainty budget. "
            "DUT groups use explicit metadata; filenames are not used to infer device identity."
        )
    return paragraphs
