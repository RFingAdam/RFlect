"""Deterministic active-measurement summaries without filename inference."""

import math
from collections import defaultdict


def active_summary(measurements, frequencies=None, dut_ids=None):
    if dut_ids is not None and len(dut_ids) != len(measurements):
        raise ValueError("Provide one DUT group for each measurement")

    raw_dut_ids = [
        dut_ids[index] if dut_ids is not None else measurement.data.get("dut_id")
        for index, measurement in enumerate(measurements)
    ]
    sorted_dut_ids = sorted({str(dut) for dut in raw_dut_ids if dut is not None})
    dut_labels = {dut: f"DUT {index + 1}" for index, dut in enumerate(sorted_dut_ids)}

    groups = defaultdict(list)
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
        dut = raw_dut_ids[index]
        if dut is None:
            label = "Unassigned DUT"
        else:
            label = dut_labels[str(dut)]
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
