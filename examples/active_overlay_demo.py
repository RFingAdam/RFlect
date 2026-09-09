"""Export a three-trace comparison using synthetic data, not customer captures."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rflect-mcp"))
from tools.active_comparison_tools import compare_active_overlay
from tools.import_tools import LoadedMeasurement, _loaded_measurements


def main(output_path: str) -> None:
    theta = np.arange(0, 181, 15)
    phi = np.arange(0, 361, 15)
    for index in range(3):
        power = (
            -20
            + 8 * np.sin(np.deg2rad(theta[:, None])) * np.cos(np.deg2rad(phi[None, :] - 30 * index))
            + 2 * index
        )
        _loaded_measurements[f"synthetic-{index}"] = LoadedMeasurement(
            "",
            "active",
            [2450.0],
            {"theta": theta, "phi": phi, "total_power_2d": power},
        )
    compare_active_overlay(
        list(_loaded_measurements),
        output_path,
        max_envelope=True,
        labels=["Synthetic A", "Synthetic B", "Synthetic C"],
    )


if __name__ == "__main__":
    main(sys.argv[1])
